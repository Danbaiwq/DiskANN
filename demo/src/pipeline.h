#pragma once

#include <vector>
#include <memory>
#include <atomic>
#include <future>
#include <cstdint>
#include <condition_variable>
#include <mutex>
#include <deque>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <new>

#include "utils.h"
#include "search.h"

#ifdef HAS_LIBAIO
extern "C" {
#include <libaio.h>
}
#endif

// 查询上下文，贯穿三阶段
struct QueryContext {
    size_t query_id{0};
    DataPoint original_query; // dim 大小

    // Stage 1 产物
    std::vector<uint32_t> candidate_ids;

    // Stage 2 产物（AIO）
#ifdef HAS_LIBAIO
    std::vector<struct iocb> iocb_requests;     // 每段一个 iocb
    std::vector<struct iocb*> iocb_pointers;    // 指向 iocb 的指针数组
#endif
    std::vector<float> full_precision_vectors;  // rows_total * dim
    std::vector<uint32_t> read_gid_order;       // 与 full_precision_vectors 行对应的全局ID
    size_t io_requests_submitted{0};
    std::atomic<size_t> io_requests_completed{0};
#ifdef HAS_LIBAIO
    // O_DIRECT 对齐读取元数据（按段）
    std::vector<void*> aligned_buffers;          // 段对齐缓冲
    std::vector<size_t> aligned_lengths;         // 段对齐读取长度
    std::vector<size_t> inner_offsets;           // 段内偏移
    // 段范围与落位
    std::vector<uint32_t> segment_first_gids;    // 段首 gid
    std::vector<uint32_t> segment_last_gids;     // 段尾 gid（含）
    std::vector<size_t> segment_row_starts;      // 段在 full_precision_vectors 中的起始行
    // 新增：用于 AIO 线程兜底 pread 的对齐起始偏移与 fd 副本
    std::vector<off_t> segment_aligned_starts;   // 每段对齐起始偏移
    int base_fd_copy{-1};
#endif

    // Stage 3 产物
    QueryResult final_result;

    // 元信息
    size_t dim{0};
    size_t top_k{0};

    // 同步
    std::promise<QueryResult> promise;
};

// 帮助函数：上取整到 2 的幂
static inline size_t roundup_pow2(size_t x) {
    if (x < 2) return 2;
    --x;
    x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16;
#if SIZE_MAX > 0xFFFFFFFFu
    x |= x >> 32;
#endif
    return x + 1;
}

// 无锁有界 MPMC 环形队列（基于 Vyukov 算法），带简易唤醒
// 接口保持与原 ThreadSafeQueue 一致
template <typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t capacity = 8192) {
        capacity_ = roundup_pow2(capacity);
        mask_ = capacity_ - 1;
        // 手动分配 Node 数组，避免 std::vector 在扩容/移动时对 std::atomic 的不兼容
        buffer_ = static_cast<Node*>(::operator new[](capacity_ * sizeof(Node)));
        for (size_t i = 0; i < capacity_; ++i) {
            new (&buffer_[i]) Node();
            buffer_[i].seq.store(i, std::memory_order_relaxed);
        }
        head_.store(0, std::memory_order_relaxed);
        tail_.store(0, std::memory_order_relaxed);
        closed_.store(false, std::memory_order_relaxed);
        avail_.store(0, std::memory_order_relaxed);
    }

    ~ThreadSafeQueue() {
        if (buffer_) {
            for (size_t i = 0; i < capacity_; ++i) {
                buffer_[i].~Node();
            }
            ::operator delete[](buffer_);
            buffer_ = nullptr;
        }
    }

    bool push(T v) {
        if (closed_.load(std::memory_order_acquire)) return false;
        size_t pos;
        Node* node;
        for (;;) {
            pos = tail_.load(std::memory_order_acquire);
            node = &buffer_[pos & mask_];
            size_t seq = node->seq.load(std::memory_order_acquire);
            intptr_t dif = (intptr_t)seq - (intptr_t)pos;
            if (dif == 0) {
                if (tail_.compare_exchange_weak(pos, pos + 1, std::memory_order_acq_rel)) break;
            } else if (dif < 0) {
                // 满，短暂让步
                std::this_thread::yield();
                if (closed_.load(std::memory_order_acquire)) return false;
            } else {
                // 其他生产者推进，重试
                std::this_thread::yield();
            }
        }
        node->value = std::move(v);
        node->seq.store(pos + 1, std::memory_order_release);
        avail_.fetch_add(1, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(cv_mtx_);
            cv_.notify_one();
        }
        return true;
    }

    bool try_pop(T& out) {
        size_t pos;
        Node* node;
        for (;;) {
            pos = head_.load(std::memory_order_acquire);
            node = &buffer_[pos & mask_];
            size_t seq = node->seq.load(std::memory_order_acquire);
            intptr_t dif = (intptr_t)seq - (intptr_t)(pos + 1);
            if (dif == 0) {
                if (head_.compare_exchange_weak(pos, pos + 1, std::memory_order_acq_rel)) break;
            } else if (dif < 0) {
                return false; // 空
            } else {
                std::this_thread::yield();
            }
        }
        out = std::move(node->value);
        node->seq.store(pos + capacity_, std::memory_order_release);
        avail_.fetch_sub(1, std::memory_order_release);
        return true;
    }

    bool wait_pop(T& out, std::chrono::milliseconds timeout) {
        if (try_pop(out)) return true;
        std::unique_lock<std::mutex> lk(cv_mtx_);
        if (!cv_.wait_for(lk, timeout, [&]{ return closed_.load(std::memory_order_acquire) || avail_.load(std::memory_order_acquire) > 0; })) return false;
        if (closed_.load(std::memory_order_acquire) && avail_.load(std::memory_order_acquire) == 0) return false;
        return try_pop(out);
    }

    void close() {
        closed_.store(true, std::memory_order_release);
        std::lock_guard<std::mutex> lk(cv_mtx_);
        cv_.notify_all();
    }

    bool empty() const {
        return avail_.load(std::memory_order_acquire) == 0;
    }

    bool closed() const {
        return closed_.load(std::memory_order_acquire);
    }

private:
    struct Node {
        std::atomic<size_t> seq{};
        T value{};
        Node() = default;
        ~Node() = default;
    };

    size_t capacity_{};
    size_t mask_{};
    Node* buffer_{nullptr};
    std::atomic<size_t> head_{};
    std::atomic<size_t> tail_{};
    std::atomic<bool> closed_{};
    std::atomic<size_t> avail_{};
    std::condition_variable cv_;
    std::mutex cv_mtx_;
};

#ifdef HAS_LIBAIO
class AIOManager {
public:
    explicit AIOManager(size_t max_concurrent_events);
    ~AIOManager();

    bool submit_reads(const std::shared_ptr<QueryContext>& ctx);

    void event_loop(ThreadSafeQueue<std::shared_ptr<QueryContext>>& rerank_queue);

    void request_stop();

    // 新增：仅入队，不在计算线程直接 io_submit
    void enqueue_reads(const std::shared_ptr<QueryContext>& ctx) { submit_q_.push(ctx); }

private:
    io_context_t ctx_{};
    std::atomic<bool> stop_{false};
    std::atomic<size_t> inflight_{0};
    std::mutex map_mtx_;
    std::unordered_map<void*, std::shared_ptr<QueryContext>> ctx_map_;
    size_t max_events_{0};
    ThreadSafeQueue<std::shared_ptr<QueryContext>> submit_q_{};
    std::deque<std::shared_ptr<QueryContext>> pending_submit_;
};
#endif

// 承载流水线执行所需的环境
struct PipelineEnv {
    diskann::Index<float, uint32_t, uint32_t>& medoid_index;
    const Buckets& buckets;
    IndexCache& index_cache;
    size_t dim{0};
    size_t f{0};
    size_t k{0};
    size_t top_k{0};
    bool use_bq{false};
    size_t bq_graph_threshold{0};
    const std::vector<uint64_t>& vector_offsets; // size == base_num
    int base_fd{-1};
#ifdef HAS_LIBAIO
    AIOManager& aio_manager;
#endif
};

// 运行异步流水线并填充结果
void run_pipeline_search(
    const FlatDataSet& queries_flat,
    size_t num_queries,
    const PipelineEnv& env,
    std::vector<QueryResult>& out_results,
    ThreadSafeQueue<std::shared_ptr<QueryContext>>*& out_stage1,
    ThreadSafeQueue<std::shared_ptr<QueryContext>>*& out_stage2,
    ThreadSafeQueue<std::shared_ptr<QueryContext>>*& out_stage3
); 