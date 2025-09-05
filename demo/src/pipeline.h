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
#endif

    // Stage 3 产物
    QueryResult final_result;

    // 元信息
    size_t dim{0};
    size_t top_k{0};

    // 同步
    std::promise<QueryResult> promise;
};

// 线程安全队列（带关闭）
template <typename T>
class ThreadSafeQueue {
public:
    void push(T v) {
        {
            std::lock_guard<std::mutex> g(m_);
            if (closed_) return;
            q_.emplace_back(std::move(v));
        }
        cv_.notify_one();
    }

    bool try_pop(T& out) {
        std::lock_guard<std::mutex> g(m_);
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    bool wait_pop(T& out, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lk(m_);
        if (!cv_.wait_for(lk, timeout, [&]{ return closed_ || !q_.empty(); })) return false;
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> g(m_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    bool empty() const {
        std::lock_guard<std::mutex> g(m_);
        return q_.empty();
    }

    bool closed() const {
        std::lock_guard<std::mutex> g(m_);
        return closed_;
    }

private:
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<T> q_;
    bool closed_{false};
};

#ifdef HAS_LIBAIO
class AIOManager {
public:
    explicit AIOManager(size_t max_concurrent_events);
    ~AIOManager();

    bool submit_reads(const std::shared_ptr<QueryContext>& ctx);

    void event_loop(ThreadSafeQueue<std::shared_ptr<QueryContext>>& rerank_queue);

    void request_stop();

private:
    io_context_t ctx_{};
    std::atomic<bool> stop_{false};
    std::atomic<size_t> inflight_{0};
    std::mutex map_mtx_;
    std::unordered_map<void*, std::shared_ptr<QueryContext>> ctx_map_;
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