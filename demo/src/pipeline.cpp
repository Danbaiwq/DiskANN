#include "pipeline.h"
#include <thread>
#include <algorithm>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <cstdlib>
#ifdef __GLIBC__
#include <malloc.h>
#endif


static inline bool getenv_bool_or(const char* k, bool defv) {
    const char* s = std::getenv(k);
    if (!s) return defv;
    std::string v(s);
    for (auto& c : v) c = (char)std::tolower(c);
    return !(v == "0" || v == "false" || v == "off");
}

// 前向声明，供回退路径直接调用
static void execute_rerank(const std::shared_ptr<QueryContext>& ctx, const PipelineEnv& env);

#ifdef HAS_LIBAIO
AIOManager::AIOManager(size_t max_concurrent_events) {
    memset(&ctx_, 0, sizeof(ctx_));
    max_events_ = (max_concurrent_events && max_concurrent_events <= 4096) ? max_concurrent_events : 4096;
    int rc = ::io_setup(max_events_, &ctx_);
    if (rc != 0) {
        ctx_ = nullptr;
        max_events_ = 0;
    }
}

AIOManager::~AIOManager() {
    if (ctx_ != nullptr) {
        ::io_destroy(ctx_);
        ctx_ = nullptr;
    }
}

void AIOManager::request_stop() {
    stop_.store(true, std::memory_order_relaxed);
    // 尝试取消所有在途 I/O，避免 io_getevents 长期阻塞
    if (ctx_ != nullptr) {
        std::lock_guard<std::mutex> g(map_mtx_);
        for (auto& kv : ctx_map_) {
            const std::shared_ptr<QueryContext>& c = kv.second;
            for (auto& iocb_ref : c->iocb_requests) {
                struct io_event ev;
                (void)::io_cancel(ctx_, &iocb_ref, &ev);
            }
        }
        // 不清零 inflight_，由事件回收路径自然递减
    }
}

bool AIOManager::submit_reads(const std::shared_ptr<QueryContext>& ctx_sp) {
    // 保留旧接口，但改为入队，由 I/O 线程统一提交
    enqueue_reads(ctx_sp);
    return true;
}

void AIOManager::event_loop(ThreadSafeQueue<std::shared_ptr<QueryContext>>& rerank_queue) {
    if (ctx_ == nullptr) return;
    const long kBatchRecv = 256;
    const long minRecv = 32;
    const long sleepNs = 20000000; // 20ms

    // 绑核（可选，忽略错误）
#ifdef __linux__
    cpu_set_t set; CPU_ZERO(&set);
    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu > 0) CPU_SET((int)(ncpu - 1), &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#endif

    int idle_ticks = 0;
    while (true) {
        size_t infl = inflight_.load(std::memory_order_relaxed);
        bool stopping = stop_.load(std::memory_order_relaxed);

        // 提交阶段：仅当未到高水位时从提交队列取任务并 io_submit
        size_t hi = (size_t)(max_events_ ? (max_events_ * 9 / 10) : 0);
        size_t lo = (size_t)(max_events_ ? (max_events_ / 2) : 0);
        if (!stopping && (max_events_ == 0 || infl < hi)) {
            // 从 submit_q_ 搬到本地 pending_submit_
            std::shared_ptr<QueryContext> one;
            int moved = 0;
            while (moved < 64 && submit_q_.try_pop(one)) { // 批量取，限 64
                pending_submit_.push_back(one);
                moved++;
            }
            // 尽量多提交，但不超过配额
            while (!pending_submit_.empty()) {
                if (max_events_ && inflight_.load(std::memory_order_relaxed) >= hi) break;
                auto ctx_sp = pending_submit_.front();
                void* key = static_cast<void*>(ctx_sp.get());
                {
                    std::lock_guard<std::mutex> g(map_mtx_);
                    ctx_map_[key] = ctx_sp;
                }
                size_t total = ctx_sp->iocb_pointers.size();
                size_t submitted = 0;
                while (submitted < total) {
                    size_t allowance = max_events_ ? (max_events_ - inflight_.load(std::memory_order_relaxed)) : total - submitted;
                    if (allowance == 0) break;
                    size_t to_submit = std::min(allowance, total - submitted);
                    long n = ::io_submit(ctx_, (long)to_submit, ctx_sp->iocb_pointers.data() + submitted);
                    if (n <= 0) {
                        // 提交失败：移出并完成空结果，避免卡住
                        {
                            std::lock_guard<std::mutex> g(map_mtx_);
                            ctx_map_.erase(key);
                        }
                        ctx_sp->io_requests_submitted = 0;
                        QueryResult out; ctx_sp->final_result = out; ctx_sp->promise.set_value(out);
                        break;
                    }
                    submitted += (size_t)n;
                    inflight_.fetch_add((size_t)n, std::memory_order_relaxed);
                }
                ctx_sp->io_requests_submitted = submitted;
                pending_submit_.pop_front();
            }
        }

        // 退出条件：stop 且 inflight==0 且 submit 队列清空
        if (stopping && inflight_.load(std::memory_order_relaxed) == 0 && pending_submit_.empty() && submit_q_.empty()) {
            break;
        }

        // 收割阶段
        size_t cur_infl = inflight_.load(std::memory_order_relaxed);
        if (cur_infl == 0) {
            struct io_event evs_empty[1];
            struct timespec ts; ts.tv_sec = 0; ts.tv_nsec = sleepNs;
            (void)::io_getevents(ctx_, 0, 1, evs_empty, &ts);
            idle_ticks++;
            continue;
        }
        long want = (long)std::min<size_t>(kBatchRecv, cur_infl);
        long min_nr = stopping ? 0 : std::min<long>(want, minRecv);
        struct io_event evs[kBatchRecv];
        struct timespec ts; ts.tv_sec = 0; ts.tv_nsec = sleepNs;
        long got = ::io_getevents(ctx_, min_nr, want, evs, &ts);
        if (got <= 0) { idle_ticks++; continue; } else idle_ticks = 0;

        for (long j = 0; j < got; ++j) {
            struct iocb* iocb = reinterpret_cast<struct iocb*>(evs[j].obj);
            void* key = reinterpret_cast<void*>(iocb->data);
            std::shared_ptr<QueryContext> ctx_sp;
            {
                std::lock_guard<std::mutex> g(map_mtx_);
                auto it = ctx_map_.find(key);
                if (it != ctx_map_.end()) ctx_sp = it->second;
            }
            if (!ctx_sp) { inflight_.fetch_sub(1, std::memory_order_relaxed); continue; }
            ctx_sp->io_requests_completed.fetch_add(1, std::memory_order_relaxed);
            inflight_.fetch_sub(1, std::memory_order_relaxed);
            if (ctx_sp->io_requests_completed.load(std::memory_order_relaxed) == ctx_sp->io_requests_submitted) {
                {
                    std::lock_guard<std::mutex> g(map_mtx_);
                    ctx_map_.erase(key);
                }
                rerank_queue.push(ctx_sp);
            }
        }
    }
    // 退出兜底：清空 submit 队列并完成空结果
    std::shared_ptr<QueryContext> rest;
    while (submit_q_.try_pop(rest)) { QueryResult out; rest->final_result = out; rest->promise.set_value(out); }
    pending_submit_.clear();
    {
        std::lock_guard<std::mutex> g(map_mtx_);
        ctx_map_.clear();
    }
}
#endif

static void execute_candidate_generation(const std::shared_ptr<QueryContext>& ctx,
                                         ThreadSafeQueue<std::shared_ptr<QueryContext>>& next_queue,
                                         const PipelineEnv& env) {
    std::vector<uint32_t> cands;
    if (env.use_bq) {
        cands = generate_bq_candidates(ctx->original_query, env.medoid_index, env.buckets, env.f, env.k, env.dim, env.bq_graph_threshold);
    } else {
        std::vector<uint32_t> nearest_bucket_ids(env.f);
        env.medoid_index.search(ctx->original_query.data(), env.f, env.f, nearest_bucket_ids.data(), nullptr);
        cands.reserve(env.f * env.k);
        std::unordered_map<uint32_t, char> seen;
        for (uint32_t bucket_id : nearest_bucket_ids) {
            auto bucket_index = env.index_cache.get_non_blocking(bucket_id);
            if (!bucket_index) continue;
            size_t actual_k = std::min(env.k, bucket_index->get_num_points());
            if (actual_k == 0) continue;
            std::vector<uint32_t> result_tags(actual_k);
            std::vector<float> result_dists(actual_k);
            bucket_index->search(ctx->original_query.data(), actual_k, actual_k, result_tags.data(), result_dists.data());
            for (size_t i = 0; i < actual_k; ++i) {
                uint32_t local_idx = result_tags[i];
                if (local_idx < env.buckets[bucket_id].size()) {
                    uint32_t gid = env.buckets[bucket_id][local_idx];
                    if (!seen[gid]) { seen[gid] = 1; cands.push_back(gid); }
                }
            }
        }
    }
    ctx->candidate_ids = std::move(cands);
    next_queue.push(ctx);
}

static void execute_io_prep(const std::shared_ptr<QueryContext>& ctx, const PipelineEnv& env) {
#ifdef HAS_LIBAIO
    // 若候选为空，直接返回空结果（设置 promise）
    if (ctx->candidate_ids.empty()) {
        QueryResult out; ctx->final_result = out; ctx->promise.set_value(out); return;
    }
    // 去重 + 升序（按 gid）
    std::vector<uint32_t>& gids_sorted = ctx->candidate_ids;
    std::sort(gids_sorted.begin(), gids_sorted.end());
    gids_sorted.erase(std::unique(gids_sorted.begin(), gids_sorted.end()), gids_sorted.end());
    if (gids_sorted.empty()) { QueryResult out; ctx->final_result = out; ctx->promise.set_value(out); return; }

    const size_t dim = env.dim;
    const size_t vec_bytes = sizeof(float) * dim;

    // 固定对齐与分段参数
    const size_t bs = 4096;
    const size_t batch_vecs_limit = 256;
    const size_t batch_mb = 32;
    const size_t batch_bytes_limit = batch_mb * 1024ULL * 1024ULL;
    const size_t gap_gids = 8;

    // 计算分段
    struct Segment { off_t aligned_start; size_t inner; size_t aligned_len; uint32_t first_gid; uint32_t last_gid; size_t rows; };
    std::vector<Segment> segs;
    segs.reserve(gids_sorted.size());
    for (size_t sidx = 0; sidx < gids_sorted.size();) {
        uint32_t first_gid = gids_sorted[sidx];
        uint32_t last_gid = first_gid;
        size_t seg_last_index = sidx;
        while (seg_last_index + 1 < gids_sorted.size()) {
            uint32_t next_gid = gids_sorted[seg_last_index + 1];
            if (static_cast<size_t>(next_gid - last_gid) > gap_gids) break;
            size_t proposed_vecs = static_cast<size_t>(next_gid - first_gid + 1);
            size_t proposed_bytes = proposed_vecs * vec_bytes;
            if (proposed_vecs > batch_vecs_limit || proposed_bytes > batch_bytes_limit) break;
            last_gid = next_gid;
            ++seg_last_index;
        }
        size_t range_vecs = static_cast<size_t>(last_gid - first_gid + 1);
        off_t range_start_off = static_cast<off_t>(env.vector_offsets[first_gid]);
        off_t aligned_start = (range_start_off / static_cast<off_t>(bs)) * static_cast<off_t>(bs);
        size_t inner = static_cast<size_t>(range_start_off - aligned_start);
        size_t aligned_len = (inner + range_vecs * vec_bytes + (bs - 1)) & ~(bs - 1);
        segs.push_back(Segment{aligned_start, inner, aligned_len, first_gid, last_gid, range_vecs});
        sidx = seg_last_index + 1;
    }

    if (segs.empty()) { QueryResult out; ctx->final_result = out; ctx->promise.set_value(out); return; }

    ctx->dim = dim;
    ctx->top_k = env.top_k;

    const size_t num_segs = segs.size();
    ctx->iocb_requests.resize(num_segs);
    ctx->aligned_buffers.resize(num_segs);
    ctx->aligned_lengths.resize(num_segs);
    ctx->inner_offsets.resize(num_segs);
    ctx->segment_first_gids.resize(num_segs);
    ctx->segment_last_gids.resize(num_segs);
    ctx->segment_row_starts.resize(num_segs);
    ctx->segment_aligned_starts.resize(num_segs);
    ctx->base_fd_copy = env.base_fd;
    ctx->iocb_pointers.clear();
    ctx->iocb_pointers.reserve(num_segs);

    size_t row_cursor = 0;
    bool alloc_fail = false;
    for (size_t i = 0; i < num_segs; ++i) {
        auto& sg = segs[i];
        ctx->segment_first_gids[i] = sg.first_gid;
        ctx->segment_last_gids[i]  = sg.last_gid;
        ctx->segment_row_starts[i] = row_cursor;
        row_cursor += sg.rows;

        void* big_buf = nullptr;
        if (posix_memalign(&big_buf, bs, sg.aligned_len) != 0 || big_buf == nullptr) {
            alloc_fail = true; break;
        }
        ctx->aligned_buffers[i] = big_buf;
        ctx->aligned_lengths[i] = sg.aligned_len;
        ctx->inner_offsets[i] = sg.inner;
        ctx->segment_aligned_starts[i] = sg.aligned_start;

        memset(&ctx->iocb_requests[i], 0, sizeof(struct iocb));
        io_prep_pread(&ctx->iocb_requests[i], env.base_fd, big_buf, sg.aligned_len, sg.aligned_start);
        ctx->iocb_requests[i].data = ctx.get();
        ctx->iocb_pointers.push_back(&ctx->iocb_requests[i]);
    }

    if (alloc_fail) { QueryResult out; ctx->final_result = out; ctx->promise.set_value(out); return; }

    // 入队，由磁盘线程统一提交
    env.aio_manager.enqueue_reads(ctx);
#else
    QueryResult out; ctx->final_result = out; ctx->promise.set_value(out); return;
#endif
}

static void execute_rerank(const std::shared_ptr<QueryContext>& ctx, const PipelineEnv& env) {
    QueryResult out;
    const size_t dim = ctx->dim;
    const bool streaming = true;

    if (streaming && !ctx->aligned_buffers.empty()) {
        const size_t vec_bytes = sizeof(float) * dim;
        std::vector<std::pair<float, uint32_t>> pairs;
        size_t rows_total = 0; for (size_t i = 0; i < ctx->segment_first_gids.size(); ++i) rows_total += (ctx->segment_last_gids[i] - ctx->segment_first_gids[i] + 1);
        pairs.reserve(rows_total);
        for (size_t i = 0; i < ctx->aligned_buffers.size(); ++i) {
            void* big_buf = ctx->aligned_buffers[i]; if (!big_buf) continue;
            size_t inner = (i < ctx->inner_offsets.size()) ? ctx->inner_offsets[i] : 0;
            uint32_t first_gid = ctx->segment_first_gids[i];
            uint32_t last_gid  = ctx->segment_last_gids[i];
            const char* base_ptr = static_cast<const char*>(big_buf) + inner;
            for (uint32_t gid = first_gid; gid <= last_gid; ++gid) {
                size_t rel = static_cast<size_t>(gid - first_gid);
                const float* vptr = reinterpret_cast<const float*>(base_ptr + rel * vec_bytes);
                float dist = l2_distance_simd(vptr, ctx->original_query.data(), dim);
                pairs.emplace_back(dist, gid);
            }
        }
        for (void*& p : ctx->aligned_buffers) { if (p) { free(p); p = nullptr; } }
        if (!pairs.empty()) {
            size_t want = std::min(ctx->top_k, pairs.size());
            std::nth_element(pairs.begin(), pairs.begin() + want, pairs.end());
            pairs.resize(want); std::sort(pairs.begin(), pairs.end());
            out.ids.reserve(want); out.distances.reserve(want);
            for (size_t i = 0; i < want; ++i) { out.distances.push_back(pairs[i].first); out.ids.push_back(pairs[i].second); }
        }
        ctx->final_result = out; ctx->promise.set_value(out);
        return;
    }

    if (dim == 0 || ctx->full_precision_vectors.empty()) {
        ctx->promise.set_value(out); return;
    }
    const float* q = ctx->original_query.data();
    size_t rows = ctx->full_precision_vectors.size() / dim;
    std::vector<std::pair<float, uint32_t>> pairs; pairs.reserve(rows);
    for (size_t i = 0; i < rows; ++i) {
        const float* v = ctx->full_precision_vectors.data() + i * dim;
        float d = l2_distance_simd(q, v, dim);
        uint32_t gid = (i < ctx->read_gid_order.size()) ? ctx->read_gid_order[i] : (uint32_t)i;
        pairs.emplace_back(d, gid);
    }
    const size_t want = std::min(ctx->top_k, pairs.size());
    if (pairs.size() > want) { std::nth_element(pairs.begin(), pairs.begin() + want, pairs.end()); pairs.resize(want); }
    std::sort(pairs.begin(), pairs.end());
    out.ids.reserve(want); out.distances.reserve(want);
    for (size_t i = 0; i < want; ++i) { out.distances.push_back(pairs[i].first); out.ids.push_back(pairs[i].second); }
    ctx->final_result = out; ctx->promise.set_value(out);
}

static void shared_worker_loop(ThreadSafeQueue<std::shared_ptr<QueryContext>>& stage1,
                               ThreadSafeQueue<std::shared_ptr<QueryContext>>& stage2,
                               ThreadSafeQueue<std::shared_ptr<QueryContext>>& stage3,
                               const PipelineEnv& env,
                               std::atomic<bool>& stop_flag) {
    using namespace std::chrono_literals;
    while (true) {
        if (stop_flag.load(std::memory_order_relaxed) && stage1.empty() && stage2.empty() && stage3.empty()) break;
        std::shared_ptr<QueryContext> ctx;
        for (int spin = 0; spin < 64; ++spin) {
            if (stage3.try_pop(ctx)) { execute_rerank(ctx, env); goto next; }
            if (stage2.try_pop(ctx)) { execute_io_prep(ctx, env); goto next; }
            if (stage1.try_pop(ctx)) { execute_candidate_generation(ctx, stage2, env); goto next; }
        }
        if (stage3.wait_pop(ctx, 2ms)) { execute_rerank(ctx, env); goto next; }
        if (stage2.wait_pop(ctx, 2ms)) { execute_io_prep(ctx, env); goto next; }
        if (stage1.wait_pop(ctx, 2ms)) { execute_candidate_generation(ctx, stage2, env); goto next; }
        next: ;
    }
}

static size_t getenv_size_t_or(const char* k, size_t defv) {
    const char* s = std::getenv(k);
    if (!s) return defv;
    try { return std::stoull(s); } catch (...) { return defv; }
}

void run_pipeline_search(
    const FlatDataSet& queries_flat,
    size_t num_queries,
    const PipelineEnv& env,
    std::vector<QueryResult>& out_results,
    ThreadSafeQueue<std::shared_ptr<QueryContext>>*& out_stage1,
    ThreadSafeQueue<std::shared_ptr<QueryContext>>*& out_stage2,
    ThreadSafeQueue<std::shared_ptr<QueryContext>>*& out_stage3
) {
    auto* stage1 = new ThreadSafeQueue<std::shared_ptr<QueryContext>>(8192);
    auto* stage2 = new ThreadSafeQueue<std::shared_ptr<QueryContext>>(8192);
    auto* stage3 = new ThreadSafeQueue<std::shared_ptr<QueryContext>>(8192);

    out_stage1 = stage1; out_stage2 = stage2; out_stage3 = stage3;

    std::atomic<bool> stop_flag{false};

#ifdef HAS_LIBAIO
    std::thread io_thread([&](){ env.aio_manager.event_loop(*stage3); });
#endif

    const size_t hw = std::max(1u, std::thread::hardware_concurrency());
    const size_t num_workers = hw;
    std::vector<std::thread> workers;
    workers.reserve(num_workers);
    for (size_t i = 0; i < num_workers; ++i) {
        workers.emplace_back([&, i]() { shared_worker_loop(*stage1, *stage2, *stage3, env, stop_flag); });
    }

    out_results.resize(num_queries);

    size_t pushed = 0;
    while (pushed < num_queries) {
        const size_t batch = std::min(static_cast<size_t>(hw * 2), num_queries - pushed);
        std::vector<std::future<QueryResult>> futs; futs.reserve(batch);
        for (size_t j = 0; j < batch; ++j) {
            auto ctx = std::make_shared<QueryContext>();
            ctx->query_id = pushed + j;
            ctx->original_query = get_point_copy(queries_flat, pushed + j, env.dim);
            auto fut = ctx->promise.get_future();
            futs.emplace_back(std::move(fut));
            stage1->push(ctx);
        }
        for (size_t j = 0; j < batch; ++j) {
            out_results[pushed + j] = futs[j].get();
        }
#ifdef __GLIBC__
        malloc_trim(0);
#endif
        pushed += batch;
    }

    // 退出顺序：先关队列，再停 worker，再停 AIO
    stage1->close(); stage2->close(); stage3->close();
    stop_flag.store(true, std::memory_order_relaxed);
    for (auto& th : workers) if (th.joinable()) th.join();
#ifdef HAS_LIBAIO
    env.aio_manager.request_stop();
    if (io_thread.joinable()) io_thread.join();
#endif
} 