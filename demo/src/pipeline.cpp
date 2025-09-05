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
    int rc = ::io_setup(max_concurrent_events ? max_concurrent_events : 64, &ctx_);
    if (rc != 0) {
        ctx_ = nullptr;
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
        inflight_.store(0, std::memory_order_relaxed);
    }
}

bool AIOManager::submit_reads(const std::shared_ptr<QueryContext>& ctx_sp) {
    if (ctx_ == nullptr) return false;
    if (ctx_sp->iocb_pointers.empty()) return false;
    void* key = static_cast<void*>(ctx_sp.get());
    {
        std::lock_guard<std::mutex> g(map_mtx_);
        ctx_map_[key] = ctx_sp;
    }

    size_t total = ctx_sp->iocb_pointers.size();
    size_t submitted = 0;
    while (submitted < total) {
        long n = ::io_submit(ctx_, static_cast<long>(total - submitted), ctx_sp->iocb_pointers.data() + submitted);
        if (n <= 0) {
            // 提交失败时，防止事件循环阻塞，立即将 ctx 移出，并返回失败
            std::lock_guard<std::mutex> g(map_mtx_);
            ctx_map_.erase(key);
            ctx_sp->io_requests_submitted = 0;
            return false;
        }
        submitted += static_cast<size_t>(n);
    }
    ctx_sp->io_requests_submitted = submitted;
    inflight_.fetch_add(submitted, std::memory_order_relaxed);
    return true;
}

void AIOManager::event_loop(ThreadSafeQueue<std::shared_ptr<QueryContext>>& rerank_queue) {
    if (ctx_ == nullptr) return;
    const long kBatch = 128;
    while (true) {
        // 收到停止信号即退出（futures 已经收齐）
        if (stop_.load(std::memory_order_relaxed)) {
            break;
        }
        // 若没有在途请求，可以更快退出
        if (inflight_.load(std::memory_order_relaxed) == 0) {
            continue; // 轻量轮询，等待 stop_ 被置位
        }
        struct io_event evs[kBatch];
        struct timespec ts; ts.tv_sec = 0; ts.tv_nsec = 10000000; // 10ms 轮询
        // 等待至少 1 个事件，但不要无限阻塞：设置超时
        long want = kBatch;
        long min_nr = stop_.load(std::memory_order_relaxed) ? 0 : 1;
        long got = ::io_getevents(ctx_, min_nr, want, evs, &ts);
        if (got <= 0) {
            if (stop_.load(std::memory_order_relaxed)) {
                // 停止阶段：不再等待事件，直接退出
                break;
            }
            // 运行阶段：超时重试
            continue;
        }
        for (long j = 0; j < got; ++j) {
            struct iocb* iocb = reinterpret_cast<struct iocb*>(evs[j].obj);
            void* key = reinterpret_cast<void*>(iocb->data);
            std::shared_ptr<QueryContext> ctx_sp;
            {
                std::lock_guard<std::mutex> g(map_mtx_);
                auto it = ctx_map_.find(key);
                if (it != ctx_map_.end()) ctx_sp = it->second;
            }
            if (!ctx_sp) continue;
            ctx_sp->io_requests_completed.fetch_add(1, std::memory_order_relaxed);
            inflight_.fetch_sub(1, std::memory_order_relaxed);
            if (ctx_sp->io_requests_completed.load(std::memory_order_relaxed) == ctx_sp->io_requests_submitted) {
                // 若存在 full_precision_vectors（非流式），将段数据拷贝并释放缓冲
                if (!ctx_sp->full_precision_vectors.empty() && !ctx_sp->aligned_buffers.empty()) {
                    const size_t dim = ctx_sp->dim;
                    const size_t vec_bytes = sizeof(float) * dim;
                    const size_t num_segs = ctx_sp->aligned_buffers.size();
                    for (size_t i = 0; i < num_segs; ++i) {
                        void* big_buf = ctx_sp->aligned_buffers[i];
                        if (!big_buf) continue;
                        size_t inner = (i < ctx_sp->inner_offsets.size()) ? ctx_sp->inner_offsets[i] : 0;
                        uint32_t first_gid = ctx_sp->segment_first_gids[i];
                        uint32_t last_gid  = ctx_sp->segment_last_gids[i];
                        size_t row_start   = ctx_sp->segment_row_starts[i];
                        const char* base_ptr = static_cast<const char*>(big_buf) + inner;
                        for (uint32_t gid = first_gid; gid <= last_gid; ++gid) {
                            size_t rel = static_cast<size_t>(gid - first_gid);
                            float* dst = ctx_sp->full_precision_vectors.data() + (row_start + rel) * dim;
                            const void* src = base_ptr + rel * vec_bytes;
                            std::memcpy(dst, src, vec_bytes);
                        }
                        free(big_buf);
                        ctx_sp->aligned_buffers[i] = nullptr;
                    }
                }
                // 流式场景：不在 I/O 线程触碰 aligned_buffers，交由 Stage3 直接消费
                {
                    std::lock_guard<std::mutex> g(map_mtx_);
                    ctx_map_.erase(key);
                }
                rerank_queue.push(ctx_sp);
            }
        }
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

    // 环境参数（沿用同步路径同名变量）
    auto read_env_size_t = [](const char* name, size_t def_val) -> size_t {
        const char* s = std::getenv(name);
        if (!s) return def_val;
        try { return std::stoull(s); } catch (...) { return def_val; }
    };
    size_t bs = read_env_size_t("RERANK_ALIGN_BS", 4096);
    if (bs == 0 || (bs % 512ULL) != 0) bs = 4096;
    const size_t batch_vecs_limit = read_env_size_t("RERANK_BATCH_VECS", 128);
    const size_t batch_mb = read_env_size_t("RERANK_BATCH_MB", 8);
    const size_t batch_bytes_limit = batch_mb * 1024ULL * 1024ULL;
    const size_t gap_gids = read_env_size_t("RERANK_GAP_GIDS", 8);
    const bool streaming = getenv_bool_or("RERANK_STREAMING", true);

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

    // 分段为空也返回空结果
    if (segs.empty()) { QueryResult out; ctx->final_result = out; ctx->promise.set_value(out); return; }

    // 若预估段总内存超过单查询上限，则直接走流式同步回退
    size_t per_query_cap_mb = read_env_size_t("PIPELINE_PER_QUERY_MB", 64);
    if (per_query_cap_mb == 0) per_query_cap_mb = 64;
    size_t total_bytes = 0; for (auto& sg : segs) total_bytes += sg.aligned_len;
    // 小 I/O 直接同步读取，避免 AIO 尾部悬挂（尤其 f=1/k=100）
    size_t aio_min_bytes = read_env_size_t("PIPELINE_AIO_MIN_BYTES", 8 * 1024ULL * 1024ULL);
    if (streaming && (total_bytes < aio_min_bytes)) {
        std::vector<std::pair<float, uint32_t>> pairs;
        pairs.reserve(gids_sorted.size());
        for (auto& sg : segs) {
            uint32_t first_gid = sg.first_gid; uint32_t last_gid = sg.last_gid;
            off_t aligned_start = sg.aligned_start; size_t inner = sg.inner;
            void* buf = nullptr; if (posix_memalign(&buf, bs, sg.aligned_len) != 0 || buf == nullptr) continue;
            ssize_t br = ::pread(env.base_fd, buf, sg.aligned_len, aligned_start);
            if (br == static_cast<ssize_t>(sg.aligned_len)) {
                const char* base_ptr = static_cast<const char*>(buf) + inner;
                for (uint32_t gid = first_gid; gid <= last_gid; ++gid) {
                    size_t rel = static_cast<size_t>(gid - first_gid);
                    const float* vptr = reinterpret_cast<const float*>(base_ptr + rel * vec_bytes);
                    float dist = l2_distance_simd(vptr, ctx->original_query.data(), dim);
                    pairs.emplace_back(dist, gid);
                }
            }
            free(buf);
        }
        if (!pairs.empty()) {
            size_t want = std::min(env.top_k, pairs.size());
            std::nth_element(pairs.begin(), pairs.begin() + want, pairs.end());
            pairs.resize(want); std::sort(pairs.begin(), pairs.end());
            QueryResult out; out.ids.reserve(want); out.distances.reserve(want);
            for (size_t i = 0; i < want; ++i) { out.distances.push_back(pairs[i].first); out.ids.push_back(pairs[i].second); }
            ctx->final_result = out; ctx->promise.set_value(out);
        } else { QueryResult out; ctx->final_result = out; ctx->promise.set_value(out); }
        return;
    }
    if (streaming && total_bytes > per_query_cap_mb * 1024ULL * 1024ULL) {
        // 直接流式同步读取，避免分配大段 buffer
        std::vector<std::pair<float, uint32_t>> pairs;
        pairs.reserve(gids_sorted.size());
        for (auto& sg : segs) {
            uint32_t first_gid = sg.first_gid; uint32_t last_gid = sg.last_gid;
            off_t aligned_start = sg.aligned_start; size_t inner = sg.inner;
            void* buf = nullptr; if (posix_memalign(&buf, bs, sg.aligned_len) != 0 || buf == nullptr) continue;
            ssize_t br = ::pread(env.base_fd, buf, sg.aligned_len, aligned_start);
            if (br == static_cast<ssize_t>(sg.aligned_len)) {
                const char* base_ptr = static_cast<const char*>(buf) + inner;
                for (uint32_t gid = first_gid; gid <= last_gid; ++gid) {
                    size_t rel = static_cast<size_t>(gid - first_gid);
                    const float* vptr = reinterpret_cast<const float*>(base_ptr + rel * vec_bytes);
                    float dist = l2_distance_simd(vptr, ctx->original_query.data(), dim);
                    pairs.emplace_back(dist, gid);
                }
            }
            free(buf);
        }
        if (!pairs.empty()) {
            size_t want = std::min(env.top_k, pairs.size());
            std::nth_element(pairs.begin(), pairs.begin() + want, pairs.end());
            pairs.resize(want); std::sort(pairs.begin(), pairs.end());
            QueryResult out; out.ids.reserve(want); out.distances.reserve(want);
            for (size_t i = 0; i < want; ++i) { out.distances.push_back(pairs[i].first); out.ids.push_back(pairs[i].second); }
            ctx->final_result = out; ctx->promise.set_value(out);
        } else { QueryResult out; ctx->final_result = out; ctx->promise.set_value(out); }
        return;
    }

    ctx->dim = dim;
    ctx->top_k = env.top_k;

    // 为每段准备 iocb + 对齐缓冲
    const size_t num_segs = segs.size();
    ctx->iocb_requests.resize(num_segs);
    ctx->aligned_buffers.resize(num_segs);
    ctx->aligned_lengths.resize(num_segs);
    ctx->inner_offsets.resize(num_segs);
    ctx->segment_first_gids.resize(num_segs);
    ctx->segment_last_gids.resize(num_segs);
    ctx->segment_row_starts.resize(num_segs); // 仅占位
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
            alloc_fail = true;
            break;
        }
        ctx->aligned_buffers[i] = big_buf;
        ctx->aligned_lengths[i] = sg.aligned_len;
        ctx->inner_offsets[i] = sg.inner;

        memset(&ctx->iocb_requests[i], 0, sizeof(struct iocb));
        io_prep_pread(&ctx->iocb_requests[i], env.base_fd, big_buf, sg.aligned_len, sg.aligned_start);
        ctx->iocb_requests[i].data = ctx.get();
        ctx->iocb_pointers.push_back(&ctx->iocb_requests[i]);
    }

    auto fallback_sync_read = [&]() {
        // 流式回退同步读取并直接复排（不分配全量矩阵）
        std::vector<std::pair<float, uint32_t>> pairs;
        pairs.reserve(row_cursor);
        for (size_t i = 0; i < num_segs; ++i) {
            uint32_t first_gid = ctx->segment_first_gids[i];
            uint32_t last_gid  = ctx->segment_last_gids[i];
            off_t base_off = static_cast<off_t>(env.vector_offsets[first_gid]);
            off_t aligned_start = (base_off / static_cast<off_t>(bs)) * static_cast<off_t>(bs);
            size_t inner = static_cast<size_t>(base_off - aligned_start);
            size_t aligned_len = (inner + (last_gid - first_gid + 1) * vec_bytes + (bs - 1)) & ~(bs - 1);
            void* buf = nullptr; if (posix_memalign(&buf, bs, aligned_len) != 0 || buf == nullptr) continue;
            ssize_t br = ::pread(env.base_fd, buf, aligned_len, aligned_start);
            if (br == static_cast<ssize_t>(aligned_len)) {
                const char* base_ptr = static_cast<const char*>(buf) + inner;
                for (uint32_t gid = first_gid; gid <= last_gid; ++gid) {
                    size_t rel = static_cast<size_t>(gid - first_gid);
                    const float* vptr = reinterpret_cast<const float*>(base_ptr + rel * vec_bytes);
                    float dist = l2_distance_simd(vptr, ctx->original_query.data(), dim);
                    pairs.emplace_back(dist, gid);
                }
            }
            free(buf);
        }
        if (!pairs.empty()) {
            size_t want = std::min(env.top_k, pairs.size());
            std::nth_element(pairs.begin(), pairs.begin() + want, pairs.end());
            pairs.resize(want); std::sort(pairs.begin(), pairs.end());
            QueryResult out; out.ids.reserve(want); out.distances.reserve(want);
            for (size_t i = 0; i < want; ++i) { out.distances.push_back(pairs[i].first); out.ids.push_back(pairs[i].second); }
            ctx->final_result = out; ctx->promise.set_value(out);
        } else {
            QueryResult out; ctx->final_result = out; ctx->promise.set_value(out);
        }
    };

    if (alloc_fail) { fallback_sync_read(); return; }

    if (!env.aio_manager.submit_reads(ctx)) { fallback_sync_read(); return; }
#else
    // 未启用 libaio：直接设置空结果，避免悬挂
    QueryResult out; ctx->final_result = out; ctx->promise.set_value(out); return;
#endif
}

static void execute_rerank(const std::shared_ptr<QueryContext>& ctx, const PipelineEnv& env) {
    QueryResult out;
    const size_t dim = ctx->dim;
    const bool streaming = getenv_bool_or("RERANK_STREAMING", true);

    if (streaming && !ctx->aligned_buffers.empty()) {
        // 直接在段缓冲上计算距离
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
        // 释放段缓冲
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

    // 非 streaming：旧路径（如有 full_precision_vectors）
    if (dim == 0 || ctx->full_precision_vectors.empty()) {
        ctx->promise.set_value(out);
        return;
    }
    const float* q = ctx->original_query.data();
    size_t rows = ctx->full_precision_vectors.size() / dim;
    std::vector<std::pair<float, uint32_t>> pairs; pairs.reserve(rows);
    for (size_t i = 0; i < rows; ++i) {
        const float* v = ctx->full_precision_vectors.data() + i * dim;
        float d = l2_distance_simd(q, v, dim);
        uint32_t gid = (i < ctx->read_gid_order.size()) ? ctx->read_gid_order[i] : static_cast<uint32_t>(i);
        pairs.emplace_back(d, gid);
    }
    const size_t want = std::min(ctx->top_k, pairs.size());
    if (pairs.size() > want) {
        std::nth_element(pairs.begin(), pairs.begin() + want, pairs.end());
        pairs.resize(want);
    }
    std::sort(pairs.begin(), pairs.end());
    out.ids.reserve(want);
    out.distances.reserve(want);
    for (size_t i = 0; i < want; ++i) {
        out.distances.push_back(pairs[i].first);
        out.ids.push_back(pairs[i].second);
    }
    ctx->final_result = out;
    ctx->promise.set_value(out);
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
        if (stage3.wait_pop(ctx, 1ms)) {
            execute_rerank(ctx, env);
            continue;
        }
        if (stage2.wait_pop(ctx, 1ms)) {
            execute_io_prep(ctx, env);
            continue;
        }
        if (stage1.wait_pop(ctx, 1ms)) {
            execute_candidate_generation(ctx, stage2, env);
            continue;
        }
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
    auto* stage1 = new ThreadSafeQueue<std::shared_ptr<QueryContext>>();
    auto* stage2 = new ThreadSafeQueue<std::shared_ptr<QueryContext>>();
    auto* stage3 = new ThreadSafeQueue<std::shared_ptr<QueryContext>>();

    out_stage1 = stage1; out_stage2 = stage2; out_stage3 = stage3;

    std::atomic<bool> stop_flag{false};

#ifdef HAS_LIBAIO
    std::thread io_thread([&](){ env.aio_manager.event_loop(*stage3); });
#endif

    const size_t hw = std::max(1u, std::thread::hardware_concurrency());
    // 限制工作线程数，避免过多线程导致堆栈/缓冲膨胀
    size_t worker_cap = getenv_size_t_or("PIPELINE_WORKERS", hw);
    worker_cap = std::max<size_t>(1, std::min<size_t>(worker_cap, hw));
    const size_t num_workers = worker_cap;
    std::vector<std::thread> workers;
    workers.reserve(num_workers);
    for (size_t i = 0; i < num_workers; ++i) {
        workers.emplace_back([&, i]() { shared_worker_loop(*stage1, *stage2, *stage3, env, stop_flag); });
    }

    out_results.resize(num_queries);

    // 控制最大在途查询数，避免内存线性增长
    size_t max_inflight = getenv_size_t_or("PIPELINE_MAX_INFLIGHT", hw * 2);
    if (max_inflight == 0) max_inflight = hw * 2;
    // 全局内存上限（MB），粗略控制总驻留
    size_t mem_cap_mb = getenv_size_t_or("PIPELINE_MEM_CAP_MB", 500);
    if (mem_cap_mb < 100) mem_cap_mb = 100;

    size_t pushed = 0;
    while (pushed < num_queries) {
        // 依据全局内存上限估算可并发的查询数（粗略：按每查询 64MB 上限）
        size_t per_query_mb = getenv_size_t_or("PIPELINE_PER_QUERY_MB", 64);
        if (per_query_mb == 0) per_query_mb = 64;
        size_t by_mem = std::max<size_t>(1, mem_cap_mb / per_query_mb);
        size_t window = std::min(max_inflight, by_mem);
        const size_t batch = std::min(window, num_queries - pushed);
        std::vector<std::future<QueryResult>> futs; futs.reserve(batch);
        for (size_t j = 0; j < batch; ++j) {
            auto ctx = std::make_shared<QueryContext>();
            ctx->query_id = pushed + j;
            ctx->original_query = get_point_copy(queries_flat, pushed + j, env.dim);
            auto fut = ctx->promise.get_future();
            futs.emplace_back(std::move(fut));
            stage1->push(ctx);
        }
        // 收集本批结果
        for (size_t j = 0; j < batch; ++j) {
            out_results[pushed + j] = futs[j].get();
        }
#ifdef __GLIBC__
        // 尝试归还未使用内存给操作系统，缓解 RSS 持续上升
        malloc_trim(0);
#endif
        pushed += batch;
    }

    stop_flag.store(true, std::memory_order_relaxed);
#ifdef HAS_LIBAIO
    // 提前通知 I/O 线程停止，避免无任务时长时间轮询
    env.aio_manager.request_stop();
#endif
    stage1->close(); stage2->close(); stage3->close();

    for (auto& th : workers) if (th.joinable()) th.join();
#ifdef HAS_LIBAIO
    // 如果 libaio 在 O_DIRECT 下仍未退出，做一次兜底睡眠后 join
    if (io_thread.joinable()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        env.aio_manager.request_stop();
        io_thread.join();
    }
#endif
} 