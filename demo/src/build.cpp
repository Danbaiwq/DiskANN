#include "build.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <array>
#include <omp.h>
#include <queue>

// rabitq deps
#include "/home/danbai.wq/DiskANN/rabitq/rabitqlib/quantization/rabitq.hpp"
#include "/home/danbai.wq/DiskANN/rabitq/rabitqlib/quantization/data_layout.hpp"
#include "/home/danbai.wq/DiskANN/rabitq/rabitqlib/fastscan/fastscan.hpp"
#include "/home/danbai.wq/DiskANN/rabitq/rabitqlib/index/query.hpp"

using namespace rabitqlib;

static void pad_dataset(const DataSet& in, size_t padded_dim, DataSet& out) {
    out = in;
    for (auto& v : out) v.resize(padded_dim, 0.0f);
}

// 评估单点估计距离（简化占位，当前未使用）
static inline float bq_est_distance(
    const uint8_t*, size_t, size_t, size_t,
    const std::vector<uint8_t>&, size_t,
    rabitqlib::SplitBatchQuery<float>&
) {
    return 0.0f;
}

// 使用图上近邻搜索收集候选（窗口C），仅在已插入节点子图上搜索，随后返回按距离升序的候选索引
static void collect_candidates_via_graph(
    const std::vector<uint32_t>& entry_points,
    const std::vector<char>& inserted,
    const std::vector<uint32_t>& adj, size_t degree,
    size_t num,
    const std::vector<uint8_t>& packed_codes, size_t code_cols,
    size_t padded_dim, size_t bq_bits,
    const std::vector<uint8_t>& ex_blob,
    const std::vector<float>& f_add, const std::vector<float>& f_rescale,
    const std::vector<float>& q_pad_u,
    size_t window_C,
    std::vector<std::pair<float,uint32_t>>& out_cand
) {
    size_t ex_bits = bq_bits > 1 ? (bq_bits - 1) : 0;
    rabitqlib::SplitBatchQuery<float> q(q_pad_u.data(), padded_dim, ex_bits, rabitqlib::METRIC_L2, false);
    float qnorm2 = 0.0f; for (size_t d = 0; d < padded_dim; ++d) qnorm2 += q_pad_u[d] * q_pad_u[d];
    float qnorm = std::sqrt(qnorm2);
    q.set_g_add(qnorm, 0.0f);

    using Node = std::pair<float, uint32_t>;
    auto cmp = [](const Node& a, const Node& b){ return a.first > b.first; };
    std::priority_queue<Node, std::vector<Node>, decltype(cmp)> pq(cmp);
    std::vector<char> visited(num, 0);

    auto eval_full_dist = [&](uint32_t idx)->float {
        size_t base = (idx / 32) * 32; size_t pos = idx % 32;
        const uint8_t* codes = packed_codes.data() + (base / 32) * (code_cols * 32);
        std::array<uint16_t, rabitqlib::fastscan::kBatchSize> accu{};
        rabitqlib::fastscan::accumulate(codes, q.lut(), accu.data(), padded_dim);
        float ip_est = q.delta() * static_cast<float>(accu[pos]) + q.sum_vl_lut();
        float dist = f_add[idx] + q.g_add() + f_rescale[idx] * (ip_est + q.k1xsumq());
        if (bq_bits > 1 && !ex_blob.empty()) {
            size_t ex_stride = ExDataMap<float>::data_bytes(padded_dim, ex_bits);
            const char* ex_ptr = reinterpret_cast<const char*>(ex_blob.data()) + (idx * ex_stride);
            auto ip_func = select_excode_ipfunc(ex_bits);
            ConstExDataMap<float> ex_map(ex_ptr, padded_dim, ex_bits);
            float ex_dist = ex_map.f_add_ex() + q.g_add() + (ex_map.f_rescale_ex() *
                (static_cast<float>(1 << ex_bits) * (ip_est) + ip_func(q_pad_u.data(), ex_map.ex_code(), padded_dim) + q.kbxsumq()));
            dist = ex_dist;
        }
        return dist;
    };

    for (auto ep : entry_points) if (ep < num && inserted[ep]) pq.emplace(eval_full_dist(ep), ep);

    while (!pq.empty() && out_cand.size() < window_C) {
        auto [d, x] = pq.top(); pq.pop();
        if (visited[x]) continue;
        visited[x] = 1;
        out_cand.emplace_back(d, x);
        if (!inserted[x]) continue;
        // 扩展邻居（仅在已插入子图上）
        for (size_t t = 0; t < degree && out_cand.size() < window_C; ++t) {
            uint32_t v = adj[x * degree + t];
            if (v >= num || visited[v] || !inserted[v]) continue;
            pq.emplace(eval_full_dist(v), v);
        }
    }
    std::sort(out_cand.begin(), out_cand.end());
}

void build_small_bucket_bqbin(
    size_t bucket_id,
    const DataSet& bucket_data,
    const std::vector<float>& centroid,
    size_t padded_dim,
    size_t bq_bits,
    const std::string& out_bin_path
) {
    DataSet bucket_padded;
    if (padded_dim != bucket_data[0].size()) {
        pad_dataset(bucket_data, padded_dim, bucket_padded);
    } else {
        bucket_padded = bucket_data;
    }
    const size_t num = bucket_padded.size();

    std::ofstream out(out_bin_path, std::ios::binary);
    if (!out.is_open()) return;

    uint64_t pd = padded_dim, bits = bq_bits, n = num;
    out.write(reinterpret_cast<char*>(&pd), sizeof(uint64_t));
    out.write(reinterpret_cast<char*>(&bits), sizeof(uint64_t));
    out.write(reinterpret_cast<char*>(&n), sizeof(uint64_t));

    quant::RabitqConfig cfg = quant::faster_config(padded_dim, bq_bits);

    std::vector<uint8_t> packed_codes((((num + 31) & ~31ULL) / 32) * (padded_dim / 8) * 32);
    std::vector<float> f_add(num), f_rescale(num), f_err(num);

    std::vector<float> flat_data(num * padded_dim, 0.0f);
    for (size_t r = 0; r < num; ++r) {
        std::copy(bucket_padded[r].begin(), bucket_padded[r].end(), flat_data.begin() + r * padded_dim);
    }
    std::vector<float> centroid_pad(padded_dim, 0.0f);
    std::copy(centroid.begin(), centroid.end(), centroid_pad.begin());

    quant::rabitq_impl::one_bit::one_bit_batch_code<float, false>(
        flat_data.data(), centroid_pad.data(), num, padded_dim,
        packed_codes.data(), f_add.data(), f_rescale.data(), f_err.data(), rabitqlib::METRIC_L2);

    out.write(reinterpret_cast<const char*>(packed_codes.data()), packed_codes.size());
    out.write(reinterpret_cast<const char*>(f_add.data()), sizeof(float) * num);
    out.write(reinterpret_cast<const char*>(f_rescale.data()), sizeof(float) * num);

    if (bq_bits > 1) {
        size_t ex_bits = bq_bits - 1;
        size_t ex_stride = ExDataMap<float>::data_bytes(padded_dim, ex_bits);
        std::vector<uint8_t> ex_blob(ex_stride * num);
        char* ex_ptr = reinterpret_cast<char*>(ex_blob.data());
        for (size_t r = 0; r < num; ++r) {
            quant::quantize_compact_ex_bits<float>(
                flat_data.data() + r * padded_dim, centroid_pad.data(), padded_dim,
                ex_bits, ex_ptr, rabitqlib::METRIC_L2, cfg);
            ex_ptr += ex_stride;
        }
        out.write(reinterpret_cast<const char*>(ex_blob.data()), ex_blob.size());
    }
    out.close();
}

void build_large_bucket_bqgraph(
    size_t bucket_id,
    const DataSet& bucket_data,
    const std::vector<float>& centroid,
    const BQBuildConfig& cfg,
    const std::string& out_graph_path
) {
    const size_t padded_dim = cfg.padded_dim;
    const size_t bq_bits = cfg.bq_bits;
    const size_t graph_degree = cfg.graph_degree;
    const size_t window_C = std::max(cfg.build_complexity, graph_degree);
    const float alpha_prune = cfg.alpha_prune;

    DataSet bucket_padded;
    if (padded_dim != bucket_data[0].size()) {
        pad_dataset(bucket_data, padded_dim, bucket_padded);
    } else {
        bucket_padded = bucket_data;
    }
    const size_t num = bucket_padded.size();

    // 量化：packed codes + factors + 可选ex
    std::vector<uint8_t> packed_codes((((num + 31) & ~31ULL) / 32) * (padded_dim / 8) * 32);
    std::vector<float> f_add(num), f_rescale(num), f_err(num);

    std::vector<float> flat_data(num * padded_dim, 0.0f);
    for (size_t r = 0; r < num; ++r) {
        std::copy(bucket_padded[r].begin(), bucket_padded[r].end(), flat_data.begin() + r * padded_dim);
    }
    std::vector<float> centroid_pad(padded_dim, 0.0f);
    std::copy(centroid.begin(), centroid.end(), centroid_pad.begin());

    quant::RabitqConfig qcfg = quant::faster_config(padded_dim, bq_bits);
    quant::rabitq_impl::one_bit::one_bit_batch_code<float, false>(
        flat_data.data(), centroid_pad.data(), num, padded_dim,
        packed_codes.data(), f_add.data(), f_rescale.data(), f_err.data(), rabitqlib::METRIC_L2);

    std::vector<uint8_t> ex_blob;
    if (bq_bits > 1) {
        size_t ex_bits = bq_bits - 1;
        size_t ex_stride = ExDataMap<float>::data_bytes(padded_dim, ex_bits);
        ex_blob.resize(ex_stride * num);
        char* ex_ptr = reinterpret_cast<char*>(ex_blob.data());
        for (size_t r = 0; r < num; ++r) {
            quant::quantize_compact_ex_bits<float>(
                flat_data.data() + r * padded_dim, centroid_pad.data(), padded_dim,
                ex_bits, ex_ptr, rabitqlib::METRIC_L2, qcfg);
            ex_ptr += ex_stride;
        }
    }

    // 生成按“距质心距离升序”的插入顺序
    std::vector<std::pair<float,uint32_t>> dist_idx; dist_idx.reserve(num);
    for (uint32_t i = 0; i < num; ++i) {
        float d2 = 0.0f; for (size_t d = 0; d < padded_dim; ++d) {
            float diff = bucket_padded[i][d] - centroid_pad[d]; d2 += diff * diff;
        }
        dist_idx.emplace_back(d2, i);
    }
    std::sort(dist_idx.begin(), dist_idx.end());

    // 初始化邻接与插入标记
    std::vector<uint32_t> adj(num * graph_degree, 0);
    std::vector<char> inserted(num, 0);

    // 入口点集合（稳定集）
    std::vector<uint32_t> stable_entries;

    // 批大小（可调）：
    const size_t B = 1024;

    for (size_t s = 0; s < num; s += B) {
        size_t e = std::min(num, s + B);

        // 构建本批入口点集合：使用已插入中的若干代表点
        std::vector<uint32_t> batch_entries;
        if (!stable_entries.empty()) batch_entries = stable_entries;
        else {
            // 没有已插入则用质心最近点作为入口（medoid）
            if (s == 0 && !dist_idx.empty()) batch_entries.push_back(dist_idx[0].second);
        }

        // Phase A：候选与剪枝（在稳定图快照上，缓存在本地）
        struct LocalNeigh { uint32_t u; std::vector<uint32_t> neigh; };
        std::vector<LocalNeigh> batch_local; batch_local.reserve(e - s);
        for (size_t p = s; p < e; ++p) {
            uint32_t u = dist_idx[p].second;
            // 收集候选（若图尚未形成，退化到全量近邻的前 M via fastscan）
            std::vector<float> q_pad_u(padded_dim, 0.0f);
            std::copy(bucket_padded[u].begin(), bucket_padded[u].end(), q_pad_u.begin());

            std::vector<std::pair<float,uint32_t>> cand;
            cand.reserve(window_C);
            if (!batch_entries.empty()) {
                collect_candidates_via_graph(
                    batch_entries, inserted, adj, graph_degree, num,
                    packed_codes, padded_dim / 8, padded_dim, bq_bits, ex_blob,
                    f_add, f_rescale, q_pad_u, window_C, cand);
            }
            // 若候选不足，使用全量 fastscan 选最近的若干（仅限初期，避免 O(n^2) 爆炸）
            if (cand.size() < graph_degree) {
                size_t num_rd = (num + 31) & ~31ULL; size_t code_cols = padded_dim / 8;
                size_t batches = num_rd / 32;
                rabitqlib::SplitBatchQuery<float> q(q_pad_u.data(), padded_dim, (bq_bits>1?bq_bits-1:0), rabitqlib::METRIC_L2, false);
                float qn2=0.0f; for (size_t d=0; d<padded_dim; ++d) qn2 += q_pad_u[d]*q_pad_u[d]; q.set_g_add(std::sqrt(qn2), 0.0f);
                std::array<uint16_t, rabitqlib::fastscan::kBatchSize> accu{};
                for (size_t b = 0; b < batches; ++b) {
                    const uint8_t* codes = packed_codes.data() + b * (code_cols * 32);
                    rabitqlib::fastscan::accumulate(codes, q.lut(), accu.data(), padded_dim);
                    size_t base = b * 32; size_t batch = std::min<size_t>(32, num - base);
                    for (size_t i = 0; i < batch; ++i) {
                        uint32_t v = static_cast<uint32_t>(base + i);
                        if (v == u) continue;
                        float ip_est = q.delta() * static_cast<float>(accu[i]) + q.sum_vl_lut();
                        float dist = f_add[v] + q.g_add() + f_rescale[v] * (ip_est + q.k1xsumq());
                        if (bq_bits > 1 && !ex_blob.empty()) {
                            size_t ex_bits = bq_bits - 1;
                            size_t ex_stride = ExDataMap<float>::data_bytes(padded_dim, ex_bits);
                            const char* ex_ptr = reinterpret_cast<const char*>(ex_blob.data()) + (v * ex_stride);
                            auto ip_func = select_excode_ipfunc(ex_bits);
                            ConstExDataMap<float> ex_map(ex_ptr, padded_dim, ex_bits);
                            float ex_dist = ex_map.f_add_ex() + q.g_add() + (ex_map.f_rescale_ex() *
                                (static_cast<float>(1 << ex_bits) * (ip_est) + ip_func(q_pad_u.data(), ex_map.ex_code(), padded_dim) + q.kbxsumq()));
                            dist = ex_dist;
                        }
                        cand.emplace_back(dist, v);
                    }
                }
                if (cand.size() > window_C) {
                    std::nth_element(cand.begin(), cand.begin() + window_C, cand.end());
                    cand.resize(window_C);
                    std::sort(cand.begin(), cand.end());
                } else {
                    std::sort(cand.begin(), cand.end());
                }
            }

            // α-遮挡剪枝
            std::vector<uint32_t> neigh; neigh.reserve(graph_degree);
            size_t ex_bits = bq_bits > 1 ? (bq_bits - 1) : 0;
            size_t code_cols = padded_dim / 8;
            for (auto& kv : cand) {
                if (neigh.size() >= graph_degree) break;
                float d_uy = kv.first; uint32_t y = kv.second; bool occluded = false;
                std::vector<float> q_pad_y(padded_dim, 0.0f);
                std::copy(bucket_padded[y].begin(), bucket_padded[y].end(), q_pad_y.begin());
                rabitqlib::SplitBatchQuery<float> qy(q_pad_y.data(), padded_dim, ex_bits, rabitqlib::METRIC_L2, false);
                float qn2y=0.0f; for (size_t d=0; d<padded_dim; ++d) qn2y += q_pad_y[d]*q_pad_y[d]; qy.set_g_add(std::sqrt(qn2y), 0.0f);
                for (uint32_t z : neigh) {
                    size_t base = (z / 32) * 32; size_t pos = z % 32;
                    const uint8_t* codes = packed_codes.data() + (base / 32) * (code_cols * 32);
                    std::array<uint16_t, rabitqlib::fastscan::kBatchSize> accu{};
                    rabitqlib::fastscan::accumulate(codes, qy.lut(), accu.data(), padded_dim);
                    float ip_est = qy.delta() * static_cast<float>(accu[pos]) + qy.sum_vl_lut();
                    float d_yz = f_add[z] + qy.g_add() + f_rescale[z] * (ip_est + qy.k1xsumq());
                    if (bq_bits > 1 && !ex_blob.empty()) {
                        size_t ex_stride = ExDataMap<float>::data_bytes(padded_dim, ex_bits);
                        const char* ex_ptr = reinterpret_cast<const char*>(ex_blob.data()) + (z * ex_stride);
                        auto ip_func = select_excode_ipfunc(ex_bits);
                        ConstExDataMap<float> ex_map(ex_ptr, padded_dim, ex_bits);
                        float ex_dist = ex_map.f_add_ex() + qy.g_add() + (ex_map.f_rescale_ex() *
                            (static_cast<float>(1 << ex_bits) * (ip_est) + ip_func(q_pad_y.data(), ex_map.ex_code(), padded_dim) + qy.kbxsumq()));
                        d_yz = ex_dist;
                    }
                    if (d_yz <= alpha_prune * d_uy) { occluded = true; break; }
                }
                if (!occluded) neigh.push_back(y);
            }
            if (neigh.size() < graph_degree) {
                for (auto& kv : cand) {
                    if (neigh.size() >= graph_degree) break;
                    if (std::find(neigh.begin(), neigh.end(), kv.second) == neigh.end()) neigh.push_back(kv.second);
                }
            }
            batch_local.push_back({u, std::move(neigh)});
        }

        // Phase B：合并（顺序写入，确保稳定性）
        for (auto& rec : batch_local) {
            uint32_t u = rec.u;
            for (size_t t = 0; t < graph_degree; ++t) {
                adj[u * graph_degree + t] = (t < rec.neigh.size() ? rec.neigh[t] : (rec.neigh.empty() ? u : rec.neigh[0]));
            }
            inserted[u] = 1;
            // 更新稳定入口集合（保留少量代表点）
            stable_entries.push_back(u);
            if (stable_entries.size() > 64) stable_entries.erase(stable_entries.begin(), stable_entries.begin() + (stable_entries.size() - 64));
        }
    }

    // 写文件
    std::ofstream gout(out_graph_path, std::ios::binary);
    if (!gout.is_open()) return;
    uint64_t pd = padded_dim, bits = bq_bits, n = num, deg = graph_degree;
    gout.write(reinterpret_cast<char*>(&pd), sizeof(uint64_t));
    gout.write(reinterpret_cast<char*>(&bits), sizeof(uint64_t));
    gout.write(reinterpret_cast<char*>(&n), sizeof(uint64_t));
    gout.write(reinterpret_cast<char*>(&deg), sizeof(uint64_t));
    gout.write(reinterpret_cast<const char*>(packed_codes.data()), packed_codes.size());
    gout.write(reinterpret_cast<const char*>(f_add.data()), sizeof(float) * num);
    gout.write(reinterpret_cast<const char*>(f_rescale.data()), sizeof(float) * num);
    if (!ex_blob.empty()) gout.write(reinterpret_cast<const char*>(ex_blob.data()), ex_blob.size());
    gout.write(reinterpret_cast<const char*>(adj.data()), sizeof(uint32_t) * adj.size());
    gout.close();
} 