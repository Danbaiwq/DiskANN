#include "build.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <array>
#include <omp.h>
#include <queue>
#include <unordered_set>
#include <limits>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include "vamana_graph.h"

// rabitq deps
#include "/home/danbai.wq/DiskANN/rabitq/rabitqlib/quantization/rabitq.hpp"
#include "/home/danbai.wq/DiskANN/rabitq/rabitqlib/quantization/data_layout.hpp"
#include "/home/danbai.wq/DiskANN/rabitq/rabitqlib/fastscan/fastscan.hpp"
#include "/home/danbai.wq/DiskANN/rabitq/rabitqlib/index/query.hpp"

using namespace rabitqlib;

// SQ 量化实现
void scalar_quantize(
    const DataSet& data,
    size_t padded_dim,
    size_t bits_per_scalar,
    SQData& sq_data
) {
    const size_t num = data.size();
    const size_t levels = (1 << bits_per_scalar) - 1;  // e.g., 255 for 8-bit
    
    sq_data.bits_per_scalar = bits_per_scalar;
    sq_data.scales.resize(num);
    sq_data.offsets.resize(num);
    sq_data.codes.resize(num * padded_dim);
    
    #pragma omp parallel for
    for (size_t i = 0; i < num; ++i) {
        // 找到每个向量的最小值和最大值
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        
        for (size_t d = 0; d < data[i].size(); ++d) {
            min_val = std::min(min_val, data[i][d]);
            max_val = std::max(max_val, data[i][d]);
        }
        
        // 计算缩放因子和偏移量
        float scale = (max_val - min_val) / levels;
        if (scale < 1e-6f) scale = 1e-6f;  // 避免除零
        
        sq_data.offsets[i] = min_val;
        sq_data.scales[i] = scale;
        
        // 量化
        for (size_t d = 0; d < padded_dim; ++d) {
            float val = (d < data[i].size()) ? data[i][d] : 0.0f;
            float normalized = (val - min_val) / scale;
            uint8_t quantized = static_cast<uint8_t>(std::round(std::min(std::max(normalized, 0.0f), static_cast<float>(levels))));
            sq_data.codes[i * padded_dim + d] = quantized;
        }
    }
}

// 计算两个 SQ 向量之间的 L2 距离
float sq_distance(
    const SQData& sq_data,
    size_t idx1,
    size_t idx2,
    size_t padded_dim
) {
    const uint8_t* codes1 = &sq_data.codes[idx1 * padded_dim];
    const uint8_t* codes2 = &sq_data.codes[idx2 * padded_dim];
    
    float scale1 = sq_data.scales[idx1];
    float scale2 = sq_data.scales[idx2];
    float offset1 = sq_data.offsets[idx1];
    float offset2 = sq_data.offsets[idx2];
    
    float dist = 0.0f;
    for (size_t d = 0; d < padded_dim; ++d) {
        float val1 = codes1[d] * scale1 + offset1;
        float val2 = codes2[d] * scale2 + offset2;
        float diff = val1 - val2;
        dist += diff * diff;
    }
    
    return dist;
}

// 计算 SQ 向量到原始向量的距离
static float sq_to_float_distance(
    const SQData& sq_data,
    size_t sq_idx,
    const std::vector<float>& float_vec,
    size_t padded_dim
) {
    const uint8_t* codes = &sq_data.codes[sq_idx * padded_dim];
    float scale = sq_data.scales[sq_idx];
    float offset = sq_data.offsets[sq_idx];
    
    float dist = 0.0f;
    for (size_t d = 0; d < padded_dim; ++d) {
        float sq_val = codes[d] * scale + offset;
        float diff = sq_val - float_vec[d];
        dist += diff * diff;
    }
    
    return dist;
}

static void pad_dataset(const DataSet& in, size_t padded_dim, DataSet& out) {
    out = in;
    for (auto& v : out) v.resize(padded_dim, 0.0f);
}

// 新增：按维度量化为 uint8（对整桶，每维 min-max 映射到 [0,255]）
static void quantize_u8_per_dim(const DataSet& padded, size_t padded_dim, std::vector<uint8_t>& flat_u8) {
    const size_t num = padded.size();
    if (num == 0) { flat_u8.clear(); return; }
    std::vector<float> minv(padded_dim, std::numeric_limits<float>::max());
    std::vector<float> maxv(padded_dim, std::numeric_limits<float>::lowest());
    for (size_t i = 0; i < num; ++i) {
        const auto& v = padded[i];
        for (size_t d = 0; d < padded_dim; ++d) {
            float x = v[d];
            if (x < minv[d]) minv[d] = x;
            if (x > maxv[d]) maxv[d] = x;
        }
    }
    flat_u8.resize(num * padded_dim);
    for (size_t i = 0; i < num; ++i) {
        const auto& v = padded[i];
        uint8_t* dst = &flat_u8[i * padded_dim];
        for (size_t d = 0; d < padded_dim; ++d) {
            float mn = minv[d], mx = maxv[d];
            float range = mx - mn;
            float val = v[d];
            if (range <= 1e-12f) { dst[d] = 0; continue; }
            float norm = (val - mn) * (255.0f / range);
            int q = static_cast<int>(std::round(norm));
            if (q < 0) q = 0; else if (q > 255) q = 255;
            dst[d] = static_cast<uint8_t>(q);
        }
    }
}

// 新增：按维度对称量化为 int8（每维映射到 [-128,127]）
static void quantize_i8_per_dim(const DataSet& padded, size_t padded_dim, std::vector<int8_t>& flat_i8) {
    const size_t num = padded.size();
    if (num == 0) { flat_i8.clear(); return; }
    std::vector<float> minv(padded_dim, std::numeric_limits<float>::max());
    std::vector<float> maxv(padded_dim, std::numeric_limits<float>::lowest());
    for (size_t i = 0; i < num; ++i) {
        const auto& v = padded[i];
        for (size_t d = 0; d < padded_dim; ++d) {
            float x = v[d];
            if (x < minv[d]) minv[d] = x;
            if (x > maxv[d]) maxv[d] = x;
        }
    }
    flat_i8.resize(num * padded_dim);
    for (size_t i = 0; i < num; ++i) {
        const auto& v = padded[i];
        int8_t* dst = &flat_i8[i * padded_dim];
        for (size_t d = 0; d < padded_dim; ++d) {
            float mn = minv[d], mx = maxv[d];
            float c = 0.5f * (mn + mx);
            float r = std::max(mx - c, c - mn);
            if (r <= 1e-12f) { dst[d] = 0; continue; }
            float norm = (v[d] - c) * (127.0f / r);
            int q = static_cast<int>(std::round(norm));
            if (q < -128) q = -128; else if (q > 127) q = 127;
            dst[d] = static_cast<int8_t>(q);
        }
    }
}

// AVX2: FP32 L2 距离（回退标量实现）
static inline float l2_distance_avx2(const float* a, const float* b, size_t dim) {
#ifdef __AVX2__
    size_t i = 0;
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }
    float buf[8];
    _mm256_storeu_ps(buf, acc);
    float sum = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
#endif
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
    const ConstructQuantization construct_mode = cfg.construct_mode;

    DataSet bucket_padded;
    if (padded_dim != bucket_data[0].size()) {
        pad_dataset(bucket_data, padded_dim, bucket_padded);
    } else {
        bucket_padded = bucket_data;
    }
    const size_t num = bucket_padded.size();

    // BQ 量化（始终需要，用于最终存储）
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

    // 若为全精度构图：直接调用 DiskANN 的构图，随后以 BQ 形式写盘
    if (construct_mode == ConstructQuantization::NO_QUANTIZATION) {
        // 1) 调用 DiskANN 构建临时 Vamana 图
        const size_t num_threads = std::max(1u, std::thread::hardware_concurrency());
        std::string tmp_graph = out_graph_path + ".tmp.vamana.index";
        build_and_save_vamana_graph(bucket_padded, {}, tmp_graph, graph_degree, cfg.build_complexity, num_threads);

        // 2) 读取邻接表
        auto adj_vecs = get_graph_neighbors(tmp_graph, num, graph_degree);
        std::vector<uint32_t> adj; adj.reserve(num * graph_degree);
        for (const auto& row : adj_vecs) adj.insert(adj.end(), row.begin(), row.end());

        // 3) 移除临时文件
        std::remove(tmp_graph.c_str());

        // 4) 用 BQ 节点向量 + DiskANN 邻接写出最终图
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
        return;
    }

    // 若为 SQ 构图：先做 SQ 量化并反量化为近似浮点，再调用 DiskANN 构建，最终仍以 BQ 写盘
    if (construct_mode == ConstructQuantization::SQ) {
        // 1) SQ 量化（8-bit，按维度 min-max 到 uint8），将维度对齐到32以提升SIMD吞吐
        const size_t sq_dim = ((padded_dim + 31) & ~31ULL);
        DataSet bucket_padded_sq;
        if (sq_dim != padded_dim) {
            pad_dataset(bucket_padded, sq_dim, bucket_padded_sq);
        } else {
            bucket_padded_sq = bucket_padded;
        }
        // 按环境变量选择量化类型：默认 u8，可设置 SQ_DATA_TYPE=i8 走 int8
        std::string sq_type = "u8";
        if (const char* env = std::getenv("SQ_DATA_TYPE")) {
            std::string v(env);
            for (auto& c : v) c = (char)std::tolower(c);
            if (v == "i8") sq_type = "i8";
        }

        const size_t num_threads = std::max(1u, std::thread::hardware_concurrency());
        std::string tmp_graph;
        if (sq_type == "i8") {
            std::vector<int8_t> flat_i8;
            quantize_i8_per_dim(bucket_padded_sq, sq_dim, flat_i8);
            tmp_graph = out_graph_path + ".tmp.i8.vamana.index";
            build_and_save_vamana_graph_i8(flat_i8, num, sq_dim, {}, tmp_graph, graph_degree, cfg.build_complexity, num_threads);
        } else {
            std::vector<uint8_t> flat_u8;
            quantize_u8_per_dim(bucket_padded_sq, sq_dim, flat_u8);
            tmp_graph = out_graph_path + ".tmp.u8.vamana.index";
            build_and_save_vamana_graph_u8(flat_u8, num, sq_dim, {}, tmp_graph, graph_degree, cfg.build_complexity, num_threads);
        }
 
        // 3) 读取邻接表
        auto adj_vecs = get_graph_neighbors(tmp_graph, num, graph_degree);
        std::vector<uint32_t> adj; adj.reserve(num * graph_degree);
        for (const auto& row : adj_vecs) adj.insert(adj.end(), row.begin(), row.end());

        // 4) 移除临时文件
        std::remove(tmp_graph.c_str());

        // 5) 用 BQ 节点向量 + DiskANN 邻接写出最终图
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
        return;
    }

    // SQ 量化（如果需要）
    SQData sq_data;
    if (construct_mode == ConstructQuantization::SQ) {
        scalar_quantize(bucket_padded, padded_dim, 8, sq_data);  // 8-bit SQ
    }

    // 生成按"距质心距离升序"的插入顺序
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

    // 定义距离计算函数（根据构图模式选择）
    auto compute_distance = [&](uint32_t idx1, uint32_t idx2) -> float {
        switch (construct_mode) {
            case ConstructQuantization::NO_QUANTIZATION: {
                // FP32 L2
                return l2_distance_avx2(bucket_padded[idx1].data(), bucket_padded[idx2].data(), padded_dim);
            }
            case ConstructQuantization::SQ:
                return sq_distance(sq_data, idx1, idx2, padded_dim);
            case ConstructQuantization::BQ:
            default:
                return 0.0f;  // 不使用
        }
    };

    // 构建图
    for (size_t s = 0; s < num; s += B) {
        size_t e = std::min(num, s + B);

        // 构建本批入口点集合：使用已插入中的若干代表点
        std::vector<uint32_t> batch_entries;
        if (!stable_entries.empty()) batch_entries = stable_entries;
        else {
            if (s == 0 && !dist_idx.empty()) batch_entries.push_back(dist_idx[0].second);
        }

        struct LocalNeigh { uint32_t u; std::vector<uint32_t> neigh; };
        std::vector<LocalNeigh> batch_local; batch_local.reserve(e - s);
        
        for (size_t p = s; p < e; ++p) {
            uint32_t u = dist_idx[p].second;
            std::vector<std::pair<float,uint32_t>> cand;
            cand.reserve(window_C);
            
            if (construct_mode == ConstructQuantization::BQ) {
                // 使用原有 BQ 候选收集（fastscan + 图扩展）
                std::vector<float> q_pad_u(padded_dim, 0.0f);
                std::copy(bucket_padded[u].begin(), bucket_padded[u].end(), q_pad_u.begin());
                if (!batch_entries.empty()) {
                    collect_candidates_via_graph(
                        batch_entries, inserted, adj, graph_degree, num,
                        packed_codes, padded_dim / 8, padded_dim, bq_bits, ex_blob,
                        f_add, f_rescale, q_pad_u, window_C, cand);
                }
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
            } else {
                // SQ / FP32：全量 Top-C 候选（质量优先），使用 AVX2 L2 或 SQ 距离
                cand.clear();
                cand.reserve(std::min(window_C, num));
                for (uint32_t v = 0; v < num; ++v) {
                    if (v == u) continue;
                    if (!inserted[v] && p != s) {
                        // 仅在图尚小的时候允许考虑未插入点；这里以简化处理为：都考虑
                    }
                    float dist;
                    if (construct_mode == ConstructQuantization::NO_QUANTIZATION) {
                        dist = l2_distance_avx2(bucket_padded[u].data(), bucket_padded[v].data(), padded_dim);
                    } else {
                        dist = sq_distance(sq_data, u, v, padded_dim);
                    }
                    cand.emplace_back(dist, v);
                }
                if (cand.size() > window_C) {
                    std::nth_element(cand.begin(), cand.begin() + window_C, cand.end());
                    cand.resize(window_C);
                }
                std::sort(cand.begin(), cand.end());
            }

            // α-遮挡剪枝
            std::vector<uint32_t> neigh; neigh.reserve(graph_degree);
            
            for (auto& kv : cand) {
                if (neigh.size() >= graph_degree) break;
                float d_uy = kv.first; uint32_t y = kv.second; bool occluded = false;
                
                for (uint32_t z : neigh) {
                    float d_yz = compute_distance(y, z);
                    if (d_yz <= alpha_prune * d_uy) { occluded = true; break; }
                }
                if (!occluded) neigh.push_back(y);
            }
            
            // 如果邻居不足，从候选中补充
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

    // 轻量级饱和/互连修复
    {
        bool enable_pass = true;
        if (const char* env = std::getenv("BQ_SATURATE_PASS")) {
            std::string v(env);
            if (v == "0" || v == "false" || v == "False") enable_pass = false;
        }
        if (enable_pass && num > 0) {
            const size_t refine_C = std::min<size_t>(std::max<size_t>(graph_degree * 2, 64), std::max<size_t>(window_C, graph_degree * 2));

            for (uint32_t u = 0; u < num; ++u) {
                std::vector<uint32_t> curr(graph_degree);
                for (size_t t = 0; t < graph_degree; ++t) curr[t] = adj[u * graph_degree + t];
                std::unordered_set<uint32_t> curr_set(curr.begin(), curr.end());
                float worst_dist = -1.0f; size_t worst_pos = 0;
                for (size_t t = 0; t < graph_degree; ++t) {
                    uint32_t v = curr[t]; if (v >= num) continue;
                    float dv = compute_distance(u, v);
                    if (dv > worst_dist) { worst_dist = dv; worst_pos = t; }
                }

                std::vector<uint32_t> entry_pts; entry_pts.reserve(graph_degree);
                for (uint32_t v : curr) if (v < num) entry_pts.push_back(v);
                if (entry_pts.empty()) entry_pts.push_back(u);

                std::vector<std::pair<float,uint32_t>> cand; cand.reserve(refine_C);
                if (construct_mode == ConstructQuantization::BQ) {
                    std::vector<float> q_pad_u(padded_dim, 0.0f);
                    std::copy(bucket_padded[u].begin(), bucket_padded[u].end(), q_pad_u.begin());
                    collect_candidates_via_graph(entry_pts, inserted, adj, graph_degree, num,
                        packed_codes, padded_dim / 8, padded_dim, bq_bits, ex_blob,
                        f_add, f_rescale, q_pad_u, refine_C, cand);
                } else {
                    // 使用全量 Top-C 生成
                    for (uint32_t v = 0; v < num; ++v) {
                        if (v == u) continue;
                        float dist = (construct_mode == ConstructQuantization::NO_QUANTIZATION)
                            ? l2_distance_avx2(bucket_padded[u].data(), bucket_padded[v].data(), padded_dim)
                            : sq_distance(sq_data, u, v, padded_dim);
                        cand.emplace_back(dist, v);
                    }
                    if (cand.size() > refine_C) {
                        std::nth_element(cand.begin(), cand.begin() + refine_C, cand.end());
                        cand.resize(refine_C);
                    }
                    std::sort(cand.begin(), cand.end());
                }

                bool replaced = false;
                for (auto& kv : cand) {
                    uint32_t y = kv.second; float d_uy = kv.first;
                    if (y == u || y >= num || curr_set.count(y)) continue;
                    if (d_uy < worst_dist) {
                        adj[u * graph_degree + worst_pos] = y;
                        curr_set.erase(curr[worst_pos]);
                        curr_set.insert(y);
                        curr[worst_pos] = y;

                        bool y_has_u = false;
                        for (size_t t = 0; t < graph_degree; ++t) if (adj[y * graph_degree + t] == u) { y_has_u = true; break; }
                        if (!y_has_u) {
                            float worst_y = -1.0f; size_t worst_pos_y = 0;
                            for (size_t t = 0; t < graph_degree; ++t) {
                                uint32_t z = adj[y * graph_degree + t]; if (z >= num) continue;
                                float d_yz = compute_distance(y, z);
                                if (d_yz > worst_y) { worst_y = d_yz; worst_pos_y = t; }
                            }
                            float d_yu = compute_distance(y, u);
                            if (d_yu < worst_y) {
                                adj[y * graph_degree + worst_pos_y] = u;
                            }
                        }
                        replaced = true;
                        break; // 每个 u 只替换一次
                    }
                }
                (void)replaced;
            }
        }
    }

    // 写文件（始终使用 BQ 格式）
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