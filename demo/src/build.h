#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include "utils.h"

// 构图量化模式
enum class ConstructQuantization {
    BQ,              // 使用 BQ 向量构图（原有模式）
    SQ,              // 使用 SQ 向量构图，最后压缩为 BQ
    NO_QUANTIZATION  // 使用全精度向量构图，最后压缩为 BQ
};

struct BQBuildConfig {
    size_t padded_dim;
    size_t bq_bits; // total bits; ex_bits = bq_bits > 1 ? bq_bits-1 : 0
    size_t graph_degree;
    size_t build_complexity; // C / efConstruction
    float alpha_prune; // Vamana occlusion
    ConstructQuantization construct_mode = ConstructQuantization::BQ; // 新增：构图模式
};

// SQ 量化数据结构
struct SQData {
    std::vector<uint8_t> codes;     // 量化后的代码
    std::vector<float> scales;      // 每个向量的缩放因子
    std::vector<float> offsets;     // 每个向量的偏移量
    size_t bits_per_scalar = 8;     // 每个标量的位数（默认8位）
};

// 执行 SQ 量化
void scalar_quantize(
    const DataSet& data,
    size_t padded_dim,
    size_t bits_per_scalar,
    SQData& sq_data
);

// 计算两个 SQ 向量之间的 L2 距离
float sq_distance(
    const SQData& sq_data,
    size_t idx1,
    size_t idx2,
    size_t padded_dim
);

// 构建大桶 bqgraph：使用 BQ 距离 + Vamana 风格剪枝
void build_large_bucket_bqgraph(
    size_t bucket_id,
    const DataSet& bucket_data,      // 原始向量（将按需pad）
    const std::vector<float>& centroid, // 桶质心（dim）
    const BQBuildConfig& cfg,
    const std::string& out_graph_path
);

// 构建小桶 bq.bin：仅量化与写出
void build_small_bucket_bqbin(
    size_t bucket_id,
    const DataSet& bucket_data,      // 原始向量（将按需pad）
    const std::vector<float>& centroid,
    size_t padded_dim,
    size_t bq_bits,
    const std::string& out_bin_path
); 