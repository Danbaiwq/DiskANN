#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include "utils.h"

struct BQBuildConfig {
    size_t padded_dim;
    size_t bq_bits; // total bits; ex_bits = bq_bits > 1 ? bq_bits-1 : 0
    size_t graph_degree;
    size_t build_complexity; // C / efConstruction
    float alpha_prune; // Vamana occlusion
};

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