#include "vamana_graph.h"
#include <iostream>
#include <vector>
#include <memory> // For std::shared_ptr
#include <thread>  // For std::thread
#include <numeric> // For std::iota
#include <algorithm> // For std::max
#include <fstream>

#include "index.h"
#include "parameters.h"
#include "index_build_params.h"

// Helper function to flatten data for diskann build
std::vector<float> flatten_data_for_build(const DataSet& data) {
    std::vector<float> flat_data;
    if (data.empty()) return flat_data;
    size_t dim = data[0].size();
    flat_data.reserve(data.size() * dim);
    for (const auto& point : data) {
        flat_data.insert(flat_data.end(), point.begin(), point.end());
    }
    return flat_data;
}

// === 新增：为 uint8_t 直接构建的辅助 ===
static std::vector<uint8_t> flatten_u8_for_build(const std::vector<std::vector<uint8_t>>& data_u8) {
    std::vector<uint8_t> flat;
    if (data_u8.empty()) return flat;
    size_t dim = data_u8[0].size();
    flat.reserve(data_u8.size() * dim);
    for (const auto& v : data_u8) flat.insert(flat.end(), v.begin(), v.end());
    return flat;
}

InMemoryIndex build_in_memory_index(const DataSet& data, const std::vector<uint32_t>& tags, size_t graph_degree, size_t build_complexity, size_t num_threads) {
    size_t num_points = data.size();
    if (num_points == 0) {
        return nullptr;
    }
    
    size_t dim = data[0].size();
    
    // 更保守的动态参数调整策略，专门防止pruned_list为空
    size_t actual_graph_degree = graph_degree;
    size_t actual_build_complexity = build_complexity;
    float alpha = 1.2f;  // 默认alpha值
    
    if (num_points < 200) {
        actual_graph_degree = std::min(graph_degree, std::max(3UL, num_points / 4));
        actual_build_complexity = std::max(actual_graph_degree * 4, std::min(build_complexity * 2, num_points));
        alpha = 2.0f;
    } else if (num_points < 1000) {
        actual_graph_degree = std::min(graph_degree, std::max(8UL, num_points / 8));
        actual_build_complexity = std::max(actual_graph_degree * 3, std::min(build_complexity * 2, num_points));
        alpha = 1.8f;
    } else {
        actual_graph_degree = std::min(graph_degree, num_points - 1);
        actual_build_complexity = std::max(actual_graph_degree * 2, std::min(build_complexity, num_points));
        alpha = 1.2f;
    }
    
    auto index_write_params = std::make_shared<diskann::IndexWriteParameters>(
        diskann::IndexWriteParametersBuilder(actual_build_complexity, actual_graph_degree)
            .with_num_threads(num_threads)
            .with_saturate_graph(true)
            .with_alpha(alpha)
            .build());
    auto index_search_params = std::make_shared<diskann::IndexSearchParams>(actual_build_complexity, 0);

    auto index = std::make_unique<diskann::Index<float, uint32_t, uint32_t>>(
        diskann::Metric::L2, dim, num_points, index_write_params, index_search_params);

    auto flat_data = flatten_data_for_build(data);

    if (tags.empty()) {
        std::vector<uint32_t> temp_tags(num_points);
        std::iota(temp_tags.begin(), temp_tags.end(), 0);
        index->build(flat_data.data(), num_points, temp_tags);
    } else {
        index->build(flat_data.data(), num_points, tags);
    }

    return index;
}

void build_and_save_vamana_graph(const DataSet& data, const std::vector<uint32_t>& tags, const std::string& graph_path, size_t graph_degree, size_t build_complexity, size_t num_threads) {
    auto index = build_in_memory_index(data, tags, graph_degree, build_complexity, num_threads);
    if (index) {
        index->save(graph_path.c_str());
    }
}

// === 新增：uint8 版本实现 ===
InMemoryIndexU8 build_in_memory_index_u8(const std::vector<uint8_t>& flat_u8, size_t num_points, size_t dim, const std::vector<uint32_t>& tags, size_t graph_degree, size_t build_complexity, size_t num_threads) {
    if (num_points == 0) return nullptr;

    size_t actual_graph_degree = graph_degree;
    size_t actual_build_complexity = build_complexity;
    float alpha = 1.2f;
    if (num_points < 200) {
        actual_graph_degree = std::min(graph_degree, std::max(3UL, num_points / 4));
        actual_build_complexity = std::max(actual_graph_degree * 4, std::min(build_complexity * 2, num_points));
        alpha = 2.0f;
    } else if (num_points < 1000) {
        actual_graph_degree = std::min(graph_degree, std::max(8UL, num_points / 8));
        actual_build_complexity = std::max(actual_graph_degree * 3, std::min(build_complexity * 2, num_points));
        alpha = 1.8f;
    } else {
        actual_graph_degree = std::min(graph_degree, num_points - 1);
        actual_build_complexity = std::max(actual_graph_degree * 2, std::min(build_complexity, num_points));
        alpha = 1.2f;
    }

    auto index_write_params = std::make_shared<diskann::IndexWriteParameters>(
        diskann::IndexWriteParametersBuilder(actual_build_complexity, actual_graph_degree)
            .with_num_threads(num_threads)
            .with_saturate_graph(true)
            .with_alpha(alpha)
            .build());
    auto index_search_params = std::make_shared<diskann::IndexSearchParams>(actual_build_complexity, 0);

    auto index = std::make_unique<diskann::Index<uint8_t, uint32_t, uint32_t>>(
        diskann::Metric::L2, dim, num_points, index_write_params, index_search_params);

    if (tags.empty()) {
        std::vector<uint32_t> temp_tags(num_points);
        std::iota(temp_tags.begin(), temp_tags.end(), 0);
        index->build(flat_u8.data(), num_points, temp_tags);
    } else {
        index->build(flat_u8.data(), num_points, tags);
    }

    return index;
}

void build_and_save_vamana_graph_u8(const std::vector<uint8_t>& flat_u8, size_t num_points, size_t dim, const std::vector<uint32_t>& tags, const std::string& graph_path, size_t graph_degree, size_t build_complexity, size_t num_threads) {
    auto index = build_in_memory_index_u8(flat_u8, num_points, dim, tags, graph_degree, build_complexity, num_threads);
    if (index) {
        index->save(graph_path.c_str());
    }
}

// === 新增：int8 版本实现 ===
InMemoryIndexI8 build_in_memory_index_i8(const std::vector<int8_t>& flat_i8, size_t num_points, size_t dim, const std::vector<uint32_t>& tags, size_t graph_degree, size_t build_complexity, size_t num_threads) {
    if (num_points == 0) return nullptr;

    size_t actual_graph_degree = graph_degree;
    size_t actual_build_complexity = build_complexity;
    float alpha = 1.2f;
    if (num_points < 200) {
        actual_graph_degree = std::min(graph_degree, std::max(3UL, num_points / 4));
        actual_build_complexity = std::max(actual_graph_degree * 4, std::min(build_complexity * 2, num_points));
        alpha = 2.0f;
    } else if (num_points < 1000) {
        actual_graph_degree = std::min(graph_degree, std::max(8UL, num_points / 8));
        actual_build_complexity = std::max(actual_graph_degree * 3, std::min(build_complexity * 2, num_points));
        alpha = 1.8f;
    } else {
        actual_graph_degree = std::min(graph_degree, num_points - 1);
        actual_build_complexity = std::max(actual_graph_degree * 2, std::min(build_complexity, num_points));
        alpha = 1.2f;
    }

    auto index_write_params = std::make_shared<diskann::IndexWriteParameters>(
        diskann::IndexWriteParametersBuilder(actual_build_complexity, actual_graph_degree)
            .with_num_threads(num_threads)
            .with_saturate_graph(true)
            .with_alpha(alpha)
            .build());
    auto index_search_params = std::make_shared<diskann::IndexSearchParams>(actual_build_complexity, 0);

    auto index = std::make_unique<diskann::Index<int8_t, uint32_t, uint32_t>>(
        diskann::Metric::L2, dim, num_points, index_write_params, index_search_params);

    if (tags.empty()) {
        std::vector<uint32_t> temp_tags(num_points);
        std::iota(temp_tags.begin(), temp_tags.end(), 0);
        index->build(flat_i8.data(), num_points, temp_tags);
    } else {
        index->build(flat_i8.data(), num_points, tags);
    }

    return index;
}

void build_and_save_vamana_graph_i8(const std::vector<int8_t>& flat_i8, size_t num_points, size_t dim, const std::vector<uint32_t>& tags, const std::string& graph_path, size_t graph_degree, size_t build_complexity, size_t num_threads) {
    auto index = build_in_memory_index_i8(flat_i8, num_points, dim, tags, graph_degree, build_complexity, num_threads);
    if (index) {
        index->save(graph_path.c_str());
    }
}

std::vector<std::vector<uint32_t>> get_graph_neighbors(const std::string& graph_path, size_t num_points, size_t degree) {
    std::ifstream in(graph_path, std::ios::binary);
    if (!in.is_open()) return {};

    // header: uint64 index_size, uint32 max_degree, uint32 start, uint64 num_frozen_points
    uint64_t index_size = 0; uint32_t max_degree = 0; uint32_t start = 0; uint64_t num_frozen = 0;
    in.read(reinterpret_cast<char*>(&index_size), sizeof(uint64_t));
    in.read(reinterpret_cast<char*>(&max_degree), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&start), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&num_frozen), sizeof(uint64_t));

    std::vector<std::vector<uint32_t>> adj(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        uint32_t k = 0;
        in.read(reinterpret_cast<char*>(&k), sizeof(uint32_t));
        std::vector<uint32_t> row;
        row.resize(degree);
        if (k > 0) {
            const size_t to_read = std::min<size_t>(k, degree);
            in.read(reinterpret_cast<char*>(row.data()), to_read * sizeof(uint32_t));
            // 若实际度数大于degree，则跳过多余部分
            if (k > degree) {
                in.seekg(static_cast<std::streamoff>((k - degree) * sizeof(uint32_t)), std::ios::cur);
            }
            // 不足补齐：用第一个邻居回填（若无则自环）
            uint32_t pad_val = row[0];
            for (size_t t = to_read; t < degree; ++t) row[t] = pad_val;
        } else {
            // k==0：全部填自身索引，避免非法值
            std::fill(row.begin(), row.end(), static_cast<uint32_t>(i));
        }
        adj[i].swap(row);
    }
    return adj;
} 