#pragma once

#include <vector>
#include <string>
#include <memory>
#include "utils.h"

// Forward declarations to avoid including diskann headers in header file
namespace diskann {
    template <typename T, typename TagT, typename LabelT> class Index;
}

using InMemoryIndex = std::unique_ptr<diskann::Index<float, uint32_t, uint32_t>>;

InMemoryIndex build_in_memory_index(const DataSet& data, const std::vector<uint32_t>& tags, size_t graph_degree, size_t build_complexity, size_t num_threads = 1);

void build_and_save_vamana_graph(const DataSet& data, const std::vector<uint32_t>& tags, const std::string& graph_path, size_t graph_degree, size_t build_complexity, size_t num_threads = 1);

// 读取 DiskANN 保存的 .index 邻接（根据 demo 的保存格式读取）
std::vector<std::vector<uint32_t>> get_graph_neighbors(const std::string& graph_path, size_t num_points, size_t degree);

// === 新增：uint8 构建接口 ===
using InMemoryIndexU8 = std::unique_ptr<diskann::Index<uint8_t, uint32_t, uint32_t>>;

InMemoryIndexU8 build_in_memory_index_u8(const std::vector<uint8_t>& flat_u8, size_t num_points, size_t dim, const std::vector<uint32_t>& tags, size_t graph_degree, size_t build_complexity, size_t num_threads = 1);

void build_and_save_vamana_graph_u8(const std::vector<uint8_t>& flat_u8, size_t num_points, size_t dim, const std::vector<uint32_t>& tags, const std::string& graph_path, size_t graph_degree, size_t build_complexity, size_t num_threads = 1);

// === 新增：int8 构建接口 ===
using InMemoryIndexI8 = std::unique_ptr<diskann::Index<int8_t, uint32_t, uint32_t>>;

InMemoryIndexI8 build_in_memory_index_i8(const std::vector<int8_t>& flat_i8, size_t num_points, size_t dim, const std::vector<uint32_t>& tags, size_t graph_degree, size_t build_complexity, size_t num_threads = 1);

void build_and_save_vamana_graph_i8(const std::vector<int8_t>& flat_i8, size_t num_points, size_t dim, const std::vector<uint32_t>& tags, const std::string& graph_path, size_t graph_degree, size_t build_complexity, size_t num_threads = 1); 