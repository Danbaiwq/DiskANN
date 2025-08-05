#include "vamana_graph.h"
#include <iostream>
#include <vector>
#include <memory> // For std::shared_ptr
#include <thread>  // For std::thread
#include <numeric> // For std::iota
#include <algorithm> // For std::max

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

InMemoryIndex build_in_memory_index(const DataSet& data, const std::vector<uint32_t>& tags, size_t graph_degree, size_t build_complexity) {
    if (data.empty()) return nullptr;
    size_t num_points = data.size();
    size_t dim = data[0].size();
    
    // 根据数据点数量动态调整参数
    size_t actual_graph_degree = std::min(graph_degree, num_points / 2);  // 图度数不能超过点数的一半
    actual_graph_degree = std::max(actual_graph_degree, (size_t)3);       // 但至少要 3
    
    size_t actual_build_complexity = std::min(build_complexity, num_points - 1);  // 构建复杂度不能超过点数-1
    actual_build_complexity = std::max(actual_build_complexity, actual_graph_degree * 2);  // 至少是图度数的2倍
    
    std::cout << "Building index for " << num_points << " points with degree=" 
              << actual_graph_degree << ", complexity=" << actual_build_complexity << std::endl;
    
    auto index_write_params = std::make_shared<diskann::IndexWriteParameters>(
        diskann::IndexWriteParametersBuilder(actual_build_complexity, actual_graph_degree)
            .with_num_threads(1)  // 使用单线程避免冲突
            .build()
    );
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

void build_and_save_vamana_graph(const DataSet& data, const std::vector<uint32_t>& tags, const std::string& graph_path, size_t graph_degree, size_t build_complexity) {
    auto index = build_in_memory_index(data, tags, graph_degree, build_complexity);
    if (index) {
        index->save(graph_path.c_str());
    }
} 