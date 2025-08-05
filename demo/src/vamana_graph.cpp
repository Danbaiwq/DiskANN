#include "vamana_graph.h"
#include <iostream>
#include <vector>
#include <memory> // For std::shared_ptr

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
    
    auto index_write_params = std::make_shared<diskann::IndexWriteParameters>(
        diskann::IndexWriteParametersBuilder(build_complexity, graph_degree).with_num_threads(std::thread::hardware_concurrency()).build()
    );
    auto index_search_params = std::make_shared<diskann::IndexSearchParams>(build_complexity, 0);

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