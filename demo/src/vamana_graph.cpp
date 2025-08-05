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
        // 对于极小的数据集，使用非常保守的参数
        actual_graph_degree = std::min(graph_degree, std::max(3UL, num_points / 4));
        actual_build_complexity = std::max(actual_graph_degree * 4, std::min(build_complexity * 2, num_points));
        alpha = 2.0f;  // 更大的alpha值确保能找到邻居
    } else if (num_points < 1000) {
        // 对于小数据集，适度降低参数但增加alpha
        actual_graph_degree = std::min(graph_degree, std::max(8UL, num_points / 8));
        actual_build_complexity = std::max(actual_graph_degree * 3, std::min(build_complexity * 2, num_points));
        alpha = 1.8f;
    } else {
        // 对于较大数据集，使用标准参数但确保不超过限制
        actual_graph_degree = std::min(graph_degree, num_points - 1);
        actual_build_complexity = std::max(actual_graph_degree * 2, std::min(build_complexity, num_points));
        alpha = 1.2f;
    }
    
    std::cout << "Building Vamana graph for " << num_points << " points with degree=" 
              << actual_graph_degree << ", complexity=" << actual_build_complexity 
              << ", alpha=" << alpha << ", threads=" << num_threads << std::endl;

    auto index_write_params = std::make_shared<diskann::IndexWriteParameters>(
        diskann::IndexWriteParametersBuilder(actual_build_complexity, actual_graph_degree)
            .with_num_threads(num_threads)
            .with_saturate_graph(true)  // 关键：启用saturate_graph避免pruned_list为空
            .with_alpha(alpha)          // 关键：设置更大的alpha值
            .build());
    auto index_search_params = std::make_shared<diskann::IndexSearchParams>(actual_build_complexity, 0);

    auto index = std::make_unique<diskann::Index<float, uint32_t, uint32_t>>(
        diskann::Metric::L2, dim, num_points, index_write_params, index_search_params);

    auto flat_data = flatten_data_for_build(data);

    try {
        if (tags.empty()) {
            std::vector<uint32_t> temp_tags(num_points);
            std::iota(temp_tags.begin(), temp_tags.end(), 0);
            index->build(flat_data.data(), num_points, temp_tags);
        } else {
            index->build(flat_data.data(), num_points, tags);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error building index for " << num_points << " points: " << e.what() << std::endl;
        std::cerr << "Attempting with even more conservative parameters..." << std::endl;
        
        // 如果构建失败，尝试更保守的参数
        actual_graph_degree = std::max(3UL, std::min(actual_graph_degree / 2, num_points / 10));
        actual_build_complexity = std::max(actual_graph_degree * 5, std::min(num_points, actual_build_complexity * 2));
        alpha = 3.0f;  // 更大的alpha值
        
        std::cout << "Retry with degree=" << actual_graph_degree 
                  << ", complexity=" << actual_build_complexity 
                  << ", alpha=" << alpha << std::endl;
        
        auto retry_write_params = std::make_shared<diskann::IndexWriteParameters>(
            diskann::IndexWriteParametersBuilder(actual_build_complexity, actual_graph_degree)
                .with_num_threads(num_threads)
                .with_saturate_graph(true)
                .with_alpha(alpha)
                .build());
        
        auto retry_index = std::make_unique<diskann::Index<float, uint32_t, uint32_t>>(
            diskann::Metric::L2, dim, num_points, retry_write_params, index_search_params);
        
        if (tags.empty()) {
            std::vector<uint32_t> temp_tags(num_points);
            std::iota(temp_tags.begin(), temp_tags.end(), 0);
            retry_index->build(flat_data.data(), num_points, temp_tags);
        } else {
            retry_index->build(flat_data.data(), num_points, tags);
        }
        
        return retry_index;
    }
    
    return index;
}

void build_and_save_vamana_graph(const DataSet& data, const std::vector<uint32_t>& tags, const std::string& graph_path, size_t graph_degree, size_t build_complexity, size_t num_threads) {
    auto index = build_in_memory_index(data, tags, graph_degree, build_complexity, num_threads);
    if (index) {
        index->save(graph_path.c_str());
    }
} 