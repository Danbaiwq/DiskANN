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