#pragma once

#include "utils.h"
#include <memory> // For std::unique_ptr

// Forward declaration
namespace diskann {
    template <typename T, typename TagT, typename LabelT>
    class Index;
}

using InMemoryIndex = std::unique_ptr<diskann::Index<float, uint32_t, uint32_t>>;

// Builds an index in memory and returns a pointer to it
InMemoryIndex build_in_memory_index(const DataSet& data, const std::vector<uint32_t>& tags, size_t graph_degree, size_t build_complexity);

// Builds an index and saves it to disk
void build_and_save_vamana_graph(const DataSet& data, const std::vector<uint32_t>& tags, const std::string& graph_path, size_t graph_degree, size_t build_complexity); 