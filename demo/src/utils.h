#pragma once

#include <vector>
#include <string>

// Define a simple vector structure
using DataPoint = std::vector<float>;
using DataSet = std::vector<DataPoint>;
using Buckets = std::vector<std::vector<uint32_t>>;
using FlatDataSet = std::vector<float>;

// Helper to get a copy of a point from a flat data structure
DataPoint get_point_copy(const FlatDataSet& flat_data, size_t point_id, size_t dim);

// Load data from a .fbin file directly into a flat vector
FlatDataSet load_fbin_flat(const std::string& filename, size_t& num_points, size_t& dim);

// Function to sample a fraction of the dataset from a flat vector
DataSet sample_data(const FlatDataSet& full_flat_data, size_t num_points, size_t dim, float alpha);

// Save and load the bucket mapping
void save_buckets(const std::string& filename, const Buckets& buckets);
Buckets load_buckets(const std::string& filename); 