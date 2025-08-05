#include "utils.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <random>
#include <numeric>
#include <algorithm>

DataPoint get_point_copy(const FlatDataSet& flat_data, size_t point_id, size_t dim) {
    DataPoint point(dim);
    const float* start_ptr = flat_data.data() + point_id * dim;
    std::copy(start_ptr, start_ptr + dim, point.begin());
    return point;
}

FlatDataSet load_fbin_flat(const std::string& filename, size_t& num_points, size_t& dim) {
    std::ifstream reader(filename, std::ios::binary);
    if (!reader.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return {};
    }

    uint32_t n_u32, dim_u32;
    reader.read(reinterpret_cast<char*>(&n_u32), sizeof(uint32_t));
    reader.read(reinterpret_cast<char*>(&dim_u32), sizeof(uint32_t));
    num_points = n_u32;
    dim = dim_u32;

    std::cout << "Loading " << num_points << " points of dimension " << dim << " from " << filename << std::endl;

    FlatDataSet data(num_points * dim);
    reader.read(reinterpret_cast<char*>(data.data()), num_points * dim * sizeof(float));

    if (!reader) {
        std::cerr << "Error reading data from file: " << filename << std::endl;
        return {};
    }
    
    reader.close();
    return data;
}

DataSet sample_data(const FlatDataSet& full_flat_data, size_t num_points, size_t dim, float alpha) {
    if (alpha <= 0.0 || alpha > 1.0) {
        throw std::invalid_argument("Alpha must be between 0 and 1.");
    }
    size_t sample_size = static_cast<size_t>(num_points * alpha);
    
    std::vector<size_t> indices(num_points);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    DataSet sampled_data;
    sampled_data.reserve(sample_size);
    for (size_t i = 0; i < sample_size; ++i) {
        sampled_data.push_back(get_point_copy(full_flat_data, indices[i], dim));
    }
    std::cout << "Sampled " << sample_size << " data points." << std::endl;
    return sampled_data;
}

void save_buckets(const std::string& filename, const Buckets& buckets) {
    std::ofstream writer(filename, std::ios::binary);
    if (!writer.is_open()) {
        std::cerr << "Error opening file for writing buckets: " << filename << std::endl;
        return;
    }

    uint64_t num_buckets = buckets.size();
    writer.write(reinterpret_cast<const char*>(&num_buckets), sizeof(uint64_t));

    for (const auto& bucket : buckets) {
        uint64_t bucket_size = bucket.size();
        writer.write(reinterpret_cast<const char*>(&bucket_size), sizeof(uint64_t));
        if (bucket_size > 0) {
            writer.write(reinterpret_cast<const char*>(bucket.data()), bucket_size * sizeof(uint32_t));
        }
    }
    writer.close();
    std::cout << "Buckets saved to " << filename << std::endl;
}

Buckets load_buckets(const std::string& filename) {
    std::ifstream reader(filename, std::ios::binary);
    if (!reader.is_open()) {
        std::cerr << "Error opening file for reading buckets: " << filename << std::endl;
        return {};
    }

    uint64_t num_buckets;
    reader.read(reinterpret_cast<char*>(&num_buckets), sizeof(uint64_t));

    Buckets buckets(num_buckets);
    for (uint64_t i = 0; i < num_buckets; ++i) {
        uint64_t bucket_size;
        reader.read(reinterpret_cast<char*>(&bucket_size), sizeof(uint64_t));
        if (bucket_size > 0) {
            buckets[i].resize(bucket_size);
            reader.read(reinterpret_cast<char*>(buckets[i].data()), bucket_size * sizeof(uint32_t));
        }
    }
    reader.close();
    std::cout << "Buckets loaded from " << filename << std::endl;
    return buckets;
} 