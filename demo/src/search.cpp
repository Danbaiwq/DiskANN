#include "search.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include "index.h"
#include "parameters.h"
#include "kmeans.h"

std::shared_ptr<diskann::Index<float, uint32_t, uint32_t>> load_index(const std::string& index_path, const size_t dim) {
    if (!std::ifstream(index_path).good()) {
        return nullptr;
    }
    auto index_write_params =
        std::make_shared<diskann::IndexWriteParameters>(diskann::IndexWriteParametersBuilder(100, 64).build());
    auto index_search_params = std::make_shared<diskann::IndexSearchParams>(100, 0);
    auto index = std::make_shared<diskann::Index<float, uint32_t, uint32_t>>(
        diskann::Metric::L2, dim, 1, index_write_params, index_search_params);
    index->load(index_path.c_str(), 1, 100);
    return index;
}


IndexCache::IndexCache(size_t max_size_bytes, size_t d, size_t gd) : max_size(max_size_bytes), current_size(0), dim(d), graph_degree(gd) {}

IndexPtr IndexCache::get(uint32_t key) {
    {
        std::shared_lock<std::shared_mutex> lock(mtx);
        if (cache.count(key)) {
            return cache.at(key);
        }
    }
    std::unique_lock<std::shared_mutex> lock(mtx);
    if (cache.count(key)) {
        return cache.at(key);
    }
    std::string index_path = "bucket_" + std::to_string(key) + "_vamana.index";
    IndexPtr index = load_index(index_path, dim);
    if (index) {
        put_locked(key, index);
    }
    return index;
}

void IndexCache::put_locked(uint32_t key, IndexPtr index) {
    auto index_size = diskann::estimate_ram_usage(index->get_num_points(), dim, sizeof(float), graph_degree);
    while (current_size + index_size > max_size && !lru.empty()) {
        auto last_key = lru.back();
        lru.pop_back();
        auto evicted_index = cache.at(last_key);
        current_size -= diskann::estimate_ram_usage(evicted_index->get_num_points(), dim, sizeof(float), graph_degree);
        cache.erase(last_key);
    }
    if (current_size + index_size <= max_size) {
        if (cache.find(key) != cache.end()) {
            current_size -= diskann::estimate_ram_usage(cache[key]->get_num_points(), dim, sizeof(float), graph_degree);
            lru.remove(key);
        }
        cache[key] = index;
        lru.push_front(key);
        current_size += index_size;
    }
}

QueryResult search_two_stage(
    const std::vector<float>& query,
    diskann::Index<float, uint32_t, uint32_t>& medoid_index,
    const Buckets& buckets,
    size_t f,
    size_t k,
    size_t top_k,
    IndexCache& index_cache,
    size_t dim
) {
    std::vector<uint32_t> nearest_bucket_ids(f);
    medoid_index.search(query.data(), f, f, nearest_bucket_ids.data(), nullptr);

    std::vector<std::pair<float, uint32_t>> candidates;
    std::set<uint32_t> visited_ids;
    
    for (uint32_t bucket_id : nearest_bucket_ids) {
        auto bucket_index = index_cache.get(bucket_id);
        if (bucket_index) {
            size_t actual_k = std::min(k, bucket_index->get_num_points());
            if (actual_k == 0) continue;

            std::vector<uint32_t> result_tags(actual_k);
            std::vector<float> result_dists(actual_k);
            bucket_index->search(query.data(), actual_k, actual_k, result_tags.data(), result_dists.data());

            for (size_t i = 0; i < actual_k; ++i) {
                // The tag returned is a LOCAL index within the bucket
                uint32_t local_idx = result_tags[i];
                
                // Check bounds before accessing buckets
                if (local_idx < buckets[bucket_id].size()) {
                    uint32_t global_id = buckets[bucket_id][local_idx];
                    if (visited_ids.find(global_id) == visited_ids.end()) {
                        candidates.push_back({result_dists[i], global_id});
                        visited_ids.insert(global_id);
                    }
                }
            }
        }
    }
    
    std::sort(candidates.begin(), candidates.end());

    QueryResult final_results;
    final_results.ids.reserve(std::min(top_k, candidates.size()));
    final_results.distances.reserve(std::min(top_k, candidates.size()));
    for (size_t i = 0; i < std::min(top_k, candidates.size()); ++i) {
        final_results.distances.push_back(candidates[i].first);
        final_results.ids.push_back(candidates[i].second);
    }
    return final_results;
}


std::vector<uint32_t> load_ground_truth(const std::string& gt_path, size_t& num_queries, size_t& gt_dim) {
    std::vector<uint32_t> gt_ids;
    std::ifstream reader(gt_path, std::ios::binary);
    if (!reader.is_open()) {
        std::cerr << "Error opening ground truth file: " << gt_path << std::endl;
        return gt_ids;
    }
    unsigned n, d;
    reader.read((char*)&n, sizeof(unsigned));
    reader.read((char*)&d, sizeof(unsigned));
    num_queries = n;
    gt_dim = d;
    gt_ids.resize(n * d);
    reader.read((char*)gt_ids.data(), sizeof(uint32_t) * n * d);
    reader.close();
    return gt_ids;
}

double calculate_recall(size_t num_queries, const uint32_t* our_results, size_t our_top_k, const uint32_t* gt_results, size_t gt_top_k, size_t recall_at) {
    if (recall_at > our_top_k) {
        recall_at = our_top_k;
    }
    size_t total_matches = 0;
    for (size_t i = 0; i < num_queries; ++i) {
        std::set<uint32_t> gt_set;
        for (size_t j = 0; j < gt_top_k; ++j) {
            gt_set.insert(gt_results[i * gt_top_k + j]);
        }
        size_t query_matches = 0;
        for (size_t j = 0; j < recall_at; ++j) {
            if (gt_set.count(our_results[i * our_top_k + j])) {
                query_matches++;
            }
        }
        total_matches += query_matches;
    }
    return (double)total_matches / (num_queries * recall_at);
} 