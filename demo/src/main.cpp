#include <iostream>
#include <vector>
#include <thread>
#include <numeric>
#include <algorithm>
#include <string>
#include <chrono>
#include <fstream> // Added for file operations

#include "utils.h"
#include "kmeans.h"
#include "vamana_graph.h"
#include "search.h"
#include "omp.h"

void build_mode(const std::string& data_path) {
    std::cout << "\n--- Running in BUILD mode ---" << std::endl;

    // --- Parameters ---
    const float alpha = 0.1f;
    const size_t m = 256;
    const int t = 10;
    const int l = 5;
    const size_t graph_degree = 32;
    const size_t build_complexity = 50;

    // --- Load Data ---
    std::cout << "Loading data from " << data_path << "..." << std::endl;
    size_t num_points, dim;
    FlatDataSet full_dataset_flat = load_fbin_flat(data_path, num_points, dim);
    if (full_dataset_flat.empty()) {
        return;
    }

    // --- K-means Clustering ---
    std::cout << "Starting K-means clustering (t=" << t << ", m=" << m << ")..." << std::endl;
    DataSet final_centroids(m, DataPoint(dim, 0.0f));
    #pragma omp parallel for
    for (int i = 0; i < t; ++i) {
        DataSet sampled_data = sample_data(full_dataset_flat, num_points, dim, alpha);
        DataSet initial_centroids = kmeans_plusplus_init(sampled_data, m);
        DataSet centroids = kmeans_lloyds(sampled_data, m, initial_centroids, 100);
        #pragma omp critical
        {
            for (size_t j = 0; j < m; ++j) {
                if(j < centroids.size()){
                    for (size_t k = 0; k < dim; ++k) {
                        final_centroids[j][k] += centroids[j][k];
                    }
                }
            }
        }
    }
    for(size_t i = 0; i < m; ++i) { for(size_t j = 0; j < dim; ++j) { final_centroids[i][j] /= t; } }
    
    // --- Data Bucketing ---
    std::cout << "Assigning full dataset to buckets..." << std::endl;
    Buckets buckets(m);
    #pragma omp parallel for
    for (size_t i = 0; i < num_points; ++i) {
        DataPoint point = get_point_copy(full_dataset_flat, i, dim);
        std::vector<std::pair<float, uint32_t>> dists;
        for (uint32_t j = 0; j < m; ++j) {
            dists.push_back({calculate_distance(point, final_centroids[j]), j});
        }
        std::sort(dists.begin(), dists.end());
        for (int k = 0; k < l; ++k) {
            #pragma omp critical
            buckets[dists[k].second].push_back(i);
        }
    }

    // --- Save Buckets & Metadata ---
    std::cout << "Saving bucket assignments and metadata..." << std::endl;
    save_buckets("buckets.bin", buckets);
    std::ofstream meta_writer("medoid_meta.txt");
    meta_writer << dim << std::endl;
    meta_writer << graph_degree << std::endl;
    meta_writer.close();

    // --- Build Medoid Vamana Graph ---
    std::cout << "Building Medoid Vamana Graph..." << std::endl;
    build_and_save_vamana_graph(final_centroids, {}, "medoid_vamana.index", graph_degree, build_complexity);

    // --- Build Bucket Vamana Graphs ---
    const size_t MIN_BUCKET_SIZE_FOR_INDEX = 50;  // 大幅增加最小bucket大小要求
    std::cout << "Building and saving Vamana graphs for each bucket..." << std::endl;
    size_t buckets_built = 0;
    #pragma omp parallel for reduction(+:buckets_built)
    for (size_t i = 0; i < m; ++i) {
        if (buckets[i].size() >= MIN_BUCKET_SIZE_FOR_INDEX) {
            DataSet bucket_data;
            bucket_data.reserve(buckets[i].size());
            for (const auto& point_idx : buckets[i]) {
                bucket_data.push_back(get_point_copy(full_dataset_flat, point_idx, dim));
            }
            std::string bucket_graph_path = "bucket_" + std::to_string(i) + "_vamana.index";
            // 传递空的tags，让DiskANN使用局部ID (0, 1, 2, ...)
            build_and_save_vamana_graph(bucket_data, {}, bucket_graph_path, graph_degree, build_complexity);
            buckets_built++;
        } else {
            std::cout << "Skipping bucket " << i << " (size: " << buckets[i].size() 
                      << ", minimum required: " << MIN_BUCKET_SIZE_FOR_INDEX << ")" << std::endl;
        }
    }
    std::cout << "Built " << buckets_built << " bucket indices out of " << m << " total buckets." << std::endl;
    std::cout << "\nBuild mode finished successfully." << std::endl;
}

void search_mode(const std::string& query_path, const std::string& gt_path) {
    std::cout << "\n--- Running in SEARCH mode ---" << std::endl;

    // --- Parameters ---
    const int f = 2; // Number of buckets to search
    const int k = 50; // Number of neighbors to retrieve per bucket
    const int num_threads = std::thread::hardware_concurrency();

    // --- Load metadata ---
    size_t dim = 0;
    size_t graph_degree = 0;
    std::ifstream meta_reader("medoid_meta.txt");
    if (!meta_reader.is_open()) {
        std::cerr << "FATAL: medoid_meta.txt not found. Please run build mode first." << std::endl;
        return;
    }
    meta_reader >> dim >> graph_degree;
    meta_reader.close();
    
    if (dim == 0 || graph_degree == 0) {
        std::cerr << "FATAL: Failed to read metadata from medoid_meta.txt or metadata is invalid." << std::endl;
        return;
    }
    std::cout << "Read metadata: dim=" << dim << ", graph_degree=" << graph_degree << std::endl;


    // --- Load Search Data ---
    std::cout << "Loading query and ground truth data..." << std::endl;
    size_t num_queries, query_dim;
    FlatDataSet queries_flat = load_fbin_flat(query_path, num_queries, query_dim);
    if(queries_flat.empty()) exit(1);
    
    size_t gt_num, gt_dim;
    std::vector<uint32_t> gt_results = load_ground_truth(gt_path, gt_num, gt_dim);
    if(gt_results.empty()) exit(1);

    // --- Validate Data Consistency ---
    if (query_dim != dim) {
        std::cerr << "Error: Mismatch between query vector dimension (" << query_dim
                  << ") and indexed vector dimension (" << dim << ")." << std::endl;
        exit(1);
    }
    if (num_queries != gt_num) {
        std::cerr << "Error: Mismatch between number of queries in query file (" << num_queries
                  << ") and ground truth file (" << gt_num << ")." << std::endl;
        exit(1);
    }

    // --- Load medoid Vamana graph ---
    std::cout << "Loading medoid Vamana graph..." << std::endl;
    auto medoid_index = load_index("medoid_vamana.index", dim);
    if (!medoid_index) exit(1);

    // --- Load bucket mapping ---
    std::cout << "Loading bucket mapping..." << std::endl;
    auto buckets = load_buckets("buckets.bin");
    if (buckets.empty()) {
        std::cerr << "Error: Failed to load buckets.bin or it is empty." << std::endl;
        exit(1);
    }

    // --- Set up the LRU Index Cache ---
    const size_t cache_size_bytes = 1ULL * 1024 * 1024 * 1024; // 1 GB
    IndexCache bucket_index_cache(cache_size_bytes, dim, graph_degree);
    std::cout << "LRU index cache initialized with a " << cache_size_bytes / (1024*1024) << "MB budget." << std::endl;

    // --- Parameters for Search Evaluation ---
    std::vector<size_t> f_values = {f};
    size_t top_k = 100;
    std::vector<uint32_t> recall_k_values = {10, 50, 100};

    // --- Perform Search Evaluation ---
    std::cout << "\n--- Starting Search Evaluation ---" << std::endl;
    for (size_t f_val : f_values) {
        std::vector<QueryResult> results(num_queries);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (size_t i = 0; i < num_queries; ++i) {
            DataPoint query = get_point_copy(queries_flat, i, dim);
            results[i] = search_two_stage(query, *medoid_index, buckets, f_val, k, top_k, bucket_index_cache, dim);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end_time - start_time;
        double qps = num_queries / diff.count();

        // Flatten results for recall calculation
        std::vector<uint32_t> our_results_flat(num_queries * top_k, 0);
        for(size_t i = 0; i < num_queries; ++i) {
            // Ensure we don't copy more than top_k or more than available results
            size_t num_to_copy = std::min(top_k, results[i].ids.size());
            if (num_to_copy > 0) {
                 std::copy(results[i].ids.begin(), results[i].ids.begin() + num_to_copy, our_results_flat.data() + i * top_k);
            }
        }

        std::cout << "\n--- Results for f=" << f_val << " ---" << std::endl;
        std::cout << "QPS: " << qps << std::endl;
        
        for (size_t recall_at : recall_k_values) {
            double recall = calculate_recall(num_queries, our_results_flat.data(), top_k, gt_results.data(), gt_dim, recall_at);
            std::cout << "Recall@" << recall_at << ": " << recall * 100 << "%" << std::endl;
        }
    }
}


int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cout << "Usage: " << std::endl;
        std::cout << "  Build Mode: " << argv[0] << " <base.fbin>" << std::endl;
        std::cout << "  Search Mode: " << argv[0] << " <query.fbin> <ground_truth.ivecs>" << std::endl;
        return 1;
    }

    if (argc == 2) {
        build_mode(argv[1]);
    } else {
        search_mode(argv[1], argv[2]);
    }

    return 0;
} 