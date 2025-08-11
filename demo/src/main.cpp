#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <omp.h>
#include <fstream>
#include <atomic>
#include <algorithm>
#include <iomanip>  // 添加这个头文件用于std::fixed和std::setprecision
#include <cmath>

#include "utils.h"
#include "kmeans.h"
#include "vamana_graph.h"
#include "search.h"

// rabitq 量化接口（构建阶段用）
#include "/home/danbai.wq/DiskANN/rabitq/rabitqlib/quantization/rabitq.hpp"
#include "/home/danbai.wq/DiskANN/rabitq/rabitqlib/quantization/data_layout.hpp"
#include "/home/danbai.wq/DiskANN/rabitq/rabitqlib/fastscan/fastscan.hpp"
#include "/home/danbai.wq/DiskANN/rabitq/rabitqlib/index/query.hpp"

void build_mode(const std::string& data_path) {
    std::cout << "\n--- Running in BUILD mode ---" << std::endl;

    // --- Parameters ---
    const float alpha = 0.01f;
    const size_t m = 1024;
    const int t = 8;
    const int l = 3;
    const float beta = 1.2f;  // 新增：距离比例约束参数
    const size_t graph_degree = 32;
    const size_t build_complexity = 50;

    // bq 开关与bits（默认从环境变量读取，不存在则默认关闭bq）
    bool use_bq = false;
    size_t bq_bits = 4; // 1~8，默认4
    if (const char* env = std::getenv("USE_BQ")) {
        use_bq = (std::string(env) == "1" || std::string(env) == "true");
    }
    if (const char* envb = std::getenv("BQ_BITS")) {
        bq_bits = std::max<size_t>(1, std::min<size_t>(8, std::stoul(envb)));
    }
    
    // 获取可用线程数，但为每个索引构建留一些余量
    const size_t max_threads = std::thread::hardware_concurrency();
    const size_t threads_per_build = std::max(1UL, max_threads / 2);  // 每个索引构建使用一半的线程
    
    std::cout << "Using " << threads_per_build << " threads per index build (total CPU cores: " << max_threads << ")" << std::endl;
    std::cout << "Distance constraint parameter beta: " << beta << std::endl;
    std::cout << "bq mode: " << (use_bq ? "ON" : "OFF") << ", bits=" << bq_bits << std::endl;

    auto build_start_time = std::chrono::high_resolution_clock::now();
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
    
    // --- 优化的数据分桶策略 (基于距离比例约束) ---
    std::cout << "Starting optimized vector bucketing with distance constraint (beta=" << beta << ")..." << std::endl;

    Buckets buckets(m);
    size_t total_bucket_assignments = 0;  // 使用普通变量进行统计
    
    #pragma omp parallel for reduction(+:total_bucket_assignments)
    for (size_t i = 0; i < num_points; ++i) {
        DataPoint point = get_point_copy(full_dataset_flat, i, dim);
        std::vector<std::pair<float, uint32_t>> dists;
        
        // 计算到所有聚类中心的距离
        for (uint32_t j = 0; j < m; ++j) {
            dists.push_back({calculate_distance(point, final_centroids[j]), j});
        }
        
        // 按距离排序
        std::sort(dists.begin(), dists.end());
        
        // 获取最近距离作为基准
        float dist1 = dists[0].first;
        size_t buckets_assigned = 0;
        
        // 应用距离比例约束的智能分桶策略
        for (int k = 0; k < l && k < static_cast<int>(dists.size()); ++k) {
            float dist_k = dists[k].first;
            
            // 第一个桶（最近的）总是分配
            // 后续桶需要满足距离约束：beta * dist1 >= dist_k
            if (k == 0 || (beta * dist1 >= dist_k)) {
                #pragma omp critical
                {
                    buckets[dists[k].second].push_back(i);
                }
                buckets_assigned++;
            } else {
                // 距离约束不满足，停止分配
                break;
            }
        }
        
        total_bucket_assignments += buckets_assigned;
    }

    double average_buckets_per_vector = static_cast<double>(total_bucket_assignments) / num_points;
    // --- Save Buckets & Metadata ---
    std::cout << "Saving bucket assignments and metadata..." << std::endl;
    save_buckets("buckets.bin", buckets);
    std::ofstream meta_writer("medoid_meta.txt");
    meta_writer << dim << std::endl;
    meta_writer << graph_degree << std::endl;
    meta_writer << beta << std::endl;  // 保存beta参数到元数据
    meta_writer << average_buckets_per_vector << std::endl;  // 保存平均分桶数
    meta_writer << (use_bq ? 1 : 0) << std::endl; // 是否使用bq
    meta_writer << bq_bits << std::endl; // bq bits
    meta_writer.close();

    // --- Build Medoid Vamana Graph ---
    std::cout << "Building Medoid Vamana Graph (using " << threads_per_build << " threads)..." << std::endl;
    build_and_save_vamana_graph(final_centroids, {}, "medoid_vamana.index", graph_degree, build_complexity, threads_per_build);

    // --- Build Bucket Vamana Graphs (bucket间并行构建) ---
    const size_t MIN_BUCKET_SIZE_FOR_INDEX = 100;
    std::cout << "Building and saving Vamana graphs for each bucket..." << std::endl;
    
    // 计算合理的并行度
    const size_t max_parallel_buckets = std::max(1UL, max_threads);  // 同时构建的bucket数量
    const size_t threads_per_bucket = std::max(1UL, max_threads / max_parallel_buckets);  // 每个bucket的线程数
    
    std::cout << "Parallel bucket building: " << max_parallel_buckets << " buckets simultaneously, " 
              << threads_per_bucket << " threads per bucket" << std::endl;
    
    std::atomic<size_t> buckets_built(0);
    
    
    // 准备所有需要构建的bucket
    std::vector<size_t> buckets_to_build;
    for (size_t i = 0; i < m; ++i) {
        if (buckets[i].size() >= MIN_BUCKET_SIZE_FOR_INDEX) {
            buckets_to_build.push_back(i);
        } else {
            std::cout << "Skipping bucket " << i << " (size: " << buckets[i].size() 
                      << ", minimum required: " << MIN_BUCKET_SIZE_FOR_INDEX << ")" << std::endl;
        }
    }
    
    std::cout << "Will build " << buckets_to_build.size() << " valid buckets out of " << m << " total buckets." << std::endl;
    
    // 使用OpenMP并行构建bucket，但限制并行度
    #pragma omp parallel for schedule(dynamic) num_threads(max_parallel_buckets)
    for (size_t idx = 0; idx < buckets_to_build.size(); ++idx) {
        size_t i = buckets_to_build[idx];
        
        // 线程安全的输出
        #pragma omp critical
        {
            std::cout << "Thread " << omp_get_thread_num() << " building bucket " << i 
                      << " (" << buckets[i].size() << " points)..." << std::endl;
        }
        
        DataSet bucket_data;
        bucket_data.reserve(buckets[i].size());
        for (const auto& point_idx : buckets[i]) {
            bucket_data.push_back(get_point_copy(full_dataset_flat, point_idx, dim));
        }
        
        std::string bucket_graph_path = "bucket_" + std::to_string(i) + "_vamana.index";
        
        if (!use_bq) {
            // 原始构图
            build_and_save_vamana_graph(bucket_data, {}, bucket_graph_path, graph_degree, build_complexity, threads_per_bucket);
        } else {
            // 读取 BQ_GRAPH_THRESHOLD 环境变量，默认 1000
            size_t bq_graph_threshold = 1000;
            if (const char* env_thr = std::getenv("BQ_GRAPH_THRESHOLD")) {
                try { bq_graph_threshold = std::stoul(env_thr); } catch (...) {}
            }

            if (buckets[i].size() >= bq_graph_threshold) {
                // --- 使用 BQ 距离构建大桶图：生成batched 1bit codes + 可选ex_data，并据此做近邻选择 ---
                size_t padded_dim = (dim + 3) / 4 * 4;
                DataSet bucket_padded = bucket_data;
                if (padded_dim != dim) {
                    for (auto& v : bucket_padded) v.resize(padded_dim, 0.0f);
                }
                const size_t num = bucket_padded.size();

                // 生成batched 1-bit codes与因子（供fastscan批量距离估计）
                std::vector<uint8_t> packed_codes((((num + 31) & ~31ULL) / 32) * (padded_dim / 8) * 32);
                std::vector<float> f_add(num), f_rescale(num), f_err(num);

                std::vector<float> flat_data(num * padded_dim, 0.0f);
                for (size_t r = 0; r < num; ++r) {
                    std::copy(bucket_padded[r].begin(), bucket_padded[r].end(), flat_data.begin() + r * padded_dim);
                }
                std::vector<float> centroid_pad(padded_dim, 0.0f);
                std::copy(final_centroids[i].begin(), final_centroids[i].end(), centroid_pad.begin());

                using namespace rabitqlib;
                using namespace rabitqlib::quant;
                RabitqConfig cfg = faster_config(padded_dim, bq_bits);

                rabitqlib::quant::rabitq_impl::one_bit::one_bit_batch_code<float, false>(
                    flat_data.data(),
                    centroid_pad.data(),
                    num,
                    padded_dim,
                    packed_codes.data(),
                    f_add.data(),
                    f_rescale.data(),
                    f_err.data(),
                    rabitqlib::METRIC_L2
                );

                // ex_data（可选）
                std::vector<uint8_t> ex_blob;
                if (bq_bits > 1) {
                    size_t ex_bits = bq_bits - 1;
                    size_t ex_stride = rabitqlib::ExDataMap<float>::data_bytes(padded_dim, ex_bits);
                    ex_blob.resize(ex_stride * num);
                    char* ex_ptr = reinterpret_cast<char*>(ex_blob.data());
                    for (size_t r = 0; r < num; ++r) {
                        rabitqlib::quant::quantize_compact_ex_bits<float>(
                            flat_data.data() + r * padded_dim,
                            centroid_pad.data(),
                            padded_dim,
                            ex_bits,
                            ex_ptr,
                            rabitqlib::METRIC_L2,
                            cfg
                        );
                        ex_ptr += ex_stride;
                    }
                }

                // 基于 BQ 距离选择每个点的 graph_degree 个邻居（O(n^2) 构建）
                std::vector<uint32_t> adj(num * graph_degree, 0);

                #pragma omp parallel for schedule(static)
                for (size_t u = 0; u < num; ++u) {
                    // 构造查询对象（按之前fastscan路径保持一致的因子设置）
                    std::vector<float> q_pad(padded_dim, 0.0f);
                    std::copy(bucket_padded[u].begin(), bucket_padded[u].end(), q_pad.begin());
                    rabitqlib::SplitBatchQuery<float> qobj(q_pad.data(), padded_dim, (bq_bits > 1 ? bq_bits - 1 : 0), rabitqlib::METRIC_L2, false);
                    float qnorm2 = 0.0f; for (size_t ddd = 0; ddd < padded_dim; ++ddd) qnorm2 += q_pad[ddd] * q_pad[ddd];
                    float qnorm = std::sqrt(qnorm2);
                    qobj.set_g_add(qnorm, 0.0f);

                    std::vector<std::pair<float, uint32_t>> local_dists;
                    local_dists.reserve(num);

                    size_t code_cols = padded_dim / 8;
                    size_t num_rd = (num + 31) & ~31ULL;
                    size_t batches = num_rd / 32;
                    std::array<uint16_t, rabitqlib::fastscan::kBatchSize> accu{};

                    for (size_t b = 0; b < batches; ++b) {
                        const uint8_t* codes = packed_codes.data() + b * (code_cols * 32);
                        rabitqlib::fastscan::accumulate(codes, qobj.lut(), accu.data(), padded_dim);
                        size_t base = b * 32;
                        size_t batch = std::min<size_t>(32, num - base);
                        for (size_t ii = 0; ii < batch; ++ii) {
                            size_t v = base + ii;
                            if (v == u) continue;
                            float ip_est = qobj.delta() * static_cast<float>(accu[ii]) + qobj.sum_vl_lut();
                            float dist_est = f_add[v] + qobj.g_add() + f_rescale[v] * (ip_est + qobj.k1xsumq());
                            if (bq_bits > 1 && !ex_blob.empty()) {
                                size_t ex_bits = bq_bits - 1;
                                size_t ex_stride = rabitqlib::ExDataMap<float>::data_bytes(padded_dim, ex_bits);
                                const char* ex_ptr = reinterpret_cast<const char*>(ex_blob.data()) + (v * ex_stride);
                                auto ip_func = rabitqlib::select_excode_ipfunc(ex_bits);
                                rabitqlib::ConstExDataMap<float> ex_map(ex_ptr, padded_dim, ex_bits);
                                float ex_dist = ex_map.f_add_ex() + qobj.g_add() + (ex_map.f_rescale_ex() *
                                    (static_cast<float>(1 << ex_bits) * (ip_est) + ip_func(q_pad.data(), ex_map.ex_code(), padded_dim) + qobj.kbxsumq()));
                                dist_est = ex_dist;
                            }
                            local_dists.emplace_back(dist_est, static_cast<uint32_t>(v));
                        }
                    }

                    size_t want = std::min(graph_degree, local_dists.size());
                    std::nth_element(local_dists.begin(), local_dists.begin() + want, local_dists.end());
                    for (size_t ttt = 0; ttt < want; ++ttt) {
                        adj[u * graph_degree + ttt] = local_dists[ttt].second;
                    }
                    for (size_t ttt = want; ttt < graph_degree; ++ttt) {
                        adj[u * graph_degree + ttt] = local_dists.empty() ? 0u : local_dists[0].second;
                    }
                }

                // 写出 bqgraph 文件：header(pd,bits,num,degree) + batched codes + f_add + f_rescale + ex_blob + 邻接表
                std::string gpath = "bucket_" + std::to_string(i) + "_bqgraph.bin";
                std::ofstream gout(gpath, std::ios::binary);
                if (!gout.is_open()) {
                    #pragma omp critical
                    std::cerr << "Failed to open " << gpath << " for write" << std::endl;
                } else {
                    uint64_t pd = padded_dim, bits = bq_bits, n = num, deg = graph_degree;
                    gout.write(reinterpret_cast<char*>(&pd), sizeof(uint64_t));
                    gout.write(reinterpret_cast<char*>(&bits), sizeof(uint64_t));
                    gout.write(reinterpret_cast<char*>(&n), sizeof(uint64_t));
                    gout.write(reinterpret_cast<char*>(&deg), sizeof(uint64_t));
                    gout.write(reinterpret_cast<const char*>(packed_codes.data()), packed_codes.size());
                    gout.write(reinterpret_cast<const char*>(f_add.data()), sizeof(float) * num);
                    gout.write(reinterpret_cast<const char*>(f_rescale.data()), sizeof(float) * num);
                    if (!ex_blob.empty()) {
                        gout.write(reinterpret_cast<const char*>(ex_blob.data()), ex_blob.size());
                    }
                    gout.write(reinterpret_cast<const char*>(adj.data()), sizeof(uint32_t) * adj.size());
                    gout.close();
                }
            } else {
                // 小桶：仅生成 bq.bin，不构建图
                size_t padded_dim = (dim + 3) / 4 * 4;
                DataSet bucket_padded = bucket_data;
                if (padded_dim != dim) {
                    for (auto& v : bucket_padded) v.resize(padded_dim, 0.0f);
                }
                const size_t num = bucket_padded.size();
                std::string bq_path = "bucket_" + std::to_string(i) + "_bq.bin";
                std::ofstream out(bq_path, std::ios::binary);
                if (!out.is_open()) {
                    #pragma omp critical
                    std::cerr << "Failed to open " << bq_path << " for write" << std::endl;
                } else {
                    uint64_t pd = padded_dim, bits = bq_bits, n = num;
                    out.write(reinterpret_cast<char*>(&pd), sizeof(uint64_t));
                    out.write(reinterpret_cast<char*>(&bits), sizeof(uint64_t));
                    out.write(reinterpret_cast<char*>(&n), sizeof(uint64_t));
                    using namespace rabitqlib::quant;
                    RabitqConfig cfg = faster_config(padded_dim, bq_bits);
                    std::vector<uint8_t> packed_codes((((num + 31) & ~31ULL) / 32) * (padded_dim / 8) * 32);
                    std::vector<float> f_add(num), f_rescale(num), f_err(num);
                    std::vector<float> flat_data(num * padded_dim, 0.0f);
                    for (size_t r = 0; r < num; ++r) {
                        std::copy(bucket_padded[r].begin(), bucket_padded[r].end(), flat_data.begin() + r * padded_dim);
                    }
                    std::vector<float> centroid_pad(padded_dim, 0.0f);
                    std::copy(final_centroids[i].begin(), final_centroids[i].end(), centroid_pad.begin());
                    rabitqlib::quant::rabitq_impl::one_bit::one_bit_batch_code<float, false>(
                        flat_data.data(),
                        centroid_pad.data(),
                        num,
                        padded_dim,
                        packed_codes.data(),
                        f_add.data(),
                        f_rescale.data(),
                        f_err.data(),
                        rabitqlib::METRIC_L2
                    );
                    out.write(reinterpret_cast<const char*>(packed_codes.data()), packed_codes.size());
                    out.write(reinterpret_cast<const char*>(f_add.data()), sizeof(float) * num);
                    out.write(reinterpret_cast<const char*>(f_rescale.data()), sizeof(float) * num);
                    if (bq_bits > 1) {
                        size_t ex_bits = bq_bits - 1;
                        size_t ex_stride = rabitqlib::ExDataMap<float>::data_bytes(padded_dim, ex_bits);
                        std::vector<uint8_t> ex_blob(ex_stride * num);
                        char* ex_ptr = reinterpret_cast<char*>(ex_blob.data());
                        for (size_t r = 0; r < num; ++r) {
                            rabitqlib::quant::quantize_compact_ex_bits<float>(
                                flat_data.data() + r * padded_dim,
                                centroid_pad.data(),
                                padded_dim,
                                ex_bits,
                                ex_ptr,
                                rabitqlib::METRIC_L2,
                                cfg
                            );
                            ex_ptr += ex_stride;
                        }
                        out.write(reinterpret_cast<const char*>(ex_blob.data()), ex_blob.size());
                    }
                    out.close();
                }
            }
        }
        
        // 原子操作更新计数器
        buckets_built.fetch_add(1);
        
        // 线程安全的完成输出
        #pragma omp critical
        {
            std::cout << "Thread " << omp_get_thread_num() << " completed bucket " << i 
                      << " (" << buckets_built.load() << "/" << buckets_to_build.size() << " finished)" << std::endl;
        }
    }
    
    auto build_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> build_time = build_end_time - build_start_time;

    // 计算并输出分桶统计信息

    std::cout << "Bucketing statistics:" << std::endl;
    std::cout << "  Total vectors: " << num_points << std::endl;
    std::cout << "  Total bucket assignments: " << total_bucket_assignments << std::endl;
    std::cout << "  Average buckets per vector: " << std::fixed << std::setprecision(2) 
              << average_buckets_per_vector << std::endl;
    std::cout << "  Bucket utilization: " << std::fixed << std::setprecision(1) 
              << (average_buckets_per_vector / l) * 100.0 << "% (vs " << l << " max)" << std::endl;

    std::cout << "Built " << buckets_built.load() << " bucket indices out of " << m << " total buckets." << std::endl;
    std::cout << "Total bucket build time: " << build_time.count() << " seconds" << std::endl;
    std::cout << "Average time per bucket: " << (buckets_built.load() > 0 ? build_time.count() / buckets_built.load() : 0) << " seconds" << std::endl;
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
    float beta = 0.0f; // 新增：加载beta参数
    double average_buckets_per_vector = 0.0; // 新增：加载平均分桶数
    int use_bq_flag = 0; // 新增：是否使用bq
    size_t bq_bits = 4;  // 新增：bq bits
    std::ifstream meta_reader("medoid_meta.txt");
    if (!meta_reader.is_open()) {
        std::cerr << "FATAL: medoid_meta.txt not found. Please run build mode first." << std::endl;
        return;
    }
    meta_reader >> dim >> graph_degree >> beta >> average_buckets_per_vector >> use_bq_flag >> bq_bits;
    meta_reader.close();
    
    if (dim == 0 || graph_degree == 0) {
        std::cerr << "FATAL: Failed to read metadata from medoid_meta.txt or metadata is invalid." << std::endl;
        return;
    }
    std::cout << "Read metadata: dim=" << dim << ", graph_degree=" << graph_degree << ", beta=" << beta << std::endl;
    std::cout << "Read metadata: average_buckets_per_vector=" << average_buckets_per_vector << std::endl;
    std::cout << "Read metadata: use_bq=" << use_bq_flag << ", bits=" << bq_bits << std::endl;

    // 读取 BQ_GRAPH_THRESHOLD 环境变量，默认 1000
    size_t bq_graph_threshold = 1000;
    if (const char* env_thr = std::getenv("BQ_GRAPH_THRESHOLD")) {
        try { bq_graph_threshold = std::stoul(env_thr); } catch (...) {}
    }

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
    std::vector<size_t> f_values = {static_cast<size_t>(f)};
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
            results[i] = search_two_stage(query, *medoid_index, buckets, f_val, k, top_k, bucket_index_cache, dim, use_bq_flag == 1, bq_graph_threshold);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end_time - start_time;
        double qps = num_queries / diff.count();

        // Flatten results for recall calculation
        std::vector<uint32_t> our_results_flat(num_queries * top_k, 0);
        for(size_t i = 0; i < num_queries; ++i) {
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
    
    // 输出缓存统计信息
    bucket_index_cache.print_stats();
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