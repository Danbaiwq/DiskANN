#pragma once

#include "utils.h"
#include "index.h"
#include <string>
#include <vector>
#include <memory>
#include <list>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <atomic>

// A struct to hold the search results for a single query
struct QueryResult {
    std::vector<uint32_t> ids;
    std::vector<float> distances;
};

using IndexPtr = std::shared_ptr<diskann::Index<float, uint32_t, uint32_t>>;

class IndexCache {
public:
    IndexCache(size_t max_size_bytes, size_t dim, size_t graph_degree);
    IndexPtr get(uint32_t key);
    
    // 非阻塞的获取方法 - 核心改进
    IndexPtr get_non_blocking(uint32_t key);
    
    // 获取缓存统计信息
    void print_stats() const;

    // 清空缓存，释放内存
    void clear();

private:
    void put_locked(uint32_t key, IndexPtr index);
    
    // 非阻塞的尝试插入方法
    bool try_put_non_blocking(uint32_t key, IndexPtr index);

    size_t max_size;
    size_t current_size;
    size_t dim;
    size_t graph_degree;
    std::unordered_map<uint32_t, IndexPtr> cache;
    std::list<uint32_t> lru;
    std::shared_mutex mtx; 
    
    // 为非阻塞实现添加统计信息
    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> cache_misses{0};
    std::atomic<uint64_t> direct_loads{0};  // 直接从文件加载，未放入缓存的次数
};

// Load a Vamana index from a file
IndexPtr load_index(const std::string& index_path, size_t dim);

// Load ground truth data from a file
std::vector<uint32_t> load_ground_truth(const std::string& gt_path, size_t& num_queries, size_t& gt_dim);

// Perform the two-stage search
QueryResult search_two_stage(
    const std::vector<float>& query,
    diskann::Index<float, uint32_t, uint32_t>& medoid_index,
    const Buckets& buckets, // Pass buckets to map local bucket-ID to global point-ID
    size_t f,
    size_t k,
    size_t top_k,
    IndexCache& index_cache,
    size_t dim,
    bool use_bq, // 新增：是否使用bq量化向量查询
    size_t bq_graph_threshold // 新增：大桶阈值，>=阈值时图搜
);

// Calculate recall
double calculate_recall(
    size_t num_queries,
    const uint32_t* our_results,
    size_t our_dim,
    const uint32_t* gt_results,
    size_t gt_dim,
    size_t recall_at
);

// Save cluster access statistics to file
void save_cluster_access_stats(const std::string& filename);

// 清理 BQ 查询缓存（桶/图），用于查询结束后及时释放内存
void clear_bq_caches();

// 释放簇访问统计占用的全局内存
void release_cluster_access_stats();

// 新增：生成 BQ 候选的工具函数（供异步流水线 Stage1 调用）
std::vector<uint32_t> generate_bq_candidates(
    const std::vector<float>& query,
    diskann::Index<float, uint32_t, uint32_t>& medoid_index,
    const Buckets& buckets,
    size_t f,
    size_t k,
    size_t dim,
    size_t bq_graph_threshold
);

// 新增：SIMD 距离计算对外接口（供 pipeline 复用 search.cpp 的实现）
float l2_distance_simd(const float* a, const float* b, size_t dim); 