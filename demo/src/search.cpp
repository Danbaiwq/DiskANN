#include "search.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include "index.h"
#include "parameters.h"
#include "kmeans.h"

// rabitq 量化依赖
#include "/home/danbai.wq/DiskANN/rabitq/rabitqlib/quantization/rabitq.hpp"
#include "/home/danbai.wq/DiskANN/rabitq/rabitqlib/quantization/data_layout.hpp"
#include "/home/danbai.wq/DiskANN/rabitq/rabitqlib/fastscan/fastscan.hpp"
#include "/home/danbai.wq/DiskANN/rabitq/rabitqlib/index/query.hpp"
#include "/home/danbai.wq/DiskANN/rabitq/rabitqlib/utils/space.hpp"

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

// 核心改进：非阻塞的获取方法
IndexPtr IndexCache::get_non_blocking(uint32_t key) {
    // 第一步：快速检查缓存中是否存在
    {
        std::shared_lock<std::shared_mutex> lock(mtx);
        auto it = cache.find(key);
        if (it != cache.end()) {
            cache_hits.fetch_add(1);
            return it->second;
        }
    }
    
    cache_misses.fetch_add(1);
    
    // 第二步：缓存未命中，直接从文件加载（不等待）
    std::string index_path = "bucket_" + std::to_string(key) + "_vamana.index";
    IndexPtr index = load_index(index_path, dim);
    
    if (!index) {
        return nullptr;
    }
    
    // 第三步：尝试非阻塞插入缓存，如果不能立即插入则直接返回索引
    if (try_put_non_blocking(key, index)) {
        // 成功插入缓存
        return index;
    } else {
        // 无法插入缓存（可能缓存满了且正在被其他线程使用），直接返回加载的索引
        direct_loads.fetch_add(1);
        return index;
    }
}

bool IndexCache::try_put_non_blocking(uint32_t key, IndexPtr index) {
    // 尝试获取写锁，如果无法立即获取则返回false
    std::unique_lock<std::shared_mutex> lock(mtx, std::try_to_lock);
    if (!lock.owns_lock()) {
        return false; // 无法立即获取锁，避免阻塞
    }
    
    // 再次检查缓存中是否已经存在该key（双重检查）
    if (cache.find(key) != cache.end()) {
        return true; // 已经在缓存中，认为成功
    }
    
    auto index_size = diskann::estimate_ram_usage(index->get_num_points(), dim, sizeof(float), graph_degree);
    
    // 如果添加这个索引会超过缓存大小限制，且需要LRU淘汰
    if (current_size + index_size > max_size) {
        // 检查是否有足够的项目可以快速淘汰
        if (lru.empty()) {
            return false; // 无法释放空间
        }
        
        // 尝试快速淘汰一些项目，但不要淘汰太多以避免阻塞
        size_t needed_space = index_size;
        size_t space_to_free = 0;
        size_t items_to_evict = 0;
        
        auto it = lru.rbegin();
        while (it != lru.rend() && current_size - space_to_free + index_size > max_size && items_to_evict < 3) {
            auto evict_key = *it;
            auto evict_index = cache.at(evict_key);
            space_to_free += diskann::estimate_ram_usage(evict_index->get_num_points(), dim, sizeof(float), graph_degree);
            ++it;
            ++items_to_evict;
        }
        
        if (current_size - space_to_free + index_size > max_size) {
            return false; // 快速淘汰仍然不够空间
        }
        
        // 执行快速淘汰
        for (size_t i = 0; i < items_to_evict; ++i) {
            auto last_key = lru.back();
            lru.pop_back();
            auto evicted_index = cache.at(last_key);
            current_size -= diskann::estimate_ram_usage(evicted_index->get_num_points(), dim, sizeof(float), graph_degree);
            cache.erase(last_key);
        }
    }
    
    // 添加到缓存
    cache[key] = index;
    lru.push_front(key);
    current_size += index_size;
    
    return true;
}

void IndexCache::print_stats() const {
    uint64_t hits = cache_hits.load();
    uint64_t misses = cache_misses.load();
    uint64_t direct = direct_loads.load();
    uint64_t total = hits + misses;
    
    std::cout << "\n--- IndexCache 统计信息 ---" << std::endl;
    std::cout << "缓存命中次数: " << hits << std::endl;
    std::cout << "缓存未命中次数: " << misses << std::endl;
    std::cout << "直接文件加载次数: " << direct << std::endl;
    std::cout << "缓存命中率: " << (total > 0 ? (double)hits / total * 100.0 : 0.0) << "%" << std::endl;
    std::cout << "当前缓存大小: " << cache.size() << " 个索引" << std::endl;
    std::cout << "当前内存使用: " << current_size / (1024 * 1024) << " MB" << std::endl;
    std::cout << "最大内存限制: " << max_size / (1024 * 1024) << " MB" << std::endl;
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

// 简单的bq桶数据结构（按桶保存一次读取的数据）
struct BQBucketData {
    size_t padded_dim{0};
    size_t bits{4};
    size_t num_points{0};
    std::vector<uint8_t> bin_codes;   // packed codes
    std::vector<float> f_add;
    std::vector<float> f_rescale;
    // ex data（可选）
    std::vector<uint8_t> ex_blob;
};

static bool load_bq_bucket(uint32_t bucket_id, BQBucketData& out) {
    std::string path = "bucket_" + std::to_string(bucket_id) + "_bq.bin";
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;
    uint64_t pd, bits, n;
    in.read(reinterpret_cast<char*>(&pd), sizeof(uint64_t));
    in.read(reinterpret_cast<char*>(&bits), sizeof(uint64_t));
    in.read(reinterpret_cast<char*>(&n), sizeof(uint64_t));
    out.padded_dim = static_cast<size_t>(pd);
    out.bits = static_cast<size_t>(bits);
    out.num_points = static_cast<size_t>(n);
    size_t cols = out.padded_dim / 8;
    size_t num_rd = (out.num_points + 31) & ~31ULL;
    size_t code_bytes = (num_rd / 32) * cols * 32;
    out.bin_codes.resize(code_bytes);
    out.f_add.resize(out.num_points);
    out.f_rescale.resize(out.num_points);
    in.read(reinterpret_cast<char*>(out.bin_codes.data()), code_bytes);
    in.read(reinterpret_cast<char*>(out.f_add.data()), sizeof(float) * out.num_points);
    in.read(reinterpret_cast<char*>(out.f_rescale.data()), sizeof(float) * out.num_points);

    // 读取 ex_data（若存在）
    if (out.bits > 1) {
        size_t ex_stride = rabitqlib::ExDataMap<float>::data_bytes(out.padded_dim, out.bits - 1);
        out.ex_blob.resize(ex_stride * out.num_points);
        in.read(reinterpret_cast<char*>(out.ex_blob.data()), out.ex_blob.size());
    }
    return true;
}

QueryResult search_two_stage(
    const std::vector<float>& query,
    diskann::Index<float, uint32_t, uint32_t>& medoid_index,
    const Buckets& buckets,
    size_t f,
    size_t k,
    size_t top_k,
    IndexCache& index_cache,
    size_t dim,
    bool use_bq
) {
    // 第一步：在medoid_vamana查找最近的bucket_id
    std::vector<uint32_t> nearest_bucket_ids(f);
    medoid_index.search(query.data(), f, f, nearest_bucket_ids.data(), nullptr);

    std::vector<std::pair<float, uint32_t>> candidates;
    std::set<uint32_t> visited_ids;

    if (!use_bq) {
        // 原始raw逻辑：使用每桶Vamana子图
        for (uint32_t bucket_id : nearest_bucket_ids) {
            auto bucket_index = index_cache.get_non_blocking(bucket_id);  // 使用非阻塞方法
            if (bucket_index) {
                size_t actual_k = std::min(k, bucket_index->get_num_points());
                if (actual_k == 0) continue;

                std::vector<uint32_t> result_tags(actual_k);
                std::vector<float> result_dists(actual_k);
                bucket_index->search(query.data(), actual_k, actual_k, result_tags.data(), result_dists.data());

                for (size_t i = 0; i < actual_k; ++i) {
                    uint32_t local_idx = result_tags[i];
                    if (local_idx < buckets[bucket_id].size()) {
                        uint32_t global_id = buckets[bucket_id][local_idx];
                        if (visited_ids.insert(global_id).second) {
                            candidates.emplace_back(result_dists[i], global_id);
                        }
                    }
                }
            }
        }
    } else {
        // bq 模式：按桶加载量化数据，使用 fastscan + 正确的估计公式；若 bits>1 再进行 ex_bits boosting
        for (uint32_t bucket_id : nearest_bucket_ids) {
            BQBucketData bd;
            if (!load_bq_bucket(bucket_id, bd) || bd.num_points == 0) continue;
            size_t code_cols = bd.padded_dim / 8;
            size_t num = bd.num_points;

            // 准备查询：pad 到 padded_dim
            std::vector<float> q_pad(bd.padded_dim, 0.0f);
            std::copy(query.begin(), query.end(), q_pad.begin());

            size_t ex_bits = bd.bits > 1 ? (bd.bits - 1) : 0;
            rabitqlib::SplitBatchQuery<float> qobj(q_pad.data(), bd.padded_dim, ex_bits, rabitqlib::METRIC_L2, false);
            float qnorm2 = 0.0f; for (size_t d = 0; d < dim; ++d) qnorm2 += query[d] * query[d];
            float qnorm = std::sqrt(qnorm2);
            qobj.set_g_add(qnorm, 0.0f);

            std::vector<std::pair<float, uint32_t>> local_cand;
            local_cand.reserve(std::min(k, num));

            std::array<uint16_t, rabitqlib::fastscan::kBatchSize> accu{};

            size_t num_rd = (num + 31) & ~31ULL;
            size_t batches = num_rd / 32;
            for (size_t b = 0; b < batches; ++b) {
                const uint8_t* codes = bd.bin_codes.data() + b * (code_cols * 32);
                rabitqlib::fastscan::accumulate(codes, qobj.lut(), accu.data(), bd.padded_dim);

                size_t base = b * 32;
                size_t batch = std::min<size_t>(32, num - base);
                for (size_t i = 0; i < batch; ++i) {
                    size_t idx = base + i;
                    float ip_est = qobj.delta() * static_cast<float>(accu[i]) + qobj.sum_vl_lut();
                    float dist_est = bd.f_add[idx] + qobj.g_add() + bd.f_rescale[idx] * (ip_est + qobj.k1xsumq());

                    // ex_bits boosting（如果存在 ex_data）
                    if (bd.bits > 1 && !bd.ex_blob.empty()) {
                        size_t ex_bits = bd.bits - 1;
                        size_t ex_stride = rabitqlib::ExDataMap<float>::data_bytes(bd.padded_dim, ex_bits);
                        const char* ex_ptr = reinterpret_cast<const char*>(bd.ex_blob.data()) + (idx * ex_stride);
                        auto ip_func = rabitqlib::select_excode_ipfunc(ex_bits);
                        rabitqlib::ConstExDataMap<float> ex_map(ex_ptr, bd.padded_dim, ex_bits);
                        float ex_dist = ex_map.f_add_ex() + qobj.g_add() + (ex_map.f_rescale_ex() *
                            (static_cast<float>(1 << ex_bits) * (ip_est) + ip_func(q_pad.data(), ex_map.ex_code(), bd.padded_dim) + qobj.kbxsumq()));
                        dist_est = ex_dist;
                    }

                    local_cand.emplace_back(dist_est, static_cast<uint32_t>(idx));
                }
            }

            size_t want = std::min(k, local_cand.size());
            if (local_cand.size() > want) {
                std::nth_element(local_cand.begin(), local_cand.begin() + want, local_cand.end());
                local_cand.resize(want);
            }
            for (auto& p : local_cand) {
                uint32_t local_idx = p.second;
                if (local_idx < buckets[bucket_id].size()) {
                    uint32_t global_id = buckets[bucket_id][local_idx];
                    if (visited_ids.insert(global_id).second) {
                        candidates.emplace_back(p.first, global_id);
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