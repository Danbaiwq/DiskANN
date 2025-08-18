#include "search.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <queue>
#include <limits>
#include <cstdlib>
#include <filesystem>
#include "index.h"
#include "parameters.h"
#include "kmeans.h"

// rabitq 量化依赖
#include "/home/danbai.wq/DiskANN/rabitq/rabitqlib/quantization/rabitq.hpp"
#include "/home/danbai.wq/DiskANN/rabitq/rabitqlib/quantization/data_layout.hpp"
#include "/home/danbai.wq/DiskANN/rabitq/rabitqlib/fastscan/fastscan.hpp"
#include "/home/danbai.wq/DiskANN/rabitq/rabitqlib/index/query.hpp"
#include "/home/danbai.wq/DiskANN/rabitq/rabitqlib/utils/space.hpp"

// 新增：mmap 读取 base.fbin 用于真距复排
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// 新增：缓存所需
#include <shared_mutex>
#include <unordered_map>
#include <list>
#include <functional>
#include <mutex>
#include <atomic>
#include <thread>
#include <condition_variable>
#include <future>
#include <deque>
#include <memory>

#ifdef HAS_IO_URING
extern "C" {
#include <liburing.h>
}
#endif

// --- 简易 I/O 线程池（用于任务 4） ---
class IOThreadPool {
public:
    explicit IOThreadPool(size_t num_threads) : stop_(false) {
        if (num_threads == 0) num_threads = 1;
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this]() {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lk(mtx_);
                        cv_.wait(lk, [this]{ return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop_front();
                    }
                    task();
                }
            });
        }
    }
    ~IOThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) { if (t.joinable()) t.join(); }
    }

    template <typename F>
    auto submit(F&& f) -> std::future<typename std::invoke_result<F>::type> {
        using R = typename std::invoke_result<F>::type;
        auto p = std::make_shared<std::packaged_task<R()>>(std::forward<F>(f));
        std::future<R> fut = p->get_future();
        {
            std::lock_guard<std::mutex> lk(mtx_);
            tasks_.emplace_back([p]{ (*p)(); });
        }
        cv_.notify_one();
        return fut;
    }
private:
    std::vector<std::thread> workers_;
    std::deque<std::function<void()>> tasks_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool stop_;
};

static inline std::string get_env_str_local(const char* key) {
    const char* v = std::getenv(key);
    return v ? std::string(v) : std::string();
}
static inline std::string join_path_local(const std::string& dir, const std::string& name) {
    if (dir.empty()) return name;
    if (dir.back() == '/') return dir + name;
    return dir + "/" + name;
}
static inline std::string resolve_read_path_local(const std::string& name) {
    std::string in_dir = get_env_str_local("DEMO_INPUT_DIR");
    if (in_dir.empty()) in_dir = get_env_str_local("DEMO_OUTPUT_DIR");
    if (in_dir.empty()) return name;
    return join_path_local(in_dir, name);
}

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

// 新增：大桶 BQ 图数据结构与加载器
struct BQGraphData {
    size_t padded_dim{0};
    size_t bits{4};
    size_t num_points{0};
    size_t degree{0};
    std::vector<uint8_t> bin_codes;   // packed codes
    std::vector<float> f_add;
    std::vector<float> f_rescale;
    std::vector<uint8_t> ex_blob;     // 可选
    std::vector<uint32_t> adj;        // 邻接表（按行存储，num_points * degree）
};

// --- 新增：BQ 产物缓存（非阻塞，内存受限 LRU） ---
namespace {
static inline size_t estimate_bucket_bytes(const BQBucketData& b) {
    size_t bytes = b.bin_codes.size();
    bytes += b.f_add.size() * sizeof(float);
    bytes += b.f_rescale.size() * sizeof(float);
    bytes += b.ex_blob.size();
    return bytes;
}
static inline size_t estimate_graph_bytes(const BQGraphData& g) {
    size_t bytes = g.bin_codes.size();
    bytes += g.f_add.size() * sizeof(float);
    bytes += g.f_rescale.size() * sizeof(float);
    bytes += g.ex_blob.size();
    bytes += g.adj.size() * sizeof(uint32_t);
    return bytes;
}

template <typename T, size_t (*estimate_bytes)(const T&)> class SimpleLRUCache {
public:
    explicit SimpleLRUCache(size_t max_bytes) : max_bytes_(max_bytes), cur_bytes_(0) {}

    std::shared_ptr<const T> get(uint32_t key) {
        std::shared_lock<std::shared_mutex> lk(mtx_);
        auto it = map_.find(key);
        if (it == map_.end()) return nullptr;
        return it->second.value;
    }

    std::shared_ptr<const T> get_or_load(uint32_t key, const std::function<bool(T&)>& loader) {
        if (max_bytes_ == 0) {
            auto obj = std::make_shared<T>();
            if (!loader(*obj)) return nullptr;
            return obj;
        }
        {
            std::shared_lock<std::shared_mutex> rlk(mtx_);
            auto it = map_.find(key);
            if (it != map_.end()) return it->second.value;
        }
        auto loaded = std::make_shared<T>();
        if (!loader(*loaded)) return nullptr;
        size_t need = estimate_bytes(*loaded);
        std::unique_lock<std::shared_mutex> wlk(mtx_, std::try_to_lock);
        if (!wlk.owns_lock()) {
            return loaded;
        }
        auto it2 = map_.find(key);
        if (it2 != map_.end()) return it2->second.value;
        size_t evicted = 0;
        while (cur_bytes_ + need > max_bytes_ && !lru_.empty() && evicted < 3) {
            uint32_t k = lru_.back();
            lru_.pop_back();
            auto it = map_.find(k);
            if (it != map_.end()) {
                cur_bytes_ -= it->second.bytes;
                map_.erase(it);
            }
            evicted++;
        }
        if (cur_bytes_ + need > max_bytes_) {
            return loaded;
        }
        lru_.push_front(key);
        map_[key] = Node{loaded, need};
        cur_bytes_ += need;
        return loaded;
    }

private:
    struct Node { std::shared_ptr<const T> value; size_t bytes; };
    size_t max_bytes_;
    size_t cur_bytes_;
    mutable std::shared_mutex mtx_;
    std::list<uint32_t> lru_;
    std::unordered_map<uint32_t, Node> map_;
};

using BQBucketCache = SimpleLRUCache<BQBucketData, estimate_bucket_bytes>;
using BQGraphCache  = SimpleLRUCache<BQGraphData,  estimate_graph_bytes>;
}

static bool load_bq_bucket(uint32_t bucket_id, BQBucketData& out) {
    std::string path = resolve_read_path_local("bucket_" + std::to_string(bucket_id) + "_bq.bin");
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

    if (out.bits > 1) {
        size_t ex_stride = rabitqlib::ExDataMap<float>::data_bytes(out.padded_dim, out.bits - 1);
        out.ex_blob.resize(ex_stride * out.num_points);
        in.read(reinterpret_cast<char*>(out.ex_blob.data()), out.ex_blob.size());
    }
    return true;
}

static bool load_bq_graph(uint32_t bucket_id, BQGraphData& g) {
    std::string path = resolve_read_path_local("bucket_" + std::to_string(bucket_id) + "_bqgraph.bin");
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;
    uint64_t pd, bits, n, deg;
    in.read(reinterpret_cast<char*>(&pd), sizeof(uint64_t));
    in.read(reinterpret_cast<char*>(&bits), sizeof(uint64_t));
    in.read(reinterpret_cast<char*>(&n), sizeof(uint64_t));
    in.read(reinterpret_cast<char*>(&deg), sizeof(uint64_t));
    g.padded_dim = static_cast<size_t>(pd);
    g.bits = static_cast<size_t>(bits);
    g.num_points = static_cast<size_t>(n);
    g.degree = static_cast<size_t>(deg);
    size_t cols = g.padded_dim / 8;
    size_t num_rd = (g.num_points + 31) & ~31ULL;
    size_t code_bytes = (num_rd / 32) * cols * 32;
    g.bin_codes.resize(code_bytes);
    g.f_add.resize(g.num_points);
    g.f_rescale.resize(g.num_points);
    in.read(reinterpret_cast<char*>(g.bin_codes.data()), code_bytes);
    in.read(reinterpret_cast<char*>(g.f_add.data()), sizeof(float) * g.num_points);
    in.read(reinterpret_cast<char*>(g.f_rescale.data()), sizeof(float) * g.num_points);
    if (g.bits > 1) {
        size_t ex_stride = rabitqlib::ExDataMap<float>::data_bytes(g.padded_dim, g.bits - 1);
        g.ex_blob.resize(ex_stride * g.num_points);
        in.read(reinterpret_cast<char*>(g.ex_blob.data()), g.ex_blob.size());
    }
    g.adj.resize(g.num_points * g.degree);
    in.read(reinterpret_cast<char*>(g.adj.data()), sizeof(uint32_t) * g.adj.size());
    return true;
}

// 使用 BQ 距离在图上进行近邻搜索（简化版：贪心+候选队列），基于已加载图
static void bq_graph_search_loaded(
    const BQGraphData& G,
    const std::vector<float>& query,
    size_t orig_dim,
    size_t k,
    std::vector<std::pair<float, uint32_t>>& out_local_cand
) {
    if (G.num_points == 0) return;
    std::vector<float> q_pad(G.padded_dim, 0.0f);
    std::copy(query.begin(), query.end(), q_pad.begin());
    size_t ex_bits = G.bits > 1 ? (G.bits - 1) : 0;
    rabitqlib::SplitBatchQuery<float> qobj(q_pad.data(), G.padded_dim, ex_bits, rabitqlib::METRIC_L2, false);
    float qnorm2 = 0.0f; for (size_t d = 0; d < orig_dim; ++d) qnorm2 += query[d] * query[d];
    float qnorm = std::sqrt(qnorm2);
    qobj.set_g_add(qnorm, 0.0f);

    const size_t code_cols = G.padded_dim / 8;

    auto eval_dist = [&](uint32_t idx)->float {
        size_t base = (idx / 32) * 32;
        size_t pos = idx % 32;
        const uint8_t* codes = G.bin_codes.data() + (base / 32) * (code_cols * 32);
        std::array<uint16_t, rabitqlib::fastscan::kBatchSize> accu{};
        rabitqlib::fastscan::accumulate(codes, qobj.lut(), accu.data(), G.padded_dim);
        float ip_est = qobj.delta() * static_cast<float>(accu[pos]) + qobj.sum_vl_lut();
        float dist_est = G.f_add[idx] + qobj.g_add() + G.f_rescale[idx] * (ip_est + qobj.k1xsumq());
        if (G.bits > 1 && !G.ex_blob.empty()) {
            size_t ex_stride = rabitqlib::ExDataMap<float>::data_bytes(G.padded_dim, ex_bits);
            const char* ex_ptr = reinterpret_cast<const char*>(G.ex_blob.data()) + (idx * ex_stride);
            auto ip_func = rabitqlib::select_excode_ipfunc(ex_bits);
            rabitqlib::ConstExDataMap<float> ex_map(ex_ptr, G.padded_dim, ex_bits);
            float ex_dist = ex_map.f_add_ex() + qobj.g_add() + (ex_map.f_rescale_ex() *
                (static_cast<float>(1 << ex_bits) * (ip_est) + ip_func(q_pad.data(), ex_map.ex_code(), G.padded_dim) + qobj.kbxsumq()));
            dist_est = ex_dist;
        }
        return dist_est;
    };

    size_t ef_search = 128;
    if (const char* env = std::getenv("BQ_EF_SEARCH")) {
        try { ef_search = std::max<size_t>(1, std::stoul(env)); } catch (...) {}
    }
    size_t num_seeds = 8;
    if (const char* env = std::getenv("BQ_SEEDS")) {
        try { num_seeds = std::max<size_t>(1, std::stoul(env)); } catch (...) {}
    }

    std::vector<uint32_t> entry_points;
    if (G.num_points > 0) {
        num_seeds = std::min(num_seeds, G.num_points);
        if (num_seeds == 1) {
            entry_points.push_back(0);
        } else {
            size_t stride = std::max<size_t>(1, G.num_points / num_seeds);
            for (size_t i = 0; i < num_seeds; ++i) {
                uint32_t ep = static_cast<uint32_t>(std::min(G.num_points - 1, i * stride));
                if (entry_points.empty() || entry_points.back() != ep) entry_points.push_back(ep);
            }
        }
    }

    std::vector<char> visited(G.num_points, 0);
    using Node = std::pair<float, uint32_t>;
    auto cmp = [](const Node& a, const Node& b){ return a.first > b.first; };
    std::priority_queue<Node, std::vector<Node>, decltype(cmp)> cand_queue(cmp);

    for (auto ep : entry_points) {
        if (ep < G.num_points && !visited[ep]) {
            float d = eval_dist(ep);
            cand_queue.emplace(d, ep);
        }
    }

    std::vector<Node> best;
    best.reserve(std::max(k, static_cast<size_t>(ef_search)));

    size_t expanded = 0;
    while (!cand_queue.empty() && expanded < ef_search) {
        auto [cd, u] = cand_queue.top();
        cand_queue.pop();
        if (visited[u]) continue;
        visited[u] = 1;
        ++expanded;
        best.emplace_back(cd, u);

        for (size_t t = 0; t < G.degree; ++t) {
            uint32_t v = G.adj[u * G.degree + t];
            if (v >= G.num_points || visited[v]) continue;
            float dv = eval_dist(v);
            cand_queue.emplace(dv, v);
        }
    }

    size_t want = std::min(k, best.size());
    if (best.size() > want) {
        std::nth_element(best.begin(), best.begin() + want, best.end());
        best.resize(want);
    }
    out_local_cand.insert(out_local_cand.end(), best.begin(), best.end());
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
    bool use_bq,
    size_t bq_graph_threshold
) {
    // 懒加载全局 BQ 缓存
    static std::once_flag s_once;
    static std::unique_ptr<BQBucketCache> s_bucket_cache;
    static std::unique_ptr<BQGraphCache>  s_graph_cache;
    std::call_once(s_once, [](){
        size_t bucket_mb = 256, graph_mb = 256;
        if (const char* env = std::getenv("BQ_BUCKET_CACHE_MB")) { try { bucket_mb = std::stoul(env); } catch (...) {} }
        if (const char* env = std::getenv("BQ_GRAPH_CACHE_MB"))  { try { graph_mb  = std::stoul(env); } catch (...) {} }
        s_bucket_cache = std::make_unique<BQBucketCache>(bucket_mb * 1024ULL * 1024ULL);
        s_graph_cache  = std::make_unique<BQGraphCache>( graph_mb * 1024ULL * 1024ULL);
        std::cout << "[BQ Cache] bucket=" << bucket_mb << "MB, graph=" << graph_mb << "MB" << std::endl;
    });

    // 可选：按桶大小预热前 N 个桶，减少前期 I/O 抖动
    static std::atomic<bool> s_prewarmed{false};
    static std::mutex s_prewarm_mtx;
    if (use_bq && !s_prewarmed.load()) {
        std::lock_guard<std::mutex> g(s_prewarm_mtx);
        if (!s_prewarmed.load()) {
            size_t topN = 0;
            if (const char* env = std::getenv("BQ_PREWARM_TOP")) { try { topN = std::stoul(env); } catch (...) {} }
            if (topN > 0) {
                std::vector<std::pair<uint32_t, size_t>> by_size;
                by_size.reserve(buckets.size());
                for (uint32_t bid = 0; bid < buckets.size(); ++bid) {
                    by_size.emplace_back(bid, buckets[bid].size());
                }
                std::sort(by_size.begin(), by_size.end(), [](auto& a, auto& b){ return a.second > b.second; });
                topN = std::min(topN, by_size.size());
                for (size_t i = 0; i < topN; ++i) {
                    uint32_t bid = by_size[i].first;
                    size_t bsz = by_size[i].second;
                    if (bsz >= bq_graph_threshold) {
                        if (s_graph_cache) { s_graph_cache->get_or_load(bid, [&](BQGraphData& dst){ return load_bq_graph(bid, dst); }); }
                    } else {
                        if (s_bucket_cache) { s_bucket_cache->get_or_load(bid, [&](BQBucketData& dst){ return load_bq_bucket(bid, dst); }); }
                    }
                }
                std::cout << "[BQ Cache] prewarmed top " << topN << " buckets by size" << std::endl;

            }
            s_prewarmed.store(true);
        }
    }

    // 第一步：在medoid_vamana查找最近的bucket_id
    std::vector<uint32_t> nearest_bucket_ids(f);
    medoid_index.search(query.data(), f, f, nearest_bucket_ids.data(), nullptr);

    std::vector<std::pair<float, uint32_t>> candidates;
    std::set<uint32_t> visited_ids;

    if (!use_bq) {
        for (uint32_t bucket_id : nearest_bucket_ids) {
            auto bucket_index = index_cache.get_non_blocking(bucket_id);
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
        for (uint32_t bucket_id : nearest_bucket_ids) {
            const size_t bucket_size = buckets[bucket_id].size();
            if (bucket_size >= bq_graph_threshold) {
                std::shared_ptr<const BQGraphData> gp;
                if (s_graph_cache) {
                    gp = s_graph_cache->get_or_load(bucket_id, [&](BQGraphData& dst){ return load_bq_graph(bucket_id, dst); });
                }
                BQGraphData localG;
                const BQGraphData* G = nullptr;
                if (gp) {
                    G = gp.get();
                } else {
                    if (load_bq_graph(bucket_id, localG)) G = &localG; else G = nullptr;
                }
                if (G && G->num_points > 0) {
                    std::vector<std::pair<float, uint32_t>> local_cand;
                    bq_graph_search_loaded(*G, query, dim, k, local_cand);
                    if (!local_cand.empty()) {
                        for (auto& p : local_cand) {
                            uint32_t local_idx = p.second;
                            if (local_idx < buckets[bucket_id].size()) {
                                uint32_t global_id = buckets[bucket_id][local_idx];
                                if (visited_ids.insert(global_id).second) {
                                    candidates.emplace_back(p.first, global_id);
                                }
                            }
                        }
                        continue;
                    }
                }
            }

            std::shared_ptr<const BQBucketData> bp;
            if (s_bucket_cache) {
                bp = s_bucket_cache->get_or_load(bucket_id, [&](BQBucketData& dst){ return load_bq_bucket(bucket_id, dst); });
            }
            BQBucketData localB;
            const BQBucketData* bd = nullptr;
            if (bp) { bd = bp.get(); }
            else {
                if (load_bq_bucket(bucket_id, localB)) bd = &localB; else bd = nullptr;
            }
            if (!bd || bd->num_points == 0) continue;

            size_t code_cols = bd->padded_dim / 8;
            size_t num = bd->num_points;

            std::vector<float> q_pad(bd->padded_dim, 0.0f);
            std::copy(query.begin(), query.end(), q_pad.begin());

            size_t ex_bits = bd->bits > 1 ? (bd->bits - 1) : 0;
            rabitqlib::SplitBatchQuery<float> qobj(q_pad.data(), bd->padded_dim, ex_bits, rabitqlib::METRIC_L2, false);
            float qnorm2 = 0.0f; for (size_t d = 0; d < dim; ++d) qnorm2 += query[d] * query[d];
            float qnorm = std::sqrt(qnorm2);
            qobj.set_g_add(qnorm, 0.0f);

            std::vector<std::pair<float, uint32_t>> local_cand;
            local_cand.reserve(std::min(k, num));

            std::array<uint16_t, rabitqlib::fastscan::kBatchSize> accu{};

            size_t num_rd = (num + 31) & ~31ULL;
            size_t batches = num_rd / 32;
            for (size_t b = 0; b < batches; ++b) {
                const uint8_t* codes = bd->bin_codes.data() + b * (code_cols * 32);
                rabitqlib::fastscan::accumulate(codes, qobj.lut(), accu.data(), bd->padded_dim);

                size_t base = b * 32;
                size_t batch = std::min<size_t>(32, num - base);
                for (size_t i = 0; i < batch; ++i) {
                    size_t idx = base + i;
                    float ip_est = qobj.delta() * static_cast<float>(accu[i]) + qobj.sum_vl_lut();
                    float dist_est = bd->f_add[idx] + qobj.g_add() + bd->f_rescale[idx] * (ip_est + qobj.k1xsumq());

                    if (bd->bits > 1 && !bd->ex_blob.empty()) {
                        size_t ex_bits2 = bd->bits - 1;
                        size_t ex_stride = rabitqlib::ExDataMap<float>::data_bytes(bd->padded_dim, ex_bits2);
                        const char* ex_ptr = reinterpret_cast<const char*>(bd->ex_blob.data()) + (idx * ex_stride);
                        auto ip_func = rabitqlib::select_excode_ipfunc(ex_bits2);
                        rabitqlib::ConstExDataMap<float> ex_map(ex_ptr, bd->padded_dim, ex_bits2);
                        float ex_dist = ex_map.f_add_ex() + qobj.g_add() + (ex_map.f_rescale_ex() *
                            (static_cast<float>(1 << ex_bits2) * (ip_est) + ip_func(q_pad.data(), ex_map.ex_code(), bd->padded_dim) + qobj.kbxsumq()));
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

    {
        const char* base_fbin_env = std::getenv("BASE_FBIN");
        if (base_fbin_env && base_fbin_env[0] != '\0' && !candidates.empty() && top_k > 0) {
            std::string base_fbin_path(base_fbin_env);
            bool force_odirect = true;
            if (const char* env_od = std::getenv("RERANK_O_DIRECT")) {
                std::string v(env_od);
                if (v == "0" || v == "false" || v == "False") force_odirect = false;
            }
            int flags = O_RDONLY | (force_odirect ? O_DIRECT : 0);
            int fd = ::open(base_fbin_path.c_str(), flags);
            if (fd >= 0) {
                do {
                    uint32_t hdr_u32[2] = {0, 0};
                    int fd_meta = ::open(base_fbin_path.c_str(), O_RDONLY);
                    if (fd_meta < 0) break;
                    ssize_t hread = ::pread(fd_meta, hdr_u32, sizeof(hdr_u32), 0);
                    ::close(fd_meta);
                    if (hread != static_cast<ssize_t>(sizeof(hdr_u32))) break;
                    size_t base_num = hdr_u32[0], base_dim = hdr_u32[1];
                    if (base_dim != dim) {
                        std::cout << "[rerank] skip: base_dim(" << base_dim << ") != dim(" << dim << ")" << std::endl;
                        break;
                    }
                    const off_t header_bytes = static_cast<off_t>(sizeof(uint32_t) * 2);
                    std::vector<std::pair<float, uint32_t>> rerank_pairs;
                    rerank_pairs.reserve(candidates.size());
                    std::vector<float> buf(base_dim);
                    const bool odirect_on = force_odirect;
                    const size_t vec_bytes = sizeof(float) * base_dim;

                    auto read_env_size_t = [](const char* name, size_t def_val) -> size_t {
                        const char* s = std::getenv(name);
                        if (s == nullptr) return def_val;
                        char* endp = nullptr;
                        unsigned long long v = std::strtoull(s, &endp, 10);
                        if (endp == s || v == 0ULL) return def_val;
                        return static_cast<size_t>(v);
                    };

                    size_t bs = read_env_size_t("RERANK_ALIGN_BS", 4096);
                    if (bs == 0 || (bs % 512ULL) != 0) bs = 4096; // 至少 512 对齐，优选 4K
                    const size_t batch_vecs_limit = read_env_size_t("RERANK_BATCH_VECS", 128);
                    const size_t batch_mb = read_env_size_t("RERANK_BATCH_MB", 8);
                    const size_t batch_bytes_limit = batch_mb * 1024ULL * 1024ULL;
                    const size_t gap_gids = read_env_size_t("RERANK_GAP_GIDS", 8);

                    // 单向量 O_DIRECT 读取的兜底缓冲
                    size_t aligned_cap_sv = ((vec_bytes + bs) + (bs - 1)) & ~(bs - 1);
                    void* sv_buf = nullptr;
                    if (odirect_on) {
                        if (posix_memalign(&sv_buf, bs, aligned_cap_sv) != 0) sv_buf = nullptr;
                    }
                    auto read_vec_od = [&](int fd_r, off_t off)->bool{
                        if (!odirect_on || sv_buf == nullptr) {
                            ssize_t br = ::pread(fd_r, reinterpret_cast<char*>(buf.data()), vec_bytes, off);
                            return br == static_cast<ssize_t>(vec_bytes);
                        } else {
                            off_t aligned_start = (off / static_cast<off_t>(bs)) * static_cast<off_t>(bs);
                            size_t inner = static_cast<size_t>(off - aligned_start);
                            size_t aligned_len = (inner + vec_bytes + (bs - 1)) & ~(bs - 1);
                            ssize_t br = ::pread(fd_r, sv_buf, aligned_len, aligned_start);
                            if (br != static_cast<ssize_t>(aligned_len)) return false;
                            const char* p = static_cast<const char*>(sv_buf) + inner;
                            std::memcpy(buf.data(), p, vec_bytes);
                            return true;
                        }
                    };

                    // 收集有效 gid 并升序，便于顺序合并大块读取
                    std::vector<uint32_t> gids;
                    gids.reserve(candidates.size());
                    for (const auto& c : candidates) {
                        uint32_t gid = c.second;
                        if (gid < base_num) gids.push_back(gid);
                    }
                    if (!gids.empty()) {
                        std::sort(gids.begin(), gids.end());
                        gids.erase(std::unique(gids.begin(), gids.end()), gids.end());
                    }

                    // 将 gid 合并为顺序段（segment）
                    struct RerankSegment { off_t aligned_start; size_t inner; size_t aligned_len; uint32_t first_gid; uint32_t last_gid; };
                    std::vector<RerankSegment> segments;
                    segments.reserve(gids.size());
                    for (size_t sidx = 0; sidx < gids.size();) {
                        uint32_t first_gid = gids[sidx];
                        uint32_t last_gid = first_gid;
                        size_t seg_last_index = sidx;
                        while (seg_last_index + 1 < gids.size()) {
                            uint32_t next_gid = gids[seg_last_index + 1];
                            if (static_cast<size_t>(next_gid - last_gid) > gap_gids) break;
                            size_t proposed_vecs = static_cast<size_t>(next_gid - first_gid + 1);
                            size_t proposed_bytes = proposed_vecs * vec_bytes;
                            if (proposed_vecs > batch_vecs_limit || proposed_bytes > batch_bytes_limit) break;
                            last_gid = next_gid;
                            seg_last_index++;
                        }
                        size_t range_vecs = static_cast<size_t>(last_gid - first_gid + 1);
                        const off_t range_start_off = header_bytes + static_cast<off_t>(sizeof(float)) * static_cast<off_t>(first_gid) * static_cast<off_t>(base_dim);
                        off_t aligned_start = (range_start_off / static_cast<off_t>(bs)) * static_cast<off_t>(bs);
                        size_t inner = static_cast<size_t>(range_start_off - aligned_start);
                        size_t aligned_len = (inner + range_vecs * vec_bytes + (bs - 1)) & ~(bs - 1);
                        segments.push_back(RerankSegment{aligned_start, inner, aligned_len, first_gid, last_gid});
                        sidx = seg_last_index + 1;
                    }

                    size_t skipped_oob = 0, short_reads = 0;

                    // --- 任务 5：优先使用 io_uring（可选，需编译支持 + 环境变量） ---
                    bool used_engine = false;
#ifdef HAS_IO_URING
                    auto getenv_bool = [](const char* k, bool defv){ const char* s = std::getenv(k); if (!s) return defv; std::string v(s); return !(v=="0"||v=="false"||v=="False"); };
                    bool want_uring = getenv_bool("RERANK_IO_URING", false);
                    if (want_uring && !segments.empty()) {
                        unsigned depth = static_cast<unsigned>(read_env_size_t("RERANK_URING_DEPTH", 64));
                        if (depth == 0) depth = 64;
                        io_uring ring{};
                        if (::io_uring_queue_init(depth, &ring, 0) == 0) {
                            struct Ctx { void* buf; size_t aligned_len; size_t inner; uint32_t first_gid; uint32_t last_gid; };
                            std::vector<Ctx> ctxs(segments.size());
                            size_t submitted = 0, completed = 0;
                            auto submit_one = [&](size_t i){
                                const auto& seg = segments[i];
                                void* big_buf = nullptr;
                                if (posix_memalign(&big_buf, bs, seg.aligned_len) != 0 || big_buf == nullptr) return false;
                                ctxs[i] = Ctx{big_buf, seg.aligned_len, seg.inner, seg.first_gid, seg.last_gid};
                                io_uring_sqe* sqe = ::io_uring_get_sqe(&ring);
                                if (!sqe) { free(big_buf); return false; }
                                ::io_uring_prep_read(sqe, fd, big_buf, seg.aligned_len, seg.aligned_start);
                                ::io_uring_sqe_set_data(sqe, &ctxs[i]);
                                ++submitted;
                                return true;
                            };
                            size_t in_flight = 0;
                            size_t next_to_submit = 0;
                            // 初始填满队列
                            while (next_to_submit < segments.size() && in_flight < depth) {
                                if (submit_one(next_to_submit)) { ++in_flight; }
                                ++next_to_submit;
                            }
                            if (submitted > 0) { ::io_uring_submit(&ring); }
                            while (completed < segments.size()) {
                                io_uring_cqe* cqe = nullptr;
                                if (::io_uring_wait_cqe(&ring, &cqe) != 0) break;
                                Ctx* c = reinterpret_cast<Ctx*>(::io_uring_cqe_get_data(cqe));
                                if (c && c->buf) {
                                    if (cqe->res == static_cast<int>(c->aligned_len)) {
                                        const char* base_ptr = static_cast<const char*>(c->buf) + c->inner;
                                        for (uint32_t gid = c->first_gid; gid <= c->last_gid; ++gid) {
                                            size_t rel = static_cast<size_t>(gid - c->first_gid);
                                            const float* vptr = reinterpret_cast<const float*>(base_ptr + rel * vec_bytes);
                                            float dist = 0.0f;
                                            for (size_t d = 0; d < base_dim; ++d) {
                                                float diff = vptr[d] - query[d];
                                                dist += diff * diff;
                                            }
                                            rerank_pairs.emplace_back(dist, gid);
                                        }
                                    } else {
                                        short_reads++;
                                    }
                                    free(c->buf);
                                }
                                ::io_uring_cqe_seen(&ring, cqe);
                                ++completed;
                                if (next_to_submit < segments.size()) {
                                    // 尝试继续提交
                                    if (submit_one(next_to_submit)) { ::io_uring_submit(&ring); ++in_flight; }
                                    ++next_to_submit;
                                }
                            }
                            ::io_uring_queue_exit(&ring);
                            used_engine = !rerank_pairs.empty();
                        }
                    }
#endif

                    // --- 任务 4：I/O 线程池并行 pread（当未使用 io_uring 时） ---
                    if (!used_engine && !segments.empty()) {
                        size_t io_threads = read_env_size_t("RERANK_IO_THREADS", 0);
                        if (io_threads > 1) {
                            IOThreadPool pool(io_threads);
                            std::vector<std::future<std::vector<std::pair<float, uint32_t>>>> futs;
                            futs.reserve(segments.size());
                            for (const auto& seg : segments) {
                                futs.emplace_back(pool.submit([&, seg]() {
                                    std::vector<std::pair<float, uint32_t>> out;
                                    void* big_buf = nullptr;
                                    if (posix_memalign(&big_buf, bs, seg.aligned_len) != 0 || big_buf == nullptr) return out;
                                    ssize_t br = ::pread(fd, big_buf, seg.aligned_len, seg.aligned_start);
                                    if (br == static_cast<ssize_t>(seg.aligned_len)) {
                                        const char* base_ptr = static_cast<const char*>(big_buf) + seg.inner;
                                        for (uint32_t gid = seg.first_gid; gid <= seg.last_gid; ++gid) {
                                            size_t rel = static_cast<size_t>(gid - seg.first_gid);
                                            const float* vptr = reinterpret_cast<const float*>(base_ptr + rel * vec_bytes);
                                            float dist = 0.0f;
                                            for (size_t d = 0; d < base_dim; ++d) {
                                                float diff = vptr[d] - query[d];
                                                dist += diff * diff;
                                            }
                                            out.emplace_back(dist, gid);
                                        }
                                    }
                                    else {
                                        // 兜底退回该段逐向量读取
                                        for (uint32_t gid = seg.first_gid; gid <= seg.last_gid; ++gid) {
                                            const off_t off = header_bytes + static_cast<off_t>(sizeof(float)) * static_cast<off_t>(gid) * static_cast<off_t>(base_dim);
                                            if (!read_vec_od(fd, off)) { continue; }
                                            float dist = 0.0f;
                                            for (size_t d = 0; d < base_dim; ++d) {
                                                float diff = buf[d] - query[d];
                                                dist += diff * diff;
                                            }
                                            out.emplace_back(dist, gid);
                                        }
                                    }
                                    if (big_buf) free(big_buf);
                                    return out;
                                }));
                            }
                            for (auto& f : futs) {
                                auto v = f.get();
                                if (v.empty()) { short_reads++; }
                                rerank_pairs.insert(rerank_pairs.end(), v.begin(), v.end());
                            }
                            used_engine = !rerank_pairs.empty();
                        }
                    }

                    // --- 原有顺序路径（作为最终兜底） ---
                    if (!used_engine) {
                        for (const auto& seg : segments) {
                            void* big_buf = nullptr;
                            bool segment_ok = false;
                            if (posix_memalign(&big_buf, bs, seg.aligned_len) == 0 && big_buf != nullptr) {
                                ssize_t br = ::pread(fd, big_buf, seg.aligned_len, seg.aligned_start);
                                if (br == static_cast<ssize_t>(seg.aligned_len)) {
                                    const char* base_ptr = static_cast<const char*>(big_buf) + seg.inner;
                                    for (uint32_t gid = seg.first_gid; gid <= seg.last_gid; ++gid) {
                                        size_t rel = static_cast<size_t>(gid - seg.first_gid);
                                        const float* vptr = reinterpret_cast<const float*>(base_ptr + rel * vec_bytes);
                                        float dist = 0.0f;
                                        for (size_t d = 0; d < base_dim; ++d) {
                                            float diff = vptr[d] - query[d];
                                            dist += diff * diff;
                                        }
                                        rerank_pairs.emplace_back(dist, gid);
                                    }
                                    segment_ok = true;
                                }
                            }
                            if (big_buf) free(big_buf);
                            if (!segment_ok) {
                                for (uint32_t gid = seg.first_gid; gid <= seg.last_gid; ++gid) {
                                    const off_t off = header_bytes + static_cast<off_t>(sizeof(float)) * static_cast<off_t>(gid) * static_cast<off_t>(base_dim);
                                    if (!read_vec_od(fd, off)) { short_reads++; continue; }
                                    float dist = 0.0f;
                                    for (size_t d = 0; d < base_dim; ++d) {
                                        float diff = buf[d] - query[d];
                                        dist += diff * diff;
                                    }
                                    rerank_pairs.emplace_back(dist, gid);
                                }
                            }
                        }
                    }

                    if (sv_buf) { free(sv_buf); }
                    if (!rerank_pairs.empty()) {
                        std::sort(rerank_pairs.begin(), rerank_pairs.end());
                        QueryResult final_results_r;
                        final_results_r.ids.reserve(std::min(top_k, rerank_pairs.size()));
                        final_results_r.distances.reserve(std::min(top_k, rerank_pairs.size()));
                        for (size_t i = 0; i < std::min(top_k, rerank_pairs.size()); ++i) {
                            final_results_r.distances.push_back(rerank_pairs[i].first);
                            final_results_r.ids.push_back(rerank_pairs[i].second);
                        }
                        if (skipped_oob > 0 || short_reads > 0) {
                            std::cout << "[rerank] pread(" << (odirect_on ? "O_DIRECT" : "plain") << "): skipped_oob="
                                      << skipped_oob << ", short_reads=" << short_reads << std::endl;
                        }
                        ::close(fd);
                        return final_results_r;
                    }
                    if (odirect_on) {
                        std::cout << "[rerank] O_DIRECT path yielded no pairs, fallback to plain pread" << std::endl;
                        int fd_plain = ::open(base_fbin_path.c_str(), O_RDONLY);
                        if (fd_plain >= 0) {
                            std::vector<std::pair<float, uint32_t>> rerank_pairs2;
                            rerank_pairs2.reserve(candidates.size());
                            size_t skipped2 = 0, short2 = 0;
                            for (size_t i = 0; i < candidates.size(); ++i) {
                                uint32_t gid = candidates[i].second;
                                if (gid >= base_num) { skipped2++; continue; }
                                const off_t off = header_bytes + static_cast<off_t>(sizeof(float)) * static_cast<off_t>(gid) * static_cast<off_t>(base_dim);
                                ssize_t br = ::pread(fd_plain, reinterpret_cast<char*>(buf.data()), vec_bytes, off);
                                if (br != static_cast<ssize_t>(vec_bytes)) { short2++; continue; }
                                float dist = 0.0f;
                                for (size_t d = 0; d < base_dim; ++d) {
                                    float diff = buf[d] - query[d];
                                    dist += diff * diff;
                                }
                                rerank_pairs2.emplace_back(dist, gid);
                            }
                            ::close(fd_plain);
                            if (!rerank_pairs2.empty()) {
                                std::sort(rerank_pairs2.begin(), rerank_pairs2.end());
                                QueryResult final_results_r;
                                final_results_r.ids.reserve(std::min(top_k, rerank_pairs2.size()));
                                final_results_r.distances.reserve(std::min(top_k, rerank_pairs2.size()));
                                for (size_t i = 0; i < std::min(top_k, rerank_pairs2.size()); ++i) {
                                    final_results_r.distances.push_back(rerank_pairs2[i].first);
                                    final_results_r.ids.push_back(rerank_pairs2[i].second);
                                }
                                if (skipped2 > 0 || short2 > 0) {
                                    std::cout << "[rerank] pread(plain): skipped_oob=" << skipped2
                                              << ", short_reads=" << short2 << std::endl;
                                }
                                ::close(fd);
                                return final_results_r;
                            }
                        }
                    }
                } while(false);

                bool allow_mmap = false;
                if (const char* env_mm = std::getenv("RERANK_USE_MMAP")) {
                    std::string v(env_mm);
                    if (v == "1" || v == "true" || v == "True") allow_mmap = true;
                }
                if (allow_mmap) {
                    struct stat st{};
                    if (::fstat(fd, &st) == 0 && st.st_size >= static_cast<off_t>(sizeof(uint32_t) * 2)) {
                        void* map = ::mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
                        if (map != MAP_FAILED) {
                            const char* base_ptr = static_cast<const char*>(map);
                            const uint32_t* hdr = reinterpret_cast<const uint32_t*>(base_ptr);
                            size_t base_num = hdr[0];
                            size_t base_dim = hdr[1];
                            if (base_dim == dim) {
                                const float* vec_base = reinterpret_cast<const float*>(base_ptr + sizeof(uint32_t) * 2);
                                std::vector<std::pair<float, uint32_t>> rerank_pairs;
                                rerank_pairs.reserve(candidates.size());
                                size_t skipped_oob = 0;
                                for (size_t i = 0; i < candidates.size(); ++i) {
                                    uint32_t gid = candidates[i].second;
                                    if (gid >= base_num) { skipped_oob++; continue; }
                                    const float* vec = vec_base + static_cast<size_t>(gid) * base_dim;
                                    float dist = 0.0f;
                                    for (size_t d = 0; d < base_dim; ++d) {
                                        float diff = vec[d] - query[d];
                                        dist += diff * diff;
                                    }
                                    rerank_pairs.emplace_back(dist, gid);
                                }
                                if (skipped_oob > 0) {
                                    std::cout << "[rerank] skipped out-of-range ids: " << skipped_oob << std::endl;
                                }
                                if (!rerank_pairs.empty()) {
                                    std::sort(rerank_pairs.begin(), rerank_pairs.end());
                                    QueryResult final_results_r;
                                    final_results_r.ids.reserve(std::min(top_k, rerank_pairs.size()));
                                    final_results_r.distances.reserve(std::min(top_k, rerank_pairs.size()));
                                    for (size_t i = 0; i < std::min(top_k, rerank_pairs.size()); ++i) {
                                        final_results_r.distances.push_back(rerank_pairs[i].first);
                                        final_results_r.ids.push_back(rerank_pairs[i].second);
                                    }
                                    ::munmap(map, st.st_size);
                                    ::close(fd);
                                    return final_results_r;
                                }
                            } else {
                                std::cout << "[rerank] skip: base_dim(" << base_dim << ") != dim(" << dim << ")" << std::endl;
                            }
                            ::munmap(map, st.st_size);
                        }
                    }
                }
                ::close(fd);
            }
        }
    }

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