#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <mutex>
#include <atomic>

// 块键：按对齐后的文件起始偏移（字节）
using ArcBlockKey = uint64_t;
// 块值：字节缓冲（长度可变，至少覆盖调用方所需的区间）
using ArcBlockValue = std::shared_ptr<std::vector<char>>;

class ArcBlockCache {
public:
    static ArcBlockCache& instance() {
        static ArcBlockCache inst;
        return inst;
    }

    // 配置两个子缓存（MB）。任一为 0 表示禁用对应子缓存。
    // 分片数量通过环境变量 RERANK_ARC_SHARDS 控制（未设置则为 1）。
    void configure(size_t t1_mb, size_t t2_mb);

    // 命中返回 true，并输出 value；need_len 用于过滤小块（小于需求视为未命中）
    bool get(ArcBlockKey key, size_t need_len, ArcBlockValue& out);

    // 插入或更新；value->size() 为块大小（字节）
    void put(ArcBlockKey key, const ArcBlockValue& value);

    void print_stats() const;

private:
    ArcBlockCache() = default;
    ~ArcBlockCache() = default;
    ArcBlockCache(const ArcBlockCache&) = delete;
    ArcBlockCache& operator=(const ArcBlockCache&) = delete;

    enum class ListId { None, T1, T2 };
    struct Node {
        ArcBlockValue value;
        ListId list{ListId::None};
        size_t bytes{0};
        std::list<ArcBlockKey>::iterator it; // 所在列表迭代器
    };

    struct Shard {
        // 预算/占用
        size_t t1_budget_bytes{0};
        size_t t2_budget_bytes{0};
        size_t t1_bytes{0};
        size_t t2_bytes{0};
        // 容器
        std::unordered_map<ArcBlockKey, Node> map;
        std::list<ArcBlockKey> t1; // 近期（一次命中）
        std::list<ArcBlockKey> t2; // 频繁（多次命中）
        std::unordered_set<ArcBlockKey> b1; // ghost for T1
        std::unordered_set<ArcBlockKey> b2; // ghost for T2
        // 锁
        std::mutex mtx;
    };

    // 内部工具
    size_t pick_shard(ArcBlockKey key) const;
    void evict_from_t1(Shard& s);
    void evict_from_t2(Shard& s);
    void evict_if_needed(Shard& s);

    // 分片集合（使用指针避免因 mutex 导致的不可移动问题）
    std::vector<std::unique_ptr<Shard>> shards_;
    size_t num_shards_{1};

    // 全局统计
    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};

    // 缓存是否启用（预算总和）
    size_t total_t1_budget_{0};
    size_t total_t2_budget_{0};
}; 