#include "arc_cache.h"
#include <iostream>
#include <cstdlib>

size_t ArcBlockCache::pick_shard(ArcBlockKey key) const {
    if (num_shards_ == 0) return 0;
    return static_cast<size_t>(key) % num_shards_;
}

void ArcBlockCache::evict_from_t1(Shard& s) {
    while (s.t1_bytes > s.t1_budget_bytes && !s.t1.empty()) {
        ArcBlockKey victim = s.t1.back();
        s.t1.pop_back();
        auto it = s.map.find(victim);
        if (it != s.map.end()) {
            s.b1.insert(victim);
            s.t1_bytes -= it->second.bytes;
            s.map.erase(it);
        }
    }
}

void ArcBlockCache::evict_from_t2(Shard& s) {
    while (s.t2_bytes > s.t2_budget_bytes && !s.t2.empty()) {
        ArcBlockKey victim = s.t2.back();
        s.t2.pop_back();
        auto it = s.map.find(victim);
        if (it != s.map.end()) {
            s.b2.insert(victim);
            s.t2_bytes -= it->second.bytes;
            s.map.erase(it);
        }
    }
}

void ArcBlockCache::evict_if_needed(Shard& s) {
    if (s.t1_budget_bytes == 0 && s.t2_budget_bytes == 0) return;
    evict_from_t1(s);
    evict_from_t2(s);
}

void ArcBlockCache::configure(size_t t1_mb, size_t t2_mb) {
    // 分片数量：默认 1，可由环境变量 RERANK_ARC_SHARDS 控制
    size_t shards = 1;
    if (const char* s = std::getenv("RERANK_ARC_SHARDS")) {
        try { shards = std::max<size_t>(1, std::stoul(s)); } catch (...) {}
    }
    num_shards_ = shards;

    total_t1_budget_ = t1_mb * 1024ULL * 1024ULL;
    total_t2_budget_ = t2_mb * 1024ULL * 1024ULL;

    shards_.clear();
    shards_.reserve(num_shards_);
    for (size_t i = 0; i < num_shards_; ++i) {
        shards_.emplace_back(std::make_unique<Shard>());
    }

    // 简单均分预算到每个分片
    size_t t1_each = (num_shards_ > 0) ? (total_t1_budget_ / num_shards_) : 0;
    size_t t2_each = (num_shards_ > 0) ? (total_t2_budget_ / num_shards_) : 0;
    for (auto& up : shards_) {
        Shard& s = *up;
        s.t1_budget_bytes = t1_each;
        s.t2_budget_bytes = t2_each;
        s.t1_bytes = 0;
        s.t2_bytes = 0;
        s.map.clear();
        s.t1.clear();
        s.t2.clear();
        s.b1.clear();
        s.b2.clear();
    }
    hits_ = 0;
    misses_ = 0;
}

bool ArcBlockCache::get(ArcBlockKey key, size_t need_len, ArcBlockValue& out) {
    if (num_shards_ == 0 || (total_t1_budget_ + total_t2_budget_) == 0) return false;
    Shard& s = *shards_[pick_shard(key)];
    std::lock_guard<std::mutex> g(s.mtx);
    auto it = s.map.find(key);
    if (it == s.map.end()) {
        misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    Node& n = it->second;
    if (!n.value || n.value->size() < need_len) {
        misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (n.list == ListId::T1) {
        s.t1.erase(n.it);
        s.t1_bytes -= n.bytes;
        s.t2.push_front(key);
        n.it = s.t2.begin();
        n.list = ListId::T2;
        s.t2_bytes += n.bytes;
    } else if (n.list == ListId::T2) {
        s.t2.erase(n.it);
        s.t2.push_front(key);
        n.it = s.t2.begin();
    }
    hits_.fetch_add(1, std::memory_order_relaxed);
    out = n.value;
    return true;
}

void ArcBlockCache::put(ArcBlockKey key, const ArcBlockValue& value) {
    if (!value) return;
    if (num_shards_ == 0 || (total_t1_budget_ + total_t2_budget_) == 0) return;
    size_t bytes = value->size();
    Shard& s = *shards_[pick_shard(key)];
    std::lock_guard<std::mutex> g(s.mtx);

    auto it = s.map.find(key);
    if (it != s.map.end()) {
        Node& n = it->second;
        if (n.list == ListId::T1) {
            s.t1_bytes -= n.bytes;
            s.t1.erase(n.it);
        } else if (n.list == ListId::T2) {
            s.t2_bytes -= n.bytes;
            s.t2.erase(n.it);
        }
        n.value = value;
        n.bytes = bytes;
        s.t2.push_front(key);
        n.it = s.t2.begin();
        n.list = ListId::T2;
        s.t2_bytes += bytes;
        evict_if_needed(s);
        return;
    }

    ListId target = (s.b2.count(key) || s.b1.count(key)) ? ListId::T2 : ListId::T1;
    Node n;
    n.value = value;
    n.bytes = bytes;
    if (target == ListId::T1 && s.t1_budget_bytes > 0) {
        s.t1.push_front(key);
        n.it = s.t1.begin();
        n.list = ListId::T1;
        s.t1_bytes += bytes;
    } else {
        s.t2.push_front(key);
        n.it = s.t2.begin();
        n.list = ListId::T2;
        s.t2_bytes += bytes;
    }
    s.map[key] = std::move(n);
    evict_if_needed(s);
}

void ArcBlockCache::print_stats() const {
    uint64_t h = hits_.load(std::memory_order_relaxed);
    uint64_t m = misses_.load(std::memory_order_relaxed);
    uint64_t total = h + m;
    std::cout << "[ARC] hits=" << h
              << ", misses=" << m
              << ", hit_rate=" << (total ? (100.0 * h / total) : 0.0) << "%\n";
    std::cout << "[ARC] shards=" << num_shards_
              << ", T1(each)=" << (num_shards_? (shards_[0]->t1_budget_bytes/(1024*1024)) : 0)
              << "MB, T2(each)=" << (num_shards_? (shards_[0]->t2_budget_bytes/(1024*1024)) : 0) << "MB\n";
} 