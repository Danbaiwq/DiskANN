#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <mutex>
#include <queue>
#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <immintrin.h>

namespace fs = std::filesystem;

struct PartInfo {
    std::string filepath;
    uint64_t data_bytes;  // excluding 8B header
    uint64_t num_points;
    uint32_t dim;
};

static inline float l2_distance_sq_avx2(const float* a, const float* b, uint32_t dim) {
    __m256 acc = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 7 < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }
    alignas(32) float buf[8];
    _mm256_store_ps(buf, acc);
    float sum = buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

static std::vector<PartInfo> scan_parts(const std::string& chunk_dir) {
    std::vector<PartInfo> parts;
    for (auto const& entry : fs::directory_iterator(chunk_dir)) {
        if (!entry.is_regular_file()) continue;
        auto path = entry.path().string();
        if (path.size() < 5) continue;
        if (entry.path().extension() != ".fbin") continue;
        std::ifstream in(path, std::ios::binary);
        if (!in) continue;
        uint32_t n = 0, d = 0;
        in.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
        in.close();
        uint64_t file_size = fs::file_size(entry);
        uint64_t data_bytes = file_size >= 8 ? file_size - 8 : 0;
        parts.push_back({path, data_bytes, n, d});
    }
    std::sort(parts.begin(), parts.end(), [](const PartInfo& a, const PartInfo& b){ return a.filepath < b.filepath; });
    if (parts.empty()) throw std::runtime_error("No .fbin parts found in chunk_dir");
    // Validate consistent dim
    uint32_t dim0 = parts.front().dim;
    for (auto& p : parts) if (p.dim != dim0) throw std::runtime_error("All parts must have the same dimension");
    return parts;
}

static void reservoir_sample(const std::vector<PartInfo>& parts, uint32_t dim, uint64_t reservoir_capacity, std::vector<float>& out_sample) {
    if (reservoir_capacity == 0) throw std::runtime_error("reservoir_capacity must be > 0");
    out_sample.clear();
    out_sample.reserve(reservoir_capacity * dim);

    struct Candidate { double key; std::vector<float> vec; };
    struct MinComp { bool operator()(const Candidate& a, const Candidate& b) const { return a.key > b.key; } };

    std::priority_queue<Candidate, std::vector<Candidate>, MinComp> global_heap; // min-heap by key
    std::mutex heap_mu;
    std::atomic<double> threshold(0.0); // approximate acceptance threshold (min key in heap), 0 until heap full

    uint64_t seen = 0;
    const size_t block_bytes = 256ull << 20; // 256MiB IO block
    const uint64_t vec_bytes = static_cast<uint64_t>(dim) * sizeof(float);
    const uint64_t max_vecs_per_block = std::max<uint64_t>(1, block_bytes / vec_bytes);

    for (const auto& part : parts) {
        std::ifstream in(part.filepath, std::ios::binary);
        if (!in) throw std::runtime_error("Failed to open part: " + part.filepath);
        uint32_t n = 0, d = 0; 
        in.read(reinterpret_cast<char*>(&n), 4); 
        in.read(reinterpret_cast<char*>(&d), 4);
        if (d != dim) throw std::runtime_error("Dimension mismatch in part: " + part.filepath);
        // read the data into a buffer, IO block size is 256MiB
        std::vector<float> buf(std::min<uint64_t>(n, max_vecs_per_block) * dim);
        uint64_t remaining = n;
        while (remaining > 0) {
            uint64_t this_vecs = std::min<uint64_t>(remaining, max_vecs_per_block);
            in.read(reinterpret_cast<char*>(buf.data()), this_vecs * vec_bytes);
            if (static_cast<uint64_t>(in.gcount()) != this_vecs * vec_bytes) throw std::runtime_error("Short read in part: " + part.filepath);

            #pragma omp parallel
            {
                // thread-local RNG and candidate buffer
                uint64_t seed_base = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(buf.data())) ^ static_cast<uint64_t>(seen);
                #ifdef _OPENMP
                int tid = omp_get_thread_num();
                #else
                int tid = 0;
                #endif
                std::mt19937_64 rng(seed_base + (static_cast<uint64_t>(tid) << 16) + 0x9e3779b97f4a7c15ULL);
                std::uniform_real_distribution<double> urand(0.0, 1.0);
                std::vector<Candidate> local;
                local.reserve(1024);

                #pragma omp for schedule(static)
                for (int64_t i = 0; i < static_cast<int64_t>(this_vecs); ++i) {
                    double key = urand(rng);
                    if (key <= threshold.load(std::memory_order_relaxed)) continue;
                    Candidate c; c.key = key; c.vec.assign(buf.data() + static_cast<size_t>(i)*dim, buf.data() + static_cast<size_t>(i+1)*dim);
                    local.push_back(std::move(c));
                }

                if (!local.empty()) {
                    std::lock_guard<std::mutex> lk(heap_mu);
                    for (auto& c : local) {
                        if (global_heap.size() < reservoir_capacity) {
                            global_heap.push(std::move(c));
                        } else if (c.key > global_heap.top().key) {
                            global_heap.pop();
                            global_heap.push(std::move(c));
                        }
                    }
                    if (global_heap.size() == reservoir_capacity) threshold.store(global_heap.top().key, std::memory_order_relaxed);
                }
            }

            seen += this_vecs;
            remaining -= this_vecs;
        }
        in.close();
    }

    if (global_heap.empty()) throw std::runtime_error("Reservoir sample is empty");

    // Flatten heap (order not important)
    std::vector<Candidate> tmp;
    tmp.reserve(global_heap.size());
    while (!global_heap.empty()) { tmp.push_back(global_heap.top()); global_heap.pop(); }
    out_sample.reserve(tmp.size() * dim);
    for (const auto& c : tmp) {
        out_sample.insert(out_sample.end(), c.vec.begin(), c.vec.end());
    }
}

static void kmeans_pp_init_from_sample(const std::vector<float>& sample, uint64_t sample_points, uint32_t dim, uint32_t k, std::vector<float>& centers) {
    if (sample_points == 0) throw std::runtime_error("Sample points cannot be zero");
    centers.assign(static_cast<size_t>(k) * dim, 0.0f);
    std::mt19937 rng(std::random_device{}());

    std::uniform_int_distribution<uint64_t> uni0(0, sample_points - 1);
    uint64_t first = uni0(rng);
    std::copy(sample.data() + first*dim, sample.data() + (first+1)*dim, centers.data());

    std::vector<float> min_dist(sample_points, std::numeric_limits<float>::max());
    for (uint32_t picked = 1; picked < k; ++picked) {
        // Update distances to nearest picked center
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(sample_points); ++i) {
            float d = l2_distance_sq_avx2(sample.data() + i*dim, centers.data() + (picked-1)*dim, dim);
            if (d < min_dist[static_cast<size_t>(i)]) min_dist[static_cast<size_t>(i)] = d;
        }
        // Weighted pick next center
        long double total = 0.0L;
        for (float v : min_dist) total += static_cast<long double>(v);
        if (total <= 0) {
            // If all zero distances, pick random
            std::uniform_int_distribution<uint64_t> uni(0, sample_points - 1);
            uint64_t idx = uni(rng);
            std::copy(sample.data() + idx*dim, sample.data() + (idx+1)*dim, centers.data() + picked*dim);
            continue;
        }
        std::uniform_real_distribution<long double> dart(0.0L, total);
        long double r = dart(rng);
        long double prefix = 0.0L;
        uint64_t chosen = 0;
        for (uint64_t i = 0; i < sample_points; ++i) {
            prefix += static_cast<long double>(min_dist[i]);
            if (r <= prefix) { chosen = i; break; }
        }
        std::copy(sample.data() + chosen*dim, sample.data() + (chosen+1)*dim, centers.data() + picked*dim);
    }
}

static void process_batch_update(const std::vector<PartInfo>& batch_parts,
                                 uint32_t dim,
                                 uint32_t k,
                                 std::vector<float>& centers,
                                 std::vector<long double>& global_counts) {
    const size_t block_bytes = 256ull << 20; // 256MiB IO block
    const uint64_t vec_bytes = static_cast<uint64_t>(dim) * sizeof(float);
    const uint64_t max_vecs_per_block = std::max<uint64_t>(1, block_bytes / vec_bytes);

    int num_threads = 1;
    #ifdef _OPENMP
    num_threads = omp_get_max_threads();
    #endif

    std::vector<long double> batch_counts(k, 0.0L);
    std::vector<float> batch_sums(static_cast<size_t>(k) * dim, 0.0f);

    std::vector<std::vector<float>> local_sums(static_cast<size_t>(num_threads), std::vector<float>(static_cast<size_t>(k)*dim, 0.0f));
    std::vector<std::vector<long double>> local_counts(static_cast<size_t>(num_threads), std::vector<long double>(k, 0.0L));

    for (const auto& part : batch_parts) {
        std::ifstream in(part.filepath, std::ios::binary);
        if (!in) throw std::runtime_error("Failed to open part: " + part.filepath);
        uint32_t n = 0, d = 0; in.read(reinterpret_cast<char*>(&n), 4); in.read(reinterpret_cast<char*>(&d), 4);
        if (d != dim) throw std::runtime_error("Dimension mismatch in part: " + part.filepath);

        std::vector<float> buf(std::min<uint64_t>(n, max_vecs_per_block) * dim);
        uint64_t remaining = n;
        while (remaining > 0) {
            uint64_t this_vecs = std::min<uint64_t>(remaining, max_vecs_per_block);
            in.read(reinterpret_cast<char*>(buf.data()), this_vecs * vec_bytes);
            if (static_cast<uint64_t>(in.gcount()) != this_vecs * vec_bytes) throw std::runtime_error("Short read in part: " + part.filepath);

            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(this_vecs); ++i) {
                int tid = 0;
                #ifdef _OPENMP
                tid = omp_get_thread_num();
                #endif
                const float* vec = buf.data() + static_cast<size_t>(i)*dim;
                // find nearest center
                uint32_t best = 0;
                float best_dist = std::numeric_limits<float>::max();
                for (uint32_t c = 0; c < k; ++c) {
                    float d2 = l2_distance_sq_avx2(vec, centers.data() + static_cast<size_t>(c)*dim, dim);
                    if (d2 < best_dist) { best_dist = d2; best = c; }
                }
                // accumulate
                long double* cnt_slot = local_counts[static_cast<size_t>(tid)].data();
                cnt_slot[best] += 1.0L;
                float* sum_slot = local_sums[static_cast<size_t>(tid)].data() + static_cast<size_t>(best)*dim;
                for (uint32_t dd = 0; dd < dim; ++dd) sum_slot[dd] += vec[dd];
            }
            remaining -= this_vecs;
        }
        in.close();
    }

    // reduce locals into batch_sums/counts
    for (int t = 0; t < num_threads; ++t) {
        for (uint32_t c = 0; c < k; ++c) {
            batch_counts[c] += local_counts[static_cast<size_t>(t)][c];
            float* dst = batch_sums.data() + static_cast<size_t>(c)*dim;
            const float* src = local_sums[static_cast<size_t>(t)].data() + static_cast<size_t>(c)*dim;
            for (uint32_t dd = 0; dd < dim; ++dd) dst[dd] += src[dd];
        }
    }

    // Update centers with weighted mean: new = (prev*global_count + batch_sum) / (global_count + batch_count)
    for (uint32_t c = 0; c < k; ++c) {
        long double bc = batch_counts[c];
        if (bc <= 0.0L) continue;
        long double gc = global_counts[c];
        long double new_total = gc + bc;
        float* center = centers.data() + static_cast<size_t>(c)*dim;
        float* bsum = batch_sums.data() + static_cast<size_t>(c)*dim;
        for (uint32_t dd = 0; dd < dim; ++dd) {
            long double num = static_cast<long double>(center[dd]) * gc + static_cast<long double>(bsum[dd]);
            center[dd] = static_cast<float>(num / new_total);
        }
        global_counts[c] = new_total;
    }
}

static void write_centroids_fbin(const std::string& out_path, const std::vector<float>& centers, uint32_t k, uint32_t dim) {
    std::ofstream out(out_path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open output file: " + out_path);
    uint32_t n = k; uint32_t d = dim;
    out.write(reinterpret_cast<const char*>(&n), 4);
    out.write(reinterpret_cast<const char*>(&d), 4);
    out.write(reinterpret_cast<const char*>(centers.data()), static_cast<size_t>(k)*dim*sizeof(float));
    out.close();
}

static void usage() {
    std::cerr << "Usage: stream_kmeans --chunk_dir DIR --k K --epochs E --batch_gib G --init_sample_gib S --out OUT_FBIN" << std::endl;
}

int main(int argc, char** argv) {
    std::string chunk_dir;
    uint32_t k = 0;
    uint32_t epochs = 5;
    double batch_gib = 4.0;
    double init_sample_gib = 2.0;
    std::string out_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](){ return (i+1<argc) ? std::string(argv[++i]) : std::string(); };
        if (arg == "--chunk_dir") chunk_dir = next();
        else if (arg == "--k") k = static_cast<uint32_t>(std::stoul(next()));
        else if (arg == "--epochs") epochs = static_cast<uint32_t>(std::stoul(next()));
        else if (arg == "--batch_gib") batch_gib = std::stod(next());
        else if (arg == "--init_sample_gib") init_sample_gib = std::stod(next());
        else if (arg == "--out") out_path = next();
        else if (arg == "-h" || arg == "--help") { usage(); return 0; }
    }

    if (chunk_dir.empty() || k == 0 || out_path.empty()) { usage(); return 2; }

    try {
        auto t_all_start = std::chrono::high_resolution_clock::now();
        auto parts = scan_parts(chunk_dir);
        uint32_t dim = parts.front().dim;
        uint64_t vec_bytes = static_cast<uint64_t>(dim) * sizeof(float);
        uint64_t total_bytes = 0; for (auto& p : parts) total_bytes += p.data_bytes;
        uint64_t total_points = 0; for (auto& p : parts) total_points += p.num_points;
        uint64_t batch_bytes = static_cast<uint64_t>(batch_gib * (1ull<<30));
        if (batch_bytes == 0) throw std::runtime_error("batch_gib must be > 0");
        uint64_t init_bytes = static_cast<uint64_t>(init_sample_gib * (1ull<<30));
        uint64_t reservoir_capacity = std::max<uint64_t>(k, init_bytes / vec_bytes);
        int num_threads = 1;
        #ifdef _OPENMP
        num_threads = omp_get_max_threads();
        #endif

        std::cout << "[stream_kmeans] parts=" << parts.size() << ", dim=" << dim
                  << ", total_points=" << total_points
                  << ", total_bytes=" << total_bytes
                  << ", batch_bytes=" << batch_bytes
                  << ", reservoir_points=" << reservoir_capacity
                  << ", threads=" << num_threads
                  << std::endl;

        // KMeans++ init from reservoir sample
        auto t_res_start = std::chrono::high_resolution_clock::now();
        std::vector<float> sample;
        reservoir_sample(parts, dim, reservoir_capacity, sample);
        uint64_t sample_points = sample.size() / dim;
        auto t_res_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> t_res = t_res_end - t_res_start;
        std::cout << "[stream_kmeans] reservoir_sample: points=" << sample_points
                  << ", size_bytes=" << sample.size() * sizeof(float)
                  << ", time_s=" << t_res.count() << std::endl;

        auto t_init_start = std::chrono::high_resolution_clock::now();
        std::vector<float> centers; centers.reserve(static_cast<size_t>(k)*dim);
        kmeans_pp_init_from_sample(sample, sample_points, dim, k, centers);
        auto t_init_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> t_init = t_init_end - t_init_start;
        std::cout << "[stream_kmeans] kmeans++ init: k=" << k << ", dim=" << dim
                  << ", time_s=" << t_init.count() << std::endl;

        // Prepare batches per epoch by packing shuffled parts into <= batch_bytes bins
        size_t num_parts = parts.size();
        std::vector<size_t> order(num_parts); std::iota(order.begin(), order.end(), 0);
        std::vector<long double> global_counts(k, 0.0L);

        for (uint32_t e = 0; e < epochs; ++e) {
            auto t_ep_start = std::chrono::high_resolution_clock::now();
            std::mt19937 rng(std::random_device{}());
            std::shuffle(order.begin(), order.end(), rng);
            std::vector<PartInfo> batch;
            uint64_t acc_bytes = 0;
            uint64_t batch_points_counter = 0;
            uint64_t epoch_points = 0, epoch_data_bytes = 0;
            size_t used = 0;
            for (size_t idx = 0; idx < num_parts; ++idx) {
                const PartInfo& p = parts[order[idx]];
                if (acc_bytes + p.data_bytes > batch_bytes) {
                    if (!batch.empty()) {
                        uint64_t flushed_bytes = acc_bytes;
                        uint64_t flushed_points = batch_points_counter;
                        process_batch_update(batch, dim, k, centers, global_counts);
                        epoch_data_bytes += flushed_bytes;
                        epoch_points += flushed_points;
                        batch.clear(); acc_bytes = 0; batch_points_counter = 0; ++used;
                    }
                }
                batch.push_back(p); acc_bytes += p.data_bytes; batch_points_counter += p.num_points;
            }
            if (!batch.empty()) {
                uint64_t flushed_bytes = acc_bytes;
                uint64_t flushed_points = batch_points_counter;
                process_batch_update(batch, dim, k, centers, global_counts); ++used;
                epoch_data_bytes += flushed_bytes;
                epoch_points += flushed_points;
            }
            auto t_ep_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> t_ep = t_ep_end - t_ep_start;
            std::cout << "[stream_kmeans] epoch " << (e+1) << "/" << epochs
                      << ": batches=" << used
                      << ", points=" << epoch_points
                      << ", bytes=" << epoch_data_bytes
                      << ", time_s=" << t_ep.count() << std::endl;
        }

        auto t_all_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> t_all = t_all_end - t_all_start;

        write_centroids_fbin(out_path, centers, k, dim);
        std::cout << "[stream_kmeans] wrote centroids: " << out_path
                  << ", k=" << k << ", dim=" << dim
                  << ", bytes=" << static_cast<uint64_t>(k)*dim*sizeof(float)
                  << std::endl;
        std::cout << "[stream_kmeans] total_time_s=" << t_all.count() << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[stream_kmeans] error: " << e.what() << std::endl;
        return 3;
    }
} 
