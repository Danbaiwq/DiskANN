#include "kmeans.h"
#include <immintrin.h>
#include <stdexcept>
#include <random>
#include <iostream>
#include <limits>
#include <algorithm>
#include <deque>
#include <chrono>

// Function to calculate squared Euclidean distance between two points
float calculate_distance(const DataPoint& p1, const DataPoint& p2) {
    if (p1.size() != p2.size()) {
        throw std::invalid_argument("Vectors must have the same dimension.");
    }
    const float* d1 = p1.data();
    const float* d2 = p2.data();
    size_t dim = p1.size();

    __m256 diff, sum = _mm256_setzero_ps();
    size_t i = 0;

    // Process 8 floats at a time
    for (; i + 7 < dim; i += 8) {
        __m256 v1 = _mm256_loadu_ps(d1 + i);
        __m256 v2 = _mm256_loadu_ps(d2 + i);
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    
    // Horizontal sum of the 8 floats in the accumulator
    float buffer[8];
    _mm256_storeu_ps(buffer, sum);
    float distance = buffer[0] + buffer[1] + buffer[2] + buffer[3] + 
                     buffer[4] + buffer[5] + buffer[6] + buffer[7];

    // Process remaining elements
    for (; i < dim; ++i) {
        float diff_scalar = d1[i] - d2[i];
        distance += diff_scalar * diff_scalar;
    }

    return distance;
}

// K-means++ initialization
DataSet kmeans_plusplus_init(const DataSet& data, size_t m) {
    if (data.empty() || m == 0) return {};
    size_t dim = data[0].size();
    DataSet centroids;
    centroids.reserve(m);

    std::random_device rd;
    std::mt19937 gen(rd());
    
    // 1. Choose one center uniformly at random from among the data points.
    std::uniform_int_distribution<> distrib(0, data.size() - 1);
    centroids.push_back(data[distrib(gen)]);

    std::vector<float> min_dists(data.size(), std::numeric_limits<float>::max());

    while (centroids.size() < m) {
        // 2. For each data point x, compute D(x), the distance between x and the nearest center that has already been chosen.
        float total_dist = 0.0f;
        #pragma omp parallel for reduction(+:total_dist)
        for (size_t i = 0; i < data.size(); ++i) {
            float dist = calculate_distance(data[i], centroids.back());
            if (dist < min_dists[i]) {
                min_dists[i] = dist;
            }
        }
        for(const auto& d : min_dists) {
            total_dist +=d;
        }


        // 3. Choose one new data point x as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)^2.
        std::uniform_real_distribution<> prob_distrib(0.0, total_dist);
        float r = prob_distrib(gen);
        float cumulative_dist = 0.0f;
        for (size_t i = 0; i < data.size(); ++i) {
            cumulative_dist += min_dists[i];
            if (cumulative_dist >= r) {
                centroids.push_back(data[i]);
                break;
            }
        }
    }
    return centroids;
}


// A basic K-means implementation using a given set of initial centroids
DataSet kmeans_lloyds(const DataSet& data, size_t m, DataSet& initial_centroids, int max_iterations) {
    if (data.empty() || m == 0) return {};
    size_t dim = data[0].size();
    
    DataSet centroids = initial_centroids;

    std::vector<size_t> assignments(data.size(), 0);

    // Create a random engine once for re-initialization if needed
    std::mt19937 reinit_gen(std::random_device{}());

    for (int iter = 0; iter < max_iterations; ++iter) {
        // 2. Assignment step
        bool changed = false;
        #pragma omp parallel for
        for (size_t i = 0; i < data.size(); ++i) {
            float min_dist = -1.0f;
            size_t best_cluster = 0;
            for (size_t j = 0; j < m; ++j) {
                float dist = calculate_distance(data[i], centroids[j]);
                if (min_dist < 0 || dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            if (assignments[i] != best_cluster) {
                #pragma omp critical
                {
                    if (assignments[i] != best_cluster){
                        assignments[i] = best_cluster;
                        changed = true;
                    }
                }
            }
        }

        if (!changed) {
            std::cout << "K-means converged after " << iter + 1 << " iterations." << std::endl;
            break; // Converged
        }

        // 3. Update step
        DataSet new_centroids(m, DataPoint(dim, 0.0f));
        std::vector<size_t> counts(m, 0);
        for (size_t i = 0; i < data.size(); ++i) {
            size_t cluster_idx = assignments[i];
            for (size_t d = 0; d < dim; ++d) {
                new_centroids[cluster_idx][d] += data[i][d];
            }
            counts[cluster_idx]++;
        }

        #pragma omp parallel for
        for (size_t j = 0; j < m; ++j) {
            if (counts[j] > 0) {
                for (size_t d = 0; d < dim; ++d) {
                    new_centroids[j][d] /= counts[j];
                }
            } else {
                // If a centroid becomes empty, re-initialize it to a random point
                // This is a simple strategy; more advanced ones exist.
                std::uniform_int_distribution<> distrib(0, data.size() - 1);
                 #pragma omp critical
                {
                    new_centroids[j] = data[distrib(reinit_gen)];
                }
            }
        }
        centroids = new_centroids;
    }

    return centroids;
} 

static inline float l2_sqr_avx2_ptr(const float* a, const float* b, size_t dim) {
#ifdef __AVX2__
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 32 <= dim; i += 32) {
        __m256 a0 = _mm256_loadu_ps(a + i + 0);
        __m256 b0 = _mm256_loadu_ps(b + i + 0);
        __m256 d0 = _mm256_sub_ps(a0, b0);
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);

        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        __m256 d1 = _mm256_sub_ps(a1, b1);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);

        __m256 a2 = _mm256_loadu_ps(a + i + 16);
        __m256 b2 = _mm256_loadu_ps(b + i + 16);
        __m256 d2 = _mm256_sub_ps(a2, b2);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);

        __m256 a3 = _mm256_loadu_ps(a + i + 24);
        __m256 b3 = _mm256_loadu_ps(b + i + 24);
        __m256 d3 = _mm256_sub_ps(a3, b3);
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);
    }
    __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vd = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(vd, vd, acc);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
#endif
}

static inline size_t argmin_center(const float* x, const DataSet& centers) {
    size_t k = centers.size();
    size_t dim = centers[0].size();
    size_t best = 0;
    float bestd = std::numeric_limits<float>::max();
    for (size_t j = 0; j < k; ++j) {
        float d = l2_sqr_avx2_ptr(x, centers[j].data(), dim);
        if (d < bestd) { bestd = d; best = j; }
    }
    return best;
}

static inline float centers_shift_avg(const DataSet& a, const DataSet& b) {
    size_t k = a.size();
    size_t dim = a[0].size();
    double sum = 0.0;
    for (size_t j = 0; j < k; ++j) {
        sum += std::sqrt(l2_sqr_avx2_ptr(a[j].data(), b[j].data(), dim));
    }
    return static_cast<float>(sum / k);
}

DataSet distributed_minibatch_kmeans(const FlatDataSet& full_flat_data,
                                     size_t num_points,
                                     size_t dim,
                                     size_t k,
                                     int workers,
                                     float alpha,
                                     int max_iter,
                                     float tol,
                                     int patience) {
    // 初始化全局中心：用一次 kmeans++ 在全量上随机选 k 个点
    DataSet init_sample;
    {
        size_t init_n = std::min<size_t>(num_points, std::max<size_t>(k * 10, k));
        init_sample.reserve(init_n);
        std::vector<size_t> idx(num_points); std::iota(idx.begin(), idx.end(), 0);
        std::mt19937 g(std::random_device{}());
        std::shuffle(idx.begin(), idx.end(), g);
        for (size_t i = 0; i < init_n; ++i) {
            init_sample.push_back(get_point_copy(full_flat_data, idx[i], dim));
        }
    }
    DataSet centers = kmeans_plusplus_init(init_sample, k);
    std::vector<uint64_t> counts(k, 0);

    std::deque<float> shift_hist;
    int satisfied_rounds = 0;

    for (int it = 1; it <= max_iter; ++it) {
        auto t0 = std::chrono::high_resolution_clock::now();
        // Master 广播 centers（在单进程里就是直接可见）

        // 每个 worker 对其分片随机采样 alpha 比例的 mini-batch
        // 分片范围：简单平均划分 [0, num_points)
        std::vector<DataSet> local_sums(workers, DataSet(k, DataPoint(dim, 0.0f)));
        std::vector<std::vector<uint64_t>> local_counts(workers, std::vector<uint64_t>(k, 0));

        #pragma omp parallel for
        for (int w = 0; w < workers; ++w) {
            size_t start = (static_cast<size_t>(w) * num_points) / workers;
            size_t end   = (static_cast<size_t>(w + 1) * num_points) / workers;
            size_t shard_n = end - start;
            size_t sample_n = static_cast<size_t>(std::max<double>(1.0, std::floor(alpha * shard_n)));

            std::vector<size_t> ids(shard_n);
            std::iota(ids.begin(), ids.end(), start);
            std::mt19937 rng(std::random_device{}());
            std::shuffle(ids.begin(), ids.end(), rng);

            for (size_t i = 0; i < sample_n; ++i) {
                size_t gid = ids[i];
                const float* x = full_flat_data.data() + gid * dim;
                size_t j = argmin_center(x, centers);
                float* sumv = local_sums[w][j].data();
                for (size_t d = 0; d < dim; ++d) sumv[d] += x[d];
                local_counts[w][j] += 1;
            }
        }

        // Master 聚合
        std::vector<uint64_t> total_count(k, 0);
        DataSet total_sum(k, DataPoint(dim, 0.0f));
        for (int w = 0; w < workers; ++w) {
            for (size_t j = 0; j < k; ++j) {
                total_count[j] += local_counts[w][j];
                float* tsum = total_sum[j].data();
                const float* lsum = local_sums[w][j].data();
                for (size_t d = 0; d < dim; ++d) tsum[d] += lsum[d];
            }
        }

        // 计算新中心（加权累积更新）
        DataSet new_centers = centers;
        for (size_t j = 0; j < k; ++j) {
            uint64_t newc = counts[j] + total_count[j];
            if (newc == 0) continue;
            float w_old = static_cast<float>(counts[j]);
            float w_new = static_cast<float>(total_count[j]);
            float* cj = new_centers[j].data();
            const float* oj = centers[j].data();
            const float* sj = total_sum[j].data();
            for (size_t d = 0; d < dim; ++d) {
                cj[d] = (w_old * oj[d] + sj[d]) / static_cast<float>(newc);
            }
        }

        // 位移与收敛判定（移动平均）
        float shift = centers_shift_avg(centers, new_centers);
        shift_hist.push_back(shift);
        if (shift_hist.size() > 5) shift_hist.pop_front();
        float smooth = 0.0f; for (float v : shift_hist) smooth += v; smooth /= static_cast<float>(shift_hist.size());

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout.setf(std::ios::fixed); std::cout.precision(6);
        std::cout << "[DistKMeans] iter=" << it << ", shift=" << shift << ", smooth=" << smooth << ", time_ms=" << ms << std::endl;

        centers.swap(new_centers);
        for (size_t j = 0; j < k; ++j) counts[j] += total_count[j];

        if (smooth < tol) {
            satisfied_rounds += 1;
            if (satisfied_rounds >= patience) {
                std::cout << "[DistKMeans] converged at iter=" << it << std::endl;
                break;
            }
        } else {
            satisfied_rounds = 0;
        }
    }

    std::cout << "[DistKMeans] done. iters<= " << max_iter << ", k=" << k << std::endl;
    return centers;
} 