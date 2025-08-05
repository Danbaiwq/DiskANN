#include "kmeans.h"
#include <immintrin.h>
#include <stdexcept>
#include <random>
#include <iostream>
#include <limits>
#include <algorithm>

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