#pragma once

#include "utils.h"
#include <vector>

// Function to calculate squared Euclidean distance between two points
float calculate_distance(const DataPoint& p1, const DataPoint& p2);

// K-means++ initialization
DataSet kmeans_plusplus_init(const DataSet& data, size_t m);

// A basic K-means implementation using a given set of initial centroids
DataSet kmeans_lloyds(const DataSet& data, size_t m, DataSet& initial_centroids, int max_iterations);

// 分布式多轮 mini-batch KMeans（Master-Worker 聚合）
// 参数：
// - full_flat_data: 全量数据（扁平）
// - num_points, dim: 数据规模
// - k: 聚类数
// - workers: worker 数（建议与 t 一致）
// - alpha: 每轮每个 worker 的 mini-batch 采样比例（相对其分片大小）
// - max_iter: 最大迭代轮数
// - tol: 位移阈值
// - patience: 连续满足阈值的耐心轮数
// 返回：最终 k 个中心
DataSet distributed_minibatch_kmeans(const FlatDataSet& full_flat_data,
                                     size_t num_points,
                                     size_t dim,
                                     size_t k,
                                     int workers,
                                     float alpha,
                                     int max_iter,
                                     float tol,
                                     int patience); 