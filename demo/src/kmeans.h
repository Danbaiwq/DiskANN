#pragma once

#include "utils.h"
#include <vector>

// Function to calculate squared Euclidean distance between two points
float calculate_distance(const DataPoint& p1, const DataPoint& p2);

// K-means++ initialization
DataSet kmeans_plusplus_init(const DataSet& data, size_t m);

// A basic K-means implementation using a given set of initial centroids
DataSet kmeans_lloyds(const DataSet& data, size_t m, DataSet& initial_centroids, int max_iterations); 