# Mini-batch K-means + Vamana 两阶段向量搜索 Demo

## 1. 项目概述

本项目旨在实现并评估一种高效的两阶段向量近邻搜索方案。该方案结合了 **Mini-batch K-means** 聚类和 **Vamana 图**索引，以应对大规模向量数据集的快速查询需求。

`demo` 被设计为 DiskANN 项目的一个子模块，它利用了 DiskANN 核心的 Vamana 图构建和搜索功能，同时围绕其构建了一个完整的数据预处理、索引构建和查询评估流程。

项目分为两种主要模式：
- **构建模式 (`build_mode`)**: 负责处理原始向量数据，通过聚类将其分桶，并为每个桶以及聚类中心构建持久化的 Vamana 图索引。
- **搜索模式 (`search_mode`)**: 负责加载构建好的索引，接收查询请求，执行两阶段搜索，并根据 Ground Truth 数据评估搜索结果的召回率和 QPS。

## 2. 核心设计思路

传统的暴力搜索或在单个大图上进行搜索，在数据规模巨大时会面临性能瓶颈。本项目采用的两阶段搜索策略旨在通过数据划分和分层索引来加速查询。

### 2.1. 数据分桶 (Clustering & Bucketing)
在索引构建阶段，我们并不直接对全量数据集构建一个庞大的 Vamana 图。而是首先通过 **Mini-batch K-means** 算法对数据进行聚类。

1.  **聚类**: 将 `N` 个数据点聚为 `m` 个簇，得到 `m` 个聚类中心（质心）。
2.  **分桶**: 将 `N` 个数据点根据它们与 `m` 个质心的距离，分配到 `l` 个最近的桶中 (`l` 通常是一个较小的数，如 5)。这意味着一个数据点可能同时属于多个桶，这有助于处理位于簇边界的数据点，提高召回率。

### 2.2. 两层 Vamana 图索引
分桶之后，我们构建两层索引结构：

1.  **Medoid Vamana 图**: 这是一个小而快的“全局索引”，由 `m` 个聚类中心（质心）构成。它的作用是在查询时快速定位到与查询向量最相似的几个数据区域。
2.  **Bucket Vamana 图**: 我们为 `m` 个桶中的每一个，都单独构建一个 Vamana 图。每个图只包含该桶内的数据点。这些是“局部索引”，用于在小的数据子集上进行精确搜索。

### 2.3. 两阶段查询流程
当一个查询向量 `q` 到来时：

1.  **粗筛 (Coarse Search)**: 首先在 `Medoid Vamana 图` 中搜索，快速找到与 `q` 最相似的 `f` 个聚类中心。这 `f` 个中心的 ID 就对应了 `f` 个最可能包含最终结果的桶。
2.  **精排 (Fine Search)**: 接着，我们只访问这 `f` 个桶对应的 `Bucket Vamana 图`。在每个桶图中搜索 `k` 个最近邻。
3.  **结果合并**: 将从 `f` 个桶中获取的 `f * k` 个候选结果合并，根据它们与 `q` 的真实距离进行全局排序，最终返回 Top-K 的结果。

这种方法的优势在于，它将一次全局搜索转化为了 `1` 次在小图上的快速定位和 `f` 次在更小图上的精确搜索，极大地减少了需要访问的节点和需要计算的距离总量。

## 3. 实现过程详解

### 3.1. 构建阶段 (`build_mode`)
- **入口**: `main.cpp: build_mode(base_path)`
- **步骤**:
    1.  **并行 K-means**: 为了加速聚类并提高质心质量，我们并行运行 `t` 次 Mini-batch K-means。每次都从全量数据中随机采样一部分数据进行训练。最终的质心是 `t` 次运行结果的平均值。
    2.  **数据分桶**: 遍历全量数据，计算每个点与所有质心的距离，并将其 ID 存入 `l` 个最近的桶中。
    3.  **索引构建**:
        - 调用 `build_and_save_vamana_graph` 函数为 `m` 个质心构建并保存 `medoid_vamana.index`。
        - 并行地为所有（数据点数 > 1 的）桶构建并保存各自的 `bucket_i_vamana.index`。
    4.  **持久化**: 将分桶的结果（一个记录了每个桶包含哪些原始数据点ID的映射表）保存到 `buckets.bin` 文件中，供搜索阶段使用。

### 3.2. 搜索与评估阶段 (`search_mode`)
- **入口**: `main.cpp: search_mode(query_path, gt_path)`
- **步骤**:
    1.  **加载数据**:
        - 加载查询文件 (`query.fbin`) 和 Ground Truth 文件 (`ground_truth.ivecs`)。
        - 加载构建阶段产出的 `buckets.bin` 和 `medoid_vamana.index`。
    2.  **LRU 缓存**: 初始化一个有大小限制（如 1GB）的 `IndexCache`，用于在搜索时按需、线程安全地加载和管理 `Bucket Vamana 图`。
    3.  **评估循环**:
        - 遍历一系列 `f` 值 (要访问的桶数量)。
        - 在每个 `f` 值下，使用 OpenMP 并行地对所有查询向量执行 `search_two_stage` 函数。
        - 计时并计算 QPS。
        - 将所有查询结果与 Ground Truth 对比，计算 R@10, R@50, R@100 并打印。

## 4. 性能优化历程

在实现过程中，我们进行了多轮性能优化：

1.  **K-means 优化**: 最初的 K-means 采用随机初始化，收敛慢。我们将其升级为 **K-means++** 初始化策略，通过选择更好的初始质心，显著减少了收敛所需的迭代次数。
2.  **计算优化**: 距离计算是整个流程中最频繁的操作。我们使用 **AVX2 指令集**重写了 `calculate_distance` 函数，利用 CPU 的 SIMD 功能并行处理多个浮点数运算，相比原始的标量循环实现了大幅加速。
3.  **并行化**: 对于数据处理和查询中的 embarassingly parallel 循环（如 K-means 的多次运行、分桶、图的构建、批量查询等），我们使用了 **OpenMP (`#pragma omp parallel for`)** 进行并行化，充分利用多核 CPU 的处理能力。
4.  **I/O 优化 (缓存策略)**:
    - **V0 (无缓存)**: 最初，每次查询都在 `search_two_stage` 内部从磁盘加载所需的桶索引。这导致了严重的 I/O 瓶颈。
    - **V1 (全量预热缓存)**: 我们引入了一个简单的 `std::vector` 缓存，在评估开始前，一次性地将所有桶索引全部加载到内存中。这极大地提升了 QPS，但对内存不友好。
    - **V2 (LRU 缓存)**: 最终，我们实现了一个功能完善的、线程安全的 **LRU (Least Recently Used) 缓存 (`IndexCache`)**。它有固定的大小限制，按需从磁盘加载索引，并在缓存满时自动淘汰最久未使用的索引。这在 QPS 和内存使用之间取得了完美的平衡，使 `demo` 更加健壮和实用。

## 5. 如何编译和运行

本 `demo` 作为 DiskANN 的子项目进行编译。

1.  **编译**:
    ```bash
    # 1. 进入 DiskANN 项目根目录
    cd /path/to/DiskANN

    # 2. 创建并进入 build 目录
    mkdir -p build
    cd build

    # 3. 运行 CMake (如果 g++ 不在标准路径，请指定)
    # cmake -D CMAKE_CXX_COMPILER=/path/to/g++ ..
    cmake ..

    # 4. 编译整个项目
    make
    ```

2.  **运行**:
    - **构建索引**:
      ```bash
      # 可执行文件位于 build/demo/
      ./demo/demo_test /path/to/your/base.fbin
      ```
      (会在 `build/demo/` 目录下生成 `medoid_vamana.index`, `bucket_*.index`, `buckets.bin` 等文件)

    - **运行搜索评估**:
      ```bash
      ./demo/demo_test /path/to/your/query.fbin /path/to/your/ground_truth.ivecs
      ```

## 6. 代码结构

- `main.cpp`: 程序入口，包含 `build_mode` 和 `search_mode` 的顶层逻辑。
- `utils.h/.cpp`: 负责数据加载/保存 (`.fbin`, `buckets.bin`) 和采样。
- `kmeans.h/.cpp`: 负责 K-means 聚类算法的实现，包括距离计算。
- `vamana_graph.h/.cpp`: 封装了与 DiskANN 库的交互，负责构建和保存 Vamana 图。
- `search.h/.cpp`: 负责查询逻辑，包括两阶段搜索、LRU 缓存、加载索引和计算召回率。 