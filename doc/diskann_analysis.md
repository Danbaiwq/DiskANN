# DiskANN SSD 索引（构建与查询）流程解析

本文基于源码梳理 DiskANN 在构建 SSD 索引与查询 SSD 索引时的关键步骤、涉及的主要文件与函数、重要参数及其作用范围，帮助快速理解端到端流程与可调优点。

## 一、构建 SSD 索引（apps/build_disk_index）

- 入口：`apps/build_disk_index.cpp` 中 `main`
  - 解析参数（必需：`--data_type`、`--dist_fn`、`--index_path_prefix`、`--data_path`、`--search_DRAM_budget(B)`、`--build_DRAM_budget(M)`；可选：`-T`、`-R`、`-L`、`--PQ_disk_bytes`、`--build_PQ_bytes`、`--use_opq`、过滤相关等）。
  - 组装字符串参数 `params` 后调用：
    - `diskann::build_disk_index<T[, LabelT]>(data_path, index_path_prefix, params, metric, use_opq, codebook_prefix, use_filters, label_file, universal_label, filter_threshold, Lf)`

- 核心实现：`src/disk_utils.cpp` 中模板函数
  - 函数签名（示例片段）如下：
```1080:1405:/home/danbai.wq/DiskANN/src/disk_utils.cpp
template <typename T, typename LabelT>
int build_disk_index(const char *dataFilePath, const char *indexFilePath, const char *indexBuildParameters,
                     diskann::Metric compareMetric, bool use_opq, const std::string &codebook_prefix, bool use_filters,
                     const std::string &label_file, const std::string &universal_label, const uint32_t filter_threshold,
                     const uint32_t Lf)
{
  // 解析 R, L, B(final_index_ram_limit), M(indexing_ram_budget), T(num_threads),
  // PQ_disk_bytes(可选), append_reorder_data(可选), build_PQ_bytes(可选), QD(可选)
  // 根据 metric 做 MIPS/COSINE 预处理，生成临时 base 文件
  // 训练/加载 PQ pivots，生成内存侧 PQ 压缩向量 (index_prefix_pq_compressed.bin)
  // 构建/合并 Vamana 内存图（可能分片）并保存 (index_prefix_mem.index)
  // 生成磁盘布局 (index_prefix_disk.index 及其元数据)
  // 采样生成 warmup queries (index_prefix_sample_data.bin)
}
```

- 关键步骤细化：
  - 数据预处理（按度量）
    - MIPS：扩一维并归一化，写入 `index_prefix_prepped_base.bin`；保存 `max_base_norm`。
    - COSINE：整体归一化到临时文件。
  - PQ 压缩（内存侧必做）
    - 计算 `num_pq_chunks`：依据 `B` 和点数估算（或 `QD` 强制覆盖）。
    - 训练 pivots 并生成 `index_prefix_pq_pivots.bin` 与 `index_prefix_pq_compressed.bin`。
  - 可选：磁盘侧 PQ（高维数据、SSD 空间受限）
    - 若 `--PQ_disk_bytes > 0` 则额外生成 `index_prefix_disk.index_pq_pivots.bin`、`index_prefix_disk.index_pq_compressed.bin`。
    - 若同时 `--append_reorder_data`，在磁盘索引中追加全精度向量用于最终重排（重排数据不长驻内存）。
  - 图构建（Vamana）
    - 若全量图能放入 `M`：一次性构建并 `save` 到 `index_prefix_mem.index`。
    - 否则：`partition_with_ram_budget` 切分子图，逐分片 `build`，`merge_shards` 合并为整体图，输出 `index_prefix_mem.index`，同时输出 `..._medoids.bin`、`..._centroids.bin`。
    - 过滤检索时涉及 `label_file`、`labels_to_medoids.txt`、`_universal_label.txt` 等。
  - 生成磁盘布局
    - 无磁盘 PQ：`create_disk_layout<T>(base_or_prepped, mem.index, disk.index [, reorder_data_file])`
    - 有磁盘 PQ：`create_disk_layout<uint8_t>(disk_pq_compressed, mem.index, disk.index [, reorder_data_file])`
    - 产出主文件：`index_prefix_disk.index`，并写入元数据（扇区对齐、邻接、坐标段等）。
  - 采样与清理
    - `gen_random_slice` 生成 `index_prefix_sample_data.bin` 供服务端预热使用。
    - 清理临时文件、将过滤相关文件从 `mem.index` 侧拷贝到 `disk.index` 侧。

- 主要生成文件（以 `--index_path_prefix PREFIX` 为例）
  - `PREFIX_pq_pivots.bin`、`PREFIX_pq_compressed.bin`
  - `PREFIX_mem.index` 及其 `.data` 等中间件（最终会删除）
  - `PREFIX_disk.index`（主磁盘索引）、`PREFIX_disk.index_medoids.bin`、`PREFIX_disk.index_centroids.bin`
  - 过滤：`PREFIX_disk.index_labels.txt`、`PREFIX_disk.index_labels_to_medoids.txt`、`PREFIX_disk.index_labels_map.txt`、`..._universal_label.txt`
  - 可选磁盘 PQ：`PREFIX_disk.index_pq_pivots.bin`、`PREFIX_disk.index_pq_compressed.bin`
  - 预热：`PREFIX_sample_data.bin`

## 二、查询 SSD 索引（apps/search_disk_index）

- 入口：`apps/search_disk_index.cpp`
  - 初始化对齐文件读器（Linux/Windows）。
  - 构造 `diskann::PQFlashIndex<T, LabelT>`，调用 `load(num_threads, index_path_prefix)`：
    - 加载内存侧 PQ 压缩向量、PQ pivots/OPQ、图/磁盘元数据与必要缓存。
  - 缓存热点节点：`cache_bfs_levels(num_nodes_to_cache) -> load_cache_list(node_list)`。
  - 可选预热：加载/合成 warmup queries，执行少量 `cached_beam_search`。
  - 针对每个 `L`：确定 `beamwidth`（可调参或 `optimize_beamwidth` 自动寻优），并并行执行查询。

- 核心检索：`src/pq_flash_index.cpp` 中
```1180:1600:/home/danbai.wq/DiskANN/src/pq_flash_index.cpp
void PQFlashIndex<T, LabelT>::cached_beam_search(const T *query, uint64_t k, uint64_t L,
                                                 uint64_t *indices, float *distances, uint64_t beamwidth,
                                                 ... ) {
  // 1) 归一化/转置到对齐缓冲；预处理 PQ 查询表
  // 2) 选择起始 medoid（无过滤用全局质心，过滤用标签到 medoid 映射）
  // 3) 以 best_medoid 入堆，循环：
  //    - 从候选集中选 beam（优先命中 _nhood_cache；否则构造对齐 4KB 读请求）
  //    - 命中缓存：直接拿邻接与坐标；SSD 读取：读取扇区，解析邻接+坐标
  //    - 计算 PQ 近似距离，扩展邻接入堆，记录 IO/CPU 统计
  //    - 控制 IO 限制与 beam 宽度
  // 4) 结束后做最终 topK 选择；若启用重排，读取全精度向量做精排
}
```

- 典型调用（检索程序内）：
```227:254:/home/danbai.wq/DiskANN/apps/search_disk_index.cpp
_pFlashIndex->cached_beam_search(query + (i * query_aligned_dim), recall_at, L,
  query_result_ids_64.data() + (i * recall_at),
  query_result_dists[test_id].data() + (i * recall_at),
  optimized_beamwidth, use_reorder_data, stats + i);
```

- 重要参数
  - `-L (--search_list)`: 候选集大小；越大 recall 越高、IO 越多。
  - `-W (--beamwidth)`: 每轮最大并发 IO 请求数；小值追求吞吐，大值追求低往返轮次。
  - `--num_nodes_to_cache`: 缓存热点节点数量，能降低 SSD 访问比例、提升延迟稳定性。
  - `--use_reorder_data`: 若构建时追加了全精度，查询阶段对 top 若干候选做精排，提高最终精度。

## 三、从“构建参数 → 运行时行为”的映射

- `B = --search_DRAM_budget`：决定内存侧 PQ 压缩字节数（`num_pq_chunks`）；直接影响常驻内存与近似精度。
- `M = --build_DRAM_budget`：决定是否分片构建与合并；影响构建耗时与中间文件体量。
- `--PQ_disk_bytes` 与 `--append_reorder_data`：影响 SSD 存储格式（压缩或全精度 + 可选重排数据）。
- `-R/-L`：影响 Vamana 图的稀疏度与构建/查询时的候选规模。
- `--use_opq`：OPQ 旋转在部分高维数据上更省空间/更准，但构建耗时略增。

## 四、参考与定位

- 用法文档：`workflows/SSD_index.md`
```1:32:/home/danbai.wq/DiskANN/workflows/SSD_index.md
To generate an SSD-friendly index, use the apps/build_disk_index program ...
To search the SSD-index, use the apps/search_disk_index program ...
```
- 构建入口与实现：`apps/build_disk_index.cpp`、`src/disk_utils.cpp`
- 查询入口与实现：`apps/search_disk_index.cpp`、`include/pq_flash_index.h`、`src/pq_flash_index.cpp`

以上内容可作为工程级别的“端到端”认知地图，帮助在不同阶段（构建/服务）定位关键参数与性能杠杆。 

## 五、内存占用与磁盘占用分析

本节从“构建阶段（Build）”与“查询阶段（Search/Serving）”两个阶段，拆解峰值/常驻内存与磁盘占用来源，并给出可估算公式与示例。

### 5.1 构建阶段峰值内存（SSD 索引）

- 相关代码与常量
  - 估算函数（内存图阶段）：`include/index.h: estimate_ram_usage()` 使用 `OVERHEAD_FACTOR=1.1`、`defaults::GRAPH_SLACK_FACTOR=1.3`
  - 扇区常量：`defaults::SECTOR_LEN=4096`
  - 参考构建流程：`src/disk_utils.cpp::build_disk_index()`、`build_merged_vamana_index()`

- 峰值内存主要构成
  - 原始或预处理后数据装载（MIPS/COSINE 会写临时文件，但构建时仍需流式/块状处理内存）
  - Vamana 图构建中间结构：邻接、锁、外层向量（由 `estimate_ram_usage()` 近似）
  - PQ 训练/压缩：pivots、临时缓冲与最终内存侧 PQ 表（`_pq_compressed.bin` 的生成期内存）
  - 过滤索引时的标签/中间结构

- 近似估算（一次性内存构建时）
  - 数据载入：`size_of_data ≈ N * ROUND_UP(D,8) * sizeof(T)`
  - 图：`size_of_graph ≈ N * R * sizeof(uint32_t) * GRAPH_SLACK_FACTOR`（R 为 `--max_degree`）
  - 锁与外层向量：`N * sizeof(mutex)` 与 `N * sizeof(ptrdiff_t)`（常量开销，数量级 O(N)）
  - 总计：`OVERHEAD_FACTOR * (size_of_data + size_of_graph + size_of_locks + size_of_outer_vector)`
  - 若 `M(--build_DRAM_budget)` 不足以一次性构建，框架会分片构建并合并，峰值内存大致受分片规模、并发与 PQ 训练影响；总体峰值低于一次性构建，但耗时更长。

- 额外内存因素
  - `--build_PQ_bytes > 0`：构建阶段用 PQ 加速距离计算，需持有额外 PQ 表与临时变量，但对总峰值影响通常小于数据与图本体。
  - 过滤：密集标签拆分（`breakup_dense_points`）会临时放大数据规模。

### 5.2 查询阶段内存（常驻与峰值）

- 常驻内存（必需）
  - 内存侧 PQ 压缩向量：`N * num_pq_chunks` 字节（每点1字节/块，256中心编码）
  - PQ pivots/码本（含可选 OPQ 旋转矩阵）：`O(D * 256)` 与若干矩阵，规模远小于数据本体
  - 部分元数据（medoids、centroids等）

- 可选常驻内存
  - 节点热点缓存（`--num_nodes_to_cache`）：缓存邻接与坐标，命中时少一次 SSD 读取，按缓存点数线性增加内存

- 查询峰值内存
  - 每线程 scratch：PQ 距离表、对齐查询缓冲、扇区缓冲（`SECTOR_LEN` 对齐）、候选集合等；与 `-T`、`-L`、`-W` 成正相关
  - 若启用重排（`--use_reorder_data` 且构建时追加全精度），最终阶段会读取 top 若干候选的全精度向量做精排，产生瞬时内存与 IO 峰值

### 5.3 磁盘占用（SSD 索引）

- 主体：`PREFIX_disk.index`（对齐扇区布局，包含每点邻接与坐标段（压缩或全精度/磁盘PQ），以及全局元数据）
- PQ 相关文件：`PREFIX_pq_pivots.bin`、`PREFIX_pq_compressed.bin`（内存侧PQ）；若 `--PQ_disk_bytes>0` 则有 `PREFIX_disk.index_pq_pivots.bin`、`PREFIX_disk.index_pq_compressed.bin`
- 过滤/路由：`..._medoids.bin`、`..._centroids.bin`、`..._labels*.txt` 等

### 5.4 示例：100 万条 × 768 维 float（SSD 索引）

- 条件设定
  - N = 1,000,000；D = 768；T = float32（4字节）
  - R = 64（默认）
  - 搜索内存预算 B = 3 GB（示例，近似决定 `num_pq_chunks`）
  - 构建内存预算 M 足够一次性构建（观察峰值）
  - 不考虑过滤/磁盘侧 PQ/重排（即 `--PQ_disk_bytes=0`、不追加重排数据）

- 构建阶段峰值（一次性构建近似）
  - 数据：`N * ROUND_UP(D,8) * 4B = 1e6 * 768 * 4 ≈ 3.07 GB`
  - 图：`N * R * 4B * 1.3 ≈ 1e6 * 64 * 4 * 1.3 ≈ 332.8 MB`
  - 锁 + 外层向量：数量级 O(N)；以 64 位平台估算：`~ (N * 8B + N * sizeof(mutex))`，粗略取 100–200 MB 量级
  - 小计：`(3.07 GB + 0.33 GB + ~0.15 GB) * 1.1 ≈ 3.75–3.9 GB`
  - PQ 训练/中间：相对上述占比小（通常 < 数百 MB），总体峰值可按 `~4.0–4.5 GB` 粗估（依实现/并发差异）
  - 若 M 不足，分片构建峰值下降，但总时长上升

- 查询常驻内存
  - 计算 `num_pq_chunks`：近似 `num_pq_chunks ≈ floor(B(字节) / N)`；B=3GB → `~ 3e9 / 1e6 = 3000`，受限于 `MAX_PQ_CHUNKS` 和 `D`，最终 `num_pq_chunks = min(D, MAX_PQ_CHUNKS)`；C++ 版本上限通常 256（按实现），则 `num_pq_chunks = 256`
  - 内存侧 PQ 向量表：`N * num_pq_chunks * 1B = 1e6 * 256 ≈ 256 MB`
  - PQ pivots/OPQ：`O(D*256*4B) ≈ 768*256*4 ≈ 0.75 MB`（远小于 1MB；若含 OPQ 矩阵也极小）
  - 元数据与索引常量：几十 MB 以内
  - 合计常驻：`~ 300 MB` 量级（不含热点缓存）
  - 热点缓存：若缓存 250k 节点，粗略估算每节点缓存邻接+坐标开销几十到上百字节，则 `~10–30 MB+`，视实现而定
  - 每线程 scratch：与线程数/参数成正比，通常若干 MB 到几十 MB 总量

- SSD 磁盘占用
  - 磁盘索引主体：若不做磁盘PQ与重排，存全精度向量：`~ (头+邻接+对齐开销) + N*D*4B`
    - 数据体：`~ 3.07 GB`
    - 邻接：`~ 256 MB` 量级（与构建时图类似，但布局不同；4KB 扇区对齐会放大）
    - 对齐/元数据：若干百 MB 内外（依 `SECTOR_LEN` 对齐与实现）
  - 合计粗估：`~ 3.6–4.2 GB`
  - 若启用磁盘 PQ（例如 64 字节/向量）：数据体变为 `~ N * 64B = 64 MB`，总占用明显下降，但召回上限受限；可选 `--append_reorder_data` 追加全精度数据作重排，磁盘占用又增加 `~3.07 GB`

### 5.5 示例：100 万条 × 768 维 float（内存索引）

- 仅内存索引（不涉及 SSD 磁盘布局）
  - 数据：`~ 3.07 GB`
  - 图：`~ 0.33 GB`
  - 锁/结构：`~ 0.15 GB`（数量级）
  - 合计常驻：`~ 3.6–3.8 GB`（不含额外功能与并发 scratch）
  - 若使用内存侧 PQ 替代全精度存储，则数据部分可压缩为 `N * num_pq_bytes`，例如 16B/向量 → `~ 16 MB`；但需保留 PQ pivots 和必要的近似表，检索需用 PQ 距离，精度下降

> 注：上述估算为工程向的数量级估算。实际占用随实现版本、编译选项、线程 scratch、缓存策略以及是否使用过滤/OPQ/重排等有差异。建议在目标环境中以 `resident set size`、文件尺寸与进程自检日志为准做校准。 