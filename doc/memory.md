# DiskANN 磁盘索引（SSD）内存占用分析

本文梳理 DiskANN 在“构建 SSD 索引”和“查询 SSD 索引（search_disk_index）”时的主要内存占用来源，并给出可操作的优化建议。分析基于源码：`apps/search_disk_index.cpp`、`src/pq_flash_index.cpp`、`src/scratch.cpp`、`include/defaults.h` 等。

## 一、构建阶段（apps/build_disk_index）
构建时主要内存来自：
- 原始向量数据加载与图构建中间结构（不在本文展开）。
- 训练/加载 PQ 质心表与可选的 OPQ 旋转矩阵。
- 生成“内存侧 PQ 压缩向量”（供查询时常驻内存使用）。
- 如果设置了 `--PQ_disk_bytes > 0`，同时会为“磁盘侧 PQ 编码”准备必要的元数据与文件。
- 如果设置了 `--append_reorder_data`，会把“全精度向量”追加写入磁盘索引用于重排（查询时不长驻内存）。

构建期的峰值内存和数据规模、并发、临时缓冲相关，通常不作为长期驻留；本文主要聚焦服务时（查询）的常驻内存。

## 二、查询阶段（apps/search_disk_index）
`search_disk_index` 典型流程：
1. 加载索引：
   - 读取“内存侧 PQ 压缩向量”（常驻，所有点）。
   - 读取 PQ 质心表/可选 OPQ 矩阵。
   - 打开磁盘索引文件，设置每线程 scratch 缓冲。
2. 可选：按 `--num_nodes_to_cache` 从 medoid 附近 BFS 选点并缓存节点（坐标与邻接），作为热点缓存。
3. 查询：对每条 query 执行 `PQFlashIndex::cached_beam_search`，beam 扩展时优先命中缓存，否则向 SSD 发起对齐的 4KB 扇区读，使用 PQ 距离近似推进候选集合；若启用 `--use_reorder_data`，对 top(k×3) 读取全精度向量做最后重排。

### 2.1 常驻内存的大头
- 内存侧 PQ 压缩向量 `this->data`
  - 大小 ≈ `N × _n_chunks` 字节（每点 PQ 字节数），随维度与 PQ 配置（build_PQ_bytes）变化。
  - 代码参考：`src/pq_flash_index.cpp` 中对 `_pq_compressed.bin` 的加载。

- 节点缓存（由 `--num_nodes_to_cache` 决定，默认为 0）
  - 坐标缓存 `_coord_cache_buf`
    - 若磁盘未做 PQ 压缩（`PQ_disk_bytes=0`），缓存为 full-precision 向量：大小 ≈ `num_nodes_to_cache × aligned_dim × sizeof(T)`，与维度线性相关，是高维数据内存上升的主要原因。
    - 若磁盘启用 PQ 压缩（`PQ_disk_bytes>0`），缓存为 PQ 码：大小 ≈ `num_nodes_to_cache × disk_pq_n_chunks`，显著小于 full-precision。
  - 邻接缓存 `_nhood_cache_buf`
    - 大小 ≈ `num_nodes_to_cache × (max_degree+1) × sizeof(uint32_t)`，与维度无关。
  - 代码参考：`src/pq_flash_index.cpp::load_cache_list` 分配与填充。

- 每线程 scratch（数量级较小）
  - `sector_scratch`：固定 `MAX_N_SECTOR_READS × SECTOR_LEN = 128 × 4096 = 512KB/线程`。
  - `coord_scratch`：≈ `aligned_dim × sizeof(T)`/线程。
  - `PQScratch`：包含 `aligned_pqtable_dist_scratch`（约 `256 × chunks × 4B`）、`aligned_pq_coord_scratch`（约 `MAX_GRAPH_DEGREE × chunks` 字节）等，通常几十到数百 KB 量级。
  - 代码参考：`src/scratch.cpp::SSDQueryScratch` 与 `PQScratch` 构造。

- PQ 质心表与（可选）OPQ：
  - 规模 ≈ `O(256 × dim × 4B)`，一般仅数 MB 级别。

- 统计数组、结果数组等：占用较小。

### 2.2 查询流程要点
- `cached_beam_search`：
  - 预处理 query，预计算 `256×chunks` 的 PQ chunk 距离表；
  - 以 medoid 为起点，循环：从缓存或 SSD 取邻接；对邻接点批量做 PQ 距离；用小根堆维护候选；
  - 可选“重排”（`--use_reorder_data`）：对 top(k×3) 读取全精度向量到 `sector_scratch`，重算距离，重新排序；不增加长驻内存。

## 三、为什么 SIFT(128D) ≈ 700MB，GIST(960D) ≈ 2GB？
- 主要差异来自“坐标缓存 `_coord_cache_buf`”的维度线性放大：
  - 若 `--num_nodes_to_cache = 500k`：
    - 128D 浮点：≈ `500k × 128 × 4B ≈ 244MB`；
    - 960D 浮点：≈ `500k × 960 × 4B ≈ 1.83GB`。
- 其它项：
  - 邻接缓存：`500k × (R+1) × 4B`，R 若 64–96，则约 130–190MB，和维度无关；
  - 内存侧 PQ 码：SIFT 可能 16–32B/点（约 16–32MB），GIST 可能 64–120B/点（约 64–120MB）；
  - PQ 表、线程 scratch 等相对较小。
- 合计后即可达到你观测的数量级：SIFT ~700MB、GIST ~2GB。根因是“缓存坐标项对维度的线性放大”。

## 四、优化建议（按影响度排序）
1. 降低 `--num_nodes_to_cache`
   - 对内存影响最大，且与维度线性相关。优先收敛该参数，结合延迟/QPS 目标逐步调优。
2. 构建时启用 `--PQ_disk_bytes > 0`
   - 使磁盘与缓存存储 PQ 码而非 full-precision，`_coord_cache_buf` 将按字节数而非维度增长，大幅降低内存。
   - 注意需重新构建索引。
3. 控制内存侧 PQ 字节数（`build_PQ_bytes`）
   - 可小幅降低 `this->data` 常驻内存，但收益通常不及前两项。
4. 线程数与 `--beamwidth` 对常驻内存影响较小（每线程 ~0.5–1MB 级别），一般无需为节省内存而下调。
5. `--use_reorder_data` 不产生额外长驻内存（仅查询时 4KB 对齐读取进入 `sector_scratch`）。

## 五、关键代码参考
- PQ 压缩向量加载（常驻）：`src/pq_flash_index.cpp` 加载 `*_pq_compressed.bin`。
- 节点缓存分配：`src/pq_flash_index.cpp::load_cache_list`（`_coord_cache_buf` 与 `_nhood_cache_buf`）。
- 线程 scratch：`src/scratch.cpp::SSDQueryScratch`（`sector_scratch`=512KB/线程）与 `PQScratch`。
- 查询主循环与重排：`src/pq_flash_index.cpp::cached_beam_search`。
- 常量：`include/defaults.h`（`SECTOR_LEN=4096`、`MAX_N_SECTOR_READS=128`、`MAX_GRAPH_DEGREE=512`）。

---
如需进一步压缩内存：优先组合“较小的 `--num_nodes_to_cache` + 启用 `--PQ_disk_bytes`”，通常能在维持可接受延迟的同时，显著降低高维数据集的内存占用。 

## 六、示例计算与估算模板

本节给出“可套用的估算公式 + 参数获取方法 + 两个数据集的样例”，帮助快速判断哪些项值得优化，以及投入额外内存大致能带来哪些性能改善（QPS/延迟/召回）。

### 6.1 统一估算公式（查询常驻内存）
- 内存侧 PQ 压缩向量：`Mem_pq_inmem ≈ N × PQ_inmem_bytes`
- 节点坐标缓存：
  - 若 SSD 未压缩：`Mem_coord_cache ≈ ncache × aligned_dim × sizeof(T)`
  - 若 SSD 启用 PQ 压缩：`Mem_coord_cache ≈ ncache × PQ_disk_bytes`
- 节点邻接缓存：`Mem_nhood_cache ≈ ncache × (R + 1) × 4B`
- 每线程 scratch（近似）：
  - `Mem_sector_scratch ≈ 512KB × threads`（固定 128×4KB/线程，见 `defaults.h`）
  - `Mem_coord_scratch ≈ aligned_dim × sizeof(T) × threads`
  - `Mem_pq_scratch ≈ (256 × chunks × 4B + MAX_GRAPH_DEGREE × chunks + MAX_GRAPH_DEGREE × 4B + 2 × aligned_dim × 4B) × threads`
- PQ 质心表与 OPQ：`Mem_pq_tables ≈ 256 × dim × 4B`（若使用 OPQ，另加 旋转矩阵，量级为 `dim × dim × 4B`，高维时数十 MB 以内）
- 其它（哈希表元数据、运行时统计/结果数组等）：`Mem_misc`（与具体实现/参数相关，可预留 10% 安全冗余）

合计：
- `Mem_total ≈ Mem_pq_inmem + Mem_coord_cache + Mem_nhood_cache + Mem_scratch(threads) + Mem_pq_tables + Mem_misc`

注意：若使用 `cache_bfs_levels`，其内部会将 `ncache` 上限裁剪为不超过总点数的 10%（见源码逻辑）。如果采用基于样本查询生成缓存（`generate_cache_list_from_sample_queries`），则可按照给定的 `num_nodes_to_cache` 精确生成，不受 10% 限制。

### 6.2 参数获取方法
- `N`、`dim`、`aligned_dim`、`R`、磁盘是否 PQ 压缩：
  - 均存于 `*_disk.index` 头部（或运行时由 `PQFlashIndex::load()` 打印/读取）。
- `PQ_inmem_bytes`：
  - 来自 `*_pq_compressed.bin` 的“每点字节数”，亦可在运行时通过 `PQFlashIndex::_n_chunks` 得到。
- `PQ_disk_bytes`：
  - 若存在 `*_disk.index_pq_pivots.bin`（或构建参数 `--PQ_disk_bytes > 0`），则启用 SSD 侧 PQ 压缩；每点字节数为 `_disk_pq_n_chunks`。
- `ncache`（实际缓存节点数）：
  - 由 CLI `--num_nodes_to_cache` 与缓存生成方式共同决定；使用 `cache_bfs_levels` 时，若输入超过 10% 总点数会被裁剪。
- `threads`：
  - CLI `--num_threads`。

建议：把上述参数与实际运行（`search_disk_index` 输出的 `Mean IOs`、`Mean IO(us)`、`CPU(s)` 和 `Recall@K`）一起记录到表格中，便于后续做“内存-性能”取舍。

### 6.3 样例 A：SIFT1M（N=1,000,000，float，dim=128）
假设：`R=64`，`threads=16`，`aligned_dim=128`。

对比不同配置（单位：MB，近似按 10^6 取整，不含 10% 冗余）：

- 情形 A1：SSD 未压缩（`PQ_disk_bytes=0`），缓存 100k（BFS 上限 10%），内存侧 PQ=32B
  - `Mem_pq_inmem ≈ 1,000,000 × 32B = 32`
  - `Mem_coord_cache ≈ 100,000 × 128 × 4B ≈ 51.2`
  - `Mem_nhood_cache ≈ 100,000 × 65 × 4B ≈ 26`
  - `Mem_scratch(16) ≈ 0.55 × 16 ≈ 8.8`（粗略）
  - `Mem_pq_tables ≈ 256 × 128 × 4B ≈ 0.13`
  - 小计 ≈ 118 MB（+ 元数据/哈希表等 10–20% 冗余）

- 情形 A2：SSD 未压缩，缓存 500k（采用样本查询生成缓存，绕过 10% 上限），内存侧 PQ=32B
  - `Mem_pq_inmem ≈ 32`
  - `Mem_coord_cache ≈ 500,000 × 128 × 4B ≈ 244`
  - `Mem_nhood_cache ≈ 500,000 × 65 × 4B ≈ 124`
  - `Mem_scratch(16) ≈ 8.8`
  - `Mem_pq_tables ≈ 0.13`
  - 小计 ≈ 409 MB（+ 冗余/哈希表开销，可能接近 500 MB）

- 情形 A3：SSD 未压缩，缓存 500k，内存侧 PQ=512B（极端：近似按 full-precision 尺度存 PQ 码）
  - `Mem_pq_inmem ≈ 1,000,000 × 512B ≈ 512`
  - `Mem_coord_cache ≈ 244`
  - `Mem_nhood_cache ≈ 124`
  - 其它合计 ≈ 10–20
  - 小计 ≈ 890 MB（+ 冗余）

- 情形 A4：SSD 启用 PQ 压缩（`PQ_disk_bytes=32`），缓存 500k，内存侧 PQ=32B
  - `Mem_pq_inmem ≈ 32`
  - `Mem_coord_cache ≈ 500,000 × 32B ≈ 16`
  - `Mem_nhood_cache ≈ 124`
  - 其它合计 ≈ 10–20
  - 小计 ≈ 182 MB（+ 冗余）

结论（SIFT）：同样缓存 500k 节点时，是否对 SSD 启用 PQ 压缩，会极大影响“坐标缓存”项；而把“内存侧 PQ 字节数”从 32B 提升到 512B，也会显著推高总内存。若你的实测为 ~700MB，通常意味着“缓存节点数较大”或“内存侧 PQ 字节数较高”（或二者皆是）。

### 6.4 样例 B：GIST1M（N=1,000,000，float，dim=960）
假设：`R=64`，`threads=16`，`aligned_dim≈960`。

- 情形 B1：SSD 未压缩，缓存 500k，内存侧 PQ=96B（示例）
  - `Mem_pq_inmem ≈ 96`
  - `Mem_coord_cache ≈ 500,000 × 960 × 4B ≈ 1,830`
  - `Mem_nhood_cache ≈ 124`
  - `Mem_scratch(16) ≈ (~0.7MB/线程) × 16 ≈ 11`（维度更大，scratch 略增）
  - `Mem_pq_tables ≈ 256 × 960 × 4B ≈ 0.94`
  - 小计 ≈ 2,062 MB（≈ 2.0 GB；与常见观测一致）

- 情形 B2：SSD 启用 PQ 压缩（`PQ_disk_bytes=96`），缓存 500k，内存侧 PQ=96B
  - `Mem_coord_cache ≈ 500,000 × 96B ≈ 48`
  - 其余同上，小计 ≈ 269 MB（+ 冗余）

结论（GIST）：维度 960 使“SSD 未压缩 + 大缓存”的坐标缓存项成为绝对大头（线性放大）；启用 SSD PQ 压缩可将该项从 ~1.83GB 降到数十 MB 量级。

### 6.5 内存-性能取舍要点（经验规律）
- 提升 `num_nodes_to_cache`（更多热点节点在内存）
  - 直接降低 `Mean IOs` 与 `Mean IO(us)`，从而降低查询延迟、提高 QPS；
  - 边际收益递减：缓存 0→50k→100k 的收益通常大于 400k→500k；
  - 对高维数据（如 GIST），此举的内存代价很高（坐标缓存线性放大）。
- 启用 `PQ_disk_bytes > 0`（SSD 侧 PQ 压缩）
  - 明显减少“坐标缓存”内存，并减少磁盘每节点的字节数，提高节点 packing（同一扇区容纳更多节点），通常减少 IO 次数；
  - 但使用 PQ 坐标参与距离估计，会轻微影响未重排时的召回；可配合 `--use_reorder_data` 在 top(k×3) 上做全精度重排以恢复召回，代价是每查询多一次小规模 IO 扇区读取与少量 CPU；
  - 对高维数据收益尤为显著。
- 增大 `PQ_inmem_bytes`（内存侧 PQ 字节数）
  - 提高 PQ 近似精度，通常减少 beam 扩展中的“误扩展”，从而降低 IO 次数与 CPU；
  - 但它是常驻内存中的线性大项（N×字节数），需与缓存内存统筹；
  - 召回的提升还可通过增大 `L` 或启用重排获得，注意整体延迟/IO 著账。
- `beamwidth (W)` 与 `L`（候选集合大小）
  - 增大 `W` 可减少迭代轮次、提升单查询延迟，但会增加总 IO 次数；应使用“自动调优 beamwidth”功能寻找最优点（已在工具中支持）；
  - 增大 `L` 提升召回但增加 CPU/IO，且也会扩大 `retset/full_retset` 的工作集（短期内存）。
- 线程数 `T`
  - 主要影响的是 scratch（约 0.5–1MB/线程）和吞吐，不是常驻大头；需结合 SSD IOPS/带宽能力与 CPU 查看最佳线程数。

### 6.6 实操建议（如何做“代价→收益”评估）
1. 固定数据集与 `L,K`，对下列参数网格做小规模跑分：
   - `num_nodes_to_cache ∈ {0, 50k, 100k, 200k, 500k}`（注意 BFS 10% 上限）
   - `PQ_disk_bytes ∈ {0, 32, 64, 96}`（重建索引）
   - `PQ_inmem_bytes ∈ {16, 32, 64, 96, 128}`（重建索引）
   - `beamwidth ∈ {1, 2, 4, 8}`（支持自动调优）
2. 每个点记录：`Mem_total`（用 pmap/smaps 或容器内存）、`Mean IOs`、`Mean IO(us)`、`CPU(s)`、`QPS`、`Recall@K`。
3. 以 `ΔMem` 对 `ΔQPS/Δ延迟/ΔRecall` 作“单位内存收益”排序，挑拣前 2–3 个性价比最高的组合。
4. 典型优先级：
   - 高维数据（如 GIST）：优先 `PQ_disk_bytes`→适量 `num_nodes_to_cache`→调 `W`；
   - 低维数据（如 SIFT）：`num_nodes_to_cache` 的收益更早体现，`PQ_inmem_bytes` 可小步试探。

---
以上样例以可复算的公式为主，具体数值会因构建参数（尤其是 PQ 字节数与是否 OPQ）、缓存生成方式以及哈希表/分配器的元数据开销而有差异。建议在你的环境中按 6.6 的方法做一次小网格跑分，得到“自己机器/磁盘”的最优取舍曲线。 

## 七、常见问答（Q&A）

### Q1. 所有 PQ 向量都会在内存？磁盘是否存原始向量？启用 SSD-PQ 后为何内存仍有 PQ？邻居以什么格式存？
- **内存侧 PQ 一定常驻**：加载 `*_pq_compressed.bin` 到内存（`PQFlashIndex::data`），用于搜索时批量 PQ 查表，驱动 beam 扩展。
- **磁盘侧向量**：`*_disk.index` 内“节点自身坐标”可为原始 float（`PQ_disk_bytes=0`）或 PQ 字节（`PQ_disk_bytes>0`）。
- **为何启用 SSD-PQ 内存仍有 PQ**：内存侧 PQ 专用于批量近似距离计算（不触盘），与磁盘侧“节点自身坐标表示”用途不同。
- **邻居列表格式**：邻接存储为“度数(uint32) + 邻居ID数组(uint32[])”，不含邻居向量。文件内与缓存 `_nhood_cache_buf` 都是该格式。

### Q2. PQ 训练时采样量与 PQ 分段（字节数）如何确定？
- **采样量**：随机下采样上限 `MAX_PQ_TRAINING_SET_SIZE=256,000`，采样率 `p_val = 256000 / N`（超过 1 则截断为 1）。
- **内存侧 PQ 分段数（字节数）**：默认由 DRAM 预算推导 `num_pq_chunks = floor(final_index_ram_limit / N)`，再裁剪到 `[1, dim, MAX_PQ_CHUNKS]`；可用 QD 参数覆盖。
- **磁盘侧 PQ 分段数（字节数）**：由 `--PQ_disk_bytes` 明确指定，独立于内存侧 PQ。
- **分段方式**：按维度连续切分成 `num_pq_chunks` 段；每段各自做 256 聚类（OPQ 时先旋转）。

### Q3. 查询不启用 `--use_reorder_data` 时，最终距离是 PQ 还是全精度？
- **扩展阶段**：一律用“内存侧 PQ 查表”估计邻居距离。
- **最终结果距离**：取决于磁盘“节点自身坐标”的存储形式：
  - SSD 未压缩：返回全精度距离（float 坐标 vs 查询）。
  - SSD-PQ：返回基于“磁盘侧 PQ”的近似距离。
- 启用重排时，会对 Top(k×3) 追加一次全精度向量读取并重排。

### Q4. `PQ_disk_bytes` 是什么含义？如何设置？
- **定义**：磁盘侧“每个向量的 PQ 字节数”＝“PQ 分段数”（每段 1 字节，256 码字）。
- **取值权衡**：更大→磁盘占用↑、磁盘距离更精、通常召回↑/IO↓；更小反之。常见 32/64/96，需结合维度、SSD 空间与召回目标重建评估。

### Q5. 启用 SSD-PQ 后，磁盘 PQ 与内存 PQ 的区别？
- **用途**：内存 PQ 用于批量近似（驱动扩展）；磁盘 PQ 表示节点自身坐标（评估该节点与查询的距离，不开重排时直接作为结果）。
- **码本与配置**：两套独立码本；内存侧 PQ 字节数由预算/QD 决定（可 OPQ），磁盘侧由 `PQ_disk_bytes` 决定（当前实现不支持 OPQ）。
- **字节数可不同**：`_n_chunks` 与 `_disk_pq_n_chunks` 互不影响，可分别调优。

### Q6. “对邻居ID批量做 PQ 查表距离”具体是什么？
- **步骤**：
  - 预计算查询到每个码字的距离表（大小约 `256 × NCHUNKS`）。
  - 将一批邻居ID对应的“内存侧 PQ 码”聚合到临时缓冲；
  - 对这批 PQ 码逐 chunk 查表累加，得到近似距离数组；
  - 将未访问的邻居按距离入堆，继续 beam 扩展。
- **目的**：批量、纯内存、零触盘地评估邻居距离，避免为每个邻居读取磁盘向量。

### Q7. beam 扩展到底是什么？
- **直觉**：每一轮只展开堆里“最近的前 W 个候选点”（`W=beamwidth`），把它们的邻居批量算近似距离入堆，重复至收敛或达上限。
- **取舍**：W 大→轮次少、延迟低但 IO 次数可能↑；W 小→轮次多、延迟可能↑。工具支持自动调 W 寻优。 

### Q8. 内存侧 PQ 段数如何设置？为何 128D 是 3 段而 960D 是 320 段？和机器 76G 内存有关系吗？
- **不看机器物理内存**：内存侧 PQ 段数与“构建时的搜索内存预算（`--search_DRAM_budget`，记作 B）”相关，而非机器物理内存大小。
- **推导公式**（单位：字节/向量）：
  - `num_pq_chunks = clamp( floor(final_index_ram_limit / N), 1, min(dim, MAX_PQ_CHUNKS) )`
  - 其中 `N` 为点数，`dim` 为维度，`MAX_PQ_CHUNKS=512`。
- **为何 128D→3 段、960D→320 段**：说明当时 `final_index_ram_limit / N` 分别约等于 3 与 320，经裁剪后得到相应段数（且均未超过维度与 512 的上限）。
- **如何控制**：
  - 调整 `--search_DRAM_budget=B`，使 `B/N ≈ 目标段数（字节/向量）`。
    - 例：`N=1,000,000`，想要 `320` 段 → 令 `B≈320 MB`（因为 `320 B/vec × 1e6 = 320 MB`）。
  - 或者用 `QD`（Quantized Dimension）参数显式覆盖段数（优先级高于预算推导）。
- **注意**：`--use_opq` 只影响码本与精度，不改变段数；段数仍受 `min(dim, 512)` 约束。 