# DiskANN 构图（Vamana）参数与优化建议

本文总结 DiskANN 构图的默认配置与可调优化点，结合 `apps/build_memory_index.cpp` 与 `apps/build_disk_index.cpp` 中的参数，给出在“高召回/低延迟/低内存”等目标下的实操建议。

## 1. 核心概念与流程
- DiskANN 构图基于 Vamana 思路：对每个点以候选集合大小 `Lbuild` 做近邻探索，
  在候选池上用“遮挡剪枝（occlude/prune，参数 `alpha` 与 `max_occlusion_size`）”得到最多 `R` 条出边。
- 过滤索引（带标签）会在构图时考虑标签约束（`FilteredLbuild`/`filter_threshold`）。
- 对“磁盘索引”构建，还会生成用于服务时的 PQ 码本/向量，及将图+坐标写入 `*_disk.index`。

## 2. 主要参数与默认值（内存索引）
源自 `apps/build_memory_index.cpp` 与 `src/index.cpp`：
- **R（max_degree）**: 默认 64。每点最大出度；越大图越密、召回更高但内存与构图/查询开销增大。
- **Lbuild（L）**: 默认 100。构图时每点的候选集合大小；越大构图更慢、图更好（召回更高）。
- **alpha**: 默认 1.2。遮挡剪枝参数；>1 可鼓励角度多样性，减少冗余边，常用范围 [1.1, 1.4]。
- **max_occlusion_size（maxc）**: 默认 750。遮挡判定的候选上限，影响剪枝的激进程度。
- **saturate_graph**: 默认 false。若为 true，剪枝后尝试“填满到恰好 R 条边”（在某些分布下能避免过稀）。
- **num_threads（T）**: 默认 `omp_get_num_procs()`，并行构图线程数。
- **build_PQ_bytes / use_opq（可选）**: 若 `build_PQ_bytes>0`，构图时用“PQ 距离”近似加速（OPQ 可提升 PQ 精度）。限制：不支持 Inner Product；动态索引不支持。

## 3. 主要参数与默认值（磁盘索引）
源自 `apps/build_disk_index.cpp` 与 `src/disk_utils.cpp`：
- 上述 R/Lbuild/alpha/maxc/saturate_graph/num_threads 同样适用。
- 额外关键参数：
  - **B: --search_DRAM_budget（GB）** 必选。决定“服务时内存侧 PQ 字节数/向量”（=分段数）。约为 `floor(B_bytes/N)` 并裁剪到 `[1, min(dim, 512)]`。可用 `QD` 覆盖。
  - **M: --build_DRAM_budget（GB）** 必选。约束构建阶段内存上限，用于分块处理等。
  - **PQ_disk_bytes**: 磁盘侧每向量 PQ 字节数（分段数）。0 表示 SSD 存原始 float；>0 表示 SSD 存 PQ 码。影响服务时 IO 与可选重排策略。
  - **append_reorder_data（bool）**: 仅当 `PQ_disk_bytes>0` 且数据类型为 float 时可用。将全精度向量追加到索引，用于服务时 top(k×3) 重排以提升召回。
  - **build_PQ_bytes / use_opq（可选）**: 同上，用于构图时基于 PQ 距离加速。
  - **QD**: 覆盖“内存侧 PQ 分段数”。

## 4. 常用调参策略（按目标）
### 4.1 高召回优先
- R: 80–128（常用 96 起），Lbuild: 150–250，alpha: 1.2–1.35。
- saturate_graph: true（减少欠连通风险）。
- 磁盘索引：
  - PQ_disk_bytes: 64–128（高维用更大值），并建议启用 append_reorder_data。
  - B（search_DRAM_budget）适度增大（例如 64–128 B/vec），提升“内存侧 PQ 精度”，减少误扩展。
- 代价：构图时间、索引尺寸、服务内存/延迟上升。

### 4.2 延迟/QPS 优先（兼顾召回）
- R: 48–64，Lbuild: 80–120，alpha: 1.15–1.25。
- saturate_graph: false（减少过度连边）。
- 构图提速：`build_PQ_bytes=32~96`（可 `use_opq`），维持可接受质量。
- 磁盘索引：
  - PQ_disk_bytes: 32–96；配合 `--use_reorder_data` 在 top(k×3) 重排，兼顾召回。
  - 合理设置 B：例如 32–96 B/vec，节省内存同时维持较好 PQ 近似。

### 4.3 内存占用优先（构图和服务侧）
- 降低 R 到 32–48，Lbuild 到 60–100，alpha 维持 1.2 左右。
- 构图时启用 `build_PQ_bytes=32~64`（可 `use_opq`），加速且降低内存峰值。
- 磁盘索引：
  - PQ_disk_bytes>0，结合 `--use_reorder_data` 保召回。
  - B（search_DRAM_budget）调小，使“内存侧 PQ 字节/向量”下降。

### 4.4 过滤索引（带标签）
- 设置 `label_file`、`universal_label`；构图时使用 `FilteredLbuild`（独立于 Lbuild）提升过滤子图质量。
- `filter_threshold F`（在磁盘构建里）：控制按标签拆分节点阈值，避免单节点标签过多导致查询低效。

## 5. 参数要点与陷阱
- `build_PQ_bytes`/`use_opq`：
  - 仅影响“构图阶段使用 PQ 距离近似”以提速；对最终图的邻接仍由剪枝输出决定。
  - 不支持 Inner Product；不支持动态索引（dynamic index）。
- `alpha`：
  - 过小→边冗余，过大→过度多样性易丢边。一般 1.2 起步；MIPS 可按实现要求微调。
- `saturate_graph`：
  - 默认为 false；若图过稀或候选不足，设 true 可强制补足到 R 条；但可能引入弱边，需结合评测。
- `B（search_DRAM_budget）` 与 `QD`：
  - B 决定“服务时的内存侧 PQ 字节/向量”；QD 可覆盖。请配合 N（点数）按 `B/N≈目标字节/vec` 估算。
- `PQ_disk_bytes` 与 `append_reorder_data`：
  - SSD-PQ + 重排在高维下性价比高；若 SSD 空间允许、追求极致召回，可设较大 PQ_disk_bytes 并开启重排。

## 6. 推荐起始配置（经验值）
- 低维（SIFT128）
  - 内存索引：R=64, Lbuild=100, alpha=1.2。
  - 磁盘索引：R=64, Lbuild=100, alpha=1.2, PQ_disk_bytes=32–64；B≈32–64 B/vec；如需召回↑，开重排。
- 高维（GIST960）
  - 内存索引：R=96, Lbuild=150–200, alpha=1.25。
  - 磁盘索引：R=96, Lbuild=150–200, alpha=1.25, PQ_disk_bytes=96–128；B≈64–128 B/vec；建议开重排。

## 7. 示例命令
- 内存索引：
```bash
build/apps/build_memory_index \
  --data_type float --dist_fn l2 \
  --data_path data/sift1m_base.fbin \
  --index_path_prefix indexes/sift_mem \
  -R 64 -L 100 --alpha 1.2 -T 32 \
  --build_PQ_bytes 0 --use_opq
```
- 磁盘索引：
```bash
build/apps/build_disk_index \
  --data_type float --dist_fn l2 \
  --data_path data/gist1m_base.fbin \
  --index_path_prefix indexes/gist_disk \
  -R 96 -L 180 -T 32 \
  --search_DRAM_budget 8   \
  --build_DRAM_budget 32   \
  --PQ_disk_bytes 96       \
  --append_reorder_data    \
  --build_PQ_bytes 64 --use_opq
```

## 8. 校验与评测
- 评估时记录：索引大小、构图时间、服务内存、QPS、Mean/99.9 延迟、Recall@K、Mean IOs。
- 小网格搜索：
  - R ∈ {48,64,96}，Lbuild ∈ {80,120,180}，alpha ∈ {1.15,1.2,1.3}；
  - 磁盘：PQ_disk_bytes ∈ {32,64,96,128}，B ∈ {32,64,96}（B/vec，单位字节）；
  - 记录“单位内存/单位时间的召回收益”，选择最优点。

---
备注：不同数据分布下最优组合不同；建议配合 `--use_reorder_data` 与 `beamwidth` 调优一起看服务侧的延迟/召回曲线。 