# Demo: Mini-batch KMeans + Vamana 两阶段向量搜索（含异步流水线）

## 1. 概述

本 Demo 在 DiskANN 基础上，提供面向大规模向量数据的两阶段搜索方案：
- 构建模式：Mini-batch KMeans 聚类 + 桶内索引/量化产物构建
- 搜索模式：Medoid 粗筛 + 桶内候选（支持 BQ/BQGraph） + 末端真距离复排

在搜索模式下，Demo 引入了高性能异步流水线：统一 CPU 工作池 + 独立 I/O 轮询线程（libaio），并配套 segment 合并、O_DIRECT 顺序读优化、流式复排与无锁队列，实现在资源约束内的高并发、低抖动与稳定退出。

## 2. 数据与文件

- 输入数据：`<base>.fbin`（float32，头部 `uint32 n, uint32 dim`），查询 `<query>.fbin`，评测 GT `<gt>.ivecs`。
- 构建产物（位于 `DEMO_OUTPUT_DIR`）：
  - `buckets.bin`：每桶局部ID → 全局ID 映射
  - `medoid_vamana.index`：质心图（粗筛）
  - 桶内：
    - RAW 模式：`bucket_<i>_vamana.index`
    - BQ 模式：小桶 `bucket_<i>_bq.bin`；大桶 `bucket_<i>_bqgraph.bin`
  - `vector_offsets.bin`：全局ID → `<base>.fbin` 字节偏移（真距复排使用）

## 3. 构建模式（build_mode）

流程：
1) Mini-batch KMeans（并行多轮采样聚类）
2) 距离比例约束分桶（参数 β，避免过度分配）
3) 保存 `buckets.bin` 与 `medoid_vamana.index`
4) 桶内产物：
   - RAW：Vamana 图（`bucket_<i>_vamana.index`）
   - BQ：小桶 `bq.bin`（fastscan 估计距）、大桶 `bqgraph.bin`（压缩图）
5) 写出 `vector_offsets.bin`（以字节偏移形式记录 `<base>.fbin` 行位置，供复排读取）

关键参数（环境变量读取）：
- `USE_BQ`（默认 0）：是否启用 BQ 模式
- `BQ_BITS`：量化比特
- `BQ_GRAPH_THRESHOLD`：大桶阈值（≥ 阈值生成 `bqgraph`）

## 4. 搜索模式（search_mode）

读取 `medoid_meta.txt`、`buckets.bin` 与 `medoid_vamana.index` 后，执行评测。核心步骤：
1) Medoid 粗筛：在 `medoid_vamana.index` 上搜索 `f` 个桶ID。
2) 桶内候选：
   - RAW：逐桶用 Vamana 子图返回 `k` 个局部ID → 映射为全局ID
   - BQ：
     - 大桶（≥ 阈值）：在 `bqgraph` 上以 `BQ_EF_SEARCH` 近邻搜索
     - 小桶：读取 `bq.bin`，fastscan 估计选 Top-k
   - 去重合并，获得候选全局ID集合
3) 末端真距离复排（`BASE_FBIN` 必需）：
   - 将候选 gid 升序，按 `.fbin` 行顺序合并为段（segment）
   - O_DIRECT 路径：对齐批量 pread；失败回退普通 pread；必要时回退 mmap
   - 计算 SIMD L2 距离（AVX2），排序返回 Top-K

## 5. 异步流水线（核心实现）

### 5.1 结构
- 统一 CPU 工作池（数量=`PIPELINE_WORKERS`，默认等于在线 CPU 数）
- 独立 I/O 轮询线程（libaio）
- 三阶段逻辑：
  - Stage1：候选生成（RAW/BQ/BQGraph）
  - Stage2：段合并 + 异步提交（libaio / 小 I/O 同步回退）
  - Stage3：复排（支持流式：直接在段缓冲上计算并释放）

### 5.2 无锁队列与调度
- 三个阶段队列：无锁有界 MPMC 环形队列（Vyukov），容量 `PIPELINE_QUEUE_CAP`（默认 8192）
- 工作线程优先级调度：Stage3 → Stage2 → Stage1，短自旋（64 次）后 `wait_pop(2ms)` 阻塞等待，降低 condvar 与锁竞争

### 5.3 I/O 策略
- segment 合并：按 gid 升序分段，受 `RERANK_BATCH_VECS`、`RERANK_BATCH_MB`、`RERANK_GAP_GIDS` 控制
- O_DIRECT：`RERANK_O_DIRECT=1` 时走对齐大块 pread；失败自动回退普通 pread
- 小 I/O 回退：当单查询总段字节 `< PIPELINE_AIO_MIN_BYTES`（默认 8MB）时，直接同步分段 pread，避免 AIO 尾部阻塞与轮询开销
- I/O 轮询：运行阶段 `io_getevents(ctx, 1, k, ...)` 阻塞等待；空闲时 `io_getevents(ctx, 0, 1, 20ms)` 轮询；停止时 `io_cancel` 唤醒并退出

### 5.4 流式复排与内存控制
- 流式复排 `RERANK_STREAMING=1`（默认）：不分配整块 `full_precision_vectors`，在段缓冲上计算后立即 `free`
- 源头限流：
  - `PIPELINE_MAX_INFLIGHT`（默认 2×CPU）按批次投喂查询
  - `PIPELINE_PER_QUERY_MB`（默认 64MB）单查询上限，超出走同步流式
  - `PIPELINE_MEM_CAP_MB`（默认 500MB）全局上限，按预算动态计算每批并发
- 每批结束尝试 `malloc_trim(0)`（glibc）回收空闲堆

### 5.5 退出与健壮性
- 每个 QueryContext 路径均设置 `promise.set_value`（候选空、分段空、分配失败、提交失败、同步回退等）
- 收集完 futures 后：先置 `stop_flag`，再 `aio_manager.request_stop()`（内部 `io_cancel` + `inflight_=0`），关闭队列并 join 所有线程

## 6. 关键参数（运行时环境变量）

通用：
- `DEMO_INPUT_DIR`/`DEMO_OUTPUT_DIR`：输入/输出目录
- `BASE_FBIN`：复排所需 base 向量文件（必需）
- `DEMO_F_PARAM`、`DEMO_K_PARAM`：粗筛桶数 f、每桶候选 k

复排/段合并（O_DIRECT 优化）：
- `RERANK_O_DIRECT`（默认 1）
- `RERANK_ALIGN_BS`（默认 4096）
- `RERANK_BATCH_VECS`（默认 128）
- `RERANK_BATCH_MB`（默认 8）
- `RERANK_GAP_GIDS`（默认 8）

异步流水线：
- `PIPELINE_WORKERS`：工作线程数（默认=在线 CPU）
- `PIPELINE_QUEUE_CAP`：各阶段队列容量（默认 8192）
- `PIPELINE_MAX_INFLIGHT`：最大在途查询数（默认 2×CPU）
- `PIPELINE_PER_QUERY_MB`：单查询内存上限（默认 64MB）
- `PIPELINE_MEM_CAP_MB`：全局内存上限（默认 500MB）
- `PIPELINE_AIO_MIN_BYTES`：小 I/O 同步阈值（默认 8MB）
- `RERANK_STREAMING`：流式复排（默认 1）

BQ 模式：
- `USE_BQ`（默认 0）
- `BQ_BITS`
- `BQ_GRAPH_THRESHOLD`、`BQ_EF_SEARCH`、`BQ_SEEDS`
- `BQ_BUCKET_CACHE_MB`（默认 256）、`BQ_GRAPH_CACHE_MB`（默认 256）
- `BQ_PREWARM_TOP`（默认 0）

其它：
- `RERANK_LIBAIO`（默认 1）/`RERANK_AIO_DEPTH`（建议 128）
- `RERANK_IO_THREADS`、`RERANK_IO_URING`（如需回退到线程池或 io_uring）

## 7. 调优建议

- CPU 利用率不足：
  - 确认 `PIPELINE_WORKERS` ≥ 目标并发且 `std::thread::hardware_concurrency()` 未被 cgroup/cpuset 限制
  - 提升 `PIPELINE_MAX_INFLIGHT`，在 `PIPELINE_MEM_CAP_MB` 与 `PIPELINE_PER_QUERY_MB` 允许范围内增大批并发
- I/O 抖动或 AIO 轮询占比高：
  - 调大 `RERANK_BATCH_VECS`/`RERANK_BATCH_MB`，提升顺序段吞吐
  - 适当调大 `RERANK_AIO_DEPTH`（如 128）；小 I/O 由 `PIPELINE_AIO_MIN_BYTES` 控制改走同步
- 内存线性增长：
  - 启用 `RERANK_STREAMING=1`（默认），设置 `PIPELINE_PER_QUERY_MB` 与 `PIPELINE_MEM_CAP_MB`，并控制 `PIPELINE_MAX_INFLIGHT`
- 退出卡住：
  - 确保 `BASE_FBIN` 正确；若 O_DIRECT 不兼容，设 `RERANK_O_DIRECT=0`
  - 小 I/O 自动走同步路径；停止阶段 `io_cancel` 保证轮询退出

## 8. 编译与运行

```bash
# 配置与编译
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j $(nproc)

# 构建
USE_BQ=1 BQ_BITS=12 ./build/demo/demo_test /path/to/base.fbin

# 搜索（推荐使用监控脚本）
BASE_FBIN=/path/to/base.fbin \
  /home/danbai.wq/DiskANN/perf/demo/monitor.sh
```

## 9. 代码定位
- `demo/src/main.cpp`：构建/搜索入口、参数读取与评测
- `demo/src/search.cpp|.h`：候选生成、BQ/BQGraph、SIMD 距离 `l2_distance_simd`
- `demo/src/pipeline.h|.cpp`：异步流水线、无锁队列、AIO 轮询、段合并与流式复排
- `demo/src/build.cpp|.h`：构建流程、BQ 产物与 bqgraph 生成
- `perf/demo/monitor.sh`：一键运行与性能采集

## 10. 变更与一致性说明
本 README 与代码实现保持一致：
- 已纳入：无锁 MPMC 队列、分段合并、O_DIRECT 批读、流式复排、内存与在途限流、AIO 阻塞等待与小 I/O 同步回退、停止阶段取消并退出
- 重要参数均可通过环境变量覆盖，监控脚本会打印并透传，便于 A/B 比较与回归验证

## 11. 向量分块组织与流式读取（for demo streaming build/search）

为支持超大规模向量在低内存环境下的构建与搜索，提供将 `<base>.fbin` 按约 1GiB 分块的脚本 `demo/scripts/split_fbin.py`。关键说明如下：

- 命名与路径：
  - 输入：`/path/to/base.fbin`
  - 输出目录：`/path/to/chunk/`
  - 输出文件：`<basename>.part_00000.fbin`, `<basename>.part_00001.fbin`, ...（从 0 递增）
  - 可通过 `--prefix` 指定前缀，默认取输入文件名去扩展名后加 `.part`

- 每个分块文件的格式：
  - 头部：`uint32 n, uint32 dim`（小端，8B）
  - 数据：连续存放 `n * dim` 个元素（默认 `float32`，即 4B），无额外对齐填充
  - 若原始 `<base>.fbin` 存在“行对齐/填充”（每向量 stride > `dim*sizeof(T)`），分块时会自动压紧为“紧凑布局”（仅写前 `dim*sizeof(T)` 有效字节），以便后续流式处理

- 目标块大小与对齐：
  - 建议用 `--target`（GiB，默认 1.0）；实际每块向量数按“向量粒度”对齐，不跨向量拆分
  - 单次缓冲用 `--buffer`（GiB，默认 0.25）；如需精确字节，`--target-bytes`/`--buffer-bytes` 可选覆盖

- 全局 ID 与分块内局部 ID 的关系：
  - 设各分块的向量数分别为 `n_0, n_1, ..., n_k`，块 `i` 内的局部 ID 记为 `j`（`0 <= j < n_i`）
  - 若原始 `<base>.fbin` 是按“行顺序”紧凑排列，则可用前缀和计算全局 ID：
    - `global_id = j + sum_{t=0}^{i-1} n_t`
  - 对于 demo 的流式聚类 / 复排，如需从 `<base>.fbin` 回源读取，可按上述映射反推原始行号，同时结合 `dim` 与元素大小定位字节偏移

- 流式读取建议（构建/搜索）：
  - 构建（聚类）阶段：按块顺序遍历，单块内再按批（batch）读取到工作缓冲，避免一次载入整块；批大小建议控制在 64–256MiB
  - 搜索（真距复排）阶段：对候选 `global_id` 升序分段，优先合并跨块连续区间，最大化顺序读吞吐；若启用 O_DIRECT，请按对齐阈值（`RERANK_ALIGN_BS`）对齐请求
  - 若上游需要“原始 `.fbin` 的字节偏移”，请在构建阶段生成 `vector_offsets.bin`（`uint64_t` 数组），记录每行在原始 `<base>.fbin` 的字节偏移；当使用分块紧凑布局时，仍以原始偏移为准进行回源

- 使用示例：
```bash
python3 /home/danbai.wq/DiskANN/demo/scripts/split_fbin.py \
  --input /data/dataset/deep/base.1B.fbin \
  --outdir /data/dataset/deep/chunk \
  --target 1.0 \
  --buffer 0.25
```

- 兼容性：
  - `.fbin` 头部与元素类型沿用 DiskANN 约定（默认 `float32`），可通过 `--dtype` 指定 `int8/uint8`
  - 分块文件可被现有的加载与处理逻辑按普通 `.fbin` 文件读取（每块独立），适合多进程/多机并行

## 12. Stream KMeans（超大规模分块流式聚类）

当数据规模 >50GiB 时，推荐使用 Stream KMeans（读取 `.part_*.fbin` 分块作为随机采样替代）。流程与参数：

- 流程概要：
  1) KMeans++ 初始化：对分块数据执行蓄水池采样，训练得到 K 个初始质心
  2) 迭代（epoch 轮）：每轮将若干分块随机打乱、按 `--batch_gib` 聚合为批，逐批：
     - 计算每个向量到 k 个中心的距离（SIMD 并行）
     - 归属最近中心，累计该中心向量和与计数
     - 用加权移动平均更新中心（全局计数累加，防止漂移）
  3) 输出最终 K 个聚类中心为 `.fbin`

- 可执行程序：`build/demo/stream_kmeans`
  - 参数：
    - `--chunk_dir`: 分块目录（内含 `*.fbin`，头部 8B + 紧凑布局）
    - `--k`: 聚类数 K
    - `--epochs`: 迭代轮数
    - `--batch_gib`: 每批最大读取 GiB（建议 2–8）
    - `--init_sample_gib`: KMeans++ 采样 GiB（建议 1–4）
    - `--out`: 输出质心 `.fbin` 路径
  - 示例：
```bash
/home/danbai.wq/DiskANN/build/demo/stream_kmeans \
  --chunk_dir /data/dataset/deep/chunk \
  --k 1024 \
  --epochs 5 \
  --batch_gib 4.0 \
  --init_sample_gib 2.0 \
  --out /data/1/demo/gist/centroids.fbin
```

- 在 `demo/scripts/build.sh` 中启用：
  - 设置 `KMEANS_MODE=stream`，并配置：`CHUNK_DIR`、`KMEANS_K`、`KMEANS_EPOCHS`、`KMEANS_BATCH_GIB`、`KMEANS_INIT_SAMPLE_GIB`、`CENTROIDS_OUT`
  - 若保持 `KMEANS_MODE=dist`，则使用原 DistKMeans 路径

注意：分块文件需来自前述“向量分块组织与流式读取”，且要求所有分块维度一致。

### 与构建流程对接（尽量不改动原流程）

- 仅使用外部质心（不修改 build.sh）：
  - 在运行构建前设置环境变量 `EXTERNAL_CENTROIDS_FBIN` 指向质心 `.fbin`，程序会自动加载并跳过 DistKMeans；未设置则按原流程执行 DistKMeans。
```bash
EXTERNAL_CENTROIDS_FBIN=/data/1/demo/gist/centroids.fbin \
./demo/scripts/build.sh
```

- 可选：一键流式聚类 + 构建（已提供开关，不影响默认行为）：
  - 使用 `KMEANS_MODE=stream` 时，脚本会先调用 `stream_kmeans` 产出质心，再继续原构建流程。
```bash
KMEANS_MODE=stream \
CHUNK_DIR=/data/dataset/deep/chunk \
KMEANS_K=1024 KMEANS_EPOCHS=5 \
KMEANS_BATCH_GIB=4.0 KMEANS_INIT_SAMPLE_GIB=2.0 \
CENTROIDS_OUT=/data/1/demo/gist/centroids.fbin \
./demo/scripts/build.sh
```

- 说明：
  - 默认不设置 `EXTERNAL_CENTROIDS_FBIN` 与 `KMEANS_MODE` 时，构建流程保持不变。