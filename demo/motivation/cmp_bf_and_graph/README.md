# cmp_bf_and_graph

对比 FAISS 暴搜（IndexFlat）与 图搜（HNSW）在不同数据量、不同维度下的构建与查询耗时，并统计召回率与延迟分位数。支持可配置的 M、efConstruction、efSearch、线程数、重复次数等参数，结果输出到 CSV。

## 依赖安装

建议使用 Python 3.8+。

```bash
pip install -U numpy faiss-cpu
```

若你的环境已安装 GPU 版 FAISS，可用 `faiss-gpu` 替代。

## 运行示例

默认会将结果写入当前目录下的 `results.csv`。

```bash
nohup python ~/DiskANN/demo/motivation/cmp_bf_and_graph/run_benchmarks.py \
  --dims 96,128,384,768,1536 \
  --num-data 2500,5000,7500,10000,25000,50000,100000 \
  --nq 1000 \
  --topk 10 \
  --metric l2 \
  --hnsw-M 32 \
  --hnsw-efC 200 \
  --hnsw-efS 64,128,256,384,512,640,768 \
  --threads 32 \
  --repeats 2 \
  --sample-latency-n 200 \
  --out-csv ./results.csv
```

若使用 Inner Product（IP）相似度，建议对向量做 L2 归一化：

```bash
python run_benchmarks.py \
  --dims 128 \
  --num-data 100000,300000 \
  --nq 1000 \
  --topk 10 \
  --metric ip \
  --normalize-for-ip \
  --hnsw-M 32 \
  --hnsw-efC 200 \
  --hnsw-efS 128,256,512 \
  --threads 24 \
  --repeats 1 \
  --out-csv ./results_ip.csv
```

## 输出字段说明（CSV）

- **repeat_id**: 第几次重复运行
- **index_type**: `flat` 或 `hnsw`
- **N**: 数据集大小
- **d**: 维度
- **metric**: `l2` 或 `ip`
- **topk**: Top-K
- **nq**: 查询数量
- **threads**: FAISS OpenMP 线程数
- **hnsw_M / hnsw_efC / hnsw_efS**: HNSW 参数（Flat 为空）
- **build_s**: 构建（add）时间（秒）
- **search_s**: 批量查询时间（秒）
- **qps**: 查询吞吐（每秒查询数）
- **avg_ms_per_q**: 均值单查询耗时（毫秒）
- **p50_ms/p95_ms/p99_ms**: 逐查询延迟采样分位数（毫秒）
- **recall_at_k**: 相对 BF 的召回率（Flat 为 1.0）
- **index_bytes**: 通过序列化估算的索引占用（字节；可能为空）

## 提示与注意

- 默认数据分布为正态（`--distribution normal`），可切换为均匀分布（`--distribution uniform`）。
- 逐查询延迟分位数通过 `--sample-latency-n` 控制采样数量；设置为 0 可跳过采样以加速。
- `--threads` 会调用 `faiss.omp_set_num_threads` 调整 OpenMP 线程数。
- 参数组合会呈乘法增长，建议先在较小规模上验证，再扩大规模以避免过长运行时间。 