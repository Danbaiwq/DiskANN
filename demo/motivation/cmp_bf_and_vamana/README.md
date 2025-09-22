# cmp_bf_and_vamana

该目录用于对比内存 Vamana 与暴力检索（BF）的数据准备、构建与评测，并提供可视化脚本。

## Shell 脚本（.sh）

- run_prepare_and_gt.sh
  - 功能：批量将 `{sift,gist}` × `{1K,10K,50K,100K,250K,500K,1M}` 的 base `.fvecs` 转为 `.fbin`，并使用 `compute_groundtruth` 生成 K=10 的 groundtruth。
  - 依赖：`./apps/utils/fvecs_to_bin`、`./apps/utils/compute_groundtruth`
  - 环境变量：
    - `DATA_ROOT`（默认 `/data/dataset`）
    - `OUT_GT_DIR`（默认 `./data`）
  - 用法示例：
    ```bash
    DATA_ROOT=/data/dataset OUT_GT_DIR=./data \
    ./run_prepare_and_gt.sh
    ```

- run_memory_vamana.sh
  - 功能：批量构建与搜索内存 Vamana，配置固定为 `R=32, L=50, alpha=1.2`，评测 L=`10 20 30 40 50 100`。
  - 依赖：`./apps/build_memory_index`、`./apps/search_memory_index`
  - 环境变量：
    - `DATA_ROOT`（默认 `/data/dataset`）
    - `OUT_GT_DIR`（默认 `./data`，用于加载 GT 文件）
  - 用法示例：
    ```bash
    DATA_ROOT=/data/dataset OUT_GT_DIR=./data \
    ./run_memory_vamana.sh
    ```

- run_bf_via_gt_qps.sh
  - 功能：以 `compute_groundtruth` 执行 BF，并对每个 `{dataset,size}` 组合统计 wall-clock 时间，打印 QPS（Q/时间）。
  - 依赖：`./apps/utils/compute_groundtruth`、Python3（用于读取 `query.fbin` 头和计算）
  - 环境变量：
    - `DATA_ROOT`（默认 `/data/dataset`）
    - `OUT_GT_DIR`（默认 `./data`）
  - 输出格式：
    ```
    [gt-bf][done] ds=sift scale=1K time=...s QPS=...
    ```
  - 用法示例：
    ```bash
    DATA_ROOT=/data/dataset OUT_GT_DIR=./data \
    ./run_bf_via_gt_qps.sh
    ```

## Python 脚本（.py）

- split_fvecs.py
  - 功能：从 1M 的 `.fvecs` 流式切分为 `{1K,10K,50K,100K,250K,500K,1M}` 子集，避免重复读盘。
  - 参数：
    - `input`：输入 `.fvecs`（1M）
    - `-o/--out_dir`：输出目录（默认 `.`）
    - `-b/--base`：输出文件前缀（默认取输入文件名）
  - 用法示例：
    ```bash
    ./split_fvecs.py /data/dataset/sift/sift_test/sift_1M.fvecs -o /data/dataset/sift/sift_test -b sift
    ```

- plot_pic_memory_vamana.py
  - 功能：读取 `pic_memory_vamana.csv` 并绘制曲线图（SIFT/GIST 两条曲线）：
    - 横轴：Dataset size（1K..1M）
    - 纵轴：QPS ratio (Vamana/BF)，图中标注参考线 `y=1`
  - 输出：`pic_memory_vamana.png`
  - 用法示例：
    ```bash
    ./plot_pic_memory_vamana.py -i ./pic_memory_vamana.csv -o ./pic_memory_vamana.png
    ```

## 数据文件

- pic_memory_vamana.csv：示例数据，列格式为：
  - 行1：`SIFT,,,GIST,`
  - 行2：`BF,Vamana/BF,,BF,Vamana/BF`
  - 行3+：`<size>,<ratio_sift>,,<size>,<ratio_gist>`

## 约定与路径

- 数据路径默认形如：`/data/dataset/{sift,gist}/{sift,gist}_test/`。
- 可通过 `DATA_ROOT` 与 `OUT_GT_DIR` 环境变量覆盖。
- 依赖的可执行文件路径按 `./apps/...` 默认在仓库根目录下构建生成。 