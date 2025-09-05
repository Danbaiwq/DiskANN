#!/usr/bin/env bash
set -euo pipefail

# =====================================
# Demo 查询脚本（直接在下方“用户配置区”修改配置）
# 运行：
#   ./monitor.sh
# =====================================

# ===== 用户配置区（请按需修改） =====
# demo 可执行文件路径（默认推导到 <repo>/build/demo/demo_test）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEMO_BIN="${DEMO_BIN:-$REPO_ROOT/build/demo/demo_test}"

# 索引产物所在目录（构建脚本写入的位置）
INDEX_DIR="${INDEX_DIR:-/data/1/demo}"

# 查询数据与评测 GT 路径（请设置为实际文件）
QUERY_FBIN="${QUERY_FBIN:-/data/dataset/gist/gist_query.fbin}"            # 例：/data/sift_query.fbin
GROUNDTRUTH_IVECS="${GROUNDTRUTH_IVECS:-/data/dataset/gist/gist_query_base_gt100}"  # 例：/data/sift_query_learn_gt100
# 原始向量库（用于真距复排）
BASE_FBIN="${BASE_FBIN:-/data/dataset/gist/gist_base.fbin}"

# 查询参数（可按需调整）
BQ_GRAPH_THRESHOLD="${BQ_GRAPH_THRESHOLD:-16000}"  # >= 阈值走 bqgraph，否则 bq.bin fastscan
BQ_EF_SEARCH="${BQ_EF_SEARCH:-196}"             # bqgraph 的 ef 宽度
BQ_SEEDS="${BQ_SEEDS:-8}"                       # bqgraph 的入口点数
# 是否允许 mmap 回退（默认关闭，保持与 DiskANN fastscan
RERANK_USE_MMAP="${RERANK_USE_MMAP:-0}"

# 新增：查询参数 F 与 K
DEMO_F_PARAM="${DEMO_F_PARAM:-1}"
DEMO_K_PARAM="${DEMO_K_PARAM:-100}"

# 新增：BQ 缓存内存预算（MB）
BQ_BUCKET_CACHE_MB="${BQ_BUCKET_CACHE_MB:-256}"
BQ_GRAPH_CACHE_MB="${BQ_GRAPH_CACHE_MB:-0}"
# 新增：预热前 N 个最大桶（按桶大小排序，0 表示不预热）
BQ_PREWARM_TOP="${BQ_PREWARM_TOP:-0}"

PIPELINE_MAX_INFLIGHT="${PIPELINE_MAX_INFLIGHT:-32}"
PIPELINE_PER_QUERY_MB="${PIPELINE_PER_QUERY_MB:-128}"
PIPELINE_MEM_CAP_MB="${PIPELINE_MEM_CAP_MB:-800}"
PIPELINE_AIO_MIN_BYTES="${PIPELINE_AIO_MIN_BYTES:-8388608}"

# 新增：复排 O_DIRECT 读优化参数（仅磁盘读，不启用 mmap）
RERANK_ALIGN_BS="${RERANK_ALIGN_BS:-4096}"       # 对齐粒度（字节），建议 4096；自动兼容 >=512
RERANK_BATCH_VECS="${RERANK_BATCH_VECS:-256}"    # 每段最大向量数
RERANK_BATCH_MB="${RERANK_BATCH_MB:-32}"          # 每段最大读取字节数（MB）
RERANK_GAP_GIDS="${RERANK_GAP_GIDS:-16}"          # 合并段允许的 gid 间隙（行数）

# 新增：并行与异步 I/O 参数（任务 4 & 任务 5）
RERANK_O_DIRECT="${RERANK_O_DIRECT:-1}"
RERANK_STREAMING="${RERANK_STREAMING:-1}"
RERANK_IO_THREADS="${RERANK_IO_THREADS:-0}"       # >1 启用线程池并行 pread（如 4/8）和libaio的线程池
RERANK_IO_URING="${RERANK_IO_URING:-0}"           # 1 启用 io_uring 异步 I/O（需 liburing）
RERANK_URING_DEPTH="${RERANK_URING_DEPTH:-128}"    # io_uring 队列深度
# 新增：libaio 异步 I/O（兼容旧内核）
RERANK_LIBAIO="${RERANK_LIBAIO:-0}"               # 1 启用 libaio（需 libaio）
RERANK_AIO_DEPTH="${RERANK_AIO_DEPTH:-128}"        # libaio 队列深度

# 新增：提示构建时使用的模式（虽然搜索时不需要，但帮助用户了解）
CONSTRUCT_QUANTIZATION="${CONSTRUCT_QUANTIZATION:-bq}"  # 仅用于展示
# 记录统计信息
CLUSTER_STATS_ENABLE=${CLUSTER_STATS_ENABLE:-1}
# 新增：禁用召回评估
DISABLE_RECALL_EVAL=${DISABLE_RECALL_EVAL:-1}
# =====================================

# 基本校验
if [[ ! -x "$DEMO_BIN" ]]; then
  echo "[error] DEMO_BIN 不存在或不可执行: $DEMO_BIN" >&2
  exit 1
fi
if [[ -z "$QUERY_FBIN" || -z "$GROUNDTRUTH_IVECS" ]]; then
  echo "[error] 请在脚本顶部配置 QUERY_FBIN 与 GROUNDTRUTH_IVECS" >&2
  exit 1
fi
if [[ ! -d "$INDEX_DIR" ]]; then
  echo "[warn] INDEX_DIR 不存在: $INDEX_DIR （请确认构建产物目录）" >&2
fi

# 展示配置
cat <<EOF
[demo search] binary             : $DEMO_BIN
[demo search] artifacts input    : $INDEX_DIR
[demo search] query.fbin         : $QUERY_FBIN
[demo search] groundtruth.ivecs  : $GROUNDTRUTH_IVECS
[demo search] base.fbin          : $BASE_FBIN
[demo search] RERANK_USE_MMAP    : $RERANK_USE_MMAP
[demo search] BQ_GRAPH_THRESHOLD : $BQ_GRAPH_THRESHOLD
[demo search] BQ_EF_SEARCH       : $BQ_EF_SEARCH
[demo search] BQ_SEEDS           : $BQ_SEEDS
[demo search] BQ_BUCKET_CACHE_MB : $BQ_BUCKET_CACHE_MB
[demo search] BQ_GRAPH_CACHE_MB  : $BQ_GRAPH_CACHE_MB
[demo search] BQ_PREWARM_TOP     : $BQ_PREWARM_TOP
[demo search] RERANK_ALIGN_BS    : $RERANK_ALIGN_BS
[demo search] RERANK_BATCH_VECS  : $RERANK_BATCH_VECS
[demo search] RERANK_BATCH_MB    : $RERANK_BATCH_MB
[demo search] RERANK_GAP_GIDS    : $RERANK_GAP_GIDS
[demo search] RERANK_IO_THREADS  : $RERANK_IO_THREADS
[demo search] RERANK_STREAMING   : $RERANK_STREAMING
[demo search] RERANK_IO_URING    : $RERANK_IO_URING
[demo search] RERANK_URING_DEPTH : $RERANK_URING_DEPTH
[demo search] RERANK_LIBAIO      : $RERANK_LIBAIO
[demo search] RERANK_AIO_DEPTH   : $RERANK_AIO_DEPTH
[demo search] DEMO_F_PARAM       : $DEMO_F_PARAM
[demo search] DEMO_K_PARAM       : $DEMO_K_PARAM
[demo search] CLUSTER_STATS_ENABLE : $CLUSTER_STATS_ENABLE
[demo search] DISABLE_RECALL_EVAL : $DISABLE_RECALL_EVAL
[demo search] PIPELINE_MAX_INFLIGHT : $PIPELINE_MAX_INFLIGHT
[demo search] PIPELINE_PER_QUERY_MB : $PIPELINE_PER_QUERY_MB
[demo search] PIPELINE_MEM_CAP_MB : $PIPELINE_MEM_CAP_MB
[demo search] PIPELINE_AIO_MIN_BYTES : $PIPELINE_AIO_MIN_BYTES
EOF

# 确保绘图不阻塞（monitor_process.py 内已支持 PLOT_SHOW 环境开关）
export PLOT_SHOW=${PLOT_SHOW:-0}

# 启动被监控进程（后台），随后对其 PID 做固定时长监控
DEMO_INPUT_DIR="$INDEX_DIR" \
BQ_GRAPH_THRESHOLD="$BQ_GRAPH_THRESHOLD" \
BQ_EF_SEARCH="$BQ_EF_SEARCH" \
BQ_SEEDS="$BQ_SEEDS" \
BASE_FBIN="$BASE_FBIN" \
RERANK_O_DIRECT="$RERANK_O_DIRECT" \
RERANK_USE_MMAP="$RERANK_USE_MMAP" \
BQ_BUCKET_CACHE_MB="$BQ_BUCKET_CACHE_MB" \
BQ_GRAPH_CACHE_MB="$BQ_GRAPH_CACHE_MB" \
BQ_PREWARM_TOP="$BQ_PREWARM_TOP" \
RERANK_ALIGN_BS="$RERANK_ALIGN_BS" \
RERANK_BATCH_VECS="$RERANK_BATCH_VECS" \
RERANK_BATCH_MB="$RERANK_BATCH_MB" \
RERANK_GAP_GIDS="$RERANK_GAP_GIDS" \
RERANK_IO_THREADS="$RERANK_IO_THREADS" \
RERANK_STREAMING="$RERANK_STREAMING" \
RERANK_IO_URING="$RERANK_IO_URING" \
RERANK_URING_DEPTH="$RERANK_URING_DEPTH" \
RERANK_LIBAIO="$RERANK_LIBAIO" \
RERANK_AIO_DEPTH="$RERANK_AIO_DEPTH" \
DEMO_F_PARAM="$DEMO_F_PARAM" \
DEMO_K_PARAM="$DEMO_K_PARAM" \
CLUSTER_STATS_ENABLE="$CLUSTER_STATS_ENABLE" \
DISABLE_RECALL_EVAL="$DISABLE_RECALL_EVAL" \
PIPELINE_MAX_INFLIGHT="$PIPELINE_MAX_INFLIGHT" \
PIPELINE_PER_QUERY_MB="$PIPELINE_PER_QUERY_MB" \
PIPELINE_MEM_CAP_MB="$PIPELINE_MEM_CAP_MB" \
PIPELINE_AIO_MIN_BYTES="$PIPELINE_AIO_MIN_BYTES" \
"$DEMO_BIN" "$QUERY_FBIN" "$GROUNDTRUTH_IVECS" &

PID=$!

echo "✅ 启动 demo_test，PID: $PID"

# 监控 600 秒（可通过 DURATION 与 INTERVAL 自定义）
DURATION=${DURATION:-600}
INTERVAL=${INTERVAL:-0.1}
python "$REPO_ROOT/perf/monitor_process.py" $PID -d "$DURATION" -i "$INTERVAL" -o "$SCRIPT_DIR/search_monitor.csv"

# 等待被监控进程退出（以确保 CSV/PNG 完整）
wait $PID

# 统计与分析开关（默认开启，设为0/false/off关闭）
if [[ "$CLUSTER_STATS_ENABLE" != "0" && "$CLUSTER_STATS_ENABLE" != "false" && "$CLUSTER_STATS_ENABLE" != "off" ]]; then
    # 分析簇向量分布（如果存在统计文件）
    CLUSTER_STATS_FILE="$INDEX_DIR/cluster_stats.txt"
    if [[ -f "$CLUSTER_STATS_FILE" ]]; then
        echo "🔍 分析簇向量数量分布..."
        python "$SCRIPT_DIR/analyze_cluster_stats.py" "$CLUSTER_STATS_FILE" -o "$SCRIPT_DIR"
        echo "📊 簇分布分析完成，结果保存在 $SCRIPT_DIR/"
    else
        echo "⚠️  未找到簇统计文件 $CLUSTER_STATS_FILE，跳过分布分析"
    fi

    # 分析簬访问统计（如果存在统计文件）
    CLUSTER_ACCESS_FILE="$SCRIPT_DIR/cluster_access_stats.txt"
    if [[ -f "$CLUSTER_ACCESS_FILE" ]]; then
        echo "🔍 分析簇访问统计..."
        python "$SCRIPT_DIR/analyze_cluster_access.py" "$CLUSTER_ACCESS_FILE" -c "$CLUSTER_STATS_FILE" -o "$SCRIPT_DIR"
        echo "📊 簇访问分析完成，结果保存在 $SCRIPT_DIR/"
    else
        echo "⚠️  未找到簇访问统计文件 $CLUSTER_ACCESS_FILE，跳过访问分析"
    fi
else
    echo "ℹ️  CLUSTER_STATS_ENABLE=0，跳过簇统计与访问分析"
fi


