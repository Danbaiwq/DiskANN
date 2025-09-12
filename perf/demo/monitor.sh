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
INDEX_DIR="${INDEX_DIR:-/data/1/demo/deep}"

# 查询数据与评测 GT 路径（请设置为实际文件）
QUERY_FBIN="${QUERY_FBIN:-/data/dataset/gist/gist_query.fbin}"            # 例：/data/sift_query.fbin
GROUNDTRUTH_IVECS="${GROUNDTRUTH_IVECS:-/data/dataset/gist/gist_query_base_gt100}"  # 例：/data/sift_query_learn_gt100
# 原始向量库（用于真距复排）
BASE_FBIN="${BASE_FBIN:-/data/dataset/gist/gist_base.fbin}"

# 查询参数（可按需调整）
BQ_GRAPH_THRESHOLD="${BQ_GRAPH_THRESHOLD:-16000}"  # >= 阈值走 bqgraph，否则 bq.bin fastscan
BQ_EF_SEARCH="${BQ_EF_SEARCH:-196}"             # bqgraph 的 ef 宽度
BQ_SEEDS="${BQ_SEEDS:-8}"                       # bqgraph 的入口点数

# 新增：查询参数 F 与 K
DEMO_F_PARAM="${DEMO_F_PARAM:-10}"
DEMO_K_PARAM="${DEMO_K_PARAM:-100}"

# 新增：BQ 缓存内存预算（MB）
BQ_BUCKET_CACHE_MB="${BQ_BUCKET_CACHE_MB:-400}"
BQ_GRAPH_CACHE_MB="${BQ_GRAPH_CACHE_MB:-0}"
# 新增：预热前 N 个最大桶（按桶大小排序，0 表示不预热）
BQ_PREWARM_TOP="${BQ_PREWARM_TOP:-0}"

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
[demo search] BQ_GRAPH_THRESHOLD : $BQ_GRAPH_THRESHOLD
[demo search] BQ_EF_SEARCH       : $BQ_EF_SEARCH
[demo search] BQ_SEEDS           : $BQ_SEEDS
[demo search] BQ_BUCKET_CACHE_MB : $BQ_BUCKET_CACHE_MB
[demo search] BQ_GRAPH_CACHE_MB  : $BQ_GRAPH_CACHE_MB
[demo search] BQ_PREWARM_TOP     : $BQ_PREWARM_TOP
[demo search] DEMO_F_PARAM       : $DEMO_F_PARAM
[demo search] DEMO_K_PARAM       : $DEMO_K_PARAM
EOF

# 确保绘图不阻塞（monitor_process.py 内已支持 PLOT_SHOW 环境开关）
export PLOT_SHOW=${PLOT_SHOW:-0}

# 启动被监控进程（后台），随后对其 PID 做固定时长监控
DEMO_INPUT_DIR="$INDEX_DIR" \
BQ_GRAPH_THRESHOLD="$BQ_GRAPH_THRESHOLD" \
BQ_EF_SEARCH="$BQ_EF_SEARCH" \
BQ_SEEDS="$BQ_SEEDS" \
BASE_FBIN="$BASE_FBIN" \
BQ_BUCKET_CACHE_MB="$BQ_BUCKET_CACHE_MB" \
BQ_GRAPH_CACHE_MB="$BQ_GRAPH_CACHE_MB" \
BQ_PREWARM_TOP="$BQ_PREWARM_TOP" \
DEMO_F_PARAM="$DEMO_F_PARAM" \
DEMO_K_PARAM="$DEMO_K_PARAM" \
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


