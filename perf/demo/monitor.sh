#!/usr/bin/env bash
set -euo pipefail

# =====================================
# Demo 查询脚本（直接在下方“用户配置区”修改配置）
# 运行：
#   ./search.sh
# =====================================

# ===== 用户配置区（请按需修改） =====
# demo 可执行文件路径（默认推导到 <repo>/build/demo/demo_test）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEMO_BIN="${DEMO_BIN:-$REPO_ROOT/build/demo/demo_test}"

# 索引产物所在目录（构建脚本写入的位置）
INDEX_DIR="${INDEX_DIR:-/data/1/demo}"

# 查询数据与评测 GT 路径（请设置为实际文件）
QUERY_FBIN="${QUERY_FBIN:-$REPO_ROOT/build/data/gist_query.fbin}"            # 例：/data/sift_query.fbin
GROUNDTRUTH_IVECS="${GROUNDTRUTH_IVECS:-$REPO_ROOT/build/data/gist_query_base_gt100}"  # 例：/data/sift_query_learn_gt100
# 原始向量库（用于真距复排）
BASE_FBIN="${BASE_FBIN:-$REPO_ROOT/build/data/gist_base.fbin}"
# 复排 I/O 模式（默认开启 O_DIRECT）
RERANK_O_DIRECT="${RERANK_O_DIRECT:-1}"

# 查询参数（可按需调整）
BQ_GRAPH_THRESHOLD="${BQ_GRAPH_THRESHOLD:-8000}"  # >= 阈值走 bqgraph，否则 bq.bin fastscan
BQ_EF_SEARCH="${BQ_EF_SEARCH:-128}"             # bqgraph 的 ef 宽度
BQ_SEEDS="${BQ_SEEDS:-8}"                       # bqgraph 的入口点数
# 是否允许 mmap 回退（默认关闭，保持与 DiskANN fastscan
RERANK_USE_MMAP="${RERANK_USE_MMAP:-0}"
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
[demo search] RERANK_O_DIRECT    : $RERANK_O_DIRECT
[demo search] RERANK_USE_MMAP    : $RERANK_USE_MMAP
[demo search] BQ_GRAPH_THRESHOLD : $BQ_GRAPH_THRESHOLD
[demo search] BQ_EF_SEARCH       : $BQ_EF_SEARCH
[demo search] BQ_SEEDS           : $BQ_SEEDS
EOF

# 运行（程序内部会在 BQ 模式自动禁用 Vamana LRU Cache）
DEMO_INPUT_DIR="$INDEX_DIR" \
BQ_GRAPH_THRESHOLD="$BQ_GRAPH_THRESHOLD" \
BQ_EF_SEARCH="$BQ_EF_SEARCH" \
BQ_SEEDS="$BQ_SEEDS" \
BASE_FBIN="$BASE_FBIN" \
RERANK_O_DIRECT="$RERANK_O_DIRECT" \
RERANK_USE_MMAP="$RERANK_USE_MMAP" \
"$DEMO_BIN" "$QUERY_FBIN" "$GROUNDTRUTH_IVECS" &

# 获取刚启动的子进程 PID
PID=$!

echo "✅ 启动程序: ../demo/scripts/search.sh"
echo "   Demo 搜索过程监控"
echo "   PID: $PID"

# 使用 Python 监控脚本监控该 PID（假设监控 10 分钟足够）
python monitor_process.py $PID -d 600 -i 0.1 -o demo/search_monitor.csv

# 等待进程结束（可选）
wait $PID
