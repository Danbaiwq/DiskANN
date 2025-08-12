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
INDEX_DIR="${INDEX_DIR:-$REPO_ROOT/build/demo_index}"

# 查询数据与评测 GT 路径（请设置为实际文件）
QUERY_FBIN="${QUERY_FBIN:-$REPO_ROOT/build/data/sift_query.fbin}"            # 例：/data/sift_query.fbin
GROUNDTRUTH_IVECS="${GROUNDTRUTH_IVECS:-$REPO_ROOT/build/data/sift_query_base_gt100}"  # 例：/data/sift_query_learn_gt100

# 查询参数（可按需调整）
BQ_GRAPH_THRESHOLD="${BQ_GRAPH_THRESHOLD:-2000}"  # >= 阈值走 bqgraph，否则 bq.bin fastscan
BQ_EF_SEARCH="${BQ_EF_SEARCH:-128}"             # bqgraph 的 ef 宽度
BQ_SEEDS="${BQ_SEEDS:-8}"                       # bqgraph 的入口点数
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
[demo search] BQ_GRAPH_THRESHOLD : $BQ_GRAPH_THRESHOLD
[demo search] BQ_EF_SEARCH       : $BQ_EF_SEARCH
[demo search] BQ_SEEDS           : $BQ_SEEDS
EOF

# 运行（程序内部会在 BQ 模式自动禁用 Vamana LRU Cache）
DEMO_INPUT_DIR="$INDEX_DIR" \
BQ_GRAPH_THRESHOLD="$BQ_GRAPH_THRESHOLD" \
BQ_EF_SEARCH="$BQ_EF_SEARCH" \
BQ_SEEDS="$BQ_SEEDS" \
"$DEMO_BIN" "$QUERY_FBIN" "$GROUNDTRUTH_IVECS" 