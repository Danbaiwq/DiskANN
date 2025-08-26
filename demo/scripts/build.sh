#!/usr/bin/env bash
set -euo pipefail

# =====================================
# Demo 构建脚本（直接在下方"用户配置区"修改配置）
# 运行：
#   ./build.sh
# =====================================

# ===== 用户配置区（请按需修改） =====
# demo 可执行文件路径（默认推导到 <repo>/build/demo/demo_test）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEMO_BIN="${DEMO_BIN:-$REPO_ROOT/build/demo/demo_test}"

# 构建产物输出目录
OUTPUT_DIR="${OUTPUT_DIR:-/data/1/demo}"

# 基础数据（base.fbin）路径（请设置为实际文件）
BASE_FBIN="${BASE_FBIN:-$REPO_ROOT/build/data/gist_base.fbin}"    # 例：/data/sift_base.fbin

# 构建参数（可按需调整）
USE_BQ="${USE_BQ:-1}"                 # 1 启用 BQ 模式；0 构 raw 的 bucket_vamana.index
BQ_BITS="${BQ_BITS:-4}"                # 量化比特总数
BQ_GRAPH_THRESHOLD="${BQ_GRAPH_THRESHOLD:-16000}"  # >= 阈值构 bqgraph，否则 bq.bin
BQ_SATURATE_PASS="${BQ_SATURATE_PASS:-1}"        # 轻量互连/饱和微修复（度不增）

# 新增：构图量化模式（仅在 USE_BQ=1 时生效）
# 可选值：bq (使用BQ向量构图), sq (使用SQ向量构图), no_quantization (使用全精度向量构图)
# 注意：无论使用哪种模式，最终保存的图都是 BQ 压缩格式
CONSTRUCT_QUANTIZATION="${CONSTRUCT_QUANTIZATION:-bq}"
# =====================================

# 基本校验
if [[ ! -x "$DEMO_BIN" ]]; then
  echo "[error] DEMO_BIN 不存在或不可执行: $DEMO_BIN" >&2
  exit 1
fi
if [[ -z "$BASE_FBIN" ]]; then
  echo "[error] 请在脚本顶部配置 BASE_FBIN" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# 展示配置
cat <<EOF
[demo build] binary               : $DEMO_BIN
[demo build] base.fbin            : $BASE_FBIN
[demo build] output dir           : $OUTPUT_DIR
[demo build] USE_BQ               : $USE_BQ
[demo build] BQ_BITS              : $BQ_BITS
[demo build] BQ_GRAPH_THRESHOLD   : $BQ_GRAPH_THRESHOLD
[demo build] BQ_SATURATE_PASS     : $BQ_SATURATE_PASS
[demo build] CONSTRUCT_QUANTIZATION: $CONSTRUCT_QUANTIZATION
EOF

# 运行构建
DEMO_OUTPUT_DIR="$OUTPUT_DIR" \
USE_BQ="$USE_BQ" \
BQ_BITS="$BQ_BITS" \
BQ_GRAPH_THRESHOLD="$BQ_GRAPH_THRESHOLD" \
BQ_SATURATE_PASS="$BQ_SATURATE_PASS" \
CONSTRUCT_QUANTIZATION="$CONSTRUCT_QUANTIZATION" \
"$DEMO_BIN" "$BASE_FBIN"

echo "[demo build] artifacts written to: $OUTPUT_DIR" 