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
STREAM_KMEANS_BIN="${STREAM_KMEANS_BIN:-$REPO_ROOT/build/demo/stream_kmeans}"

# 构建产物输出目录
OUTPUT_DIR="${OUTPUT_DIR:-/data/1/demo/gist}"

# 基础数据（base.fbin）路径（请设置为实际文件）
BASE_FBIN="${BASE_FBIN:-/data/dataset/gist/gist_base.fbin}"    # 例：/data/sift_base.fbin

# 构建参数（可按需调整）
USE_BQ="${USE_BQ:-1}"                 # 1 启用 BQ 模式；0 构 raw 的 bucket_vamana.index
BQ_BITS="${BQ_BITS:-4}"                # 量化比特总数
BQ_GRAPH_THRESHOLD="${BQ_GRAPH_THRESHOLD:-16000}"  # >= 阈值构 bqgraph，否则 bq.bin
BQ_SATURATE_PASS="${BQ_SATURATE_PASS:-1}"        # 轻量互连/饱和微修复（度不增）

# 新增：构图量化模式（仅在 USE_BQ=1 时生效）
# 可选值：bq (使用BQ向量构图), sq (使用SQ向量构图), no_quantization (使用全精度向量构图)
# 注意：无论使用哪种模式，最终保存的图都是 BQ 压缩格式
CONSTRUCT_QUANTIZATION="${CONSTRUCT_QUANTIZATION:-no_quantization}"
MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-}"

# 新增：KMeans 模式切换（dist：使用 DistKMeans；stream：使用分块流式 KMeans）
KMEANS_MODE="${KMEANS_MODE:-dist}"
# 当 KMEANS_MODE=stream 时的参数
EXTERNAL_CENTROIDS_FBIN="${EXTERNAL_CENTROIDS_FBIN:-/data/1/demo/gist/centroids.fbin}"
CHUNK_DIR="${CHUNK_DIR:-/data/dataset/deep/chunk}"              # 例如：/data/dataset/deep/chunk
BUCKET_BUILD_START="${BUCKET_BUILD_START:-0}"
BUCKET_BUILD_END="${BUCKET_BUILD_END:-1023}"
OMP_MAX_ACTIVE_LEVELS="${OMP_MAX_ACTIVE_LEVELS:-1}"
OMP_NESTED="${OMP_NESTED:-FALSE}"

KMEANS_K="${KMEANS_K:-1024}"
KMEANS_EPOCHS="${KMEANS_EPOCHS:-5}"
KMEANS_BATCH_GIB="${KMEANS_BATCH_GIB:-20.0}"          # 每批最大GiB
KMEANS_INIT_SAMPLE_GIB="${KMEANS_INIT_SAMPLE_GIB:-10.0}"  # KMeans++ reservoir 采样GiB
CENTROIDS_OUT="${CENTROIDS_OUT:-/data/1/demo/deep/centroids.fbin}"
# 并行构桶个数（可选，默认=CPU核数）
BUCKET_BUILD_PARALLELISM="${BUCKET_BUILD_PARALLELISM:-}"
# 新增：构建阶段总内存上限（GiB），用于流式分桶批量大小和并行度控制
CONSTRUCT_MAX_GB="${CONSTRUCT_MAX_GB:-50}"
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
[demo build] KMEANS_MODE          : $KMEANS_MODE
[demo build] BUCKET_BUILD_PARALLELISM: ${BUCKET_BUILD_PARALLELISM:-auto}
[demo build] CONSTRUCT_MAX_GB     : ${CONSTRUCT_MAX_GB:-unset}
[demo build] MKL_THREADING_LAYER : $MKL_THREADING_LAYER
EOF

# 仅在 stream 模式下展示流式相关参数
if [[ "$KMEANS_MODE" == "stream" ]]; then
  cat <<EOF
[demo build] EXTERNAL_CENTROIDS_FBIN: $EXTERNAL_CENTROIDS_FBIN
[demo build] CHUNK_DIR            : $CHUNK_DIR
[demo build] BUCKET_BUILD_START    : $BUCKET_BUILD_START
[demo build] BUCKET_BUILD_END      : $BUCKET_BUILD_END
EOF
fi

# 若选择流式 KMeans，先产出质心
if [[ "$KMEANS_MODE" == "stream" ]]; then
  if [[ ! -x "$STREAM_KMEANS_BIN" ]]; then
    echo "[error] STREAM_KMEANS_BIN 不存在或不可执行: $STREAM_KMEANS_BIN" >&2
    exit 1
  fi
  if [[ -z "$CHUNK_DIR" ]]; then
    echo "[error] KMEANS_MODE=stream 需要设置 CHUNK_DIR 指向 .part_*.fbin 所在目录" >&2
    exit 1
  fi
  # 优先使用已存在的质心文件
  if [[ -n "${EXTERNAL_CENTROIDS_FBIN:-}" && -f "$EXTERNAL_CENTROIDS_FBIN" ]]; then
    echo "[demo build] reuse external centroids: $EXTERNAL_CENTROIDS_FBIN" >&2
  elif [[ -f "$CENTROIDS_OUT" ]]; then
    echo "[demo build] reuse existing centroids: $CENTROIDS_OUT" >&2
    export EXTERNAL_CENTROIDS_FBIN="$CENTROIDS_OUT"
  else
    echo "[demo build] run stream_kmeans to compute centroids" >&2
    echo "[stream_kmeans] k=$KMEANS_K epochs=$KMEANS_EPOCHS batch_gib=$KMEANS_BATCH_GIB init_sample_gib=$KMEANS_INIT_SAMPLE_GIB" >&2
    "$STREAM_KMEANS_BIN" \
      --chunk_dir "$CHUNK_DIR" \
      --k "$KMEANS_K" \
      --epochs "$KMEANS_EPOCHS" \
      --batch_gib "$KMEANS_BATCH_GIB" \
      --init_sample_gib "$KMEANS_INIT_SAMPLE_GIB" \
      --out "$CENTROIDS_OUT"
    echo "[demo build] centroids written: $CENTROIDS_OUT" >&2
    # 将 centroids 交给后续 demo 使用，避免重复聚类
    export EXTERNAL_CENTROIDS_FBIN="$CENTROIDS_OUT"
  fi
fi

# 运行构建
if [[ "$KMEANS_MODE" == "stream" ]]; then
DEMO_OUTPUT_DIR="$OUTPUT_DIR" \
USE_BQ="$USE_BQ" \
BQ_BITS="$BQ_BITS" \
BQ_GRAPH_THRESHOLD="$BQ_GRAPH_THRESHOLD" \
BQ_SATURATE_PASS="$BQ_SATURATE_PASS" \
CONSTRUCT_QUANTIZATION="$CONSTRUCT_QUANTIZATION" \
BUCKET_BUILD_PARALLELISM="${BUCKET_BUILD_PARALLELISM}" \
EXTERNAL_CENTROIDS_FBIN="${EXTERNAL_CENTROIDS_FBIN}" \
KMEANS_MODE="$KMEANS_MODE" \
CHUNK_DIR="$CHUNK_DIR" \
CONSTRUCT_MAX_GB="${CONSTRUCT_MAX_GB}" \
MKL_THREADING_LAYER="$MKL_THREADING_LAYER" \
BUCKET_BUILD_START="$BUCKET_BUILD_START" \
BUCKET_BUILD_END="$BUCKET_BUILD_END" \
"$DEMO_BIN" "$BASE_FBIN"
else
DEMO_OUTPUT_DIR="$OUTPUT_DIR" \
USE_BQ="$USE_BQ" \
BQ_BITS="$BQ_BITS" \
BQ_GRAPH_THRESHOLD="$BQ_GRAPH_THRESHOLD" \
BQ_SATURATE_PASS="$BQ_SATURATE_PASS" \
CONSTRUCT_QUANTIZATION="$CONSTRUCT_QUANTIZATION" \
BUCKET_BUILD_PARALLELISM="${BUCKET_BUILD_PARALLELISM}" \
KMEANS_MODE="$KMEANS_MODE" \
CONSTRUCT_MAX_GB="${CONSTRUCT_MAX_GB}" \
MKL_THREADING_LAYER="$MKL_THREADING_LAYER" \
"$DEMO_BIN" "$BASE_FBIN"
fi

echo "[demo build] artifacts written to: $OUTPUT_DIR" 