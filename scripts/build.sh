#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   DATA_TYPE=float DIST_FN=l2 INDEX_PREFIX=out/sift DISK=1 DATA_PATH=data/sift_base.fbin R=64 L=100 B=8 M=32 PQ_DISK_BYTES=64 APPEND_REORDER=1 ./scripts/build.sh
#   # in-memory index
#   DATA_TYPE=float DIST_FN=l2 INDEX_PREFIX=out/sift_mem DATA_PATH=data/sift_base.fbin R=64 L=100 BUILD_PQ_BYTES=0 USE_OPQ=0 DISK=0 ./scripts/build.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Required-like params (with defaults)
DATA_TYPE=${DATA_TYPE:-float}           # float|int8|uint8
DIST_FN=${DIST_FN:-l2}                  # l2|mips|cosine
INDEX_PREFIX=${INDEX_PREFIX:-/data/1/diskann}
DATA_PATH=${DATA_PATH:-$REPO_ROOT/build/data/sift_base.fbin}

# Common build params
R=${R:-32}
L=${L:-50}
NUM_THREADS=${NUM_THREADS:-$(nproc)}
ALPHA=${ALPHA:-1.2}

# Memory build extras
BUILD_PQ_BYTES=${BUILD_PQ_BYTES:-0}     # >0 to enable PQ distance build
USE_OPQ=${USE_OPQ:-0}

# Disk build extras
B=${B:-0.003}                               # search_DRAM_budget in GB
M=${M:-32}                              # build_DRAM_budget in GB
PQ_DISK_BYTES=${PQ_DISK_BYTES:-0}       # 0=no SSD PQ; >0=SSD PQ bytes/vec
APPEND_REORDER=${APPEND_REORDER:-0}     # 1=true (requires PQ_DISK_BYTES>0 and float)
QD=${QD:-0}                             # override in-memory PQ bytes/vec
CODEBOOK_PREFIX=${CODEBOOK_PREFIX:-}
LABEL_FILE=${LABEL_FILE:-}
UNIVERSAL_LABEL=${UNIVERSAL_LABEL:-}
FILTERED_LBUILD=${FILTERED_LBUILD:-0}
FILTER_THRESHOLD=${FILTER_THRESHOLD:-0}
LABEL_TYPE=${LABEL_TYPE:-uint}

# Select builder: DISK=1 for build_disk_index, else build_memory_index
DISK=${DISK:-1}

if [[ -z "$DATA_PATH" ]]; then
  echo "[build.sh] ERROR: DATA_PATH is required." >&2
  exit 1
fi

if [[ "$DISK" == "1" ]]; then
  # build_disk_index
  APPEND_REORDER_FLAG=""
  [[ "$APPEND_REORDER" == "1" ]] && APPEND_REORDER_FLAG="--append_reorder_data"
  USE_OPQ_FLAG=""
  [[ "$USE_OPQ" == "1" ]] && USE_OPQ_FLAG="--use_opq"

  CMD=( "${REPO_ROOT}/build/apps/build_disk_index"
        --data_type "$DATA_TYPE" --dist_fn "$DIST_FN"
        --index_path_prefix "$INDEX_PREFIX" --data_path "$DATA_PATH"
        --search_DRAM_budget "$B" --build_DRAM_budget "$M"
        -T "$NUM_THREADS" -R "$R" -L "$L"
        # --PQ_disk_bytes "$PQ_DISK_BYTES"
        # --build_PQ_bytes "$BUILD_PQ_BYTES"
        $USE_OPQ_FLAG $APPEND_REORDER_FLAG )
  [[ -n "$CODEBOOK_PREFIX" ]] && CMD+=( --codebook_prefix "$CODEBOOK_PREFIX" )
  [[ -n "$LABEL_FILE" ]] && CMD+=( --label_file "$LABEL_FILE" )
  [[ -n "$UNIVERSAL_LABEL" ]] && CMD+=( --universal_label "$UNIVERSAL_LABEL" )
  [[ "$FILTERED_LBUILD" != "0" ]] && CMD+=( --FilteredLbuild "$FILTERED_LBUILD" )
  [[ "$FILTER_THRESHOLD" != "0" ]] && CMD+=( --filter_threshold "$FILTER_THRESHOLD" )
#   [[ -n "$LABEL_TYPE" ]] && CMD+=( --label_type "$LABEL_TYPE" )
  [[ "$QD" != "0" ]] && CMD+=( --QD "$QD" )
else
  # build_memory_index
  USE_OPQ_FLAG=""
  [[ "$USE_OPQ" == "1" ]] && USE_OPQ_FLAG="--use_opq"
  CMD=( "build/apps/build_memory_index"
        --data_type "$DATA_TYPE" --dist_fn "$DIST_FN"
        --index_path_prefix "$INDEX_PREFIX" --data_path "$DATA_PATH"
        -T "$NUM_THREADS" -R "$R" -L "$L" --alpha "$ALPHA"
        --build_PQ_bytes "$BUILD_PQ_BYTES" $USE_OPQ_FLAG )
  [[ -n "$LABEL_FILE" ]] && CMD+=( --label_file "$LABEL_FILE" )
  [[ -n "$UNIVERSAL_LABEL" ]] && CMD+=( --universal_label "$UNIVERSAL_LABEL" )
fi

echo "[build.sh] Running: ${CMD[*]}"
exec "${CMD[@]}" 