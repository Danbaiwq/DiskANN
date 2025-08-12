#!/usr/bin/env bash
set -euo pipefail

# Usage example:
#   DATA_TYPE=float DIST_FN=l2 INDEX_PREFIX=indexes/gist_disk QUERY=data/gist_query.fbin \
#   RESULT=out/gist_res K=10 L_LIST="100 150 200" W=4 THREADS=32 NODES_TO_CACHE=200000 \
#   USE_REORDER=1 IO_LIMIT=4294967295 ./scripts/search.sh

DATA_TYPE=${DATA_TYPE:-float}         # float|int8|uint8
DIST_FN=${DIST_FN:-l2}                # l2|mips|cosine
INDEX_PREFIX=${INDEX_PREFIX:-}
QUERY=${QUERY:-}
RESULT=${RESULT:-results/out}
K=${K:-10}
L_LIST_STR=${L_LIST:-"100 150"}
W=${W:-2}
THREADS=${THREADS:-$(nproc)}
NODES_TO_CACHE=${NODES_TO_CACHE:-0}
IO_LIMIT=${IO_LIMIT:-4294967295}
GT_FILE=${GT_FILE:-null}
USE_REORDER=${USE_REORDER:-0}
FILTER_LABEL=${FILTER_LABEL:-}
QUERY_FILTERS_FILE=${QUERY_FILTERS_FILE:-}
LABEL_TYPE=${LABEL_TYPE:-uint}

if [[ -z "$INDEX_PREFIX" || -z "$QUERY" ]]; then
  echo "[search.sh] ERROR: INDEX_PREFIX and QUERY are required." >&2
  exit 1
fi

# Normalize L list into args
read -r -a L_ARR <<< "$L_LIST_STR"
L_ARGS=( -L )
for v in "${L_ARR[@]}"; do L_ARGS+=( "$v" ); done

USE_REORDER_FLAG=""
[[ "$USE_REORDER" == "1" ]] && USE_REORDER_FLAG="--use_reorder_data"

CMD=( "build/apps/search_disk_index"
      --data_type "$DATA_TYPE" --dist_fn "$DIST_FN"
      --index_path_prefix "$INDEX_PREFIX" --result_path "$RESULT"
      --query_file "$QUERY" --gt_file "$GT_FILE"
      -K "$K" --beamwidth "$W" --num_nodes_to_cache "$NODES_TO_CACHE"
      --search_io_limit "$IO_LIMIT" --num_threads "$THREADS"
      ${L_ARGS[@]} $USE_REORDER_FLAG )

[[ -n "$FILTER_LABEL" ]] && CMD+=( --filter_label "$FILTER_LABEL" )
[[ -n "$QUERY_FILTERS_FILE" ]] && CMD+=( --query_filters_file "$QUERY_FILTERS_FILE" )
[[ -n "$LABEL_TYPE" ]] && CMD+=( --label_type "$LABEL_TYPE" )

echo "[search.sh] Running: ${CMD[*]}"
exec "${CMD[@]}" 