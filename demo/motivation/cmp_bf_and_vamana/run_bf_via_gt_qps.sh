#!/usr/bin/env bash
set -euo pipefail
# 使用 diskann 的 compute_groundtruth 计算 gt，并计算 qps


DATA_ROOT=${DATA_ROOT:-/data/dataset}
OUT_GT_DIR=${OUT_GT_DIR:-./data}

DATASETS=(sift gist)
SCALES=(1K 10K 50K 100K 250K 500K 1M)

GT=./apps/utils/compute_groundtruth

# 读取 fbin 头以获取查询数
read_fbin_header() {
  local fbin=$1
  python3 - "$fbin" <<'PY'
import struct,sys
p=sys.argv[1]
with open(p,'rb') as f:
    n=struct.unpack('I',f.read(4))[0]
    d=struct.unpack('I',f.read(4))[0]
print(f"{n} {d}")
PY
}

run_one() {
  local ds=$1
  local sc=$2
  local ds_dir="${DATA_ROOT}/${ds}/${ds}_test"
  local base_fbin="${ds_dir}/${ds}_${sc}.fbin"
  local query_fbin="${ds_dir}/${ds}_query.fbin"
  local gt_out="${OUT_GT_DIR}/${ds}_query_${sc}_gt10"

  mkdir -p "${OUT_GT_DIR}"

  # 读取查询数量
  local qn_dim
  qn_dim=$(read_fbin_header "${query_fbin}")
  local qn=$(echo "$qn_dim" | awk '{print $1}')

  echo "[gt-bf] ds=${ds} scale=${sc} => base=${base_fbin}, query=${query_fbin} (Q=${qn})"
  local t0 t1 dt
  t0=$(date +%s.%N)
  ${GT} --data_type float --dist_fn l2 --base_file "${base_fbin}" --query_file "${query_fbin}" --gt_file "${gt_out}" --K 10 >/dev/null 2>&1 || true
  t1=$(date +%s.%N)
  dt=$(python3 - "$t0" "$t1" <<'PY'
import sys
print(max(1e-9, float(sys.argv[2]) - float(sys.argv[1])))
PY
)
  local qps
  qps=$(python3 - "$qn" "$dt" <<'PY'
import sys
qn=float(sys.argv[1]); dt=float(sys.argv[2])
print(qn/dt)
PY
)
  echo "[gt-bf][done] ds=${ds} scale=${sc} time=${dt}s QPS=${qps}"
}

main() {
  for ds in "${DATASETS[@]}"; do
    for sc in "${SCALES[@]}"; do
      run_one "${ds}" "${sc}"
    done
  done
}

main "$@" 