#!/usr/bin/env bash
# 将fvecs数据集转为fbin，并计算gt
set -euo pipefail

# 可选：覆盖数据根目录，默认 /data/dataset
DATA_ROOT=${DATA_ROOT:-/data/dataset}
OUT_GT_DIR=${OUT_GT_DIR:-./data}

# 数据集与规模
DATASETS=(sift gist)
SCALES=(1K 10K 50K 100K 250K 500K 1M)

# 可执行
FVEC2BIN=./apps/utils/fvecs_to_bin
GT=./apps/utils/compute_groundtruth

mkdir -p "${OUT_GT_DIR}"

convert_fvecs_to_fbin() {
  local ds=$1
  local scale=$2
  local ds_dir="${DATA_ROOT}/${ds}/${ds}_test"
  local src_fvecs="${ds_dir}/${ds}_${scale}.fvecs"
  local dst_fbin="${ds_dir}/${ds}_${scale}.fbin"
  echo "[convert] ${src_fvecs} -> ${dst_fbin}"
  ${FVEC2BIN} float "${src_fvecs}" "${dst_fbin}"
}

compute_gt() {
  local ds=$1
  local scale=$2
  local ds_dir="${DATA_ROOT}/${ds}/${ds}_test"
  local base_fbin="${ds_dir}/${ds}_${scale}.fbin"
  local query_fbin="${ds_dir}/${ds}_query.fbin"
  local gt_out="${OUT_GT_DIR}/${ds}_query_${scale}_gt10"
  echo "[gt] base=${base_fbin} query=${query_fbin} -> ${gt_out}"
  ${GT} --data_type float --dist_fn l2 --base_file "${base_fbin}" --query_file "${query_fbin}" --gt_file "${gt_out}" --K 10
}

main() {
  for ds in "${DATASETS[@]}"; do
    for sc in "${SCALES[@]}"; do
      convert_fvecs_to_fbin "${ds}" "${sc}"
    done
  done

  for ds in "${DATASETS[@]}"; do
    for sc in "${SCALES[@]}"; do
      compute_gt "${ds}" "${sc}"
    done
  done
}

main "$@" 