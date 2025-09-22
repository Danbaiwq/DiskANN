#!/usr/bin/env bash
set -euo pipefail
# 运行不同数据量大小的 memory vamana 索引，配置参数采用文档默认值


# 可选环境变量：
# DATA_ROOT: 数据根目录（默认 /data/dataset）
# OUT_GT_DIR: GT 所在目录（默认 ./data，与前序脚本一致）
DATA_ROOT=${DATA_ROOT:-/data/dataset}
OUT_GT_DIR=${OUT_GT_DIR:-./data}

DATASETS=(gist)
SCALES=(1K 10K 50K 100K 250K 500K 1M)

BUILD=./apps/build_memory_index
SEARCH=./apps/search_memory_index

# 固定构建参数
R=32
LBUILD=50
ALPHA=1.2

# 搜索 L 列表（与用户给定顺序一致）
SEARCH_L=(40 50 100 120 140 160 180 200 400 600 800 1000 1200 1400 1600 1800 2000)

build_index() {
  local ds=$1
  local sc=$2
  local ds_dir="${DATA_ROOT}/${ds}/${ds}_test"
  local data_path="${ds_dir}/${ds}_${sc}.fbin"
  local prefix="${ds_dir}/index_${ds}_${sc}_R${R}_L${LBUILD}_A${ALPHA}"
  echo "[build] ${data_path} -> ${prefix}"
  ${BUILD} --data_type float --dist_fn l2 --data_path "${data_path}" \
           --index_path_prefix "${prefix}" -R ${R} -L ${LBUILD} --alpha ${ALPHA}
}

search_index() {
  local ds=$1
  local sc=$2
  local ds_dir="${DATA_ROOT}/${ds}/${ds}_test"
  local prefix="${ds_dir}/index_${ds}_${sc}_R${R}_L${LBUILD}_A${ALPHA}"
  local qfile="${ds_dir}/${ds}_query.fbin"
  local gtfile="${OUT_GT_DIR}/${ds}_query_${sc}_gt10"
  local res_dir="${ds_dir}/res"
  mkdir -p "${res_dir}"

  echo "[search] prefix=${prefix} query=${qfile} gt=${gtfile}"
  ${SEARCH} --data_type float --dist_fn l2 --index_path_prefix "${prefix}" \
            --query_file "${qfile}" --gt_file "${gtfile}" -K 10 -L ${SEARCH_L[@]} \
            --result_path "${res_dir}"
}

main() {
  # for ds in "${DATASETS[@]}"; do
  #   for sc in "${SCALES[@]}"; do
  #     build_index "${ds}" "${sc}"
  #   done
  # done

  for ds in "${DATASETS[@]}"; do
    for sc in "${SCALES[@]}"; do
      search_index "${ds}" "${sc}"
    done
  done
}

main "$@" 