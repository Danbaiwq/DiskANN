#!/usr/bin/env bash
set -euo pipefail

# =====================================
# 测试三种构图模式的脚本
# 运行：
#   ./test_construct_modes.sh
# =====================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEMO_BIN="${DEMO_BIN:-$REPO_ROOT/build/demo/demo_test}"

# 数据路径（请根据实际情况修改）
BASE_FBIN="${BASE_FBIN:-$REPO_ROOT/build/data/gist_base.fbin}"
QUERY_FBIN="${QUERY_FBIN:-$REPO_ROOT/build/data/gist_query.fbin}"
GROUNDTRUTH_IVECS="${GROUNDTRUTH_IVECS:-$REPO_ROOT/build/data/gist_query_base_gt100}"

# 基础输出目录
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-/home/danbai.wq/DiskANN/perf/build_graph/}"

# 确保输出目录存在
mkdir -p "$BASE_OUTPUT_DIR"

# 测试参数
USE_BQ=1
BQ_BITS=4
BQ_GRAPH_THRESHOLD=8000
BQ_SATURATE_PASS=1

echo "======================================="
echo "测试三种构图模式"
echo "======================================="
echo "基础数据: $BASE_FBIN"
echo "查询数据: $QUERY_FBIN"
echo "Ground Truth: $GROUNDTRUTH_IVECS"
echo "输出目录: $BASE_OUTPUT_DIR"
echo "======================================="

# 编译项目（如果需要）
if [[ ! -x "$DEMO_BIN" ]]; then
  echo "编译项目..."
  cd "$REPO_ROOT"
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j $(nproc)
fi

# 测试三种模式
for mode in "no_quantization"; do
  echo ""
  echo "======================================="
  echo "测试模式: $mode"
  echo "======================================="
  
  OUTPUT_DIR="$BASE_OUTPUT_DIR/mode_$mode"
  mkdir -p "$OUTPUT_DIR"
  
  # 构建阶段
  echo "构建索引..."
#   DEMO_OUTPUT_DIR="$OUTPUT_DIR" \
#   USE_BQ="$USE_BQ" \
#   BQ_BITS="$BQ_BITS" \
#   BQ_GRAPH_THRESHOLD="$BQ_GRAPH_THRESHOLD" \
#   BQ_SATURATE_PASS="$BQ_SATURATE_PASS" \
#   CONSTRUCT_QUANTIZATION="$mode" \
#   "$DEMO_BIN" "$BASE_FBIN"
  
  # 搜索阶段
  echo ""
  echo "执行搜索测试..."
  DEMO_INPUT_DIR="$OUTPUT_DIR" \
  BQ_GRAPH_THRESHOLD="$BQ_GRAPH_THRESHOLD" \
  BQ_EF_SEARCH="256" \
  BQ_SEEDS="16" \
  BASE_FBIN="$BASE_FBIN" \
  RERANK_O_DIRECT="1" \
  DEMO_F_PARAM="2" \
  DEMO_K_PARAM="200" \
  "$DEMO_BIN" "$QUERY_FBIN" "$GROUNDTRUTH_IVECS" > "$OUTPUT_DIR/search_results.log" 2>&1
  
  # 提取关键性能指标
  echo ""
  echo "性能结果："
  grep -E "(QPS|Recall@|queries/second)" "$OUTPUT_DIR/search_results.log" | tail -10
  
  # 统计索引文件大小
  echo ""
  echo "索引文件大小统计："
  du -sh "$OUTPUT_DIR"/*.bin 2>/dev/null | head -5
  
  echo ""
  echo "结果保存在: $OUTPUT_DIR"
done

echo ""
echo "======================================="
echo "测试完成！结果对比："
echo "======================================="

# 生成对比报告
for mode in "bq" "sq" "no_quantization"; do
  OUTPUT_DIR="$BASE_OUTPUT_DIR/mode_$mode"
  if [[ -f "$OUTPUT_DIR/search_results.log" ]]; then
    echo ""
    echo "模式: $mode"
    echo "-------------"
    grep -E "(QPS|Recall@100)" "$OUTPUT_DIR/search_results.log" | tail -2
    echo "索引总大小: $(du -sh "$OUTPUT_DIR" | cut -f1)"
  fi
done

echo ""
echo "详细结果保存在: $BASE_OUTPUT_DIR" 