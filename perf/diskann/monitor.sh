#!/bin/bash

# 启动目标程序（在后台）
../../build/apps/search_disk_index \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix /data/1/perf/diskann/disk_index_gist_base_R32_L50_A1.2 \
  --query_file ../../build/data/gist_query.fbin \
  --gt_file ../../build/data/gist_query_base_gt100 \
  -K 10 \
  -L 200 400 600 800 1000 1200 1400\
  --result_path ../../build/data/res \
  --num_nodes_to_cache 100000 \
  -T 10 &

# 获取刚启动的子进程 PID
PID=$!

echo "✅ 启动程序: ../../build/apps/search_disk_index"
echo "   PID: $PID"

# 使用 Python 监控脚本监控该 PID（假设监控 10 分钟足够）
python monitor_process.py $PID -d 600 -i 0.1 -o diskann/search_monitor.csv

# 等待进程结束（可选）
wait $PID
