#!/bin/bash

# 启动目标程序（在后台）
/home/danbai.wq/DiskANN/build/apps/search_disk_index \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix /data/1/perf/diskann/disk_index_gist_base_R32_L50_A1.03 \
  --query_file /data/dataset/gist/gist_query.fbin \
  --gt_file /data/dataset/gist/gist_query_base_gt100 \
  -K 10 \
  -L 600 800 1000 1200 1400 1600 1800 2000 2200 2400 2600 2800 3000\
  --result_path /data/1/perf/diskann/res \
  --num_nodes_to_cache 1000 \
  -W 32 \
  -T 32 &

# 获取刚启动的子进程 PID
PID=$!

echo "✅ 启动程序: /home/danbai.wq/DiskANN/build/apps/search_disk_index"
echo "   PID: $PID"

# 使用 Python 监控脚本监控该 PID（假设监控 10 分钟足够）
python /home/danbai.wq/DiskANN/perf/monitor_process.py $PID -d 600 -i 0.1 -o /home/danbai.wq/DiskANN/perf/diskann/search_monitor.csv

# 等待进程结束（可选）
wait $PID
