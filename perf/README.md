# perf 工具使用说明

本目录包含两个工具：
- `monitor_perf.sh`：采集指定进程的内存 RSS 与 IO 指标并输出 CSV。
- `plot_perf.py`：读取 CSV，输出图表与统计汇总。

## 依赖与环境
- Linux（已在 3.10+ 内核上工作），需要 `/proc/<pid>/status` 与 `/proc/<pid>/io` 字段可读。
- 采集脚本需要 `root` 权限（读取 `/proc/<pid>/io`）。
- `bash`、`awk` 等常用工具。
- Python 3 以及依赖：`pandas`、`matplotlib`
  - 如未安装，执行：
    ```bash
    python3 -m pip install --user pandas matplotlib
    ```

## 目录结构
```text
perf/
  ├─ monitor_perf.sh   # 采集脚本（需 root）
  ├─ plot_perf.py      # 绘图与统计
  └─ README.md         # 本说明
```

## 采集脚本：monitor_perf.sh
- 作用：
  - 监控已存在的 PID，或启动一条命令并对其进行监控。
  - 采集项：
    - `VmRSS`（MB）
    - IO 吞吐：读/写 MB/s（由 `/proc/<pid>/io` 的累积字节差分得到）
    - IO IOPS：读/写 ops/s（由 `/proc/<pid>/io` 的 `syscr/syscw` 差分得到）

- 用法：
  ```bash
  sudo /home/danbai.wq/DiskANN/perf/monitor_perf.sh \
    -p <pid|command> \
    -o /home/danbai.wq/DiskANN/perf/run.csv \
    [-i interval_sec] \
    [--duration seconds]
  ```

- 示例（监控已有 PID 300 秒，每 1 秒采样一次）：
  ```bash
  sudo /home/danbai.wq/DiskANN/perf/monitor_perf.sh -p 12345 -o /home/danbai.wq/DiskANN/perf/run.csv -i 1 --duration 300
  ```

- 示例（启动并监控命令，0.5 秒采样一次，随进程生命周期）：
  ```bash
  sudo /home/danbai.wq/DiskANN/perf/monitor_perf.sh -p "./your_binary --arg1 --arg2" -o /home/danbai.wq/DiskANN/perf/run.csv -i 0.5
  ```

- CSV 字段说明：
  - `timestamp`：时间戳（ISO 格式）
  - `pid`：被监控进程 PID
  - `rss_mb`：当前常驻内存（MB），来自 `/proc/<pid>/status` 的 `VmRSS`
  - `read_bytes_delta` / `write_bytes_delta`：本采样间隔内的读/写字节增量
  - `read_MBps` / `write_MBps`：本采样间隔内的读/写吞吐（MB/s）
  - `cum_read_bytes` / `cum_write_bytes`：自进程启动以来累计读/写字节数
  - `syscr_delta` / `syscw_delta`：本间隔内读/写相关 syscalls 次数增量
  - `read_IOPS` / `write_IOPS`：读/写 IOPS（ops/s）
  - `cum_syscr` / `cum_syscw`：自进程启动以来累计读/写相关 syscalls 次数

- 结束条件：
  - 监控的进程退出，或达到 `--duration` 指定的时长（`--duration=0` 表示仅随进程生命周期）。

- 注意：
  - 需要 `sudo` 执行以读取 `/proc/<pid>/io`。
  - 指标基于单进程 `/proc/<pid>`，不自动汇总子进程。
  - RSS 使用 `VmRSS`（瞬时驻留内存），峰值/均值由绘图阶段从采样值计算；如需历史峰值可关注 `VmHWM`。

## 绘图与统计：plot_perf.py
- 作用：
  - 读取采集生成的 CSV，输出图表与统计汇总文件。

- 用法：
  ```bash
  python3 /home/danbai.wq/DiskANN/perf/plot_perf.py \
    -i /home/danbai.wq/DiskANN/perf/run.csv \
    -o /home/danbai.wq/DiskANN/perf/out
  ```

- 输出：
  - `out/rss_over_time.png`：RSS 随时间变化
  - `out/io_throughput.png`：读/写 MB/s 随时间变化
  - `out/io_iops.png`：读/写 IOPS 随时间变化（若 CSV 中存在 IOPS 列）
  - `out/summary.txt`：关键统计指标，包括：
    - RSS 峰值（MB）、RSS 均值（MB）
    - 平均读吞吐（MB/s）、平均写吞吐（MB/s）

## 快速开始
```bash
# 1) 采集 120 秒
sudo /home/danbai.wq/DiskANN/perf/monitor_perf.sh -p 12345 -o /home/danbai.wq/DiskANN/perf/run.csv -i 1 --duration 120

# 2) 绘图与统计
python3 /home/danbai.wq/DiskANN/perf/plot_perf.py -i /home/danbai.wq/DiskANN/perf/run.csv -o /home/danbai.wq/DiskANN/perf/out
```

## 常见问题
- 权限不足：请使用 `sudo` 执行采集脚本。
- 找不到依赖：使用 `python3 -m pip install --user pandas matplotlib` 安装依赖。
- CSV 列缺失：老旧内核或权限限制可能导致 `/proc/<pid>/io` 字段不完整；绘图脚本会尽量跳过缺失项并生成可用结果。 