#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import psutil
import time
import argparse
import sys
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime


def find_process(pid_or_name):
    """Find process by PID or partial name"""
    try:
        pid = int(pid_or_name)
        proc = psutil.Process(pid)
        return proc
    except ValueError:
        for proc in psutil.process_iter(['pid', 'name']):
            if pid_or_name.lower() in proc.info['name'].lower():
                return proc
        return None
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def monitor_process(pid_or_name, duration, interval, log_file):
    """Monitor memory, I/O, and CPU usage of a process"""
    proc = find_process(pid_or_name)
    if not proc:
        print("Error: Cannot find process '%s'" % pid_or_name)
        sys.exit(1)

    print("✅ Starting monitoring process: %s (PID: %d)" % (proc.name(), proc.pid))
    print("   Duration: %d seconds | Sampling interval: %.1f seconds" % (duration, interval))
    print("   Log file: %s" % log_file)
    print("")

    # Print header
    # print("%-10s %-12s %-10s %-12s %-12s" % ("Time(s)", "Mem(MB)", "CPU(%)", "Read(kB/s)", "Write(kB/s)"))

    start_time = time.time()
    end_time = start_time + duration
    data = []
    max_rss_mb = 0
    total_rss_mb = 0
    max_cpu = 0
    total_cpu = 0
    total_read = 0
    total_write = 0
    sample_count = 0
    # 新增：累计“逻辑 IO”字符计数的速率
    total_read_chars = 0
    total_write_chars = 0
    # 新增：累计缺页速率
    total_minor_faults = 0
    total_major_faults = 0

    # Initial I/O and CPU
    try:
        io_start = proc.io_counters()
        read_bytes_prev = io_start.read_bytes
        write_bytes_prev = io_start.write_bytes
        # 新增：字符级计数（包含页缓存命中的读取）
        read_chars_prev = getattr(io_start, 'read_chars', 0)
        write_chars_prev = getattr(io_start, 'write_chars', 0)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        read_bytes_prev = 0
        write_bytes_prev = 0
        read_chars_prev = 0
        write_chars_prev = 0

    # 新增：初始化缺页计数（来自 /proc/<pid>/stat 的累积值）
    def read_faults(pid):
        try:
            with open(f"/proc/{pid}/stat", "r") as f:
                s = f.read()
            # comm 字段包含在括号内，先找到右括号后再 split
            rpar = s.rfind(')')
            rest = s[rpar+2:].split()
            # 字段序号参考 proc(5)：
            # 10 minflt, 11 cminflt, 12 majflt, 13 cmajflt （基于 1 起始）
            minflt = int(rest[7])   # 10th -> index 7（因为前面去掉了前 2 个字段）
            majflt = int(rest[9])   # 12th -> index 9
            return minflt, majflt
        except Exception:
            return 0, 0

    try:
        minflt_prev, majflt_prev = read_faults(proc.pid)
    except Exception:
        minflt_prev, majflt_prev = 0, 0

    # Note: We call cpu_percent() once to initialize (returns 0), then sample in loop
    try:
        proc.cpu_percent()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    while time.time() < end_time:
        try:
            current_time = time.time()

            # Memory
            mem_info = proc.memory_info()
            rss_mb = mem_info.rss / 1024 / 1024  # MB
            total_rss_mb += rss_mb
            if rss_mb > max_rss_mb:
                max_rss_mb = rss_mb

            # CPU usage since last call
            try:
                cpu_percent = proc.cpu_percent(interval=interval)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                cpu_percent = 0
            total_cpu += cpu_percent
            if cpu_percent > max_cpu:
                max_cpu = cpu_percent

            # I/O
            try:
                io_curr = proc.io_counters()
                read_bytes_curr = io_curr.read_bytes
                write_bytes_curr = io_curr.write_bytes
                # 新增：字符级计数（read_chars/write_chars）
                read_chars_curr = getattr(io_curr, 'read_chars', 0)
                write_chars_curr = getattr(io_curr, 'write_chars', 0)

                read_rate = (read_bytes_curr - read_bytes_prev) / interval / 1024  # kB/s
                write_rate = (write_bytes_curr - write_bytes_prev) / interval / 1024
                # 新增：字符级速率（更能反映页缓存命中场景）
                read_chars_rate = (read_chars_curr - read_chars_prev) / interval / 1024
                write_chars_rate = (write_chars_curr - write_chars_prev) / interval / 1024

                read_bytes_prev = read_bytes_curr
                write_bytes_prev = write_bytes_curr
                read_chars_prev = read_chars_curr
                write_chars_prev = write_chars_curr
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                read_rate = 0
                write_rate = 0
                read_chars_rate = 0
                write_chars_rate = 0

            total_read += read_rate
            total_write += write_rate
            # 累计字符级速率
            total_read_chars += read_chars_rate
            total_write_chars += write_chars_rate

            # 新增：缺页统计（每秒）
            try:
                minflt_curr, majflt_curr = read_faults(proc.pid)
                minor_faults_rate = (minflt_curr - minflt_prev) / interval
                major_faults_rate = (majflt_curr - majflt_prev) / interval
                minflt_prev = minflt_curr
                majflt_prev = majflt_curr
            except Exception:
                minor_faults_rate = 0
                major_faults_rate = 0

            total_minor_faults += minor_faults_rate
            total_major_faults += major_faults_rate

            # Record data
            timestamp = datetime.now()
            data.append({
                'time': timestamp,
                'rss_mb': round(rss_mb, 2),
                'cpu_percent': round(cpu_percent, 2),
                'read_kbs': round(max(read_rate, 0), 2),
                'write_kbs': round(max(write_rate, 0), 2),
                # 新增：字符级速率字段
                'read_chars_kbs': round(max(read_chars_rate, 0), 2),
                'write_chars_kbs': round(max(write_chars_rate, 0), 2),
                # 新增：缺页速率字段
                'minor_faults_s': round(max(minor_faults_rate, 0), 2),
                'major_faults_s': round(max(major_faults_rate, 0), 2)
            })

            # Print real-time
            # elapsed = int(current_time - start_time)
            # print("%3ds      %8.2f     %6.1f   %8.2f     %8.2f" % (
            #     elapsed, rss_mb, cpu_percent, read_rate, write_rate))

            sample_count += 1

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            print("\n❌ Process %d has exited or is no longer accessible." % proc.pid)
            break
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring interrupted by user.")
            break

    if data:
        df = pd.DataFrame(data)
        df.to_csv(log_file, index=False)
        print("\n📊 Data saved to: %s" % log_file)

        # Summary
        avg_rss_mb = total_rss_mb / sample_count if sample_count > 0 else 0
        avg_cpu = total_cpu / sample_count if sample_count > 0 else 0
        avg_read = total_read / sample_count if sample_count > 0 else 0
        avg_write = total_write / sample_count if sample_count > 0 else 0
        # 新增：字符级平均速率
        avg_read_chars = total_read_chars / sample_count if sample_count > 0 else 0
        avg_write_chars = total_write_chars / sample_count if sample_count > 0 else 0
        # 新增：缺页平均速率
        avg_minor_faults = total_minor_faults / sample_count if sample_count > 0 else 0
        avg_major_faults = total_major_faults / sample_count if sample_count > 0 else 0

        print("\n📈 Summary:")
        print("   Peak Memory Usage:     %.2f MB" % max_rss_mb)
        print("   Average Memory Usage:  %.2f MB" % avg_rss_mb)
        print("   Peak CPU Usage:        %.1f %%" % max_cpu)
        print("   Average CPU Usage:     %.1f %%" % avg_cpu)
        print("   Average Read Rate:     %.2f kB/s" % avg_read)
        print("   Average Write Rate:    %.2f kB/s" % avg_write)
        # 新增：字符级平均速率打印
        print("   Avg Read Chars Rate:   %.2f kB/s" % avg_read_chars)
        print("   Avg Write Chars Rate:  %.2f kB/s" % avg_write_chars)
        # 新增：缺页平均速率打印
        print("   Avg Minor Faults:      %.2f /s" % avg_minor_faults)
        print("   Avg Major Faults:      %.2f /s" % avg_major_faults)
        print("   Total Samples:         %d" % sample_count)

        # Plot
        plot_data(df, log_file)
    else:
        print("❌ No data collected.")


def plot_data(df, log_file):
    """Plot monitoring data: Memory, CPU, I/O"""
    plt.style.use('seaborn-v0_8')
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Subplot 1: Memory
    ax1.plot(df['time'], df['rss_mb'], color='tab:blue', marker='o', markersize=3, label='Memory Usage (MB)')
    ax1.set_ylabel('Memory (MB)')
    ax1.set_title('Process Resource Monitoring')
    ax1.legend()
    ax1.grid(True)

    # Subplot 2: CPU
    ax2.plot(df['time'], df['cpu_percent'], color='tab:orange', marker='s', markersize=3, label='CPU Usage (%)')
    ax2.set_ylabel('CPU (%)')
    ax2.legend()
    ax2.grid(True)

    # Subplot 3: I/O
    ax3.plot(df['time'], df['read_kbs'], color='tab:green', alpha=0.8, label='Disk Read (kB/s)')
    ax3.plot(df['time'], df['write_kbs'], color='tab:red', alpha=0.8, label='Disk Write (kB/s)')
    # 新增：字符级速率曲线（更能反映页缓存命中场景）
    if 'read_chars_kbs' in df.columns:
        ax3.plot(df['time'], df['read_chars_kbs'], color='tab:green', linestyle='--', alpha=0.6, label='Read Chars (kB/s)')
    if 'write_chars_kbs' in df.columns:
        ax3.plot(df['time'], df['write_chars_kbs'], color='tab:red', linestyle='--', alpha=0.6, label='Write Chars (kB/s)')
    # 新增：缺页曲线，用于识别 mmap 触发的真实磁盘读（major faults）
    if 'minor_faults_s' in df.columns:
        ax3.plot(df['time'], df['minor_faults_s'], color='tab:purple', linestyle=':', alpha=0.8, label='Minor Faults (/s)')
    if 'major_faults_s' in df.columns:
        ax3.plot(df['time'], df['major_faults_s'], color='tab:brown', linestyle='-.', alpha=0.9, label='Major Faults (/s)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('I/O Rate (kB/s) / Faults (/s)')
    ax3.legend()
    ax3.grid(True)

    # Format time axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax3.xaxis.set_major_locator(mdates.SecondLocator(interval=max(1, len(df)//10)))
    plt.xticks(rotation=45)

    plt.tight_layout()
    output_image = log_file.replace(".csv", ".png").replace(".log", ".png")
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    print("📈 Chart saved as: %s" % output_image)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Monitor memory, CPU, and I/O usage of a process")
    parser.add_argument("process", help="Process name or PID (e.g., firefox or 1234)")
    parser.add_argument("-d", "--duration", type=int, default=60, help="Monitoring duration in seconds (default: 60)")
    parser.add_argument("-i", "--interval", type=float, default=1.0, help="Sampling interval in seconds (default: 1.0)")
    parser.add_argument("-o", "--output", default="process_monitor.csv", help="Output log file (default: process_monitor.csv)")

    args = parser.parse_args()
    monitor_process(args.process, args.duration, args.interval, args.output)


if __name__ == "__main__":
    main()
