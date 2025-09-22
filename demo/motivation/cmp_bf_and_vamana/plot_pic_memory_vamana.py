#!/usr/bin/env python3
import argparse
import csv
import os
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

ORDER = ["1K", "10K", "50K", "100K", "250K", "500K", "1M"]


def parse_csv(path: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    # 返回两个映射：size->ratio (SIFT), size->ratio (GIST)
    ratios_sift: Dict[str, float] = {}
    ratios_gist: Dict[str, float] = {}
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    # 预期格式：
    # L1: SIFT,,,GIST,
    # L2: BF,Vamana/BF,,BF,Vamana/BF
    # 后续：size,ratio_sift,,size,ratio_gist
    for row in rows[2:]:
        if not row:
            continue
        # 容错：行长度可能小于 5
        size_sift = row[0].strip() if len(row) > 0 else ""
        ratio_sift = row[1].strip() if len(row) > 1 else ""
        size_gist = row[3].strip() if len(row) > 3 else ""
        ratio_gist = row[4].strip() if len(row) > 4 else ""
        if size_sift and ratio_sift:
            try:
                ratios_sift[size_sift] = float(ratio_sift)
            except ValueError:
                pass
        if size_gist and ratio_gist:
            try:
                ratios_gist[size_gist] = float(ratio_gist)
            except ValueError:
                pass
    return ratios_sift, ratios_gist


def plot_lines(ratios_sift: Dict[str, float], ratios_gist: Dict[str, float], out_path: str):
    sizes = ORDER
    x_pos = list(range(len(sizes)))
    sift_vals = [ratios_sift.get(s, float("nan")) for s in sizes]
    gist_vals = [ratios_gist.get(s, float("nan")) for s in sizes]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_pos, sift_vals, marker="o", label="SIFT", color="#1f77b4")
    ax.plot(x_pos, gist_vals, marker="s", label="GIST", color="#ff7f0e")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("QPS ratio (Vamana/BF)")
    ax.set_title("Memory Vamana vs BF QPS ratio")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.legend(loc="best")
    fig.tight_layout()

    fig.savefig(out_path, dpi=200)
    print(f"Saved figure: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="绘制 pic_memory_vamana.csv 的 Vamana/BF 比值图")
    parser.add_argument("-i", "--input", default="pic_memory_vamana.csv", help="输入 CSV 路径")
    parser.add_argument("-o", "--output", default="pic_memory_vamana.png", help="输出 PNG 路径")
    args = parser.parse_args()

    ratios_sift, ratios_gist = parse_csv(args.input)
    plot_lines(ratios_sift, ratios_gist, args.output)


if __name__ == "__main__":
    main() 