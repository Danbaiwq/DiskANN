#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def configure_matplotlib_style() -> None:
	plt.rcParams.update({
		"figure.dpi": 120,
		"savefig.dpi": 300,
		"font.size": 12,
		"font.family": "serif",
		"axes.titlesize": 13,
		"axes.labelsize": 12,
		"legend.fontsize": 10,
		"xtick.labelsize": 10,
		"ytick.labelsize": 10,
		"axes.grid": True,
		"grid.alpha": 0.3,
		"grid.linestyle": "--",
		"lines.linewidth": 2.0,
		"lines.markersize": 5.5,
		"axes.spines.top": False,
		"axes.spines.right": False,
	})


def main() -> None:
	script_dir = Path(__file__).resolve().parent
	csv_path = script_dir / "spann_cpu.csv"
	out_png = script_dir / "spann_cpu.png"

	df = pd.read_csv(csv_path)
	# 标准化列名
	assert {"F", "CPU%"}.issubset(set(df.columns)), "csv 需包含列: F, CPU%"
	df = df.copy()
	# 确保数值类型（去掉百分号）
	df["F"] = pd.to_numeric(df["F"], errors="coerce")
	df["CPU%"] = pd.to_numeric(df["CPU%"].astype(str).str.replace('%', '', regex=False).str.strip(), errors="coerce")
	# 过滤无效并按 F 升序
	df = df.dropna(subset=["F", "CPU%"]).sort_values(by=["F"]).reset_index(drop=True)

	configure_matplotlib_style()
	fig, ax = plt.subplots(figsize=(6.4, 4.0))
	ax.plot(df["F"], df["CPU%"], marker="o", color="#1f77b4", label="SPANN CPU usage")
	ax.set_title("SPANN: CPU% vs F")
	ax.set_xlabel("F")
	ax.set_ylabel("CPU%")
	# 将横坐标设置为整数刻度
	xticks = sorted(df["F"].unique().tolist())
	ax.set_xticks(xticks)
	ax.legend(loc="best", frameon=False)
	fig.tight_layout()
	fig.savefig(out_png, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved: {out_png}")


if __name__ == "__main__":
	main() 