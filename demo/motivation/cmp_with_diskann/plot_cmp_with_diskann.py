#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
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


def _to_num(series: pd.Series) -> pd.Series:
	# 去掉百分号与空白，转为 float
	return pd.to_numeric(series.astype(str).str.replace('%', '', regex=False).str.strip(), errors="coerce")


def main() -> None:
	p = argparse.ArgumentParser(description="绘制 DiskANN vs LiteANN 的 QPS-Recall 曲线")
	p.add_argument("csv", type=str, help="输入CSV。支持两种格式：1) 列: Recall, DiskANN, LiteANN；2) 列: DiskANN, Recall@10, (空), LiteANN, Recall@10")
	p.add_argument("--out", type=str, default=None, help="输出PNG路径（默认与CSV同名）")
	args = p.parse_args()

	csv_path = Path(args.csv)
	out_png = Path(args.out) if args.out else csv_path.with_suffix(".png")

	df = pd.read_csv(csv_path)

	# 兼容两种表头格式
	if {"DiskANN", "LiteANN"}.issubset(set(df.columns)):
		# 新格式：DiskANN, Recall@10, [空], LiteANN, Recall@10(重复名)
		recall_cols = [c for c in df.columns if "Recall" in c]
		if len(recall_cols) == 0:
			raise SystemExit("CSV 缺少 Recall 列")
		# 第一列 Recall -> DiskANN，对应第二个 Recall -> LiteANN（若存在）
		recall_disk_col = recall_cols[0]
		recall_lite_col = recall_cols[-1] if len(recall_cols) > 1 else recall_cols[0]

		disk = pd.DataFrame({
			"Recall": _to_num(df[recall_disk_col]),
			"QPS": pd.to_numeric(df["DiskANN"], errors="coerce"),
		})
		lite = pd.DataFrame({
			"Recall": _to_num(df[recall_lite_col]),
			"QPS": pd.to_numeric(df["LiteANN"], errors="coerce"),
		})
	else:
		# 旧格式：Recall, DiskANN, LiteANN
		for col in ["Recall", "DiskANN", "LiteANN"]:
			if col not in df.columns:
				raise SystemExit(f"CSV 缺少列: {col}")
		disk = pd.DataFrame({
			"Recall": _to_num(df["Recall"]),
			"QPS": pd.to_numeric(df["DiskANN"], errors="coerce"),
		})
		lite = pd.DataFrame({
			"Recall": _to_num(df["Recall"]),
			"QPS": pd.to_numeric(df["LiteANN"], errors="coerce"),
		})

	# 清洗无效行并按 Recall 升序
	disk = disk.dropna(subset=["Recall", "QPS"]).sort_values(by=["Recall"]).reset_index(drop=True)
	lite = lite.dropna(subset=["Recall", "QPS"]).sort_values(by=["Recall"]).reset_index(drop=True)

	configure_matplotlib_style()
	fig, ax = plt.subplots(figsize=(6.8, 4.6))
	if len(disk) > 0:
		ax.plot(disk["Recall"], disk["QPS"], marker="o", color="#1f77b4", label="DiskANN")
	if len(lite) > 0:
		ax.plot(lite["Recall"], lite["QPS"], marker="s", color="#ff7f0e", label="LiteANN")
	ax.set_title("QPS vs Recall@10")
	ax.set_xlabel("Recall@10 (%)")
	ax.set_ylabel("QPS")
	ax.legend(loc="best", frameon=False)
	fig.tight_layout()
	fig.savefig(out_png, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved: {out_png}")


if __name__ == "__main__":
	main() 