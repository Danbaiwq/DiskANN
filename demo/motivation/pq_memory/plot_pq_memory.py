#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def configure_matplotlib_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 12,
        "font.family": "serif",
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 5,
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


def read_curves_from_csv(csv_path: Path) -> List[Tuple[str, pd.DataFrame]]:
    df = pd.read_csv(csv_path)
    # Forward-fill B and E to propagate block headers into following rows
    df[["B(MB)", "E(K)"]] = df[["B(MB)", "E(K)"]].ffill()

    # Ensure numeric types
    df["B(MB)"] = pd.to_numeric(df["B(MB)"], errors="coerce")
    df["E(K)"] = pd.to_numeric(df["E(K)"] , errors="coerce")
    df["QPS"] = pd.to_numeric(df["QPS"], errors="coerce")
    df["Recall@10"] = pd.to_numeric(df["Recall@10"], errors="coerce")

    # Drop rows with missing QPS/Recall
    df = df.dropna(subset=["QPS", "Recall@10"]).copy()

    curves: List[Tuple[str, pd.DataFrame]] = []
    for (b_mb, e_k), g in df.groupby(["B(MB)", "E(K)"], sort=False):
        label = f"B={int(b_mb)}MB, E={int(e_k)}K"
        # Keep original order; also provide a stable sort by Recall for nicer left-to-right lines
        g_sorted = g.sort_values(by=["Recall@10"]).reset_index(drop=True)
        curves.append((label, g_sorted))
    return curves


def _compute_axis_limits(curves: List[Tuple[str, pd.DataFrame]]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    x_min = math.inf
    x_max = -math.inf
    y_min = math.inf
    y_max = -math.inf

    for _, df in curves:
        if len(df) == 0:
            continue
        # x is Recall@10, y is QPS
        x_min = min(x_min, float(df["Recall@10"].min()))
        x_max = max(x_max, float(df["Recall@10"].max()))
        y_min = min(y_min, float(df["QPS"].min()))
        y_max = max(y_max, float(df["QPS"].max()))

    if not math.isfinite(x_min):
        x_min, x_max = 0.0, 100.0
    if not math.isfinite(y_min):
        y_min, y_max = 0.0, 1.0

    # Add small margins
    x_pad = 0.02 * (x_max - x_min) if x_max > x_min else 1.0
    y_pad = 0.02 * (y_max - y_min) if y_max > y_min else 1.0
    return (max(0.0, x_min - x_pad), min(100.0, x_max + x_pad)), (max(0.0, y_min - y_pad), y_max + y_pad)


def plot_curves(
    curves: List[Tuple[str, pd.DataFrame]],
    title: str,
    out_path: Path,
    line_styles: List[str] = None,
    colors: List[str] = None,
) -> None:
    configure_matplotlib_style()

    if line_styles is None:
        line_styles = ["-", "-", "-", "-", "-", "-"]
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4", "C5"])  # type: ignore

    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    for idx, (label, df) in enumerate(curves):
        style = line_styles[idx % len(line_styles)]
        color = colors[idx % len(colors)]
        ax.plot(
            df["Recall@10"],
            df["QPS"],
            linestyle=style,
            marker="o",
            color=color,
            label=label,
        )

    (x0, x1), (y0, y1) = _compute_axis_limits(curves)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    ax.set_title(title)
    ax.set_xlabel("Recall@10 (%)")
    ax.set_ylabel("QPS")

    ax.legend(loc="upper right", frameon=False, ncol=2, handlelength=2.5)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    csv_320b = script_dir / "diskann_gist1M_320B.csv"
    csv_32b = script_dir / "diskann_gist1M_32B.csv"

    curves_320b = read_curves_from_csv(csv_320b)
    curves_32b = read_curves_from_csv(csv_32b)

    # 1) 320B: three curves
    out1 = script_dir / "gist1m_pq_320b_qps_recall.png"
    plot_curves(
        curves_320b,
        title="GIST1M, PQ=320B: QPS vs Recall@10 by (B,E)",
        out_path=out1,
        line_styles=["-", "-", "-"],
    )

    # 2) 32B: three curves
    out2 = script_dir / "gist1m_pq_32b_qps_recall.png"
    plot_curves(
        curves_32b,
        title="GIST1M, PQ=32B: QPS vs Recall@10 by (B,E)",
        out_path=out2,
        line_styles=["-", "-", "-"],
    )

    # 3) Combined: six curves
    combined: List[Tuple[str, pd.DataFrame]] = []
    # Solid lines for 320B
    for label, df in curves_320b:
        combined.append((f"PQ=320B, {label}", df))
    # Dashed lines for 32B
    for label, df in curves_32b:
        combined.append((f"PQ=32B, {label}", df))

    out3 = script_dir / "gist1m_pq_compare_32b_vs_320b.png"
    plot_curves(
        combined,
        title="GIST1M: PQ=320B vs 32B — QPS vs Recall@10",
        out_path=out3,
        line_styles=["-", "-", "-", "--", "--", "--"],
    )


if __name__ == "__main__":
    main() 