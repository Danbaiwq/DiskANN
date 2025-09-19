#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import time
from typing import Iterable, List, Optional, Tuple

import numpy as np

try:
	import faiss  # type: ignore
except Exception as exc:  # pragma: no cover
	raise RuntimeError(
		"无法导入 faiss。请先安装 faiss-cpu，例如: pip install -U faiss-cpu"
	) from exc


# ------------------------------
# 辅助函数
# ------------------------------

def parse_int_list(arg: str) -> List[int]:
	values = []
	for part in arg.split(","):
		part = part.strip()
		if not part:
			continue
		values.append(int(part))
	return values


def quantiles_ms(samples_ms: List[float], qs: Iterable[float]) -> List[float]:
	if not samples_ms:
		return [0.0 for _ in qs]
	arr = np.asarray(samples_ms, dtype=np.float64)
	return [float(np.quantile(arr, q)) for q in qs]


def l2_normalize_inplace(vectors: np.ndarray) -> None:
	# vectors: float32, shape (n, d)
	norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
	vectors /= norms


def ensure_dir(path: str) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)


def compute_recall_at_k(I_true: np.ndarray, I_pred: np.ndarray, k: int) -> float:
	# I_*: shape (nq, k)
	if I_true.shape != I_pred.shape:
		raise ValueError("I_true 与 I_pred 形状不一致")
	nq = I_true.shape[0]
	total = 0.0
	for i in range(nq):
		truth = set(int(x) for x in I_true[i])
		pred = set(int(x) for x in I_pred[i])
		inter = truth.intersection(pred)
		total += len(inter) / float(k)
	return total / float(nq)


def estimate_index_bytes(index: "faiss.Index") -> Optional[int]:  # noqa: F821
	# 优先使用序列化估算内存尺寸
	try:
		buf = faiss.serialize_index(index)
		return len(buf)
	except Exception:
		return None


# ------------------------------
# 数据生成
# ------------------------------

def generate_dataset(
	num_base: int,
	num_query: int,
	dimension: int,
	metric: str,
	normalize_for_ip: bool,
	seed: int,
	distribution: str = "normal",
) -> Tuple[np.ndarray, np.ndarray]:
	"""生成 base 与 query 数据集。

	metric: "l2" 或 "ip"。
	distribution: "normal" 或 "uniform"。
	"""
	rng = np.random.RandomState(seed)
	if distribution == "uniform":
		base = rng.rand(num_base, dimension).astype(np.float32)
		query = rng.rand(num_query, dimension).astype(np.float32)
	else:
		base = rng.standard_normal(size=(num_base, dimension)).astype(np.float32)
		query = rng.standard_normal(size=(num_query, dimension)).astype(np.float32)

	if metric == "ip" and normalize_for_ip:
		l2_normalize_inplace(base)
		l2_normalize_inplace(query)

	return base, query


# ------------------------------
# 索引构建与查询
# ------------------------------

def set_num_threads(threads: int) -> None:
	if threads and threads > 0:
		faiss.omp_set_num_threads(threads)


def build_index_flat(dimension: int, metric: str) -> "faiss.Index":  # noqa: F821
	if metric == "ip":
		return faiss.IndexFlatIP(dimension)
	return faiss.IndexFlatL2(dimension)


def build_index_hnsw(dimension: int, metric: str, M: int, efC: int, efS: int) -> "faiss.Index":  # noqa: F821
	metric_type = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
	index = faiss.index_factory(dimension, f"HNSW{M}", metric_type)
	# 设置 HNSW 参数
	index.hnsw.efConstruction = int(efC)
	index.hnsw.efSearch = int(efS)
	return index


def time_add(index: "faiss.Index", xb: np.ndarray) -> float:  # noqa: F821
	start = time.perf_counter()
	index.add(xb)
	return time.perf_counter() - start


def time_search_batch(index: "faiss.Index", xq: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, float]:  # noqa: F821
	start = time.perf_counter()
	D, I = index.search(xq, k)
	elapsed = time.perf_counter() - start
	return D, I, elapsed


def sample_per_query_latencies_ms(index: "faiss.Index", xq: np.ndarray, k: int, sample_n: int) -> List[float]:  # noqa: F821
	if sample_n <= 0:
		return []
	sample_n = min(sample_n, xq.shape[0])
	latencies_ms: List[float] = []
	for i in range(sample_n):
		q = xq[i : i + 1]
		start = time.perf_counter()
		index.search(q, k)
		latencies_ms.append((time.perf_counter() - start) * 1000.0)
	return latencies_ms


# ------------------------------
# 基准执行
# ------------------------------

def run_bench(
	output_csv: str,
	dimensions: List[int],
	num_bases: List[int],
	num_queries: int,
	topk: int,
	repeats: int,
	metric: str,
	normalize_for_ip: bool,
	hnsw_M_list: List[int],
	hnsw_efC_list: List[int],
	hnsw_efS_list: List[int],
	threads: int,
	sample_latency_n: int,
	seed: int,
	distribution: str,
) -> None:
	ensure_dir(output_csv)
	is_new_file = not os.path.exists(output_csv)
	with open(output_csv, mode="a", newline="") as f:
		writer = csv.writer(f)
		if is_new_file:
			writer.writerow(
				[
					"repeat_id",
					"index_type",
					"N",
					"d",
					"metric",
					"topk",
					"nq",
					"threads",
					"hnsw_M",
					"hnsw_efC",
					"hnsw_efS",
					"build_s",
					"search_s",
					"qps",
					"avg_ms_per_q",
					"p50_ms",
					"p95_ms",
					"p99_ms",
					"recall_at_k",
					"index_bytes",
				]
			)

		for d in dimensions:
			for N in num_bases:
				for rep in range(repeats):
					cur_seed = seed + rep * 9973 + d * 13 + N
					set_num_threads(threads)
					xb, xq = generate_dataset(
						N,
						num_queries,
						d,
						metric,
						normalize_for_ip,
						cur_seed,
						distribution=distribution,
					)

					# BF: IndexFlat
					flat = build_index_flat(d, metric)
					build_s_flat = time_add(flat, xb)
					_, I_bf, search_s_flat = time_search_batch(flat, xq, topk)
					lat_ms_flat = sample_per_query_latencies_ms(flat, xq, topk, sample_latency_n)
					p50, p95, p99 = quantiles_ms(lat_ms_flat, [0.5, 0.95, 0.99])
					qps_flat = float(num_queries) / search_s_flat if search_s_flat > 0 else 0.0
					avg_ms_flat = (search_s_flat / max(1, num_queries)) * 1000.0
					bytes_flat = estimate_index_bytes(flat)

					writer.writerow(
						[
							rep,
							"flat",
							N,
							d,
							metric,
							topk,
							num_queries,
							threads,
							"",
							"",
							"",
							f"{build_s_flat:.6f}",
							f"{search_s_flat:.6f}",
							f"{qps_flat:.3f}",
							f"{avg_ms_flat:.3f}",
							f"{p50:.3f}",
							f"{p95:.3f}",
							f"{p99:.3f}",
							f"{1.0:.6f}",
							bytes_flat if bytes_flat is not None else "",
						]
					)
					print(
						f"[flat metric={metric}] N={N} d={d} topk={topk} nq={num_queries} threads={threads} "
						f"build={build_s_flat:.3f}s search={search_s_flat:.3f}s qps={qps_flat:.1f} "
						f"avg={avg_ms_flat:.3f}ms p50={p50:.3f}ms p95={p95:.3f}ms p99={p99:.3f}ms recall=1.000"
					)
					# HNSW 组合
					for M in hnsw_M_list:
						for efC in hnsw_efC_list:
							for efS in hnsw_efS_list:
								set_num_threads(threads)
								idx = build_index_hnsw(d, metric, M, efC, efS)
								build_s_hnsw = time_add(idx, xb)
								_, I_hnsw, search_s_hnsw = time_search_batch(idx, xq, topk)
								lat_ms_hnsw = sample_per_query_latencies_ms(idx, xq, topk, sample_latency_n)
								p50, p95, p99 = quantiles_ms(lat_ms_hnsw, [0.5, 0.95, 0.99])
								qps_hnsw = float(num_queries) / search_s_hnsw if search_s_hnsw > 0 else 0.0
								avg_ms_hnsw = (search_s_hnsw / max(1, num_queries)) * 1000.0
								recall = compute_recall_at_k(I_bf, I_hnsw, topk)
								bytes_hnsw = estimate_index_bytes(idx)

								writer.writerow(
									[
										rep,
										"hnsw",
										N,
										d,
										metric,
										topk,
										num_queries,
										threads,
										M,
										efC,
										efS,
										f"{build_s_hnsw:.6f}",
										f"{search_s_hnsw:.6f}",
										f"{qps_hnsw:.3f}",
										f"{avg_ms_hnsw:.3f}",
										f"{p50:.3f}",
										f"{p95:.3f}",
										f"{p99:.3f}",
										f"{recall:.6f}",
										bytes_hnsw if bytes_hnsw is not None else "",
									]
								)
								print(
									f"[hnsw M={M} efC={efC} efS={efS} metric={metric}] N={N} d={d} topk={topk} "
									f"nq={num_queries} threads={threads} build={build_s_hnsw:.3f}s "
									f"search={search_s_hnsw:.3f}s qps={qps_hnsw:.1f} avg={avg_ms_hnsw:.3f}ms "
									f"p50={p50:.3f}ms p95={p95:.3f}ms p99={p99:.3f}ms recall={recall:.4f}"
								)
					print(
						f"--------------------------------------------------------------------------------------------------------"
					)


# ------------------------------
# CLI
# ------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(
		prog="run_benchmarks",
		description="比较 FAISS 暴搜(IndexFlat) 与 图搜(HNSW) 在不同 N/d 下的构建与查询耗时",
	)
	p.add_argument("--dims", type=str, default="64,128,256", help="逗号分隔的维度列表")
	p.add_argument("--num-data", type=str, default="10000,50000,100000", help="逗号分隔的数据量列表 N")
	p.add_argument("--nq", type=int, default=1000, help="查询向量数量")
	p.add_argument("--topk", type=int, default=10, help="Top-K")
	p.add_argument("--repeats", type=int, default=1, help="重复次数")
	p.add_argument("--metric", type=str, choices=["l2", "ip"], default="l2", help="度量：l2 或 ip")
	p.add_argument("--normalize-for-ip", action="store_true", help="当 metric=ip 时对向量做 L2 归一化")
	p.add_argument("--hnsw-M", type=str, default="32", help="HNSW 的 M 值列表，逗号分隔")
	p.add_argument("--hnsw-efC", type=str, default="200", help="HNSW efConstruction 列表，逗号分隔")
	p.add_argument("--hnsw-efS", type=str, default="64,128,256", help="HNSW efSearch 列表，逗号分隔")
	p.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1), help="FAISS OpenMP 线程数")
	p.add_argument("--sample-latency-n", type=int, default=200, help="用于分位数的逐查询采样数量，0 则不采样")
	p.add_argument("--seed", type=int, default=12345, help="随机种子")
	p.add_argument("--distribution", type=str, choices=["normal", "uniform"], default="normal", help="数据分布")
	default_out = os.path.join(os.path.dirname(__file__), "results.csv")
	p.add_argument("--out-csv", type=str, default=default_out, help="结果 CSV 路径")
	return p


def main() -> None:
	args = build_arg_parser().parse_args()

	dimensions = parse_int_list(args.dims)
	num_bases = parse_int_list(args.num_data)
	hnsw_M_list = parse_int_list(args.hnsw_M)
	hnsw_efC_list = parse_int_list(args.hnsw_efC)
	hnsw_efS_list = parse_int_list(args.hnsw_efS)

	# 当 metric=ip 且未显式指定 --normalize-for-ip 时默认启用归一化
	normalize_for_ip = bool(args.normalize_for_ip)
	if args.metric == "ip" and not args.normalize_for_ip:
		normalize_for_ip = True

	run_bench(
		output_csv=args.out_csv,
		dimensions=dimensions,
		num_bases=num_bases,
		num_queries=args.nq,
		topk=args.topk,
		repeats=args.repeats,
		metric=args.metric,
		normalize_for_ip=normalize_for_ip,
		hnsw_M_list=hnsw_M_list,
		hnsw_efC_list=hnsw_efC_list,
		hnsw_efS_list=hnsw_efS_list,
		threads=args.threads,
		sample_latency_n=args.sample_latency_n,
		seed=args.seed,
		distribution=args.distribution,
	)


if __name__ == "__main__":
	main() 