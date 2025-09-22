#!/usr/bin/env python3
import argparse
import os
import struct
from typing import List


def read_fvecs_header(path: str) -> int:
    with open(path, 'rb') as f:
        dim_bytes = f.read(4)
        if len(dim_bytes) != 4:
            raise ValueError("无法读取维度头部")
        (d,) = struct.unpack('i', dim_bytes)
        if d <= 0 or d > 10_000_000:
            raise ValueError(f"维度异常: {d}")
        return d


def iter_fvecs(path: str, expected_dim: int):
    rec_header = struct.Struct('i')
    vec_struct = struct.Struct(f'{expected_dim}f')
    rec_size = 4 + expected_dim * 4
    with open(path, 'rb') as f:
        while True:
            dim_raw = f.read(4)
            if not dim_raw:
                return
            if len(dim_raw) != 4:
                raise EOFError("遇到截断记录: 维度头不足 4 字节")
            (d,) = rec_header.unpack(dim_raw)
            if d != expected_dim:
                raise ValueError(f"记录维度不一致，期望 {expected_dim} 实际 {d}")
            payload = f.read(expected_dim * 4)
            if len(payload) != expected_dim * 4:
                raise EOFError("遇到截断记录: 向量体不足")
            yield payload


def open_writers(out_dir: str, base: str, counts: List[int]):
    os.makedirs(out_dir, exist_ok=True)
    writers = {}
    for n in counts:
        out_path = os.path.join(out_dir, f"{base}_{n}K.fvecs" if n < 1000 else f"{base}_{n//1000}M.fvecs")
        writers[n] = open(out_path, 'wb')
    return writers


def close_writers(writers):
    for w in writers.values():
        w.close()


def write_record(writer, dim: int, payload: bytes):
    writer.write(struct.pack('i', dim))
    writer.write(payload)


def main():
    parser = argparse.ArgumentParser(description='从 1M fvecs 切分出子集 (1K/10K/50K/100K/250K/500K/1M)')
    parser.add_argument('input', help='输入 fvecs 文件 (1M)')
    parser.add_argument('-o', '--out_dir', default='.', help='输出目录 (默认当前目录)')
    parser.add_argument('-b', '--base', default=None, help='输出文件前缀（默认使用输入文件名不含后缀）')
    args = parser.parse_args()

    in_path = args.input
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"找不到输入文件: {in_path}")

    dim = read_fvecs_header(in_path)

    base = args.base
    if base is None:
        base = os.path.splitext(os.path.basename(in_path))[0]

    # 目标数量（单位：向量数）
    targets = [1_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]

    writers = open_writers(args.out_dir, base, [1, 10, 50, 100, 250, 500, 1000])
    try:
        # 已写入计数
        written = {1: 0, 10: 0, 50: 0, 100: 0, 250: 0, 500: 0, 1000: 0}
        thresholds = {1: 1_000, 10: 10_000, 50: 50_000, 100: 100_000, 250: 250_000, 500: 500_000, 1000: 1_000_000}

        total = 0
        for payload in iter_fvecs(in_path, dim):
            total += 1
            if total <= thresholds[1]:
                write_record(writers[1], dim, payload)
                written[1] += 1
            if total <= thresholds[10]:
                write_record(writers[10], dim, payload)
                written[10] += 1
            if total <= thresholds[50]:
                write_record(writers[50], dim, payload)
                written[50] += 1
            if total <= thresholds[100]:
                write_record(writers[100], dim, payload)
                written[100] += 1
            if total <= thresholds[250]:
                write_record(writers[250], dim, payload)
                written[250] += 1
            if total <= thresholds[500]:
                write_record(writers[500], dim, payload)
                written[500] += 1
            if total <= thresholds[1000]:
                write_record(writers[1000], dim, payload)
                written[1000] += 1

            if total >= thresholds[1000]:
                break

    finally:
        close_writers(writers)

    # 基本校验
    for k, n in [(1, 1_000), (10, 10_000), (50, 50_000), (100, 100_000), (250, 250_000), (500, 500_000), (1000, 1_000_000)]:
        out_path = os.path.join(args.out_dir, f"{base}_{k}K.fvecs" if k < 1000 else f"{base}_{k//1000}M.fvecs")
        if os.path.getsize(out_path) != n * (4 + dim * 4):
            raise RuntimeError(f"输出文件大小异常: {out_path}")

    print("完成: ", args.out_dir)


if __name__ == '__main__':
    main() 