#!/usr/bin/env python3
"""
QAF scalability study: large-K representation and streaming compilation proxy.

Scientific scope:
This program does NOT instantiate or solve a 1M-50M variable full QAF MPS/TDVP
model. It does two complementary things:
1) Computes exact storage/interaction scaling laws for dense QUBO, sparse banded
   QUBO, and fixed-bond-dimension MPS representations.
2) Executes a chunked O(K*b) streaming contextual-compilation proxy at
   K = 1M, 5M, 10M, 25M, 50M to measure practical linear-pass throughput
   without materializing a KxK matrix.

The streaming proxy is a scalability harness, not evidence that the complete QAF
pipeline runs at these K values. It isolates a necessary property for such scale:
linear-memory, locality-bounded processing.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

K_VALUES = np.array([1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000], dtype=np.int64)
CHI_VALUES = [4, 8, 16, 32, 64]
BANDWIDTHS = [2, 4, 8, 16, 32]
DECIMAL = 1000.0


def human_bytes(x: float) -> str:
    if x < 1e3:
        return f"{x:.0f} B"
    units = ["KB", "MB", "GB", "TB", "PB", "EB"]
    v = float(x)
    for u in units:
        v /= 1000.0
        if v < 1000.0:
            return f"{v:.3g} {u}"
    return f"{v:,.3e} B"


def dense_qubo_bytes(K: int, symmetric_packed: bool = False) -> float:
    elems = K * (K + 1) / 2 if symmetric_packed else K * K
    return elems * 8.0  # Float64


def sparse_banded_nnz(K: int, b: int) -> int:
    # Symmetric full band with diagonal; edge correction exact for K > b.
    if K <= b:
        return K * K
    return int(K * (2 * b + 1) - b * (b + 1))


def csr_bytes(K: int, b: int) -> float:
    # Conservative 64-bit data + 64-bit column index + 64-bit row pointer.
    nnz = sparse_banded_nnz(K, b)
    return nnz * 16.0 + (K + 1) * 8.0


def mps_bytes(K: int, chi: int, d: int = 2, bytes_per_complex: int = 8) -> float:
    # Interior-site upper-bound parameter count K * d * chi^2, complex64 = 8 bytes
    return float(K) * d * chi * chi * bytes_per_complex


def tdvp_work_units(K: int, chi: int, d: int = 2) -> float:
    # Normalized dominant scaling proxy O(K * d^3 * chi^3); not seconds.
    return float(K) * (d**3) * (chi**3)


def build_theory_table() -> pd.DataFrame:
    rows = []
    for K in K_VALUES:
        row = {
            "K": int(K),
            "dense_qubo_bytes": dense_qubo_bytes(int(K)),
            "dense_qubo_packed_bytes": dense_qubo_bytes(int(K), True),
            "pairs": int(K) * (int(K) - 1) // 2,
        }
        for b in BANDWIDTHS:
            row[f"csr_b{b}_bytes"] = csr_bytes(int(K), b)
            row[f"csr_b{b}_nnz"] = sparse_banded_nnz(int(K), b)
        for chi in CHI_VALUES:
            row[f"mps_chi{chi}_bytes"] = mps_bytes(int(K), chi)
            row[f"tdvp_chi{chi}_work"] = tdvp_work_units(int(K), chi)
        rows.append(row)
    return pd.DataFrame(rows)


def hash_uniform(u: np.ndarray, salt: np.uint64) -> np.ndarray:
    """Fast deterministic [0,1) values from uint64 ids using SplitMix-like mixing."""
    x = u ^ salt
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    x = x ^ (x >> np.uint64(31))
    # 53 high-quality bits -> float64 [0,1)
    return ((x >> np.uint64(11)).astype(np.float64)) * (1.0 / 9007199254740992.0)


def streaming_proxy(K: int, block: int = 500_000, offsets: Tuple[int, ...] = (1, 2, 4, 8), gamma: float = 0.15) -> dict:
    """Chunked O(K*b) contextual compilation proxy.
    Per variable constructs deterministic confidence, damage, local context and a
    scalar QUBO-like diagonal score, then accumulates local pair interactions for
    fixed offsets. No KxK matrix is materialized.
    """
    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss
    peak_rss = rss0

    # Keep an overlap tail so pair terms across chunk boundaries are counted.
    max_off = max(offsets)
    prev_tail = np.empty(0, dtype=np.float64)
    t0 = time.perf_counter()

    checksum_diag = 0.0
    checksum_pair = 0.0
    positive = 0
    n_done = 0

    for start in range(0, K, block):
        end = min(K, start + block)
        u = np.arange(start, end, dtype=np.uint64)
        p = 0.05 + 0.90 * hash_uniform(u, np.uint64(0x123456789abcdef0))
        dmg = 0.30 + 0.70 * hash_uniform(u, np.uint64(0x9e3779b97f4a7c15))
        ctx = hash_uniform(u, np.uint64(0x243f6a8885a308d3)) - 0.5
        # Contextual update bounded to [0.005, 0.995).
        p_ctx = np.clip(p + gamma * ctx * p * (1.0 - p), 0.005, 0.995)
        score = p_ctx * dmg - 0.05

        checksum_diag += float(score.sum(dtype=np.float64))
        positive += int(np.count_nonzero(score > 0.0))

        if prev_tail.size:
            joined = np.concatenate([prev_tail, score])
            base_start = start - max_off
        else:
            joined = score
            base_start = 0

        # Count interactions whose left endpoint belongs to current output region,
        # with bounded local offsets. This is an arithmetic proxy for banded terms.
        for off in offsets:
            if joined.size > off:
                left = joined[:-off]
                right = joined[off:]
                if prev_tail.size:
                    # use the portion that includes current chunk and cross-boundary terms
                    sidx = max(0, prev_tail.size - off)
                    checksum_pair += float(np.dot(left[sidx:], right[sidx:]))
                else:
                    checksum_pair += float(np.dot(left, right))

        prev_tail = score[-max_off:].copy() if score.size > max_off else score.copy()
        n_done += (end - start)
        peak_rss = max(peak_rss, proc.memory_info().rss)

    elapsed = time.perf_counter() - t0
    return {
        "K": int(K),
        "elapsed_s": elapsed,
        "throughput_Mvars_s": (K / elapsed) / 1e6,
        "peak_delta_MB": max(0, peak_rss - rss0) / 1e6,
        "rss_end_MB": proc.memory_info().rss / 1e6,
        "checksum_diag": checksum_diag,
        "checksum_pair": checksum_pair,
        "positive_fraction": positive / K,
        "offset_count": len(offsets),
        "block": block,
    }


def benchmark_streaming(repeats: int = 2, block: int = 500_000) -> pd.DataFrame:
    rows = []
    # warm-up
    streaming_proxy(100_000, block=min(block, 100_000))

    for K in K_VALUES:
        for rep in range(repeats):
            r = streaming_proxy(int(K), block=block)
            r["repeat"] = rep + 1
            rows.append(r)
            print(f"K={K:,}, rep={rep+1}: {r['elapsed_s']:.3f} s, {r['throughput_Mvars_s']:.2f} Mvars/s")

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--block", type=int, default=500_000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    theory = build_theory_table()
    theory.to_csv(out / "scaling_theory.csv", index=False)

    stream = benchmark_streaming(args.repeats, args.block)
    stream.to_csv(out / "streaming_benchmark_raw.csv", index=False)

    summary = stream.groupby("K").agg(
        elapsed_s_mean=("elapsed_s", "mean"),
        elapsed_s_std=("elapsed_s", "std"),
        throughput_Mvars_s_mean=("throughput_Mvars_s", "mean"),
        throughput_Mvars_s_std=("throughput_Mvars_s", "std"),
        peak_delta_MB_max=("peak_delta_MB", "max"),
        positive_fraction=("positive_fraction", "mean"),
    ).reset_index()

    summary.to_csv(out / "streaming_benchmark_summary.csv", index=False)

    print("\nTheory summary:")
    for _, r in theory.iterrows():
        K = int(r.K)
        print(f"K={K:,}: dense QUBO={human_bytes(r.dense_qubo_bytes)}, "
              f"banded b=8={human_bytes(r.csr_b8_bytes)}, MPS chi=16={human_bytes(r.mps_chi16_bytes)}")


if __name__ == "__main__":
    main()
