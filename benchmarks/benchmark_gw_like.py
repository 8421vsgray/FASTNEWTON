"""CUDA timing for synthetic GW-like projection candidates."""

from __future__ import annotations

import argparse
import math
import statistics

import torch

from benchmarks.gw_like import generate_gw_like, gw_candidate
from fastnewton import project


def timed_projection(matrix, repeats: int, warmups: int):
    for _ in range(warmups):
        project(matrix, relative_tol=1.0e-3, max_iter=50, return_info=True)
    torch.cuda.synchronize(matrix.device)
    samples = []
    last = None
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        last = project(matrix, relative_tol=1.0e-3, max_iter=50, return_info=True)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples, last[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=512)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    for mr in (0.0, 0.2, 0.5, 0.8):
        sample = generate_gw_like(
            B=args.B, dim=args.dim, mr=mr, seed=20261020,
            dtype=torch.float32, device="cuda",
        )
        candidate = gw_candidate(sample["transport"], sample["d_source"], sample["d_target"])
        times, info = timed_projection(candidate, args.repeats, args.warmups)
        print({
            "B": args.B,
            "mr": mr,
            "median_ms": statistics.median(times),
            "p95_ms": sorted(times)[min(len(times) - 1, max(0, math.ceil(0.95 * len(times)) - 1))],
            "times_ms": times,
            "status": info["status"],
            "solver": info["solver"],
            "iterations": info["iterations"],
            "relative_residual": info["relative_residual"],
        })


if __name__ == "__main__":
    main()
