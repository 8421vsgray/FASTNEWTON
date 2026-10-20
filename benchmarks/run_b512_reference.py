"""Save a reproducible B=512 CVXPY/FASTNEWTON comparison artifact."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from benchmarks.cvxpy_reference import solve_reference
from benchmarks.gw_like import generate_gw_like, gw_candidate
from fastnewton import primal_dual_certificate, project


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    B, dim, mr, seed = 512, 64, 0.5, 20261020
    output_dir = Path("experiments")
    output_dir.mkdir(exist_ok=True)
    sample = generate_gw_like(B=B, dim=dim, mr=mr, seed=seed, dtype=torch.float64)
    matrix = gw_candidate(sample["transport"], sample["d_source"], sample["d_target"])
    cvx_start = time.perf_counter()
    reference, reference_objective, cvx_status = solve_reference(
        matrix, eps_abs=1.0e-8, eps_rel=1.0e-8, max_iter=200_000
    )
    cvx_seconds = time.perf_counter() - cvx_start
    reference = torch.as_tensor(reference, dtype=torch.float64)
    fast_start = time.perf_counter()
    fast, info = project(matrix, relative_tol=1.0e-3, max_iter=50, return_info=True)
    fast_seconds = time.perf_counter() - fast_start
    fast_objective = 0.5 * (fast - matrix).square().sum().item()
    artifact = {
        "source": "synthetic_gw_like",
        "B": B,
        "dim": dim,
        "mr": mr,
        "seed": seed,
        "dtype": str(matrix.dtype),
        "device": str(matrix.device),
        "cvxpy": {
            "solver": "OSQP",
            "status": cvx_status,
            "eps_abs": 1.0e-8,
            "eps_rel": 1.0e-8,
            "max_iter": 200_000,
            "seconds": cvx_seconds,
            "objective": reference_objective,
            "row_residual": float((reference.sum(1) - 1.0 / B).abs().max()),
            "col_residual": float((reference.sum(0) - 1.0 / B).abs().max()),
            "min": float(reference.min()),
        },
        "fastnewton": {
            "relative_tol": 1.0e-3,
            "max_iter": 50,
            "seconds": fast_seconds,
            "objective": fast_objective,
            "max_abs_error": float((fast - reference).abs().max()),
            "mean_abs_error": float((fast - reference).abs().mean()),
            "objective_abs_gap": abs(fast_objective - reference_objective),
            "objective_rel_gap": abs(fast_objective - reference_objective) / abs(reference_objective),
            "row_residual": float((fast.sum(1) - 1.0 / B).abs().max()),
            "col_residual": float((fast.sum(0) - 1.0 / B).abs().max()),
            "min": float(fast.min()),
            "info": {k: v for k, v in info.items() if isinstance(v, (str, int, float, bool))},
            "certificate": primal_dual_certificate(matrix, fast, info.get("dual_state")),
        },
        "sha256": {
            "input": tensor_sha256(matrix),
            "cvxpy_solution": tensor_sha256(reference),
            "fastnewton_solution": tensor_sha256(fast),
        },
    }
    torch.save({"input": matrix, "cvxpy_solution": reference, "fastnewton_solution": fast}, output_dir / "b512_cvxpy_mr05_20261020.pt")
    with (output_dir / "b512_cvxpy_mr05_20261020.json").open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
