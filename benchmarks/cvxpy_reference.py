"""Optional CVXPY/OSQP reference for small Euclidean projections."""

from __future__ import annotations

import numpy as np


def solve_reference(matrix, *, eps_abs: float = 1.0e-8, eps_rel: float = 1.0e-8, max_iter: int = 200_000):
    import cvxpy as cp

    x = np.asarray(matrix.detach().cpu() if hasattr(matrix, "detach") else matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("matrix must be square")
    n = x.shape[0]
    variable = cp.Variable((n, n), nonneg=True)
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.sum_squares(variable - x)),
        [cp.sum(variable, axis=1) == 1.0 / n, cp.sum(variable, axis=0) == 1.0 / n],
    )
    problem.solve(
        solver=cp.OSQP,
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        max_iter=max_iter,
        polish=True,
        verbose=False,
    )
    status = str(problem.status)
    if status != "optimal":
        raise RuntimeError(f"CVXPY reference status is {status}, not optimal")
    return np.asarray(variable.value), float(problem.value), status
