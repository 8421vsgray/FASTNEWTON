"""Public API for the extracted FASTNEWTON projector."""

from __future__ import annotations

from typing import Any

import torch

from ._stride_projection import EuclideanProjector


class ProjectionFailure(RuntimeError):
    """Raised when the baseline solver returns an uncertified failure."""


def project(
    matrix: torch.Tensor,
    *,
    relative_tol: float = 1.0e-3,
    max_iter: int = 50,
    return_info: bool = False,
    raise_on_failure: bool = False,
    **solver_options: Any,
):
    """Project a square matrix onto the uniform transport polytope.

    The extracted STRIDE algorithm is used without changing its numerical
    path.  This wrapper derives the marginal as ``1 / B`` and adds explicit
    input and failure-state validation.  The operation is intentionally
    non-differentiable in this first release.
    """
    if not isinstance(matrix, torch.Tensor):
        raise TypeError("matrix must be a torch.Tensor")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be a square 2-D tensor")
    if not matrix.is_floating_point():
        raise TypeError("matrix must have floating-point dtype")
    if not torch.isfinite(matrix).all():
        raise ValueError("matrix must contain only finite values")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not isinstance(relative_tol, (int, float)) or relative_tol <= 0:
        raise ValueError("relative_tol must be positive")

    n = matrix.shape[0]
    transport, info = EuclideanProjector.project(
        matrix,
        1.0 / float(n),
        max_iter=max_iter,
        relative_tol=float(relative_tol),
        **solver_options,
    )
    info = dict(info)
    solver = info.get("solver")
    failed = solver == "projection_failure"
    info.update(
        {
            "status": "failed" if failed else "solved",
            "success": not failed,
            "feasibility_verified": not failed,
            "optimality_status": "unverified",
            "autograd_supported": False,
        }
    )
    if failed and raise_on_failure:
        raise ProjectionFailure(
            "FASTNEWTON baseline returned projection_failure; uniform output "
            "is not reported as a successful projection"
        )
    if return_info:
        return transport, info
    if failed:
        raise ProjectionFailure(
            "FASTNEWTON baseline returned projection_failure; pass "
            "return_info=True to inspect the failure state"
        )
    return transport
