"""Public API for the extracted FASTNEWTON projector."""

from __future__ import annotations

from typing import Any

import torch

from ._stride_projection import EuclideanProjector
from .diagnostics import primal_dual_certificate


class ProjectionFailure(RuntimeError):
    """Raised when the baseline solver returns an uncertified failure."""


def _default_certificate_tolerance(size: int, relative_tol: float) -> float:
    """Convert the public relative marginal tolerance to an absolute scale."""
    return abs(1.0 / float(size)) * float(relative_tol)


@torch.no_grad()
def _gauge_normalize_uniform_transport_matrix(
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove row and column potentials without changing the uniform projection.

    For fixed row and column marginals, adding any row/column potential to the
    candidate changes the quadratic objective by a constant over the feasible
    set.  Removing those components gives the solver a better-scaled, doubly
    centered matrix while preserving its exact Euclidean projection.
    """
    row_mean = matrix.mean(dim=1, keepdim=True)
    col_mean = matrix.mean(dim=0, keepdim=True)
    global_mean = matrix.mean()
    centered = matrix - row_mean - col_mean + global_mean
    return centered, row_mean, col_mean, global_mean


def _transform_dual_state(
    dual_state,
    row_mean: torch.Tensor,
    col_mean: torch.Tensor,
    global_mean: torch.Tensor,
    *,
    to_centered: bool,
):
    """Convert public-coordinate dual potentials to/from the centered gauge."""
    if not isinstance(dual_state, (tuple, list)) or len(dual_state) != 2:
        return dual_state
    u, v = dual_state
    n = row_mean.shape[0]
    if (
        not torch.is_tensor(u)
        or not torch.is_tensor(v)
        or tuple(u.shape) != (n,)
        or tuple(v.shape) != (n,)
        or u.device != row_mean.device
        or v.device != col_mean.device
    ):
        return dual_state
    u = u.to(dtype=row_mean.dtype)
    v = v.to(dtype=col_mean.dtype)
    row = row_mean[:, 0]
    col = col_mean[0, :]
    if to_centered:
        # Xc = X - row - col + global, so uc=u-row+global and vc=v-col.
        return u - row + global_mean, v - col
    # Convert the solver's centered duals back to the coordinates of X.
    return u + row - global_mean, v + col


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
    non-differentiable in this first release. A failed result raises by
    default when diagnostics are not requested. With ``return_info=True``, a
    failed result is returned for inspection unless ``raise_on_failure=True``.
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
    if n < 1:
        raise ValueError("matrix must be non-empty")
    default_certificate_tolerance = _default_certificate_tolerance(n, relative_tol)
    certificate_tolerance = float(
        solver_options.pop("certificate_tolerance", default_certificate_tolerance)
    )
    if certificate_tolerance <= 0:
        raise ValueError("certificate_tolerance must be positive")
    centered_matrix, row_mean, col_mean, global_mean = (
        _gauge_normalize_uniform_transport_matrix(matrix)
    )
    centered_dual_warm_start = _transform_dual_state(
        solver_options.pop("dual_warm_start", None),
        row_mean,
        col_mean,
        global_mean,
        to_centered=True,
    )
    if centered_dual_warm_start is not None:
        solver_options["dual_warm_start"] = centered_dual_warm_start

    transport, info = EuclideanProjector.project(
        centered_matrix,
        1.0 / float(n),
        max_iter=max_iter,
        relative_tol=float(relative_tol),
        **solver_options,
    )
    info = dict(info)
    centered_dual_state = info.get("dual_state")
    original_dual_state = _transform_dual_state(
        centered_dual_state,
        row_mean,
        col_mean,
        global_mean,
        to_centered=False,
    )
    if centered_dual_state is not None:
        # Keep the public warm-start state in the coordinates of the caller's X.
        info["dual_state"] = original_dual_state
    solver = info.get("solver")
    failed = solver == "projection_failure"
    sparse_schur = bool(solver_options.get("sparse_schur", False))
    certificate = primal_dual_certificate(
        matrix, transport, original_dual_state, tolerance=certificate_tolerance
    )
    feasibility_verified = bool(
        certificate["certificate_row_residual"] <= certificate_tolerance
        and certificate["certificate_col_residual"] <= certificate_tolerance
        and certificate["certificate_min_transport"] >= -certificate_tolerance
    )
    certificate_verified = bool(certificate["certificate_verified"])
    if failed or not feasibility_verified:
        status = "failed"
        success = False
    elif certificate_verified:
        status = "verified"
        success = True
    else:
        status = "feasible_only"
        success = True
    info.update(
        {
            "status": status,
            "success": success,
            "feasibility_verified": feasibility_verified,
            "optimality_status": certificate["certificate_status"],
            "optimality_verified": certificate_verified,
            "certificate_tolerance": certificate_tolerance,
            "solver_path": "experimental_sparse_schur" if sparse_schur else "dense_schur",
            "autograd_supported": False,
            "gauge_normalized": True,
        }
    )
    info.update(certificate)
    if status == "failed" and (raise_on_failure or not return_info):
        raise ProjectionFailure(
            "FASTNEWTON projection failed: "
            f"solver={solver}, "
            f"feasibility_verified={feasibility_verified}, "
            f"certificate_status={certificate['certificate_status']}"
        )
    if return_info:
        return transport, info
    return transport
