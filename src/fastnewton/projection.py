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


def _centered_dual_state_to_original(
    dual_state,
    row_mean: torch.Tensor,
    col_mean: torch.Tensor,
    global_mean: torch.Tensor,
):
    """Map dual potentials from a centered retry back to the caller's X."""
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
    # Xc = X - row - col + global, so uc=u-row+global and vc=v-col.
    # This is the inverse map, preserving X-u-v = Xc-uc-vc.
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

    The extracted STRIDE algorithm is used on the caller's matrix first. If
    the final public status is ``failed``, the wrapper retries once
    after removing row and column potentials, which preserve the exact
    projection for fixed uniform marginals. Retry warm-start state is cleared.
    The operation is intentionally non-differentiable in this first release.
    A failed result raises by default when diagnostics are not requested. With
    ``return_info=True``, a failed result is returned for inspection unless
    ``raise_on_failure=True``.
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
    transport, info = EuclideanProjector.project(
        matrix,
        1.0 / float(n),
        max_iter=max_iter,
        relative_tol=float(relative_tol),
        **solver_options,
    )
    sparse_schur = bool(solver_options.get("sparse_schur", False))

    def finalize(candidate, solver_info, *, gauge_retry=False):
        solver_info = dict(solver_info)
        certificate = primal_dual_certificate(
            matrix,
            candidate,
            solver_info.get("dual_state"),
            tolerance=certificate_tolerance,
        )
        feasibility_verified = bool(
            certificate["certificate_row_residual"] <= certificate_tolerance
            and certificate["certificate_col_residual"] <= certificate_tolerance
            and certificate["certificate_min_transport"] >= -certificate_tolerance
        )
        certificate_verified = bool(certificate["certificate_verified"])
        failed = solver_info.get("solver") == "projection_failure"
        if failed or not feasibility_verified:
            status, success = "failed", False
        elif certificate_verified:
            status, success = "verified", True
        else:
            status, success = "feasible_only", True
        solver_info.update(
            {
                "status": status,
                "success": success,
                "feasibility_verified": feasibility_verified,
                "optimality_status": certificate["certificate_status"],
                "optimality_verified": certificate_verified,
                "certificate_tolerance": certificate_tolerance,
                "solver_path": "experimental_sparse_schur" if sparse_schur else "dense_schur",
                "autograd_supported": False,
                "gauge_normalization_retry": gauge_retry,
                "gauge_normalized": gauge_retry,
            }
        )
        solver_info.update(certificate)
        return candidate, solver_info

    transport, info = finalize(transport, info)

    # Preserve the established raw-input path for both verified and
    # feasible-only results. Normalize only after an explicit failed status,
    # and never reuse dual/active state in the retry: those states belong to
    # the original coordinate system and may be stale after centering.
    initial_status = info["status"]
    initial_solver = info.get("solver")
    initial_iterations = info.get("iterations")
    info.update(
        {
            "recovery_attempted": False,
            "recovery_succeeded": False,
            "original_status": initial_status,
            "retry_status": None,
            "gauge_normalization_retry": False,
            "gauge_normalized": False,
        }
    )
    if initial_status == "failed":
        raw_info = info
        centered_matrix, row_mean, col_mean, global_mean = (
            _gauge_normalize_uniform_transport_matrix(matrix)
        )
        retry_options = dict(solver_options)
        retry_options.pop("dual_warm_start", None)
        retry_options.pop("active_warm_start", None)
        retry_transport, retry_info = EuclideanProjector.project(
            centered_matrix,
            1.0 / float(n),
            max_iter=max_iter,
            relative_tol=float(relative_tol),
            **retry_options,
        )
        centered_dual_state = retry_info.get("dual_state")
        original_dual_state = _centered_dual_state_to_original(
            centered_dual_state, row_mean, col_mean, global_mean
        )
        if centered_dual_state is not None:
            retry_info["dual_state"] = original_dual_state
        retry_transport, retry_info = finalize(
            retry_transport, retry_info, gauge_retry=True
        )
        retry_info.update(
            {
                "recovery_attempted": True,
                "recovery_succeeded": retry_info["status"] != "failed",
                "original_solver": initial_solver,
                "original_status": initial_status,
                "original_iterations": initial_iterations,
                "retry_status": retry_info["status"],
                "retry_solver": retry_info.get("solver"),
                "gauge_retry_iterations": retry_info.get("iterations"),
                "gauge_retry_warm_started": False,
                "gauge_normalization_retry": True,
                "gauge_normalized": True,
            }
        )
        transport, info = retry_transport, retry_info

    if info["status"] == "failed" and (raise_on_failure or not return_info):
        raise ProjectionFailure(
            "FASTNEWTON projection failed: "
            f"solver={info.get('solver')}, "
            f"feasibility_verified={info.get('feasibility_verified')}, "
            f"certificate_status={info.get('certificate_status')}, "
            f"recovery_attempted={info.get('recovery_attempted')}, "
            f"original_status={info.get('original_status')}, "
            f"retry_status={info.get('retry_status')}"
        )
    if return_info:
        return transport, info
    return transport
