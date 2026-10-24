"""Independent primal and KKT diagnostics for Euclidean projection results."""

from __future__ import annotations

import torch


@torch.no_grad()
def kkt_diagnostics(matrix: torch.Tensor, transport: torch.Tensor, *, active_tol: float = 1.0e-10, tolerance: float = 1.0e-7):
    """Check feasibility and a reconstructed dual KKT residual."""
    if matrix.shape != transport.shape or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix and transport must be square tensors of equal shape")
    n = matrix.shape[0]
    target = matrix.new_tensor(1.0 / float(n))
    row_residual = (transport.sum(1) - target).abs().amax()
    col_residual = (transport.sum(0) - target).abs().amax()
    min_transport = transport.amin()
    active = transport > float(active_tol)
    row_degree = active.sum(1)
    valid_support = bool((row_degree > 0).all())
    dual_residual = matrix.new_tensor(float("inf"))
    complementarity_residual = matrix.new_tensor(float("inf"))
    if valid_support:
        active_f = active.to(dtype=matrix.dtype)
        rhs_edges = (matrix - transport) * active_f
        row_degree_f = row_degree.to(dtype=matrix.dtype)
        inv_row_degree = row_degree_f.reciprocal()
        row_rhs = rhs_edges.sum(1)
        col_rhs = rhs_edges[:, :-1].sum(0)
        block = active_f[:, :-1]
        col_degree = active_f[:, :-1].sum(0)
        schur = torch.diag(col_degree) - (block.t() * inv_row_degree[None, :]) @ block
        schur = schur + 1.0e-10 * torch.eye(n - 1, dtype=matrix.dtype, device=matrix.device)
        rhs = col_rhs - block.t() @ (inv_row_degree * row_rhs)
        try:
            v_short = torch.linalg.solve(schur, rhs)
            u = inv_row_degree * (row_rhs - block @ v_short)
            v = torch.cat((v_short, matrix.new_zeros(1)))
            dual_slack = matrix - u[:, None] - v[None, :]
            dual_residual = ((transport - dual_slack).abs() * active).amax()
            complementarity_residual = dual_slack.clamp_min(0.0).masked_fill(active, 0.0).amax()
        except RuntimeError:
            pass
    feasible = bool(
        (row_residual <= tolerance)
        and (col_residual <= tolerance)
        and (min_transport >= -tolerance)
    )
    kkt_verified = bool(
        feasible
        and valid_support
        and dual_residual <= tolerance
        and complementarity_residual <= tolerance
    )
    return {
        "row_residual": float(row_residual.cpu()),
        "col_residual": float(col_residual.cpu()),
        "min_transport": float(min_transport.cpu()),
        "dual_stationarity_residual": float(dual_residual.cpu()),
        "inactive_complementarity_residual": float(complementarity_residual.cpu()),
        "feasibility_verified": feasible,
        "kkt_verified": kkt_verified,
        "status": "verified" if kkt_verified else "unverified",
    }


@torch.no_grad()
def primal_dual_certificate(
    matrix: torch.Tensor,
    transport: torch.Tensor,
    dual_state,
    *,
    tolerance: float = 1.0e-6,
):
    """Compute a primal--dual gap certificate without support connectivity.

    For dual potentials ``u, v``, the dual lower bound is

    ``0.5*||X||^2 - 0.5*||[X-u-v]_+||^2 - m*(sum(u)+sum(v))``.

    The certificate is valid only when the returned transport is primal
    feasible and agrees with the positive-part dual minimizer. No graph
    decomposition or single-component gauge assumption is used.
    """
    if matrix.shape != transport.shape or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix and transport must be square tensors of equal shape")
    n = matrix.shape[0]
    x64 = matrix.detach().to(dtype=torch.float64)
    t64 = transport.detach().to(dtype=torch.float64)
    marginal = 1.0 / float(n)
    target64 = x64.new_tensor(marginal)
    row_deficit = target64 - t64.sum(dim=1)
    col_deficit = target64 - t64.sum(dim=0)
    result = {
        "certificate_status": "unavailable",
        "certificate_verified": False,
        "primal_objective": float("nan"),
        "dual_lower_bound": float("nan"),
        "primal_dual_gap": float("nan"),
        "relative_primal_dual_gap": float("nan"),
        "dual_projection_residual": float("inf"),
        "certificate_row_residual": float(row_deficit.abs().amax().cpu()),
        "certificate_col_residual": float(col_deficit.abs().amax().cpu()),
        "certificate_min_transport": float(t64.amin().cpu()),
    }
    if not isinstance(dual_state, (tuple, list)) or len(dual_state) != 2:
        return result
    u, v = dual_state
    if tuple(u.shape) != (n,) or tuple(v.shape) != (n,) or u.device != matrix.device or v.device != matrix.device:
        return result
    if not bool(torch.isfinite(u).all() and torch.isfinite(v).all()):
        return result
    u64 = u.detach().to(dtype=torch.float64)
    v64 = v.detach().to(dtype=torch.float64)
    z = x64 - u64[:, None] - v64[None, :]
    dual_transport = z.clamp_min(0.0)
    inactive_slack = (-z).clamp_min(0.0)
    primal_objective = 0.5 * (t64 - x64).square().sum()
    gap = (
        0.5 * (t64 - dual_transport).square().sum()
        + (t64 * inactive_slack).sum()
        + torch.dot(row_deficit, u64)
        + torch.dot(col_deficit, v64)
    )
    dual_lower_bound = primal_objective - gap
    primal_feasible = bool(
        result["certificate_row_residual"] <= tolerance
        and result["certificate_col_residual"] <= tolerance
        and result["certificate_min_transport"] >= -tolerance
    )
    dual_projection_residual = (t64 - dual_transport).abs().amax()
    gap_value = float(gap.cpu())
    result.update(
        {
            "certificate_status": "verified"
            if primal_feasible
            and float(dual_projection_residual.cpu()) <= tolerance
            and gap_value <= tolerance
            and gap_value >= -10.0 * tolerance
            else "unverified",
            "certificate_verified": bool(
                primal_feasible
                and float(dual_projection_residual.cpu()) <= tolerance
                and gap_value <= tolerance
                and gap_value >= -10.0 * tolerance
            ),
            "primal_objective": float(primal_objective.cpu()),
            "dual_lower_bound": float(dual_lower_bound.cpu()),
            "primal_dual_gap": gap_value,
            "relative_primal_dual_gap": gap_value / max(abs(float(primal_objective.cpu())), 1.0e-12),
            "dual_projection_residual": float(dual_projection_residual.cpu()),
        }
    )
    return result
