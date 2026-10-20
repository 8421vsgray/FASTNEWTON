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
