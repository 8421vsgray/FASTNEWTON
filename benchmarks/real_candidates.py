"""Build frozen candidate tensors from real STRIDE feature outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from .gw_like import candidate_sequence, gw_candidate


@torch.no_grad()
def prepare_real_features(query: torch.Tensor, key: torch.Tensor):
    """Prepare STRIDE-like feature outputs and the diagonal initialization."""
    if query.ndim != 2 or key.shape != query.shape:
        raise ValueError("query and key must have equal 2-D shapes")
    n = query.shape[0]
    q = torch.nn.functional.normalize(query, dim=1)
    k = torch.nn.functional.normalize(key, dim=1)
    d_source = (1.0 - q @ q.t()).clamp(0.0, 2.0)
    d_target = (1.0 - k @ k.t()).clamp(0.0, 2.0)
    d_source = 0.5 * (d_source + d_source.t())
    d_target = 0.5 * (d_target + d_target.t())
    transport = torch.eye(n, dtype=q.dtype, device=q.device) / float(n)
    return {"d_source": d_source, "d_target": d_target, "initial_transport": transport}


def candidates_from_reference_transports(prepared, transports, *, step_size: float = 0.05):
    """Build X_k from externally supplied reference transports."""
    return candidate_sequence(
        prepared["d_source"], prepared["d_target"], transports, step_size
    )


def first_candidate(prepared, *, step_size: float = 0.05):
    """Build X_0 without invoking FASTNEWTON."""
    return gw_candidate(
        prepared["initial_transport"],
        prepared["d_source"],
        prepared["d_target"],
        step_size,
    )


def save_candidate_bundle(bundle, path: Union[str, Path]):
    """Save a candidate bundle without committing it to Git automatically."""
    torch.save(bundle, Path(path))
