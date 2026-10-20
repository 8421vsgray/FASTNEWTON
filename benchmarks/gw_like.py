"""Synthetic GW-like candidate generation matching pinned STRIDE."""

from __future__ import annotations

from typing import Iterable, List, Union

import torch


def generate_gw_like(
    B: int = 32,
    dim: int = 64,
    mr: float = 0.8,
    seed: int = 1,
    *,
    dtype: torch.dtype = torch.float64,
    device: Union[str, torch.device] = "cpu",
):
    """Generate normalized two-view features and their symmetric distances."""
    if B < 2 or dim < 1:
        raise ValueError("B must be at least 2 and dim must be positive")
    if not 0.0 <= mr <= 1.0:
        raise ValueError("mr must lie in [0, 1]")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    clusters = min(15, B)
    labels = torch.randint(clusters, (B,), generator=generator)
    centers_q = torch.randn(clusters, dim, generator=generator, dtype=dtype)
    centers_k = torch.randn(clusters, dim, generator=generator, dtype=dtype)
    q = centers_q[labels] + 0.3 * torch.randn(B, dim, generator=generator, dtype=dtype)
    k = centers_k[labels] + 0.3 * torch.randn(B, dim, generator=generator, dtype=dtype)

    count = int(mr * B)
    indices = torch.randperm(B, generator=generator)[:count]
    permuted = indices[torch.randperm(count, generator=generator)] if count else indices
    permutation = torch.arange(B)
    if count:
        permutation[indices] = permutation[permuted]
    k = k[permutation]

    q = torch.nn.functional.normalize(q, dim=1).to(device=device, dtype=dtype)
    k = torch.nn.functional.normalize(k, dim=1).to(device=device, dtype=dtype)
    d_source = (1.0 - q @ q.t()).clamp(0.0, 2.0)
    d_target = (1.0 - k @ k.t()).clamp(0.0, 2.0)
    d_source = 0.5 * (d_source + d_source.t())
    d_target = 0.5 * (d_target + d_target.t())
    transport = torch.eye(B, dtype=dtype, device=device) / float(B)
    return {
        "q": q,
        "k": k,
        "labels": labels.to(device=device),
        "permutation": permutation.to(device=device),
        "d_source": d_source,
        "d_target": d_target,
        "transport": transport,
        "marginal": 1.0 / float(B),
        "mr": float(mr),
        "seed": int(seed),
    }


def gw_candidate(transport: torch.Tensor, d_source: torch.Tensor, d_target: torch.Tensor, step_size: float = 0.05):
    """Return the exact pinned-STRIDE candidate for symmetric distances."""
    if transport.ndim != 2 or transport.shape[0] != transport.shape[1]:
        raise ValueError("transport must be square")
    if d_source.shape != transport.shape or d_target.shape != transport.shape:
        raise ValueError("distance matrices and transport must have equal shapes")
    return transport + 4.0 * float(step_size) * (d_source @ transport @ d_target)


def candidate_sequence(
    d_source: torch.Tensor,
    d_target: torch.Tensor,
    transports: Iterable[torch.Tensor],
    step_size: float = 0.05,
) -> List[torch.Tensor]:
    """Build candidates from externally supplied reference transports."""
    return [gw_candidate(t, d_source, d_target, step_size) for t in transports]
