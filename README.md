# FASTNEWTON

FASTNEWTON is an independent PyTorch solver for the Euclidean projection

`min_T 0.5 * ||T - X||_F^2`

over nonnegative square matrices with uniform row and column marginals
`1 / B`. The first implementation extracts the semismooth Newton, active-set,
Schur-complement, line-search, ridge-retry, and finite Dykstra fallback logic
from the pinned STRIDE source without tuning it.

## Development installation

```bash
python -m pip install -e .
```

## Minimal use

```python
import torch
from fastnewton import project

X = torch.randn(512, 512, device="cuda")
T, info = project(X, relative_tol=1e-3, max_iter=50, return_info=True)
if not info["success"]:
    raise RuntimeError(info)
```

The input is not modified. The output stays on the input device. CPU and CUDA
are supported when the installed PyTorch build supports them. Autograd is not
supported in this first extraction (`info["autograd_supported"]` is false).

The public status has three values: `verified` means feasibility and the
primal--dual certificate both passed, `feasible_only` means the returned plan
is feasible but an optimality certificate is unavailable or did not pass, and
`failed` means the result must not be used as a projection. `success=True`
means that a usable feasible plan was returned; callers that require a
certified Euclidean projection must also require `status == "verified"` (or
`optimality_verified`). With `return_info=True`, a failure returns the
baseline defensive uniform tensor together with `status="failed"` and
`success=False`; without `return_info=True`, the public API raises
`ProjectionFailure`.

The optional `sparse_schur=True` path is explicitly reported as
`solver_path="experimental_sparse_schur"`; it remains subject to the same
post-solve feasibility and certificate checks.

## Verification

```bash
python -m unittest discover -s tests -v
```

The current tests cover the B=2 closed form, invalid inputs, failure-state
semantics, small-matrix feasibility, and CVXPY/OSQP comparisons on random and
synthetic GW-like candidates for B=2, 4, 8, 16, and 32. The CVXPY reference is
optional and is not a runtime dependency of FASTNEWTON.

Synthetic candidates follow the pinned STRIDE model exactly for symmetric
distance matrices:

`X = T + 4 * step_size * D_source @ T @ D_target`

The independent `kkt_diagnostics` helper reports feasibility and a conservative
dual KKT reconstruction. Disconnected active supports may remain
`unverified`, even when the primal solution agrees with CVXPY.

`primal_dual_certificate` uses the solver's dual potentials directly and does
not assume that the positive-support graph is connected. It verifies primal
feasibility, `T = [X-u-v]_+`, and the primal--dual gap. The public `info`
dictionary reports `optimality_status="verified"` only when this certificate
passes a stated numerical tolerance. This is an approximate floating-point KKT
check, not a strict mathematical proof. In particular, a tiny negative gap can
occur when the returned plan is only approximately marginal-feasible; it is
accepted only within the configured tolerance and must be interpreted with the
reported feasibility residuals.

The synthetic CUDA benchmark can be reproduced with:

```bash
PYTHONPATH=src:. python -m benchmarks.benchmark_gw_like \
  --B 512 --dim 64 --warmups 2 --repeats 5
```

No real STRIDE checkpoint is bundled in this repository. Real candidate
capture is therefore a separate data-ingestion step and is not silently
replaced by synthetic data.

An isolated workload harness is provided in
`benchmarks/collect_rebuild_features.py`. On the development server it ran
REBUILD's Scene15 diagonal-only epochs 0--19 (`m_ratio=0.8`, seed 1, B=512)
from REBUILD commit `45fd5c25b75035435ed0bb8cdfe47f6c54b967d8`, without
modifying REBUILD or saving a REBUILD checkpoint. The frozen feature metadata
is recorded in
`experiments/rebuild_scene15_mr08_e20_s1/real_batch_000.json`.

`benchmarks/benchmark_real_stride.py` then evaluates P12 and P21 for three GW
steps. The recorded same-input comparison completed all six FASTNEWTON and
DROT sequences successfully. The largest same-input plan difference was
`8.52e-7` (P12) and `6.88e-7` (P21); the largest corresponding pseudo-label
difference was `4.36e-4` and `3.52e-4`. On the occupied RTX 4090, FASTNEWTON
took roughly 62--103 ms per projection while the DROT adapter took roughly
6.7--8.7 s per projection. These are measurements under concurrent GPU load,
not a hardware-independent speedup claim. The full JSON record is
`experiments/rebuild_scene15_mr08_e20_s1/projection_comparison.json`; the
tensor bundle is intentionally Git-ignored. All solver calls returned a
successful feasibility status with no fallback; some float32 real-workload
steps remain `optimality_status="unverified"` because the absolute dual gap
was larger than the conservative certificate tolerance. They are not promoted
to certified optima by the harness.

The reproducible B=512 CVXPY artifact can be regenerated with:

```bash
PYTHONPATH=src:. python -m benchmarks.run_b512_reference
```

It writes metadata to `experiments/b512_cvxpy_mr05_20261020.json` and keeps the
large tensor bundle in the same directory as a Git-ignored `.pt` file.

## Provenance

See `vendor/BASELINE_PROVENANCE.md`. The formal source is
`8421vsgray/STRIDE` commit `12b6d3cd813d5573492c0fd22c700bd01e7b8783`.
