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

`projection_failure` is never reported as success. With `return_info=True`, a
failure returns the baseline defensive uniform tensor together with
`status="failed"` and `success=False`; without `return_info=True`, the public
API raises `ProjectionFailure`.

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

The synthetic CUDA benchmark can be reproduced with:

```bash
PYTHONPATH=src:. python -m benchmarks.benchmark_gw_like \
  --B 512 --dim 64 --warmups 2 --repeats 5
```

No real STRIDE checkpoint is bundled in this repository. Real candidate
capture is therefore a separate data-ingestion step and is not silently
replaced by synthetic data.

## Provenance

See `vendor/BASELINE_PROVENANCE.md`. The formal source is
`8421vsgray/STRIDE` commit `12b6d3cd813d5573492c0fd22c700bd01e7b8783`.
