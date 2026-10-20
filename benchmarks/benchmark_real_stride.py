"""Compare FASTNEWTON and REBUILD's DROT adapter on frozen STRIDE-like data."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path("/home/dingyn/code/FASTNEWTON")
REBUILD = Path("/home/dingyn/code/REBUILD")
RUN_DIR = ROOT / "experiments/rebuild_scene15_mr08_e20_s1"
BUNDLE = RUN_DIR / "real_batch_000.pt"
OUT_JSON = RUN_DIR / "projection_comparison.json"
OUT_PT = RUN_DIR / "projection_comparison_tensors.pt"
ETA = 0.05
STEPS = 3


def candidate(transport: torch.Tensor, source_distance: torch.Tensor, target_distance: torch.Tensor):
    return transport + 4.0 * ETA * (source_distance @ transport @ target_distance)


def distances(query: torch.Tensor, key: torch.Tensor):
    q = torch.nn.functional.normalize(query, dim=1)
    k = torch.nn.functional.normalize(key, dim=1)
    ds = (1.0 - q @ q.t()).clamp(0.0, 2.0)
    dt = (1.0 - k @ k.t()).clamp(0.0, 2.0)
    return 0.5 * (ds + ds.t()), 0.5 * (dt + dt.t())


def timed(callable_):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = callable_()
    end.record()
    end.synchronize()
    return result, float(start.elapsed_time(end))


def plan_metrics(plan, x, marginal):
    row = (plan.sum(dim=1) - marginal).abs().max().item()
    col = (plan.sum(dim=0) - marginal).abs().max().item()
    return {
        "min": float(plan.min().item()),
        "row_abs": float(row),
        "col_abs": float(col),
        "objective": float(0.5 * (plan - x).square().sum().item()),
        "active_fraction": float((plan > 0).to(dtype=torch.float32).mean().item()),
    }


def fast_project(x, marginal, previous=None):
    sys.path.insert(0, str(ROOT / "src"))
    from fastnewton import project

    options = {"return_info": True}
    if previous is not None:
        options["dual_warm_start"] = previous["info"].get("dual_state")
        options["active_warm_start"] = previous["info"].get("active_state")
    return project(x, relative_tol=1e-3, max_iter=50, **options)


def drot_project(x, marginal, projector):
    result = projector.project(x, marginal, marginal)
    return result.plan, result.diagnostics


def summarize_fast(plan, info, x, marginal, elapsed_ms):
    result = plan_metrics(plan, x, marginal)
    result.update(
        {
            "solver": info.get("solver"),
            "status": info.get("status"),
            "success": bool(info.get("success", False)),
            "iterations": int(info.get("iterations", 0)),
            "relative_residual": float(info.get("relative_residual", float("inf"))),
            "optimality_status": info.get("optimality_status"),
            "certificate_gap": info.get("primal_dual_gap"),
            "elapsed_ms": elapsed_ms,
        }
    )
    return result


def summarize_drot(plan, diagnostics, x, marginal, elapsed_ms):
    result = plan_metrics(plan, x, marginal)
    result.update(
        {
            "solver": "drot",
            "status": diagnostics.status,
            "success": diagnostics.status == "success",
            "iterations": int(diagnostics.iterations),
            "relative_residual": float(max(diagnostics.row_relative_residual, diagnostics.col_relative_residual)),
            "optimality_status": "unverified",
            "elapsed_ms": elapsed_ms,
        }
    )
    return result


def run_fast_sequence(ds, dt, marginal, *, warm_start=False):
    transport = torch.eye(ds.shape[0], device=ds.device, dtype=ds.dtype) / ds.shape[0]
    previous = None
    records = []
    xs = []
    plans = []
    for step in range(STEPS):
        x = candidate(transport, ds, dt)
        xs.append(x.detach().clone())
        if warm_start and previous is not None:
            (plan, info), elapsed = timed(lambda: fast_project(x, marginal, previous))
        else:
            (plan, info), elapsed = timed(lambda: fast_project(x, marginal))
        records.append(summarize_fast(plan, info, x, marginal, elapsed))
        plans.append(plan.detach().clone())
        if not info.get("success", False) or info.get("status") == "failed":
            break
        transport = plan
        previous = {"info": info} if info.get("status") != "failed" else None
    return records, xs, plans


def run_drot_sequence(ds, dt, marginal, projector):
    transport = torch.eye(ds.shape[0], device=ds.device, dtype=ds.dtype) / ds.shape[0]
    records = []
    xs = []
    plans = []
    for step in range(STEPS):
        x = candidate(transport, ds, dt)
        xs.append(x.detach().clone())
        (plan, diagnostics), elapsed = timed(lambda: drot_project(x, marginal, projector))
        records.append(summarize_drot(plan, diagnostics, x, marginal, elapsed))
        plans.append(plan.detach().clone())
        if diagnostics.status != "success":
            break
        transport = plan
    return records, xs, plans


def run_fixed_inputs_fast(xs, marginal, *, warm_start=False):
    previous = None
    records = []
    plans = []
    for x in xs:
        if warm_start and previous is not None:
            (plan, info), elapsed = timed(lambda: fast_project(x, marginal, previous))
        else:
            (plan, info), elapsed = timed(lambda: fast_project(x, marginal))
        records.append(summarize_fast(plan, info, x, marginal, elapsed))
        plans.append(plan.detach().clone())
        previous = {"info": info} if info.get("status") != "failed" else None
    return records, plans


def run_fixed_inputs_drot(xs, marginal, projector):
    records = []
    plans = []
    for x in xs:
        (plan, diagnostics), elapsed = timed(lambda: drot_project(x, marginal, projector))
        records.append(summarize_drot(plan, diagnostics, x, marginal, elapsed))
        plans.append(plan.detach().clone())
    return records, plans


def main():
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=True)
    p = bundle["p"]
    z_t = bundle["z_t"]
    directions = {
        "P12": distances(p[0].to(device), z_t[1].to(device)),
        "P21": distances(p[1].to(device), z_t[0].to(device)),
    }
    n = p[0].shape[0]
    marginal = torch.full((n,), 1.0 / n, dtype=torch.float32, device=device)

    sys.path.insert(0, str(REBUILD))
    from solver.drot_adapter import DrotEuclideanProjector
    import reg_drot

    drot = DrotEuclideanProjector(
        rho=4.0,
        max_iterations=20_000,
        solver_eps=1e-6,
        relative_tolerance=1e-3,
        absolute_tolerance=1e-8,
        backend=reg_drot,
    )

    # Warm up both compiled/GPU paths before collecting timings.
    for ds, dt in directions.values():
        x0 = candidate(torch.eye(n, device=device) / n, ds, dt)
        timed(lambda: fast_project(x0, marginal))
        timed(lambda: drot_project(x0, marginal, drot))
    torch.cuda.synchronize()

    result = {
        "metadata": {
            "feature_bundle": str(BUNDLE),
            "rebuild_commit": "45fd5c25b75035435ed0bb8cdfe47f6c54b967d8",
            "gpu": torch.cuda.get_device_name(device),
            "device": str(device),
            "dtype": "float32",
            "batch_size": n,
            "eta": ETA,
            "steps": STEPS,
            "drot_rho": 4.0,
            "drot_solver_eps": 1e-6,
            "drot_max_iterations": 20_000,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "directions": {},
    }
    tensor_payload = {}

    for name, (ds, dt) in directions.items():
        cold_fast, fixed_xs, cold_fast_plans = run_fast_sequence(ds, dt, marginal)
        warm_fast, warm_plans = run_fixed_inputs_fast(fixed_xs, marginal, warm_start=True)
        fixed_drot, fixed_drot_plans = run_fixed_inputs_drot(fixed_xs, marginal, drot)
        full_drot, drot_xs, full_drot_plans = run_drot_sequence(ds, dt, marginal, drot)

        same_input = []
        for i in range(min(len(cold_fast_plans), len(fixed_drot_plans))):
            diff = cold_fast_plans[i] - fixed_drot_plans[i]
            same_input.append(
                {
                    "step": i,
                    "max_abs_plan_diff": float(diff.abs().max().item()),
                    "mean_abs_plan_diff": float(diff.abs().mean().item()),
                    "l1_plan_diff": float(diff.abs().sum().item()),
                    "max_abs_pseudolabel_diff": float((n * diff).abs().max().item()),
                }
            )

        result["directions"][name] = {
            "same_input": {
                "fastnewton_cold": cold_fast,
                "fastnewton_warm": warm_fast,
                "drot": fixed_drot,
                "plan_differences": same_input,
            },
            "full_gw": {
                "fastnewton": cold_fast,
                "drot": full_drot,
                "fastnewton_vs_drot_final": (
                    {
                        "max_abs_plan_diff": float((cold_fast_plans[-1] - full_drot_plans[-1]).abs().max().item()),
                        "l1_plan_diff": float((cold_fast_plans[-1] - full_drot_plans[-1]).abs().sum().item()),
                        "max_abs_pseudolabel_diff": float((n * (cold_fast_plans[-1] - full_drot_plans[-1])).abs().max().item()),
                    }
                    if cold_fast_plans and full_drot_plans
                    else None
                ),
            },
        }
        tensor_payload[name] = {
            "fixed_inputs": [x.cpu() for x in fixed_xs],
            "fastnewton_cold_plans": [x.cpu() for x in cold_fast_plans],
            "fastnewton_warm_plans": [x.cpu() for x in warm_plans],
            "drot_fixed_input_plans": [x.cpu() for x in fixed_drot_plans],
            "drot_full_gw_inputs": [x.cpu() for x in drot_xs],
            "drot_full_gw_plans": [x.cpu() for x in full_drot_plans],
        }

    torch.save(tensor_payload, OUT_PT)
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
