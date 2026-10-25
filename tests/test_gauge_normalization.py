import json
import unittest
from pathlib import Path

import torch

from fastnewton import primal_dual_certificate, project
from fastnewton._stride_projection import EuclideanProjector


FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_TENSORS = FIXTURE_DIR / "scene15_mr00_seed1_eta003_k5_failures.pt"
FIXTURE_METADATA = FIXTURE_DIR / "scene15_mr00_seed1_eta003_k5_failures.json"


def _devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    return devices


class GaugeNormalizationTests(unittest.TestCase):
    def test_row_column_potentials_preserve_projection(self):
        for device in _devices():
            for dtype in (torch.float32, torch.float64):
                with self.subTest(device=str(device), dtype=str(dtype)):
                    torch.manual_seed(20261025)
                    x = torch.randn(16, 16, dtype=dtype, device=device)
                    row_potential = torch.randn(16, dtype=dtype, device=device)
                    col_potential = torch.randn(16, dtype=dtype, device=device)
                    shifted = x + row_potential[:, None] + col_potential[None, :] + 8.0
                    double_centered = (
                        x
                        - x.mean(dim=1, keepdim=True)
                        - x.mean(dim=0, keepdim=True)
                        + x.mean()
                    )
                    variants = {
                        "raw": x,
                        "global_centered": x - x.mean(),
                        "row_column_shifted": shifted,
                        "double_centered": double_centered,
                    }
                    solved = {}
                    for name, candidate in variants.items():
                        transport, info = project(
                            candidate,
                            relative_tol=1.0e-5 if dtype == torch.float64 else 1.0e-3,
                            max_iter=100,
                            return_info=True,
                        )
                        self.assertEqual(info["status"], "verified", (name, info))
                        self.assertTrue(info["gauge_normalized"])
                        certificate = primal_dual_certificate(
                            candidate,
                            transport,
                            info.get("dual_state"),
                            tolerance=info["certificate_tolerance"],
                        )
                        self.assertTrue(certificate["certificate_verified"], certificate)
                        solved[name] = transport

                    tolerance = 5.0e-5 if dtype == torch.float32 else 2.0e-7
                    reference = solved["raw"]
                    for name, transport in solved.items():
                        self.assertLessEqual(
                            float((reference - transport).abs().max().item()), tolerance,
                            name,
                        )

    def test_large_row_column_potentials_preserve_projection_and_input(self):
        for device in _devices():
            with self.subTest(device=str(device)):
                torch.manual_seed(20261026)
                x = 0.25 * torch.randn(24, 24, dtype=torch.float64, device=device)
                row_potential = torch.linspace(
                    -2.0e3, 2.0e3, 24, dtype=x.dtype, device=device
                )
                col_potential = torch.linspace(
                    1.0e3, -1.0e3, 24, dtype=x.dtype, device=device
                )
                shifted = (
                    x + row_potential[:, None] + col_potential[None, :] + 1.0e6
                )
                original_copy = shifted.clone()

                expected, expected_info = project(
                    x, relative_tol=1.0e-8, max_iter=100, return_info=True
                )
                actual, actual_info = project(
                    shifted, relative_tol=1.0e-8, max_iter=100, return_info=True
                )
                self.assertEqual(expected_info["status"], "verified", expected_info)
                self.assertEqual(actual_info["status"], "verified", actual_info)
                self.assertTrue(torch.equal(shifted, original_copy))
                self.assertLessEqual(
                    float((expected - actual).abs().amax().item()), 2.0e-7
                )
                self.assertLessEqual(
                    actual_info["certificate_row_residual"],
                    actual_info["certificate_tolerance"],
                )
                self.assertLessEqual(
                    actual_info["certificate_col_residual"],
                    actual_info["certificate_tolerance"],
                )

    @unittest.skipUnless(torch.cuda.is_available(), "captured regression is CUDA-specific")
    def test_raw_captured_scene15_candidates_reproduce_pre_normalization_failure(self):
        fixture = torch.load(FIXTURE_TENSORS, map_location="cuda:0", weights_only=False)
        for case in fixture["cases"]:
            with self.subTest(failure_index=case["failure_index"]):
                raw, raw_info = EuclideanProjector.project(
                    case["candidate"].to(device="cuda:0"),
                    1.0 / 512.0,
                    max_iter=200,
                    relative_tol=1.0e-3,
                )
                del raw
                self.assertEqual(raw_info["solver"], "projection_failure", raw_info)

    def test_captured_scene15_failures_are_verified_and_equivalent(self):
        if not FIXTURE_TENSORS.is_file() or not FIXTURE_METADATA.is_file():
            self.fail("the captured Scene15 failure fixture is missing")
        metadata = json.loads(FIXTURE_METADATA.read_text(encoding="utf-8"))
        self.assertEqual(metadata["dataset"], "Scene15")
        self.assertEqual(metadata["mismatch_ratio"], 0.0)
        self.assertEqual(metadata["seed"], 1)
        self.assertEqual(metadata["batch_size"], 512)
        self.assertEqual(len(metadata["cases"]), 3)
        fixture = torch.load(FIXTURE_TENSORS, map_location="cpu", weights_only=False)
        self.assertEqual(len(fixture["cases"]), 3)

        for case, provenance in zip(fixture["cases"], metadata["cases"]):
            with self.subTest(failure_index=case["failure_index"]):
                self.assertEqual(case["fastnewton_info"]["solver"], "projection_failure")
                self.assertEqual(provenance["solver"], "projection_failure")
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                original = case["candidate"].to(device=device, dtype=torch.float32)
                equivalent = (
                    original
                    - original.mean(dim=1, keepdim=True)
                    - original.mean(dim=0, keepdim=True)
                    + original.mean()
                )

                actual, info = project(
                    original, relative_tol=1.0e-3, max_iter=200, return_info=True
                )
                reference, reference_info = project(
                    equivalent, relative_tol=1.0e-3, max_iter=200, return_info=True
                )
                self.assertEqual(info["status"], "verified", info)
                self.assertEqual(reference_info["status"], "verified", reference_info)
                self.assertTrue(info["certificate_verified"], info)
                self.assertTrue(reference_info["certificate_verified"], reference_info)
                self.assertTrue(info["feasibility_verified"], info)
                self.assertLessEqual(info["iterations"], 200)

                max_t_error = float((actual - reference).abs().max().item())
                fro_t_error = float(torch.linalg.vector_norm(actual - reference).item())
                max_q_error = float((512.0 * (actual - reference)).abs().max().item())
                self.assertLessEqual(max_t_error, 3.0e-6)
                self.assertLessEqual(fro_t_error, 1.0e-5)
                self.assertLessEqual(max_q_error, 1.5e-3)

                actual_objective = 0.5 * (actual - original).double().square().sum()
                reference_objective = 0.5 * (reference - original).double().square().sum()
                self.assertLessEqual(
                    abs(float(actual_objective - reference_objective)), 5.0e-5
                )
                certificate = primal_dual_certificate(
                    original,
                    actual,
                    info.get("dual_state"),
                    tolerance=info["certificate_tolerance"],
                )
                self.assertTrue(certificate["certificate_verified"], certificate)

    def test_dual_warm_start_round_trips_through_gauge(self):
        for device in _devices():
            with self.subTest(device=str(device)):
                torch.manual_seed(8128)
                x = torch.randn(12, 12, dtype=torch.float64, device=device)
                base, base_info = project(x, relative_tol=1.0e-7, max_iter=100, return_info=True)
                shifted = x + torch.linspace(-2.0, 2.0, 12, dtype=x.dtype, device=device)[:, None]
                warm, warm_info = project(
                    shifted,
                    relative_tol=1.0e-7,
                    max_iter=100,
                    dual_warm_start=base_info.get("dual_state"),
                    active_warm_start=base_info.get("active_state"),
                    return_info=True,
                )
                self.assertEqual(base_info["status"], "verified", base_info)
                self.assertEqual(warm_info["status"], "verified", warm_info)
                self.assertTrue(warm_info["warm_started"], warm_info)
                self.assertLessEqual(float((base - warm).abs().max().item()), 2.0e-7)
                certificate = primal_dual_certificate(
                    shifted,
                    warm,
                    warm_info.get("dual_state"),
                    tolerance=warm_info["certificate_tolerance"],
                )
                self.assertTrue(certificate["certificate_verified"], certificate)


if __name__ == "__main__":
    unittest.main()
