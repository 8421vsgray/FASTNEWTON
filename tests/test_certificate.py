import unittest

import torch

from fastnewton import kkt_diagnostics, primal_dual_certificate, project
from fastnewton.projection import _default_certificate_tolerance


class CertificateTests(unittest.TestCase):
    def test_certificate_tolerance_scales_with_batch_size(self):
        self.assertEqual(_default_certificate_tolerance(1024, 1.0e-3), 1.0e-3 / 1024.0)
        self.assertLess(_default_certificate_tolerance(1024, 1.0e-3), 1.0e-6)

    def test_primal_dual_certificate_does_not_require_connected_support(self):
        torch.manual_seed(704)
        x = torch.randn(4, 4, dtype=torch.float64)
        t, info = project(x, relative_tol=1.0e-6, max_iter=100, return_info=True)
        certificate = primal_dual_certificate(x, t, info["dual_state"], tolerance=1.0e-6)
        self.assertTrue(certificate["certificate_verified"], certificate)
        self.assertLessEqual(abs(certificate["primal_dual_gap"]), 1.0e-6)
        self.assertEqual(info["optimality_status"], "verified")

    def test_support_diagnostic_can_remain_conservative(self):
        torch.manual_seed(704)
        x = torch.randn(4, 4, dtype=torch.float64)
        t, _ = project(x, relative_tol=1.0e-6, max_iter=100, return_info=True)
        diagnosis = kkt_diagnostics(x, t, tolerance=1.0e-6)
        self.assertTrue(diagnosis["feasibility_verified"])

    def test_fallback_is_feasible_only_and_reports_its_iterations(self):
        torch.manual_seed(9100)
        x = torch.randn(8, 8, dtype=torch.float64)
        _, info = project(
            x,
            relative_tol=1.0e-8,
            max_iter=1,
            retry_with_stronger_ridge=False,
            fallback_iters=10_000,
            return_info=True,
        )
        self.assertEqual(info["solver"], "fallback")
        self.assertEqual(info["status"], "feasible_only")
        self.assertTrue(info["success"])
        self.assertFalse(info["optimality_verified"])
        self.assertGreater(info["iterations"], 1)

    def test_failed_projection_does_not_expose_old_dual_state(self):
        torch.manual_seed(704)
        x = torch.randn(8, 8, dtype=torch.float64)
        _, good = project(x, relative_tol=1.0e-6, max_iter=100, return_info=True)
        _, failed = project(
            torch.tensor([[1.0, -1.0], [-1.0, 1.0]]),
            relative_tol=1.0e-12,
            max_iter=1,
            fallback_iters=1,
            dual_warm_start=good["dual_state"],
            active_warm_start=good["active_state"][:2, :2],
            return_info=True,
        )
        self.assertEqual(failed["status"], "failed")
        self.assertFalse(failed["success"])
        self.assertNotIn("dual_state", failed)

    def test_warm_start_preserves_projection(self):
        torch.manual_seed(20261020)
        x = torch.randn(8, 8, dtype=torch.float64)
        _, base = project(x, relative_tol=1.0e-8, max_iter=100, return_info=True)
        perturbed = x + 0.001 * torch.randn_like(x)
        cold, cold_info = project(
            perturbed, relative_tol=1.0e-8, max_iter=100, return_info=True
        )
        warm, warm_info = project(
            perturbed,
            relative_tol=1.0e-8,
            max_iter=100,
            dual_warm_start=base["dual_state"],
            active_warm_start=base["active_state"],
            return_info=True,
        )
        self.assertTrue(warm_info["warm_started"])
        self.assertTrue(cold_info["success"], cold_info)
        self.assertTrue(warm_info["success"], warm_info)
        self.assertTrue(torch.allclose(warm, cold, atol=1.0e-7, rtol=1.0e-7))
