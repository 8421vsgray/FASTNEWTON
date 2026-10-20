import unittest

import torch

from fastnewton import kkt_diagnostics, primal_dual_certificate, project


class CertificateTests(unittest.TestCase):
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
