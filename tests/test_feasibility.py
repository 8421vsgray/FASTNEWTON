import unittest
import torch

from fastnewton import project


class FeasibilityTests(unittest.TestCase):
    def test_small_random_feasibility(self):
        torch.manual_seed(11)
        x = torch.randn(8, 8, dtype=torch.float64)
        t, info = project(x, relative_tol=1.0e-8, max_iter=100, return_info=True)
        self.assertTrue(info["success"])
        self.assertGreaterEqual(t.min().item(), -1.0e-10)
        self.assertLessEqual(torch.max(torch.abs(t.sum(0) - 1.0 / 8)).item(), 1.0e-8)
        self.assertLessEqual(torch.max(torch.abs(t.sum(1) - 1.0 / 8)).item(), 1.0e-8)

    def test_float32_and_float64_paths(self):
        for dtype in (torch.float32, torch.float64):
            torch.manual_seed(123)
            x = torch.randn(8, 8, dtype=dtype)
            t, info = project(x, relative_tol=1.0e-3, max_iter=100, return_info=True)
            self.assertTrue(info["success"], info)
            self.assertTrue(info["feasibility_verified"], info)
            self.assertTrue(torch.isfinite(t).all())

    def test_sparse_schur_path_is_explicitly_marked(self):
        torch.manual_seed(704)
        x = torch.randn(8, 8, dtype=torch.float64)
        t, info = project(
            x,
            relative_tol=1.0e-8,
            max_iter=100,
            sparse_schur=True,
            return_info=True,
        )
        self.assertEqual(info["solver_path"], "experimental_sparse_schur")
        self.assertTrue(info["success"], info)
        self.assertTrue(info["feasibility_verified"], info)
        self.assertLessEqual(torch.abs(t.sum(0) - 1.0 / 8).max().item(), 1.0e-8)
        self.assertLessEqual(torch.abs(t.sum(1) - 1.0 / 8).max().item(), 1.0e-8)
