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
