import unittest
import torch

from fastnewton import project


class AnalyticTests(unittest.TestCase):
    def test_b2_matches_closed_form(self):
        x = torch.tensor([[0.7, -0.2], [0.1, 0.4]], dtype=torch.float64)
        t, info = project(x, relative_tol=1.0e-10, max_iter=100, return_info=True)
        a = torch.clamp((1.0 + x[0, 0] + x[1, 1] - x[0, 1] - x[1, 0]) / 4.0, 0.0, 0.5)
        expected = torch.stack((torch.stack((a, 0.5 - a)), torch.stack((0.5 - a, a))))
        self.assertTrue(info["success"])
        self.assertTrue(torch.allclose(t, expected, atol=1.0e-8, rtol=1.0e-8))
