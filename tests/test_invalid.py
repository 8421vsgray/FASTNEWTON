import unittest
import torch

from fastnewton import project


class InvalidInputTests(unittest.TestCase):
    def test_invalid_matrix(self):
        cases = (
            (torch.ones(2, 3), ValueError, {}),
            (torch.empty(0, 0), ValueError, {}),
            (torch.ones(2, 2, dtype=torch.int64), TypeError, {}),
            (torch.tensor([[0.0, float("nan")], [0.0, 0.0]]), ValueError, {}),
            (torch.ones(2, 2), ValueError, {"max_iter": 0}),
        )
        for value, error, kwargs in cases:
            with self.subTest(error=error.__name__):
                with self.assertRaises(error):
                    project(value, **kwargs)

    def test_failure_is_not_success(self):
        x = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
        _, info = project(x, relative_tol=1.0e-12, max_iter=1, fallback_iters=1, return_info=True)
        self.assertEqual(info["solver"], "projection_failure")
        self.assertEqual(info["status"], "failed")
        self.assertFalse(info["success"])
        with self.assertRaises(RuntimeError):
            project(x, relative_tol=1.0e-12, max_iter=1, fallback_iters=1)
