import unittest
from unittest import mock
import torch

from fastnewton import ProjectionFailure, project


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

    def test_outer_feasibility_failure_raises_without_info(self):
        x = torch.zeros(2, 2, dtype=torch.float64)
        infeasible = torch.zeros_like(x)
        fake_info = {"solver": "newton", "dual_state": (torch.zeros(2), torch.zeros(2))}
        with mock.patch(
            "fastnewton.projection.EuclideanProjector.project",
            return_value=(infeasible, fake_info),
        ):
            with self.assertRaisesRegex(ProjectionFailure, "feasibility_verified=False"):
                project(x)

    def test_return_info_exposes_outer_feasibility_failure(self):
        x = torch.zeros(2, 2, dtype=torch.float64)
        infeasible = torch.zeros_like(x)
        fake_info = {"solver": "newton", "dual_state": (torch.zeros(2), torch.zeros(2))}
        with mock.patch(
            "fastnewton.projection.EuclideanProjector.project",
            return_value=(infeasible, fake_info),
        ):
            result, info = project(x, return_info=True)
            self.assertIs(result, infeasible)
            self.assertEqual(info["status"], "failed")
            self.assertFalse(info["success"])
            self.assertFalse(info["feasibility_verified"])

    def test_raise_on_failure_overrides_return_info(self):
        x = torch.zeros(2, 2, dtype=torch.float64)
        infeasible = torch.zeros_like(x)
        fake_info = {"solver": "newton", "dual_state": (torch.zeros(2), torch.zeros(2))}
        with mock.patch(
            "fastnewton.projection.EuclideanProjector.project",
            return_value=(infeasible, fake_info),
        ):
            with self.assertRaises(ProjectionFailure):
                project(x, return_info=True, raise_on_failure=True)
