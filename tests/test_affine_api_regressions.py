import unittest
from unittest.mock import patch

import torch

from fastnewton import project
from fastnewton.projection import ProjectionFailure


def _affine_boundary_matrix(n, device):
    target = 1.0 / float(n)
    tolerance = 1.0e-3 * target
    negative_off_diagonal = -0.75 * tolerance
    matrix = torch.full(
        (n, n), negative_off_diagonal, dtype=torch.float64, device=device
    )
    matrix.diagonal().fill_(target - (n - 1) * negative_off_diagonal)
    return matrix


class AffineShortcutRegressionTests(unittest.TestCase):
    def _check_actual_marginals(self, n, device):
        matrix = _affine_boundary_matrix(n, device)
        transport, info = project(
            matrix, relative_tol=1.0e-3, max_iter=100, return_info=True
        )
        target = 1.0 / float(n)
        actual_residual = max(
            float((transport.sum(dim=1) - target).abs().amax()),
            float((transport.sum(dim=0) - target).abs().amax()),
        )
        actual_relative_residual = actual_residual / target
        self.assertNotEqual(info["status"], "failed", info)
        self.assertNotEqual(info["solver"], "affine", info)
        self.assertTrue(info["feasibility_verified"], info)
        self.assertGreaterEqual(float(transport.min()), -target * 1.0e-3)
        self.assertLessEqual(actual_relative_residual, 1.0e-3)
        self.assertAlmostEqual(
            float(info["relative_residual"]), actual_relative_residual, delta=1.0e-10
        )

    def test_clamped_affine_shortcut_rechecks_batch_three(self):
        self._check_actual_marginals(3, torch.device("cpu"))

    def test_small_negative_affine_candidate_does_not_take_affine_shortcut(self):
        n = 3
        target = 1.0 / n
        negative = 1.0e-5
        matrix = torch.full((n, n), -negative, dtype=torch.float64)
        matrix.diagonal().fill_(target + (n - 1) * negative)
        clipped = matrix.clamp_min(0.0)
        clipped_relative_residual = n * (clipped.sum(dim=1) - target).abs().amax()
        self.assertLess(float(clipped_relative_residual), 1.0e-3)

        transport, info = project(matrix, relative_tol=1.0e-3, return_info=True)
        expected = torch.eye(n, dtype=torch.float64) / float(n)
        self.assertEqual(info["solver"], "newton", info)
        self.assertIn(info["status"], {"verified", "feasible_only"}, info)
        actual_relative_residual = n * max(
            float((transport.sum(dim=1) - target).abs().amax()),
            float((transport.sum(dim=0) - target).abs().amax()),
        )
        self.assertLessEqual(actual_relative_residual, 1.0e-3)
        # The configured tolerance permits a small residual-level deviation;
        # the point of this case is that clipping is not reported as the
        # exact affine fast path.
        self.assertLessEqual(
            float((transport - expected).abs().amax()), 1.0e-3 * target
        )

    def test_nonnegative_affine_projection_keeps_fast_path(self):
        matrix = torch.eye(3, dtype=torch.float64) / 3.0
        transport, info = project(matrix, return_info=True)
        self.assertEqual(info["solver"], "affine", info)
        # The affine shortcut has no dual-state certificate, so it remains
        # conservatively classified as feasible-only.
        self.assertEqual(info["status"], "feasible_only", info)
        self.assertTrue(torch.equal(transport, matrix))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_clamped_affine_shortcut_rechecks_batches_512_and_1024(self):
        device = torch.device("cuda")
        for n in (512, 1024):
            with self.subTest(batch=n):
                self._check_actual_marginals(n, device)

    def test_final_failed_status_raises_unless_diagnostics_requested(self):
        matrix = torch.eye(3, dtype=torch.float64)
        invalid_transport = torch.zeros_like(matrix)
        fake_result = (invalid_transport, {"solver": "affine"})

        with patch("fastnewton.projection.EuclideanProjector.project", return_value=fake_result):
            _, info = project(matrix, return_info=True)
        self.assertEqual(info["status"], "failed", info)

        with patch("fastnewton.projection.EuclideanProjector.project", return_value=fake_result):
            with self.assertRaises(ProjectionFailure):
                project(matrix)

        with patch("fastnewton.projection.EuclideanProjector.project", return_value=fake_result):
            with self.assertRaises(ProjectionFailure):
                project(matrix, return_info=True, raise_on_failure=True)


if __name__ == "__main__":
    unittest.main()
