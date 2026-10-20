import unittest

import torch

from benchmarks.cvxpy_reference import solve_reference
from benchmarks.gw_like import candidate_sequence, generate_gw_like, gw_candidate
from fastnewton import kkt_diagnostics, project


try:
    import cvxpy  # noqa: F401
except ImportError:
    CVXPY_AVAILABLE = False
else:
    CVXPY_AVAILABLE = True


@unittest.skipUnless(CVXPY_AVAILABLE, "CVXPY is not installed")
class CVXPYTests(unittest.TestCase):
    def _check_against_reference(self, matrix):
        reference, reference_value, status = solve_reference(matrix)
        actual, info = project(matrix, relative_tol=1.0e-6, max_iter=100, return_info=True)
        self.assertEqual(status, "optimal")
        self.assertTrue(info["success"], info)
        reference_t = torch.as_tensor(reference, dtype=matrix.dtype)
        max_error = (actual - reference_t).abs().max().item()
        actual_value = 0.5 * (actual - matrix).square().sum().item()
        self.assertLessEqual(max_error, 5.0e-5)
        # The Newton solve uses relative_tol=1e-6; the fixed CVXPY/OSQP
        # suite observed a worst absolute objective gap of 1.58e-6.
        self.assertLessEqual(abs(actual_value - reference_value), 2.0e-6)
        diagnosis = kkt_diagnostics(matrix, actual, tolerance=5.0e-5)
        self.assertTrue(diagnosis["feasibility_verified"], diagnosis)

    def test_random_and_gw_like_sizes(self):
        for size in (2, 4, 8, 16, 32):
            torch.manual_seed(700 + size)
            self._check_against_reference(torch.randn(size, size, dtype=torch.float64))
            for mr in (0.0, 0.8):
                sample = generate_gw_like(B=size, dim=16, mr=mr, seed=9000 + size, dtype=torch.float64)
                self._check_against_reference(
                    gw_candidate(sample["transport"], sample["d_source"], sample["d_target"])
                )

    def test_three_gw_steps_use_reference_transports(self):
        sample = generate_gw_like(B=8, dim=24, mr=0.5, seed=42, dtype=torch.float64)
        transports = []
        transport = sample["transport"]
        for _ in range(3):
            candidate = gw_candidate(transport, sample["d_source"], sample["d_target"])
            reference, _, _ = solve_reference(candidate)
            transport = torch.as_tensor(reference, dtype=torch.float64)
            transports.append(transport)
        for candidate in candidate_sequence(sample["d_source"], sample["d_target"], transports):
            self._check_against_reference(candidate)
