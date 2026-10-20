import unittest

import torch

from benchmarks.gw_like import generate_gw_like, gw_candidate


class GWLikeTests(unittest.TestCase):
    def test_candidate_matches_symmetric_stride_formula(self):
        sample = generate_gw_like(B=8, dim=12, mr=0.5, seed=5, dtype=torch.float64)
        candidate = gw_candidate(sample["transport"], sample["d_source"], sample["d_target"])
        expected = sample["transport"] + 4.0 * 0.05 * (
            sample["d_source"] @ sample["transport"] @ sample["d_target"]
        )
        self.assertTrue(torch.equal(candidate, expected))
        self.assertTrue(torch.allclose(sample["d_source"], sample["d_source"].t()))
        self.assertTrue(torch.allclose(sample["d_target"], sample["d_target"].t()))
