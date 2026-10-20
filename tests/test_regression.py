import importlib.util
import unittest
from importlib.machinery import SourceFileLoader

import torch

from fastnewton import project


class RegressionTests(unittest.TestCase):
    def test_extracted_matches_pinned_github_source(self):
        source_path = "vendor/stride_projection.py.github"
        loader = SourceFileLoader("pinned_stride_projection", source_path)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for size in (2, 4, 8):
            torch.manual_seed(100 + size)
            x = torch.randn(size, size, dtype=torch.float64)
            expected, expected_info = module.EuclideanProjector.project(
                x, 1.0 / size, max_iter=50, relative_tol=1.0e-3
            )
            actual, actual_info = project(
                x, max_iter=50, relative_tol=1.0e-3, return_info=True
            )
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(actual_info["solver"], expected_info["solver"])
            self.assertEqual(actual_info["iterations"], expected_info["iterations"])
