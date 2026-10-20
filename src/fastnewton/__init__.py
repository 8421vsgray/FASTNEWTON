"""FASTNEWTON: extracted GPU Euclidean transport projection."""

from .projection import ProjectionFailure, project
from .diagnostics import kkt_diagnostics, primal_dual_certificate

__all__ = ["ProjectionFailure", "kkt_diagnostics", "primal_dual_certificate", "project"]

__version__ = "0.1.0.dev0"
