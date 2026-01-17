"""Utility package vendored from utils.zip.

Note: some modules inside this package may depend on optional coursework files
(e.g. `knn.py`). To keep the SVM lab self-contained we make these imports
optional, so importing `MetricsEstimator` works even if those extras are absent.
"""

from __future__ import annotations

from .metrics import MetricsEstimator, evaluate

__all__ = ["MetricsEstimator", "evaluate"]

# Optional helpers (may require additional coursework modules).
try:
    from .kernels import kernel_function  # type: ignore
except Exception:  # pragma: no cover
    kernel_function = None  # type: ignore
else:
    __all__.append("kernel_function")

try:
    from .anchors import select_anchors  # type: ignore
except Exception:  # pragma: no cover
    select_anchors = None  # type: ignore
else:
    __all__.append("select_anchors")

try:
    from .grid_search import LOO_grid_search  # type: ignore
except Exception:  # pragma: no cover
    LOO_grid_search = None  # type: ignore
else:
    __all__.append("LOO_grid_search")
