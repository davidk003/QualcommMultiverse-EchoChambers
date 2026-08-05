"""Pure-numpy reference signature library -- no web framework dependency.

Shared by `deploy/cloud_ai100/signature_service.py` (the real FastAPI service,
importing this for its logic) and `signature_library.py` (the in-process
local fallback) so the fallback path doesn't need FastAPI installed just to
answer a similarity query. See `deploy/cloud_ai100/README.md` for why this is
a local reference implementation rather than a real Cloud AI 100 deployment.
"""

from __future__ import annotations

import numpy as np

from . import CLASSES
from .dataset import build_dataset

_LIBRARY: dict[str, np.ndarray] | None = None


def reference_library() -> dict[str, np.ndarray]:
    """One mean embedding per known covert-channel family (see module docstring)."""
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    X, y = build_dataset(n_per_class=200)
    _LIBRARY = {
        label: X[y == idx].mean(axis=0)
        for idx, label in enumerate(CLASSES)
        if label != "background"
    }
    return _LIBRARY


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))
