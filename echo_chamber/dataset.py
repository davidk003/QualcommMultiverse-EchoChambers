"""Build a labeled spectral-feature dataset from the synthetic signal generators."""

from __future__ import annotations

import numpy as np

from . import CLASSES, N_FFT, SAMPLE_RATE_HZ
from .dsp import extract_band_features
from .synth import default_rng, make_example


def build_dataset(
    n_per_class: int = 400,
    sr: int = SAMPLE_RATE_HZ,
    n_fft: int = N_FFT,
    seed: int = 20260803,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, y): X is (N, n_bands) float32 features, y is (N,) int labels."""
    rng = default_rng(seed)
    feats: list[np.ndarray] = []
    labels: list[int] = []
    for class_idx, label in enumerate(CLASSES):
        for _ in range(n_per_class):
            waveform = make_example(label, n_fft, sr, rng)
            feats.append(extract_band_features(waveform, sr=sr, n_fft=n_fft))
            labels.append(class_idx)
    X = np.stack(feats).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def train_val_split(X: np.ndarray, y: np.ndarray, val_frac: float = 0.2):
    n_val = int(len(y) * val_frac)
    return X[n_val:], y[n_val:], X[:n_val], y[:n_val]
