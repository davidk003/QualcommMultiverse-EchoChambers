"""FFT feature extraction for the ultrasonic band.

Shared by training (echo_chamber/dataset.py) and the live capture pipeline
(echo_chamber/capture.py) so the exact feature contract the NPU model was
trained on is also what streaming inference feeds it.
"""

from __future__ import annotations

import numpy as np

from . import BAND_HZ, N_BANDS, N_FFT, SAMPLE_RATE_HZ


def _band_bin_edges(sr: int, n_fft: int, band_hz: tuple[int, int], n_bands: int) -> np.ndarray:
    lo, hi = band_hz
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    lo_bin = int(np.searchsorted(freqs, lo))
    hi_bin = int(np.searchsorted(freqs, hi))
    hi_bin = max(hi_bin, lo_bin + n_bands)  # guarantee enough bins to pool
    hi_bin = min(hi_bin, len(freqs))
    return np.linspace(lo_bin, hi_bin, n_bands + 1).astype(int)


_EDGES = _band_bin_edges(SAMPLE_RATE_HZ, N_FFT, BAND_HZ, N_BANDS)


def extract_band_features(
    frame: np.ndarray,
    sr: int = SAMPLE_RATE_HZ,
    n_fft: int = N_FFT,
    band_hz: tuple[int, int] = BAND_HZ,
    n_bands: int = N_BANDS,
) -> np.ndarray:
    """Log-magnitude spectrum of `frame`, pooled into `n_bands` bands over `band_hz`.

    Returns a (n_bands,) float32 vector, roughly zero-mean / unit-scale so it
    feeds a small NPU classifier without a separate normalization layer.
    """
    if len(frame) < n_fft:
        frame = np.pad(frame, (0, n_fft - len(frame)))
    else:
        frame = frame[:n_fft]

    windowed = frame * np.hanning(n_fft)
    spectrum = np.abs(np.fft.rfft(windowed, n=n_fft))
    log_mag = np.log1p(spectrum)

    edges = _EDGES if (sr, n_fft, band_hz, n_bands) == (SAMPLE_RATE_HZ, N_FFT, BAND_HZ, N_BANDS) \
        else _band_bin_edges(sr, n_fft, band_hz, n_bands)

    pooled = np.empty(n_bands, dtype=np.float32)
    for i in range(n_bands):
        lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
        pooled[i] = log_mag[lo:hi].mean()

    mean, std = pooled.mean(), pooled.std() + 1e-6
    return ((pooled - mean) / std).astype(np.float32)


def frame_rms(frame: np.ndarray) -> float:
    """Raw-waveform RMS -- used as a silence gate before feature extraction (see __init__.SILENCE_RMS_FLOOR)."""
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))


def frame_generator(waveform: np.ndarray, n_fft: int = N_FFT, hop: int | None = None):
    """Yield successive overlapping frames from a longer waveform (streaming use)."""
    hop = hop or n_fft // 2
    for start in range(0, max(1, len(waveform) - n_fft + 1), hop):
        yield waveform[start:start + n_fft]
