"""Host-side audio capture.

Standard PC audio interfaces sample at up to 44.1/48 kHz (Nyquist 22-24 kHz),
which covers the 16-24 kHz band this classifier was trained on but NOT the
full ultrasonic range the SPH0641LU4H MEMS mic on the Arduino UNO Q reaches
(up to 80 kHz) -- see ARCHITECTURE.md for why both capture paths exist and
what each one can and can't see.

Falls back to the synthetic signal generator when no microphone is present
(e.g. this build environment, CI) so the rest of the pipeline stays testable
without physical audio hardware -- same pattern samples/mobilenetv2 uses for
its test image.
"""

from __future__ import annotations

import sys
import time
from typing import Iterator

import numpy as np

from . import N_FFT, SAMPLE_RATE_HZ


def mic_available() -> bool:
    try:
        import sounddevice as sd

        return len(sd.query_devices()) > 0 and sd.query_devices(kind="input") is not None
    except Exception:
        return False


def iter_live_frames(sample_rate: int = SAMPLE_RATE_HZ, n_fft: int = N_FFT) -> Iterator[np.ndarray]:
    """Yield consecutive `n_fft`-sample frames from the default input device."""
    import sounddevice as sd

    with sd.InputStream(samplerate=sample_rate, channels=1, blocksize=n_fft, dtype="float32") as stream:
        while True:
            frame, overflowed = stream.read(n_fft)
            if overflowed:
                print("[warn] audio input overflow -- a frame was dropped upstream", file=sys.stderr)
            yield frame[:, 0]


def iter_synthetic_frames(
    sample_rate: int = SAMPLE_RATE_HZ,
    n_fft: int = N_FFT,
    seed: int = 42,
    realtime: bool = True,
) -> Iterator[np.ndarray]:
    """Deterministic synthetic frame stream -- used when no mic is present.

    Cycles through the labeled classes so a self-test can exercise every
    branch of the classify -> consensus -> alert pipeline without hardware.
    """
    from .synth import default_rng, make_example
    from . import CLASSES

    rng = default_rng(seed)
    i = 0
    hop_s = n_fft / sample_rate
    while True:
        label = CLASSES[i % len(CLASSES)]
        yield make_example(label, n_fft, sample_rate, rng)
        i += 1
        if realtime:
            time.sleep(hop_s)


def frame_source(prefer_live: bool = True) -> tuple[Iterator[np.ndarray], str]:
    """Return (frame_iterator, source_label). Falls back to synthetic if no mic."""
    if prefer_live and mic_available():
        return iter_live_frames(), "live_mic"
    return iter_synthetic_frames(), "synthetic_fallback"
