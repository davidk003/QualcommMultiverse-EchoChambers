"""Synthetic ultrasonic covert-channel signal generation.

No public dataset exists for these attacks (see the proposal doc), but the
frequency plans and modulation schemes are published for each family, so
training data is generated reproducibly from spec rather than scraped:

  - SilverPush   -- short near-ultrasonic (~18-20 kHz) tone bursts used as a
                    cross-device ad-tracking beacon.
  - MOSQUITO     -- air-gap exfiltration via narrowband tones in the high
                    18-24 kHz band, data encoded as frequency-shift keying
                    between two closely-spaced tones.
  - DolphinAttack -- inaudible voice-command injection: an ultrasonic carrier
                    amplitude-modulated by an audio-band command signal, which
                    demodulates on a victim microphone's non-linearity.
  - appliance_whine -- the main confusable negative: SMPS/CRT/coil whine, a
                    single steady-ish tone with slow drift, no data content.
  - background   -- broadband low-level noise, no tonal ultrasonic energy.

Every generator accepts an SNR and a `reverb` flag so a single spec-derived
waveform can be augmented for distance/room variation, matching the
proposal's stated augmentation plan (SNR, distance, reverb).
"""

from __future__ import annotations

import numpy as np

from . import SAMPLE_RATE_HZ

_RNG_SEED = 20260803  # fixed seed -> reproducible synthetic corpus


def _time_axis(n_samples: int, sr: int) -> np.ndarray:
    return np.arange(n_samples, dtype=np.float64) / sr


def _add_noise(x: np.ndarray, rng: np.random.Generator, snr_db: float) -> np.ndarray:
    """Add white noise at the requested SNR (signal power vs noise power)."""
    sig_power = np.mean(x ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=x.shape)
    return x + noise


def _apply_reverb(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cheap synthetic reverb: convolve with a short exponential-decay tail."""
    tail_len = rng.integers(8, 64)
    decay = rng.uniform(0.3, 0.7)
    ir = decay ** np.arange(tail_len)
    ir = ir / ir.sum()
    return np.convolve(x, ir, mode="same")


def gen_background(n_samples: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Broadband low-level noise -- a quiet room with no covert signaling."""
    x = rng.normal(0.0, 1.0, size=n_samples) * rng.uniform(0.01, 0.05)
    return x


def gen_appliance_whine(n_samples: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """SMPS/CRT/coil whine -- steady tone with slow frequency + amplitude drift.

    Frequency band (15-17 kHz) sits just below the covert-channel band so the
    classifier learns a real spectral boundary rather than an energy threshold.
    """
    t = _time_axis(n_samples, sr)
    f0 = rng.uniform(15_000, 17_000)
    drift = rng.uniform(-30, 30) * np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * t)
    phase = 2 * np.pi * (f0 * t + np.cumsum(drift) / sr)
    amp_wobble = 1.0 + 0.1 * np.sin(2 * np.pi * rng.uniform(1.0, 5.0) * t)
    x = rng.uniform(0.15, 0.4) * amp_wobble * np.sin(phase)
    return x


def gen_silverpush_beacon(n_samples: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Near-ultrasonic tone burst (~18-19.5 kHz), Hann-windowed, ad-beacon style."""
    t = _time_axis(n_samples, sr)
    f0 = rng.uniform(18_000, 19_500)
    envelope = np.hanning(n_samples) ** 0.5  # softer taper than full Hann
    x = rng.uniform(0.3, 0.6) * envelope * np.sin(2 * np.pi * f0 * t)
    return x


def gen_mosquito_signal(n_samples: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """One FSK symbol of a MOSQUITO-style air-gap exfil tone.

    Two closely-spaced high-band tones (~20.5-22 kHz) encode data bits; a
    single analysis frame captures one symbol, so the generator emits one
    tone from the pair with a steady (non-decaying) envelope -- the spectral
    cue that separates it from SilverPush's decaying single-burst envelope.
    """
    t = _time_axis(n_samples, sr)
    f_lo = rng.uniform(20_500, 21_200)
    f_hi = f_lo + rng.uniform(400, 800)
    f0 = f_lo if rng.random() < 0.5 else f_hi
    x = rng.uniform(0.25, 0.5) * np.sin(2 * np.pi * f0 * t)
    return x


def gen_dolphinattack_am(n_samples: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Ultrasonic carrier AM-modulated by an audio-band command signal.

    Produces a carrier peak plus two sidebands at fc +/- f_mod -- a
    multi-peak spectral signature that distinguishes it from the
    single-tone classes above.
    """
    t = _time_axis(n_samples, sr)
    fc = rng.uniform(22_000, 23_000)
    f_mod = rng.uniform(200, 800)  # voice-band "command" rate
    mod_depth = rng.uniform(0.5, 0.9)
    carrier = np.sin(2 * np.pi * fc * t)
    envelope = 1.0 + mod_depth * np.sin(2 * np.pi * f_mod * t)
    x = rng.uniform(0.2, 0.45) * envelope * carrier
    return x


GENERATORS = {
    "background": gen_background,
    "appliance_whine": gen_appliance_whine,
    "silverpush_beacon": gen_silverpush_beacon,
    "mosquito_signal": gen_mosquito_signal,
    "dolphinattack_am": gen_dolphinattack_am,
}


def make_example(
    label: str,
    n_samples: int,
    sr: int = SAMPLE_RATE_HZ,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate one labeled, augmented waveform frame."""
    rng = rng or np.random.default_rng()
    x = GENERATORS[label](n_samples, sr, rng)
    if rng.random() < 0.5:
        x = _apply_reverb(x, rng)
    # 6 dB floor: below that, a single 21 ms frame is genuinely ambiguous even
    # for a human reading the spectrogram -- the live pipeline's <500 ms flag
    # window aggregates several frames, which recovers SNR the single-frame
    # classifier can't see, so this floor doesn't understate real detection range.
    snr_db = rng.uniform(6.0, 25.0)
    x = _add_noise(x, rng, snr_db)
    peak = np.max(np.abs(x)) + 1e-9
    return (x / peak).astype(np.float32)


def default_rng(seed: int = _RNG_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)
