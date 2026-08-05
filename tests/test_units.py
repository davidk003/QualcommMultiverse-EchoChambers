import numpy as np
import pytest

from echo_chamber import CLASSES, N_BANDS, N_FFT, SAMPLE_RATE_HZ
from echo_chamber.alert import AlertManager
from echo_chamber.consensus import ALERT_FAMILIES, ConsensusEngine
from echo_chamber.dataset import build_dataset, train_val_split
from echo_chamber.dsp import extract_band_features, frame_rms
from echo_chamber.synth import default_rng, gen_dolphinattack_am, gen_silverpush_beacon, make_example


# -- dsp ----------------------------------------------------------------------

def test_extract_band_features_shape_and_normalization():
    rng = default_rng(0)
    waveform = make_example("silverpush_beacon", N_FFT, SAMPLE_RATE_HZ, rng)
    feats = extract_band_features(waveform)
    assert feats.shape == (N_BANDS,)
    assert feats.dtype == np.float32
    assert abs(feats.mean()) < 1e-3
    assert abs(feats.std() - 1.0) < 1e-3


def test_extract_band_features_pads_short_frames():
    short = np.ones(10, dtype=np.float32)
    feats = extract_band_features(short)
    assert feats.shape == (N_BANDS,)
    assert np.all(np.isfinite(feats))


def test_frame_rms_scales_with_amplitude():
    quiet = np.full(N_FFT, 1e-5, dtype=np.float32)
    loud = np.full(N_FFT, 0.5, dtype=np.float32)
    assert frame_rms(quiet) < frame_rms(loud)


# -- synth ----------------------------------------------------------------------

def test_all_generators_produce_finite_normalized_audio():
    rng = default_rng(1)
    for label in CLASSES:
        waveform = make_example(label, N_FFT, SAMPLE_RATE_HZ, rng)
        assert waveform.shape == (N_FFT,)
        assert np.all(np.isfinite(waveform))
        assert np.max(np.abs(waveform)) <= 1.0 + 1e-6


def test_silverpush_and_dolphinattack_have_distinct_spectral_shape():
    """DolphinAttack's AM sidebands should give it a different, less peaky
    spectrum than SilverPush's single tone -- a sanity check that the two
    classes aren't accidentally identical after feature extraction."""
    rng = default_rng(2)
    sp = extract_band_features(gen_silverpush_beacon(N_FFT, SAMPLE_RATE_HZ, rng))
    da = extract_band_features(gen_dolphinattack_am(N_FFT, SAMPLE_RATE_HZ, rng))
    assert not np.allclose(sp, da, atol=0.5)


# -- dataset ----------------------------------------------------------------------

def test_build_dataset_shapes_and_labels():
    X, y = build_dataset(n_per_class=10, seed=3)
    assert X.shape == (10 * len(CLASSES), N_BANDS)
    assert y.shape == (10 * len(CLASSES),)
    assert set(np.unique(y).tolist()) == set(range(len(CLASSES)))


def test_train_val_split_sizes():
    X, y = build_dataset(n_per_class=20, seed=4)
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_frac=0.25)
    assert len(y_val) == int(len(y) * 0.25)
    assert len(y_train) == len(y) - len(y_val)


# -- consensus ----------------------------------------------------------------------

def test_consensus_ignores_low_confidence():
    engine = ConsensusEngine()
    for _ in range(5):
        assert engine.observe("dev-a", "mosquito_signal", confidence=0.4) is None


def test_consensus_ignores_non_alert_families():
    engine = ConsensusEngine()
    for _ in range(5):
        assert engine.observe("dev-a", "appliance_whine", confidence=0.99) is None
        assert engine.observe("dev-a", "background", confidence=0.99) is None


def test_consensus_single_device_streak():
    engine = ConsensusEngine()
    t = 0.0
    for i in range(engine.single_device_streak):
        decision = engine.observe("dev-a", "silverpush_beacon", confidence=0.9, t=t)
        t += 0.02
    assert decision is not None
    assert decision["mode"] == "single_device_streak"
    assert decision["contributing_devices"] == ["dev-a"]


def test_consensus_multi_device_corroboration_beats_single_frame():
    engine = ConsensusEngine()
    t = 0.0
    assert engine.observe("dev-a", "mosquito_signal", confidence=0.9, t=t) is None
    decision = engine.observe("dev-b", "mosquito_signal", confidence=0.9, t=t + 0.05)
    assert decision is not None
    assert decision["mode"] == "multi_device_consensus"
    assert set(decision["contributing_devices"]) == {"dev-a", "dev-b"}


def test_consensus_does_not_spam_within_window():
    engine = ConsensusEngine()
    t = 0.0
    for i in range(engine.single_device_streak):
        first = engine.observe("dev-a", "silverpush_beacon", confidence=0.9, t=t)
        t += 0.02
    assert first is not None
    second = engine.observe("dev-a", "silverpush_beacon", confidence=0.9, t=t + 0.01)
    assert second is None  # cooldown -- shouldn't re-fire immediately


def test_alert_families_exclude_negatives():
    assert "background" not in ALERT_FAMILIES
    assert "appliance_whine" not in ALERT_FAMILIES


# -- alert ----------------------------------------------------------------------

def test_alert_manager_latency_and_budget():
    mgr = AlertManager()
    mgr.note_candidate("mosquito_signal", now=10.0)
    alert = mgr.raise_alert("mosquito_signal", confidence=0.95, contributing_devices=["dev-a"], now=10.2)
    assert alert.latency_s == pytest.approx(0.2)
    assert alert.within_budget is True
    assert mgr.history == [alert]


def test_alert_manager_over_budget():
    mgr = AlertManager()
    mgr.note_candidate("mosquito_signal", now=0.0)
    alert = mgr.raise_alert("mosquito_signal", confidence=0.95, contributing_devices=["dev-a"], now=0.6)
    assert alert.within_budget is False


def test_alert_to_dict_roundtrip():
    mgr = AlertManager()
    mgr.note_candidate("dolphinattack_am", now=1.0)
    alert = mgr.raise_alert("dolphinattack_am", confidence=0.8, contributing_devices=["dev-a", "dev-b"], now=1.1)
    d = alert.to_dict()
    assert d["family"] == "dolphinattack_am"
    assert d["contributing_devices"] == ["dev-a", "dev-b"]
    assert d["within_budget"] is True
