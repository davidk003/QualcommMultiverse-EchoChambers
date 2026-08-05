"""End-to-end pipeline tests: capture (synthetic) -> DSP -> inference ->
consensus -> signature match -> alert, and the WebSocket fusion hub ingest
path -- run against real localhost sockets (no external hardware needed).
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from echo_chamber import CLASSES, N_FFT, SAMPLE_RATE_HZ, SILENCE_RMS_FLOOR
from echo_chamber.consensus import ALERT_FAMILIES
from echo_chamber.fusion_server import FusionHub
from echo_chamber.inference import SpectralClassifier
from echo_chamber.signature_library import SignatureClient
from echo_chamber.synth import default_rng, make_example


def test_classifier_loads_and_runs(trained_model_path):
    clf = SpectralClassifier(trained_model_path, prefer_npu=False)
    assert clf.provider == "CPUExecutionProvider"
    rng = default_rng(5)
    waveform = make_example("background", N_FFT, SAMPLE_RATE_HZ, rng)
    from echo_chamber.dsp import extract_band_features
    result = clf.classify(extract_band_features(waveform))
    assert result.label in CLASSES
    assert 0.0 <= result.confidence <= 1.0
    assert result.latency_ms >= 0.0


@pytest.mark.parametrize("family", sorted(ALERT_FAMILIES))
def test_sustained_signal_raises_alert_within_budget(trained_model_path, family):
    clf = SpectralClassifier(trained_model_path, prefer_npu=False)
    alerts = []
    hub = FusionHub(classifier=clf, signature_client=SignatureClient(), on_alert=alerts.append)
    rng = default_rng(6)
    for _ in range(5):
        waveform = make_example(family, N_FFT, SAMPLE_RATE_HZ, rng)
        hub.observe_primary(waveform)
    assert len(alerts) >= 1
    alert = alerts[0]
    assert alert.within_budget
    assert alert.signature_match is not None
    assert alert.signature_match["provider"] == "local-reference-cpu"


def test_silence_never_reaches_the_classifier_or_alerts(trained_model_path):
    clf = SpectralClassifier(trained_model_path, prefer_npu=False)
    alerts = []
    hub = FusionHub(classifier=clf, signature_client=SignatureClient(), on_alert=alerts.append)
    silence = np.zeros(N_FFT, dtype=np.float32)
    for _ in range(10):
        result = hub.observe_primary(silence)
        assert result is None
    assert alerts == []


def test_background_class_does_not_alert(trained_model_path):
    clf = SpectralClassifier(trained_model_path, prefer_npu=False)
    alerts = []
    hub = FusionHub(classifier=clf, signature_client=SignatureClient(), on_alert=alerts.append)
    rng = default_rng(7)
    for _ in range(10):
        waveform = make_example("background", N_FFT, SAMPLE_RATE_HZ, rng)
        # Background is synthesized quietly; skip the cases the silence gate
        # would already reject so this test targets the classifier, not the gate.
        from echo_chamber.dsp import frame_rms
        if frame_rms(waveform) < SILENCE_RMS_FLOOR:
            continue
        hub.observe_primary(waveform)
    assert alerts == []


# -- fusion hub WebSocket ingest (real localhost socket, no external hardware) --

async def _run_fusion_hub_ingest_check(model_path: str) -> list:
    import websockets

    clf = SpectralClassifier(model_path, prefer_npu=False)
    alerts: list = []
    hub = FusionHub(classifier=clf, signature_client=SignatureClient(), on_alert=alerts.append)

    device_id_by_ws: dict = {}

    async def handler(ws):
        async for raw in ws:
            await hub._handle_message(device_id_by_ws, ws, raw)

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        rng = default_rng(8)
        async with websockets.connect(f"ws://127.0.0.1:{port}") as client:
            import json

            await client.send(json.dumps({
                "type": "hello", "device_id": "unoq-test", "device_kind": "arduino_uno_q",
            }))
            for _ in range(5):
                waveform = make_example("silverpush_beacon", N_FFT, SAMPLE_RATE_HZ, rng)
                await client.send(json.dumps({
                    "type": "audio_frame", "device_id": "unoq-test",
                    "t_capture": 0.0, "samples": waveform.tolist(),
                }))
            await asyncio.sleep(0.2)  # let the handler drain the queue
    return alerts


def test_fusion_hub_websocket_ingest_from_secondary_device(trained_model_path):
    alerts = asyncio.run(_run_fusion_hub_ingest_check(trained_model_path))
    assert len(alerts) >= 1
    assert alerts[0].family == "silverpush_beacon"
    assert alerts[0].contributing_devices == ["unoq-test"]
