"""Cross-device fusion hub.

Runs on the X-Elite host. The host's own primary mic stream is fed in
directly via `observe_primary` (no network hop); the Arduino UNO Q
(`deploy/arduino_uno_q/capture_agent.py`) and mobile
(`deploy/mobile/AudioCaptureClient.kt`) secondary streams connect over
WebSocket and are fed in via `_handle_message`. All three funnel into one
`ConsensusEngine` so a covert signal only gets flagged once, regardless of
which device(s) saw it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from . import SILENCE_RMS_FLOOR
from .alert import Alert, AlertManager
from .consensus import ALERT_FAMILIES, ConsensusEngine
from .dsp import extract_band_features, frame_rms
from .inference import SpectralClassifier
from .signature_library import SignatureClient

AlertCallback = Callable[[Alert], None]


@dataclass
class FusionHub:
    classifier: SpectralClassifier
    signature_client: SignatureClient = field(default_factory=SignatureClient)
    consensus: ConsensusEngine = field(default_factory=ConsensusEngine)
    alerts: AlertManager = field(default_factory=AlertManager)
    on_alert: AlertCallback | None = None

    def observe_primary(self, waveform_frame: np.ndarray, device_id: str = "x-elite-host") -> Alert | None:
        if frame_rms(waveform_frame) < SILENCE_RMS_FLOOR:
            return None
        features = extract_band_features(waveform_frame)
        result = self.classifier.classify(features)
        return self._handle_classification(device_id, result.label, result.confidence, features)

    def _handle_classification(
        self, device_id: str, label: str, confidence: float, features: np.ndarray, t: float | None = None,
    ) -> Alert | None:
        t = t if t is not None else time.monotonic()
        if label in ALERT_FAMILIES:
            self.alerts.note_candidate(label, now=t)

        decision = self.consensus.observe(device_id, label, confidence, t=t)
        if decision is None:
            return None

        sig_match = self.signature_client.match(features.tolist())
        alert = self.alerts.raise_alert(
            family=decision["family"],
            confidence=decision["confidence"],
            contributing_devices=decision["contributing_devices"],
            signature_match=sig_match,
            now=t,
        )
        if self.on_alert:
            self.on_alert(alert)
        return alert

    # -- WebSocket ingest for secondary devices --------------------------------

    async def _handle_message(self, device_id_by_ws: dict, ws, raw: str) -> None:
        msg: dict[str, Any] = json.loads(raw)
        if msg["type"] == "hello":
            device_id_by_ws[ws] = msg["device_id"]
        elif msg["type"] == "audio_frame":
            device_id = msg.get("device_id") or device_id_by_ws.get(ws, "unknown_device")
            waveform = np.asarray(msg["samples"], dtype=np.float32)
            if frame_rms(waveform) < SILENCE_RMS_FLOOR:
                return
            features = extract_band_features(waveform)
            result = self.classifier.classify(features)
            self._handle_classification(device_id, result.label, result.confidence, features)

    async def serve(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        import websockets

        device_id_by_ws: dict = {}

        async def handler(ws) -> None:
            try:
                async for raw in ws:
                    await self._handle_message(device_id_by_ws, ws, raw)
            finally:
                device_id_by_ws.pop(ws, None)

        print(f"Fusion hub listening on ws://{host}:{port}  (Arduino UNO Q / mobile secondary streams)")
        async with websockets.serve(handler, host, port):
            import asyncio

            await asyncio.Future()  # run forever
