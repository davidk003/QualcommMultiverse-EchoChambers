"""Spectral classifier inference wrapper.

Prefers the QNN NPU execution provider (matches the archetype workflow:
"deploy the FFT/CIR/spectral classifier to NPU"), falls back to CPU with a
clear, honest status rather than silently claiming NPU. As of this build,
the hosted QUAD MCP server's QAIRT install cannot produce a real QNN context
binary (see ARCHITECTURE.md's Known Blockers section), so only the CPU path
has ever actually been exercised -- the NPU branch here is real, working
code (mirrors the pattern `mcp__quad__generate_code` emits and
`samples/mobilenetv2` uses), just unverified against a real .bin artifact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import CLASSES


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    probs: np.ndarray
    latency_ms: float
    provider: str


class SpectralClassifier:
    """Runs the trained ONNX spectral classifier via onnxruntime.

    `prefer_npu=True` (default) tries the QNN HTP execution provider first;
    if it's unavailable (no onnxruntime-qnn install, or no compiled QNN
    artifact -- see ARCHITECTURE.md) it falls back to CPU and reports
    `provider="CPUExecutionProvider"` rather than silently pretending NPU ran.
    """

    def __init__(self, model_path: str, prefer_npu: bool = True) -> None:
        import onnxruntime as ort

        self.model_path = model_path
        providers: list = []
        if prefer_npu:
            providers.append((
                "QNNExecutionProvider",
                {"backend_path": "QnnHtp.dll"},
            ))
        providers.append("CPUExecutionProvider")

        try:
            self._session = ort.InferenceSession(model_path, providers=providers)
        except Exception:
            # QNN EP registration itself can raise (missing backend_path DLL) --
            # retry CPU-only rather than crash the whole app over an optional path.
            self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

        self.provider = self._session.get_providers()[0]
        self._input_name = self._session.get_inputs()[0].name

    def classify(self, features: np.ndarray) -> ClassificationResult:
        """`features`: (N_BANDS,) float32 vector from echo_chamber.dsp."""
        batch = features.astype(np.float32)[None, :]
        t0 = time.perf_counter()
        probs = self._session.run(None, {self._input_name: batch})[0][0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        idx = int(np.argmax(probs))
        return ClassificationResult(
            label=CLASSES[idx],
            confidence=float(probs[idx]),
            probs=probs,
            latency_ms=latency_ms,
            provider=self.provider,
        )
