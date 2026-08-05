"""Echo Chamber — Ultrasonic Covert Channel Detector.

Capture -> DSP (FFT band features) -> inference (NPU-preferred spectral
classifier) -> cross-device consensus -> Cloud AI 100 signature match ->
alert, end to end, per the archetype workflow this app was built against.

Usage
-----
    python run_echo_chamber.py --self-test          # offline, no mic/network needed
    python run_echo_chamber.py                       # live: host mic (or synthetic fallback)
    python run_echo_chamber.py --duration 10          # live for 10s then exit
    python run_echo_chamber.py --serve-fusion-hub     # host the WebSocket ingest for
                                                       # the Arduino UNO Q / mobile secondary streams
    python run_echo_chamber.py --signature-url http://<ai100-host>:8100
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows (cp1252 consoles can't render the checkmarks below).
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from echo_chamber import CLASSES, N_FFT, SAMPLE_RATE_HZ  # noqa: E402
from echo_chamber.alert import Alert  # noqa: E402
from echo_chamber.capture import frame_source  # noqa: E402
from echo_chamber.fusion_server import FusionHub  # noqa: E402
from echo_chamber.inference import SpectralClassifier  # noqa: E402
from echo_chamber.signature_library import SignatureClient  # noqa: E402

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "echo_chamber_classifier.onnx"


def _print_alert(alert: Alert) -> None:
    verdict = "OK" if alert.within_budget else "OVER BUDGET"
    print(
        f"  [ALERT] {alert.family}  conf={alert.confidence:.2f}  "
        f"latency={alert.latency_s * 1000:.1f}ms ({verdict})  "
        f"devices={alert.contributing_devices}"
    )
    if alert.signature_match:
        best = alert.signature_match.get("best_match", {})
        provider = alert.signature_match.get("provider", "?")
        print(f"          signature match: {best.get('family')} "
              f"(similarity={best.get('similarity', 0):.3f}, provider={provider})")
    print(f"          {json.dumps(alert.to_dict())}")


def run_self_test(model_path: Path, signature_url: str | None) -> int:
    """Deterministic, offline pipeline exercise -- Phase 4.7 clean-bootstrap check.

    No microphone or network required: synthesizes repeated bursts of each
    alert-family class (so the consensus streak logic actually fires) and
    confirms every family is both classified correctly-enough and flagged
    within the 500ms budget.
    """
    print("=" * 60)
    print("Echo Chamber self-test (offline, synthetic signals)")
    print("=" * 60)

    classifier = SpectralClassifier(str(model_path), prefer_npu=False)
    print(f"  Classifier provider: {classifier.provider}")

    alerts_seen: list[Alert] = []
    hub = FusionHub(
        classifier=classifier,
        signature_client=SignatureClient(base_url=signature_url),
        on_alert=alerts_seen.append,
    )

    from echo_chamber.consensus import ALERT_FAMILIES
    from echo_chamber.signature_reference import reference_library
    from echo_chamber.synth import default_rng, make_example

    # Pre-warm the synthetic reference signature library (a one-time ~1000-sample
    # dataset build) so the per-alert latency measured below is the real
    # capture->classify->consensus->flag path, not this one-off setup cost.
    reference_library()

    rng = default_rng(seed=99)
    ok = True
    for family in ALERT_FAMILIES:
        alerts_seen.clear()
        for _ in range(5):  # > SINGLE_DEVICE_STREAK
            waveform = make_example(family, N_FFT, SAMPLE_RATE_HZ, rng)
            hub.observe_primary(waveform)
        if alerts_seen:
            alert = alerts_seen[0]
            status = "PASS" if alert.family == family else f"FAIL (flagged {alert.family})"
            budget_note = "within 500ms budget" if alert.within_budget else "OVER 500ms budget"
            print(f"  {family:<20} -> flagged, signal-to-flag latency "
                  f"{alert.latency_s * 1000:.1f}ms ({budget_note})  [{status}]")
            if alert.family != family:
                ok = False
            if not alert.within_budget:
                ok = False
        else:
            print(f"  {family:<20} -> NOT flagged after 5 repeated frames  [FAIL]")
            ok = False

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def run_live(model_path: Path, signature_url: str | None, duration_s: float | None, prefer_npu: bool) -> int:
    print("=" * 60)
    print("Echo Chamber — live capture")
    print("=" * 60)

    classifier = SpectralClassifier(str(model_path), prefer_npu=prefer_npu)
    print(f"  Classifier provider : {classifier.provider}")
    if classifier.provider != "QNNExecutionProvider":
        print("  [info] Running on CPU, not the Hexagon NPU -- see ARCHITECTURE.md "
              "for why (the hosted QUAD MCP server's QAIRT install currently can't "
              "compile a QNN context binary for this model).")

    hub = FusionHub(classifier=classifier, signature_client=SignatureClient(base_url=signature_url),
                     on_alert=_print_alert)

    frames, source = frame_source(prefer_live=True)
    print(f"  Audio source        : {source}")
    if source == "synthetic_fallback":
        print("  [info] No microphone detected -- streaming synthetic signals so the "
              "pipeline is still exercisable end to end.")
    print()

    t0 = time.monotonic()
    n = 0
    try:
        for waveform in frames:
            hub.observe_primary(waveform)
            n += 1
            if duration_s is not None and (time.monotonic() - t0) >= duration_s:
                break
    except KeyboardInterrupt:
        pass

    print(f"\nProcessed {n} frames in {time.monotonic() - t0:.1f}s. "
          f"{len(hub.alerts.history)} alert(s) raised.")
    return 0


def run_fusion_hub(model_path: Path, signature_url: str | None, host: str, port: int) -> int:
    import asyncio

    classifier = SpectralClassifier(str(model_path), prefer_npu=False)
    hub = FusionHub(classifier=classifier, signature_client=SignatureClient(base_url=signature_url),
                     on_alert=_print_alert)
    asyncio.run(hub.serve(host=host, port=port))
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH),
                   help="Path to the trained spectral classifier ONNX (see train_classifier.py)")
    p.add_argument("--self-test", action="store_true",
                   help="Offline deterministic pipeline check -- no mic/network needed")
    p.add_argument("--serve-fusion-hub", action="store_true",
                   help="Host the WebSocket ingest for Arduino UNO Q / mobile secondary streams")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--duration", type=float, default=None, help="Live mode: stop after N seconds")
    p.add_argument("--signature-url", default=None,
                   help="Cloud AI 100 signature service base URL (default: in-process local reference)")
    p.add_argument("--npu", dest="prefer_npu", action="store_true", default=True,
                   help="Prefer the QNN NPU execution provider (default)")
    p.add_argument("--cpu", dest="prefer_npu", action="store_false", help="Force CPU inference")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[error] Model not found: {model_path}")
        print("        Run: python train_classifier.py")
        sys.exit(1)

    if args.self_test:
        sys.exit(run_self_test(model_path, args.signature_url))
    elif args.serve_fusion_hub:
        sys.exit(run_fusion_hub(model_path, args.signature_url, args.host, args.port))
    else:
        sys.exit(run_live(model_path, args.signature_url, args.duration, args.prefer_npu))


if __name__ == "__main__":
    main()
