# Echo Chamber — Ultrasonic Covert Channel Detector

Continuously monitors the ultrasonic spectrum (16-24 kHz+) for covert
device-to-device signaling — cross-device ad-tracking beacons (SilverPush),
air-gap exfiltration tones (MOSQUITO), inaudible voice-command injection
(DolphinAttack) — and flags it within 500ms, cross-checked against a
Cloud AI 100 signature library. See `ARCHITECTURE.md` for the full design,
and its **Known Blockers** section before trusting any NPU-labeled number.

## Quick start

```powershell
.\run.ps1
.venv\Scripts\python.exe train_classifier.py          # synthesize data, train, export ONNX (~10s)
.venv\Scripts\python.exe run_echo_chamber.py --self-test    # offline pipeline check, no mic needed
.venv\Scripts\python.exe run_echo_chamber.py                # live: host mic (or synthetic fallback)
```

## What's real vs. simulated vs. not-yet-tested (read this before demoing)

| Piece | Status |
|---|---|
| Capture → DSP → inference → consensus → signature match → alert pipeline (host, CPU) | **Real, tested, working** — see `tests/` |
| Classifier training on synthetic data | **Real** — trains in ~10s, no GPU needed |
| `hardware_detect` / `convert_model` / `profile_workload` / `generate_code` MCP calls | **Real calls against the hosted QUAD MCP server** — `convert_model` failed on a server-side infra bug (broken QAIRT install); `profile_workload` self-reported its numbers as simulated as a result. See `ARCHITECTURE.md` |
| NPU inference | **Not verified** — no compiled QNN artifact exists (see above); the code path is real and matches the `generate_code` template, just unexercised against real hardware |
| Arduino UNO Q / Mobile / Cloud AI 100 | **Scaffolded, not deployed or run on real hardware/cloud** — see `deploy/*/README.md` |
| Classifier robustness on real audio | **Known issue, not silently fixed** — misclassified real ambient mic noise during this build. Needs real calibration data (proposal's own Day-1 plan). See `ARCHITECTURE.md` |

## CLI

```powershell
.venv\Scripts\python.exe run_echo_chamber.py --self-test          # offline, deterministic, no mic/network
.venv\Scripts\python.exe run_echo_chamber.py                       # live, host mic (falls back to synthetic if none)
.venv\Scripts\python.exe run_echo_chamber.py --duration 30          # live for 30s then exit
.venv\Scripts\python.exe run_echo_chamber.py --cpu                  # force CPU inference
.venv\Scripts\python.exe run_echo_chamber.py --serve-fusion-hub     # accept Arduino UNO Q / mobile streams
.venv\Scripts\python.exe run_echo_chamber.py --signature-url http://<ai100-host>:8100
```

## Tests

```powershell
.venv\Scripts\python.exe -m pytest tests/ -q
```

## Layout

```
train_classifier.py         synthesize data + train + export ONNX
run_echo_chamber.py         main app: capture/DSP/inference/consensus/alert
echo_chamber/                the actual pipeline package
deploy/arduino_uno_q/        primary listener (SPH0641LU4H MEMS mic) -- not yet run on real hardware
deploy/mobile/                secondary audio channel (Android) -- not yet run on real hardware
deploy/cloud_ai100/           embedding signature-library service -- local reference, not real AI100
tests/                        pytest suite
ARCHITECTURE.md               full design + honest blocker/limitation report
```
