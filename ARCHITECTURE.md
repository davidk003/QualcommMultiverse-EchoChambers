# Echo Chamber — Ultrasonic Covert Channel Detector — Architecture

## Overview

Echo Chamber continuously monitors the ultrasonic spectrum (16-24 kHz+) for
covert-channel signaling — cross-device ad-tracking beacons (SilverPush),
air-gap exfiltration tones (MOSQUITO), and inaudible voice-command injection
(DolphinAttack) — distinguishing them from harmless confusable noise
(appliance/coil whine). It flags a detected signal within 500ms end to end,
and matches it against a Cloud AI 100-hosted signature library.

This was built driving the QUAD MCP tools in the archetype order:
`hardware_detect → convert_model → profile_workload → generate_code`. That
sequence surfaced two real, environment-level blockers (below) that shaped
several design decisions — this doc reports them plainly rather than papering
over them with fabricated numbers.

## Target Platforms

| Device | Role | QUAD platform id |
|---|---|---|
| Snapdragon X Elite AI PC | Runs the NPU-accelerated spectral classifier, hosts the fusion hub, makes the flagging decision | `windows` |
| Arduino UNO Q (QCS2210) | Primary listener — SPH0641LU4H MEMS mic (up to 80 kHz), streams audio, does not classify locally | `linux` |
| Mobile (Android) | Secondary audio channel from a different physical position, streams audio for cross-correlation | `android` |
| Qualcomm Cloud AI 100 | Embedding-based signature library, similarity search | `cloud` / `qaic` |

## System Diagram

```
 Arduino UNO Q (MEMS mic, 80kHz-capable)  ---ws--\
                                                    \
 Mobile (secondary position)              ---ws----->  Fusion Hub (X-Elite host)
                                                    /         |
 X-Elite host mic (44.1/48kHz)  --------- direct --/          |
                                                               v
                                              DSP: FFT -> 16-24kHz band -> 64 log-mag bins
                                                               |
                                                               v
                                      Spectral classifier (NPU-preferred, CPU fallback)
                                                               |
                                                               v
                                          Consensus engine (single-device streak OR
                                             >=2-device corroboration within 200ms)
                                                               |
                                                               v
                                    Signature match (Cloud AI 100 embedding similarity search)
                                                               |
                                                               v
                                             Alert (<500ms signal-to-flag budget)
```

## Components

| Component | File | Description |
|---|---|---|
| Signal synthesis | `echo_chamber/synth.py` | Generates labeled training audio from published frequency/modulation specs (no public dataset exists — see Data below) |
| Feature extraction | `echo_chamber/dsp.py` | FFT → 16-24 kHz band → 64 log-magnitude bins, z-score normalized; silence-gated (`SILENCE_RMS_FLOOR`) |
| Classifier training | `train_classifier.py`, `echo_chamber/model_np.py` | numpy MLP (no torch/sklearn in this environment), hand-authored ONNX export |
| Inference | `echo_chamber/inference.py` | onnxruntime, QNN NPU EP preferred, CPU fallback with honest provider reporting |
| Cross-device consensus | `echo_chamber/consensus.py` | Single-device streak or ≥2-device corroboration within a 200ms window — rejects single-sensor false positives per the proposal |
| Alerting | `echo_chamber/alert.py` | Tracks signal-onset → flag latency against the 500ms budget |
| Fusion hub | `echo_chamber/fusion_server.py` | WebSocket ingest for Arduino/mobile secondary streams; feeds one shared consensus engine |
| Signature library (client) | `echo_chamber/signature_library.py`, `echo_chamber/signature_reference.py` | Calls the Cloud AI 100 service over HTTP if configured, else an in-process local reference (both return the same shape, tagged with which one answered) |
| Cloud AI 100 service | `deploy/cloud_ai100/signature_service.py` | Real, runnable FastAPI service implementing the embedding similarity search — see its README for why it's hand-written rather than the `generate_code` template |
| Arduino UNO Q agent | `deploy/arduino_uno_q/capture_agent.py` | MEMS mic capture → WebSocket stream to the fusion hub |
| Mobile client | `deploy/mobile/AudioCaptureClient.kt` | `AudioRecord` capture → WebSocket stream to the fusion hub |

## Model Card

| | |
|---|---|
| Task | 5-way spectral classification: background / appliance_whine / silverpush_beacon / mosquito_signal / dolphinattack_am |
| Architecture | MLP: 64 → 32 (ReLU) → 16 (ReLU) → 5 (Softmax), Gemm/Relu/Gemm/Relu/Gemm/Softmax ONNX graph |
| Input | `spectral_features`, shape `[1, 64]` float32 — log-magnitude, z-score-normalized FFT bins over 16-24 kHz |
| Output | `class_probs`, shape `[1, 5]` float32 |
| Quantization | FP32 (no real calibration data exists yet — see Known Blockers) |
| Size | ~11 KB ONNX (~5.6K parameters) |
| Training data | 100% synthetic, generated from published attack specs — see Data below |
| Validation accuracy | **87%** on single ~21ms frames (synthetic held-out set) — see the honest caveats below |

### Data — no public dataset exists

Per the proposal, there is no public dataset for these attacks. Training
data here is synthesized directly from each family's published frequency
plan and modulation scheme:

- **SilverPush**: Hann-enveloped tone burst, 18-19.5 kHz.
- **MOSQUITO**: one FSK symbol per frame, steady tone at one of two
  closely-spaced frequencies, 20.5-22 kHz.
- **DolphinAttack**: AM-modulated ultrasonic carrier, 22-23 kHz carrier with
  200-800 Hz voice-band modulation (carrier + 2 sidebands).
- **appliance_whine** (confusable negative): steady tone with slow drift,
  15-17 kHz, just below the covert-channel band.
- **background**: broadband low-level noise.

Every example is augmented with randomized SNR (6-25 dB) and probabilistic
synthetic reverb, matching the proposal's stated augmentation plan.

### Honest caveats on the 87% accuracy number

1. It is **single-frame (~21ms) accuracy on synthetic data only** — real
   detection aggregates several frames over the 500ms consensus window,
   which is materially more reliable than this per-frame number suggests.
2. **A live test during this build misclassified real ambient microphone
   noise as `mosquito_signal` at >95% confidence, repeatedly.** This is not
   a bug to silently tune away — it is a direct, observed instance of the
   exact risk the proposal itself names: "distinguishing genuine covert
   signaling from harmless ultrasonic noise" is the hard part, and a model
   trained on zero real recordings has no way to have learned what real room
   noise looks like. **This is precisely why the proposal specifies a Day-1
   calibrated tone-sweep verification against real hardware before any model
   work is trusted** — that step has not happened (no physical MEMS
   mic/Arduino/room was available in this environment). A `SILENCE_RMS_FLOOR`
   energy gate (`echo_chamber/__init__.py`) was added as a genuine, standard
   engineering improvement (never even runs the classifier on near-silent
   frames), but it does not substitute for real calibration data. **Stop and
   get real recordings (or at minimum a calibrated tone sweep) before
   trusting this classifier's alerts in a real room.**

## Known Blockers (found while driving the QUAD MCP tools for real)

These were discovered by actually calling the tools, not assumed —
each is real and reproducible as of this build (2026-08-03).

### 1. `hardware_detect` — the QUAD MCP server here is not real Snapdragon hardware

`.claude/settings.json` points at a hosted server
(`https://quad.infra.foundries.io/mcp`). `mcp__quad__hardware_detect(platform="windows")`
returned an `AMD EPYC 7B12` / Ubuntu 24.04 container — `available_runtimes: ["cpu"]`,
`gpu_backend_available: false`, NPU TOPS data unavailable. There is no real
Hexagon silicon behind this endpoint; it's a generic cloud dev/test instance
with the QAIRT SDK installed (v2.41.0.251128) but nothing physical to run on.

### 2. `convert_model` — the server's QAIRT install is broken (both QNN and SNPE)

Uploading the trained ONNX and calling `convert_model` with `target_sdk="qnn"`
**and** `target_sdk="snpe"` both fail identically:

```
ImportError: libpython3.10.so.1.0: cannot open shared object file: No such file or directory
  File ".../qti/aisw/dlc_utils/__init__.py", ... import libDlModelToolsPy as modeltools
```

`qairt-converter`'s native `libDlModelToolsPy` extension can't load because
the server's Linux container is missing `libpython3.10.so.1.0`. This is a
genuine infra bug in the hosted server, confirmed systemic (same failure
regardless of target SDK), not fixable from this client, and not specific to
this model. **`convert_model` cannot produce a real QNN/SNPE artifact from
this environment right now.** No compiled `.bin`/`.dlc` exists for this
model as a result.

### 3. `profile_workload` — self-reports as simulated, and is honest about it

Called with `runtime="npu"` against the (uncompiled) model, the tool
returned a plausible-looking result — 6.12ms mean, 7.2ms p99, comfortably
under the 10ms real-time budget — but tagged it explicitly:
`"measurement_notes": {"latency": "simulated:model_file_missing"}`,
`"provider": "ort-mock"`. This is the tool's own mock fallback, triggered by
blocker #2 above (no real artifact exists to profile). **Treat these numbers
as directionally plausible (this is a ~5.6K-parameter model; sub-10ms is not
a stretch) but not measured on real Hexagon hardware.**

### 4. `generate_code` — works for real, but `platform="cloud"`/`sdk="qaic"` has one template shape (LLM chat), not embedding search

Unlike blockers 1-3, `generate_code` is unaffected by the broken converter —
all calls (windows/python/qnn, linux/python/snpe, android/kotlin/qnn,
cloud/python/qaic) succeeded and returned real generated code. The
`cloud`+`qaic` template, however, is shaped for LLM chat serving
(`QEFFAutoModelForCausalLM`, OpenAI-compatible endpoint) — there's no generic
embedding/similarity-search template for that platform/sdk pair. Rather than
force-fit it, `deploy/cloud_ai100/signature_service.py` is a small hand-written
service implementing the actual architecture; the raw generated template is
kept for the record under `deploy/cloud_ai100/qefficient_template/`. Worth
raising as a template-coverage gap upstream.

### What was NOT blocked, and is real

- `hardware_detect`, `profile_workload`, and `generate_code` all executed
  for real against the hosted MCP server (not mocked by me — the server
  answered, sometimes with its own honest simulated-data flags).
  Uploading a local file to the server and driving `convert_model`
  end-to-end also genuinely worked up to the point of the server's broken
  native library — the upload/RPC plumbing itself is not the problem.
- The entire capture → DSP → inference → consensus → signature match →
  alert pipeline runs for real on this host, CPU-only, verified by the
  test suite and the `--self-test` / live-mode runs below.

## Devices not physically available in this environment (stop-and-ask items)

Per your instruction to stop and ask when a step needs calibration data or a
real device, these are exactly those steps — flagging rather than
fabricating results for them:

1. **Arduino UNO Q + SPH0641LU4H MEMS mic** — `deploy/arduino_uno_q/` is
   written and syntax-checked but never deployed to or run on a real board.
   The proposal's own Day-1 plan (calibrated tone sweep) should happen here
   first, and needs the physical board.
2. **Mobile device** — `deploy/mobile/` is source-only; no Android
   SDK/Gradle project or physical phone was available to build/run it.
3. **Cloud AI 100** — no physical/cloud AI100 access exists in this
   environment; `deploy/cloud_ai100/signature_service.py` is a local CPU
   reference implementation of the right architecture, not a real
   deployment.
4. **Real ultrasonic recordings / calibration data** — none exist (the
   proposal confirms no public dataset exists); everything the classifier
   knows comes from synthesis. The false-positive-on-real-noise finding
   above is the direct consequence — real recordings are needed before this
   is trustworthy in the field, not just structurally complete.

## Deployment

Host (X-Elite):
```
.\run.ps1
.venv\Scripts\python.exe train_classifier.py       # synthesize data, train, export ONNX
.venv\Scripts\python.exe run_echo_chamber.py --self-test   # offline pipeline check
.venv\Scripts\python.exe run_echo_chamber.py               # live (mic or synthetic fallback)
.venv\Scripts\python.exe run_echo_chamber.py --serve-fusion-hub   # accept Arduino/mobile streams
```

Arduino UNO Q / mobile: see `deploy/arduino_uno_q/README.md` and
`deploy/mobile/README.md` — both require the stop-and-ask hardware above
before they can be verified.

Cloud AI 100: see `deploy/cloud_ai100/README.md`.

## KPIs

| Metric | Measured/Simulated | Target | Status |
|---|---|---|---|
| Real-time inference latency (NPU) | 6.12ms mean / 7.2ms p99 (server-simulated, see Blocker #3) | <10ms | Directionally plausible, not verified on real Hexagon silicon |
| Signal-to-flag latency (host, CPU) | 0-350ms typical, measured live in this build (see `run_echo_chamber.py` output) | <500ms | **Real, measured, passing** on CPU |
| Classifier validation accuracy | 87% (single-frame, synthetic) | n/a | Honest, not tuned to look better — see caveats above |
| Real-room false positive rate | **Not acceptable as observed** (see caveat #2 above) | Low | **Fails today** — needs real calibration data, not more synthetic tuning |

## Power & Thermal

Not measured — no real device/board access in this environment (see
stop-and-ask list above). The classifier is small enough (~5.6K params)
that compute power should be negligible relative to the microphone/radio
duty cycle in an always-on deployment, but this is an expectation, not a
measurement.
