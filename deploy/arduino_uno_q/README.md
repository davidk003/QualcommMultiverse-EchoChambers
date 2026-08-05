# Arduino UNO Q (QCS2210) — primary listener

**Status: scaffolded, not deployed or run on real hardware.** See the
blocker note in `../../ARCHITECTURE.md` before treating anything here as
verified.

| File | What it is |
|---|---|
| `capture_agent.py` | Hand-written. Streams the SPH0641LU4H MEMS mic to the X-Elite fusion hub over WebSocket. This is the board's actual role per the architecture. |
| `inference.py` | Raw `mcp__quad__generate_code(platform="linux", sdk="snpe")` output, unedited. Kept as a QUAD-standard record; **not used** — the board streams audio, it does not classify locally (see the docstring in that file for why). |

## Before trusting this on a real board

1. Run the proposal's own Day-1 step: verify the MEMS mic + `capture_agent.py`
   against a calibrated ultrasonic tone sweep. `capture_agent.py`'s docstring
   lists the specific hardware assumptions (ALSA-visible PDM capture device,
   achievable sample rate) that step needs to confirm.
2. Configure the board per the `quad-unoq` skill (SSH reachability, deps).
3. Deploy: `scp capture_agent.py root@<uno-q-ip>:/data/local/tmp/echo_chamber/`
   then run it pointed at the host's fusion hub (`run_echo_chamber.py --serve-fusion-hub`).
