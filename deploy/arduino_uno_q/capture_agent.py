"""Arduino UNO Q (QCS2210) primary listener -- captures the SPH0641LU4H MEMS
mic and streams raw PCM frames to the X-Elite fusion hub over WebSocket.

*** NOT YET RUN ON REAL HARDWARE -- see the blocker note in ARCHITECTURE.md ***
This script has been reviewed against the board's published capabilities
(2 GB RAM, DSP V66 INT8-only, SSH-managed per the quad-unoq skill) but has
not been deployed to or exercised on a physical UNO Q + SPH0641LU4H pair, and
per the proposal's own Day-1 plan, it should not be trusted before that
tone-sweep calibration happens. Stop and hand this to the real board before
relying on it.

Assumptions this script makes (call these out explicitly to whoever brings up
the board, per the proposal's own Day-1 hardware-verification plan):

  1. The SPH0641LU4H is wired to the board's PDM/I2S input and enumerates as
     a standard ALSA capture device (`arecord -l` should list it) at a PDM
     clock that decimates to >= 48 kHz PCM. If the board instead exposes only
     a raw PDM bitstream with no ALSA driver, this script's `sounddevice`
     capture call will need to be replaced with a direct PDM decimation read
     -- that is a hardware-bring-up task, not a software one, and is exactly
     the Day-1 calibration step the proposal calls out.
  2. The mic can be captured at the full SAMPLE_RATE_HZ used everywhere else
     in this app (48 kHz) even though the SPH0641LU4H's rated bandwidth goes
     to 80 kHz -- the X-Elite host classifier only needs the 16-24 kHz band,
     so 48 kHz capture (Nyquist 24 kHz) is sufficient for parity with the
     host's own audio-interface capture. Capturing at a higher rate (e.g.
     96/192 kHz, if the board's ADC path supports it) would let a *future*
     model see covert channels above 24 kHz that a standard PC audio
     interface structurally cannot -- that headroom is the whole reason the
     proposal specifies this MEMS mic instead of relying on the laptop mic
     alone. Revisit once real hardware is available to characterize the
     board's actual achievable sample rate.

Deploy (once the board is reachable over SSH -- see the quad-unoq skill):
    scp capture_agent.py root@<uno-q-ip>:/data/local/tmp/echo_chamber/
    ssh root@<uno-q-ip> "python3 /data/local/tmp/echo_chamber/capture_agent.py \
        --hub-url ws://<x-elite-host>:8765/ingest --device-id unoq-1"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import numpy as np

SAMPLE_RATE_HZ = 48_000
FRAME_SAMPLES = 1024  # matches echo_chamber.N_FFT on the host


async def stream_loop(hub_url: str, device_id: str, duration_s: float | None) -> None:
    try:
        import sounddevice as sd
        import websockets
    except ImportError as exc:
        print(f"[error] missing dependency on the board: {exc}")
        print("        pip3 install sounddevice websockets  (see run.ps1 / requirements.txt)")
        sys.exit(1)

    print(f"Connecting to fusion hub: {hub_url}")
    async with websockets.connect(hub_url) as ws:
        await ws.send(json.dumps({
            "type": "hello",
            "device_id": device_id,
            "device_kind": "arduino_uno_q",
            "mic": "SPH0641LU4H",
            "sample_rate_hz": SAMPLE_RATE_HZ,
        }))

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

        def _on_audio(indata, frames, time_info, status):  # sounddevice callback (sync)
            if status:
                print(f"[warn] capture status: {status}", file=sys.stderr)
            loop.call_soon_threadsafe(queue.put_nowait, indata[:, 0].copy())

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE_HZ,
            channels=1,
            blocksize=FRAME_SAMPLES,
            dtype="float32",
            callback=_on_audio,
        )
        t0 = time.monotonic()
        with stream:
            while duration_s is None or (time.monotonic() - t0) < duration_s:
                frame = await queue.get()
                await ws.send(json.dumps({
                    "type": "audio_frame",
                    "device_id": device_id,
                    "t_capture": time.time(),
                    "samples": frame.tolist(),
                }))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hub-url", required=True, help="ws://<x-elite-host>:8765/ingest")
    p.add_argument("--device-id", default="unoq-1")
    p.add_argument("--duration", type=float, default=None, help="seconds; default: run forever")
    args = p.parse_args()
    asyncio.run(stream_loop(args.hub_url, args.device_id, args.duration))


if __name__ == "__main__":
    main()
