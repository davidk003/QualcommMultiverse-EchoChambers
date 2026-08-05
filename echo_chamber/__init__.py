"""Echo Chamber — Ultrasonic Covert Channel Detector."""

CLASSES = [
    "background",
    "appliance_whine",
    "silverpush_beacon",
    "mosquito_signal",
    "dolphinattack_am",
]

SAMPLE_RATE_HZ = 48_000
N_FFT = 1024
BAND_HZ = (16_000, 24_000)
N_BANDS = 64
FRAME_HOP_S = N_FFT / SAMPLE_RATE_HZ  # ~21.3 ms analysis frame
FLAG_LATENCY_BUDGET_S = 0.5           # signal-to-flag budget from the proposal
CONSENSUS_WINDOW_S = 0.2              # cross-device consensus window

# Below this raw-waveform RMS, a frame is treated as silence and never even
# reaches the classifier. Without this gate, near-digital-silence input (a
# disconnected/muted mic, or a headless capture device returning near-zero
# samples) produces a near-uniform, tiny log-magnitude spectrum; the
# per-frame z-score normalization in dsp.extract_band_features then divides
# by a near-zero std and amplifies floating-point noise into confident-looking
# nonsense classifications -- observed directly during this build (a
# synthetic/near-silent "live" capture device was consistently misclassified
# as mosquito_signal at >95% confidence). This floor is a conservative
# placeholder; the proposal's own Day-1 tone-sweep calibration against real
# hardware is what should tune it for a real room.
SILENCE_RMS_FLOOR = 1e-3
