"""Cross-device consensus / cross-correlation.

Per the proposal: "Consensus + cross-correlation within a 200ms window
rejects single-sensor false positives." A lone device seeing one noisy frame
above threshold should not page anyone; either the *same* device needs to see
it consistently for a few frames, or a *second* device needs to see it too,
close in time.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from . import CONSENSUS_WINDOW_S, FLAG_LATENCY_BUDGET_S

# appliance_whine is a confusable negative (see synth.py) -- it is the
# thing this classifier must NOT confuse for a covert channel, not itself
# a family to alert on. background is the null class.
ALERT_FAMILIES = frozenset({"silverpush_beacon", "mosquito_signal", "dolphinattack_am"})

CONF_THRESHOLD = 0.6
SINGLE_DEVICE_STREAK = 3  # consecutive same-family frames from one device


@dataclass
class _Event:
    device_id: str
    family: str
    confidence: float
    t: float


@dataclass
class ConsensusEngine:
    conf_threshold: float = CONF_THRESHOLD
    single_device_streak: int = SINGLE_DEVICE_STREAK
    consensus_window_s: float = CONSENSUS_WINDOW_S
    _events: deque = field(default_factory=lambda: deque(maxlen=512))
    _last_alerted: dict[str, float] = field(default_factory=dict)

    def observe(self, device_id: str, family: str, confidence: float, t: float | None = None) -> dict | None:
        """Record one classification event; return a consensus decision if reached.

        Returns a dict with keys {family, confidence, contributing_devices,
        mode} the first time consensus is reached for a family, else None.
        Re-arms after `mode`'s cooldown so a sustained signal doesn't spam.
        """
        t = t if t is not None else time.monotonic()
        if family not in ALERT_FAMILIES or confidence < self.conf_threshold:
            return None

        self._events.append(_Event(device_id, family, confidence, t))
        self._prune(t)

        recent = [e for e in self._events if e.family == family]

        # -- multi-device corroboration: >=2 distinct devices within the window
        window_events = [e for e in recent if t - e.t <= self.consensus_window_s]
        distinct_devices = {e.device_id for e in window_events}
        if len(distinct_devices) >= 2:
            return self._maybe_alert(family, window_events, "multi_device_consensus", t)

        # -- single-device sustained streak within the overall flag budget
        same_device_recent = [e for e in recent if t - e.t <= FLAG_LATENCY_BUDGET_S]
        by_device: dict[str, list[_Event]] = {}
        for e in same_device_recent:
            by_device.setdefault(e.device_id, []).append(e)
        for dev, evs in by_device.items():
            if len(evs) >= self.single_device_streak:
                return self._maybe_alert(family, evs, "single_device_streak", t)

        return None

    def _maybe_alert(self, family: str, evidence: list[_Event], mode: str, t: float) -> dict | None:
        last = self._last_alerted.get(family)
        if last is not None and (t - last) < self.consensus_window_s:
            return None  # already alerted very recently -- don't spam
        self._last_alerted[family] = t
        return {
            "family": family,
            "confidence": max(e.confidence for e in evidence),
            "contributing_devices": sorted({e.device_id for e in evidence}),
            "mode": mode,
        }

    def _prune(self, now: float) -> None:
        horizon = max(FLAG_LATENCY_BUDGET_S, self.consensus_window_s) * 2
        while self._events and (now - self._events[0].t) > horizon:
            self._events.popleft()
