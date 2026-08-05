"""Flag emission + the proposal's <500ms signal-to-flag latency budget."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from . import FLAG_LATENCY_BUDGET_S


@dataclass
class Alert:
    family: str
    confidence: float
    t_signal: float          # monotonic time the covert signal was first observed
    t_flagged: float         # monotonic time the flag was emitted
    contributing_devices: list[str]
    signature_match: dict[str, Any] | None = None

    @property
    def latency_s(self) -> float:
        return self.t_flagged - self.t_signal

    @property
    def within_budget(self) -> bool:
        return self.latency_s <= FLAG_LATENCY_BUDGET_S

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "confidence": round(self.confidence, 4),
            "latency_ms": round(self.latency_s * 1000, 1),
            "budget_ms": FLAG_LATENCY_BUDGET_S * 1000,
            "within_budget": self.within_budget,
            "contributing_devices": self.contributing_devices,
            "signature_match": self.signature_match,
        }


@dataclass
class AlertManager:
    """Tracks signal-onset time per family so latency is measured from the
    first frame that suggested covert signaling, not from the consensus
    decision -- the proposal's <500ms budget is signal-to-flag, end to end.
    """

    _onset: dict[str, float] = field(default_factory=dict)
    history: list[Alert] = field(default_factory=list)

    def note_candidate(self, family: str, now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        self._onset.setdefault(family, now)

    def clear_candidate(self, family: str) -> None:
        self._onset.pop(family, None)

    def raise_alert(
        self,
        family: str,
        confidence: float,
        contributing_devices: list[str],
        signature_match: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> Alert:
        now = now if now is not None else time.monotonic()
        t_signal = self._onset.pop(family, now)
        alert = Alert(
            family=family,
            confidence=confidence,
            t_signal=t_signal,
            t_flagged=now,
            contributing_devices=contributing_devices,
            signature_match=signature_match,
        )
        self.history.append(alert)
        return alert
