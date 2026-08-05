"""Client for the Cloud AI 100 embedding signature-library service.

Calls out to `deploy/cloud_ai100/signature_service.py` over HTTP when a
`base_url` is configured and reachable. Falls back to an in-process copy of
the same reference library when it isn't (default for a single-laptop demo,
and for tests) -- both paths return the exact same shape of result, so the
caller (run_echo_chamber.py) doesn't need to know which one answered; the
result always says which one did, since "ran the real cloud service" and
"ran the local fallback" are not the same claim.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class SignatureClient:
    def __init__(self, base_url: str | None = None, timeout_s: float = 0.3) -> None:
        self.base_url = base_url.rstrip("/") if base_url else None
        self.timeout_s = timeout_s

    def match(self, spectral_features: list[float]) -> dict[str, Any]:
        if self.base_url:
            try:
                return self._match_remote(spectral_features)
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                return {**self._match_local(spectral_features), "remote_error": str(exc)}
        return self._match_local(spectral_features)

    def _match_remote(self, spectral_features: list[float]) -> dict[str, Any]:
        body = json.dumps({"spectral_features": spectral_features}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/match", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read())

    def _match_local(self, spectral_features: list[float]) -> dict[str, Any]:
        import numpy as np

        from .signature_reference import cosine, reference_library

        query = np.asarray(spectral_features, dtype=np.float32)
        library = reference_library()
        scored = sorted(
            ({"family": name, "similarity": cosine(query, emb)} for name, emb in library.items()),
            key=lambda m: m["similarity"], reverse=True,
        )
        return {"best_match": scored[0], "all_matches": scored, "provider": "local-reference-cpu"}
