"""Cloud AI 100 signature-library service -- embedding similarity search.

*** RUNS LOCALLY AS A CPU REFERENCE IMPLEMENTATION. NOT DEPLOYED TO REAL
CLOUD AI 100 HARDWARE -- see the blocker note in ARCHITECTURE.md. ***

Why this file exists instead of the qefficient_template/ next to it:
mcp__quad__generate_code(platform="cloud", sdk="qaic") only has one deploy
template (deploy_target="qefficient"), and it is shaped for LLM chat serving
(QEFFAutoModelForCausalLM + an OpenAI-compatible /v1/chat/completions route --
see qefficient_template/serve.py). Echo Chamber's Cloud AI 100 role per the
proposal is different: "maintains an embedding-based signature library ...
matched at query time via similarity search" -- a nearest-neighbor lookup
over fixed-length feature vectors, not causal-LM text generation. Rather than
force an LLM-serving template onto a non-LLM workload, this is a small,
honest, purpose-built FastAPI service that implements the real architecture.

Production path (once real AI100 access + credentials exist): replace
`echo_chamber.signature_reference.reference_library()` with a
QEfficient-compiled embedding model running on-card (a proper embedding
model, e.g. a distilled spectral
autoencoder trained on much more signature data than we could synthesize
here), and the /match handler's cosine-similarity core stays the same --
only where the embedding comes from changes. That swap needs real Cloud AI
100 access this environment does not have; stop and ask before wiring it in.

Run:
    pip install -r requirements.txt
    uvicorn signature_service:app --host 0.0.0.0 --port 8100
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from echo_chamber import N_BANDS  # noqa: E402
from echo_chamber.signature_reference import cosine as _cosine  # noqa: E402
from echo_chamber.signature_reference import reference_library as _reference_library  # noqa: E402

app = FastAPI(title="Echo Chamber — Cloud AI 100 Signature Library (local reference)")


class MatchRequest(BaseModel):
    spectral_features: list[float]  # len == N_BANDS, from echo_chamber.dsp.extract_band_features


class MatchResult(BaseModel):
    family: str
    similarity: float


class MatchResponse(BaseModel):
    best_match: MatchResult
    all_matches: list[MatchResult]
    provider: str = "local-reference-cpu"  # NOT "cloud-ai100" -- see module docstring


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "provider": "local-reference-cpu", "families": list(_reference_library())}


@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest) -> MatchResponse:
    if len(req.spectral_features) != N_BANDS:
        raise ValueError(f"expected {N_BANDS} features, got {len(req.spectral_features)}")
    query = np.asarray(req.spectral_features, dtype=np.float32)
    library = _reference_library()
    scored = sorted(
        (MatchResult(family=name, similarity=_cosine(query, emb)) for name, emb in library.items()),
        key=lambda m: m.similarity,
        reverse=True,
    )
    return MatchResponse(best_match=scored[0], all_matches=scored)
