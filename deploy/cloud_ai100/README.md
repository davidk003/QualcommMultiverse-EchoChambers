# Cloud AI 100 — embedding signature library

**Status: local CPU reference implementation. Not deployed to real Cloud AI
100 hardware** — no physical/cloud AI100 access exists in this environment.
See the blocker note in `../../ARCHITECTURE.md`.

| File | What it is |
|---|---|
| `signature_service.py` | Hand-written FastAPI service. Implements the actual architecture — embedding similarity search over known covert-channel families — and is real, runnable, tested code (see `../../tests/`). Runs anywhere with Python; the embedding step is CPU-only here. |
| `qefficient_template/` | Raw `mcp__quad__generate_code(platform="cloud", sdk="qaic", deploy_target="qefficient")` output, unedited. Kept as a QUAD-standard record. **Not used** — see below. |

## Why the QUAD-generated template isn't what's deployed

`generate_code`'s only `platform="cloud"` + `sdk="qaic"` template
(`deploy_target="qefficient"`) is shaped for LLM chat serving
(`QEFFAutoModelForCausalLM`, an OpenAI-compatible `/v1/chat/completions`
route). Echo Chamber's Cloud AI 100 role is a nearest-neighbor similarity
search over fixed-length spectral embeddings, not causal-LM text generation
— a structurally different workload. Rather than force-fit an LLM template
onto it, `signature_service.py` implements the real thing directly. This is
a genuine gap in `generate_code`'s template coverage (no generic
embedding/similarity-search template for `platform="cloud"`), worth raising
upstream rather than working around silently.

## Production path (needs real Cloud AI 100 access — stop and ask)

Replace `_reference_library()` in `signature_service.py` with a
QEfficient-compiled embedding model actually running on Cloud AI 100
silicon, trained on far more signature data than this project could
synthesize. The `/match` endpoint's cosine-similarity logic doesn't change —
only where the embedding comes from does. This needs real AI100 hardware/
cloud credentials this environment doesn't have.

## Run the reference service locally

```
pip install -r requirements.txt
uvicorn signature_service:app --host 0.0.0.0 --port 8100
curl -X POST localhost:8100/match -H "content-type: application/json" \
    -d '{"spectral_features": [0.1, 0.2, ...]}'   # 64 values
```
