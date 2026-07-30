# RagFintechVoice — How to Run

## Overview

The app has three runtimes that work together:

| Component | What it does | Port |
|-----------|-------------|------|
| **Text API** (`src/api/main.py`) | REST endpoint for chat/text queries | 8000 |
| **Voice server** (`src/voice/server.py`) | LiveKit token + agent launcher | 8001 |
| **Frontend** (`index.html`) | Browser UI (Chat + Voice tabs) | served statically |

The RAG pipeline (Qdrant vectors + chunked docstore) must be built once before either server starts.

---

## Prerequisites

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Requires Python 3.10+. First install downloads BGE-M3 (~2 GB) and BGE-reranker-base (~1 GB) from HuggingFace — this only happens once; models are cached in `~/.cache/huggingface/`.

### Environment variables

Create a `.env` file in the project root:

```bash
# Required for both servers
GROQ_API_KEY=gsk_...

# Required for voice server only
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=APImEY...
LIVEKIT_API_SECRET=TDO2PW...

# TTS — use one of these
DEEPGRAM_API_KEY=...         # primary (always required as fallback)
CARTESIA_API_KEY=...         # optional — if set, takes over as TTS primary

# Optional
LIVEKIT_ROOM=rag-voice       # default: rag-voice
QDRANT_URL=                  # leave empty for local file-based Qdrant (default)
                             # set to http://localhost:6333 for Docker Qdrant
```

### Source PDFs

Place RBI source documents in `data/raw/`:

```
data/raw/rbi_kyc_master_direction.pdf
data/raw/rbi_housing_loan_guidelines.pdf
data/raw/rbi_priority_sector_lending.pdf
```

---

## Step 1 — Build the RAG index (one-time, ~15 min)

This parses PDFs, creates the three-level chunk hierarchy (leaf/mid/parent), enriches metadata, embeds leaf nodes with BGE-M3, and persists everything to `storage/` and `qdrant_storage/`.

```bash
# Parse + chunk + enrich
python -m src.ingestion.chunker

# Embed + index into Qdrant
python -m src.retrieval.vector_store
```

Subsequent runs skip this — the servers load from `storage/` and `qdrant_storage/` on startup (sub-second).

**To force a full rebuild** (required after changing chunk sizes):

```bash
rm -rf storage/ qdrant_storage/
python -m src.ingestion.chunker
python -m src.retrieval.vector_store
```

---

## Step 2 — Start the Text API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Verify:

```bash
curl http://localhost:8000/health
# → {"status": "ok"}

curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What documents does an NRI need for KYC?"}'
```

The `--reload` flag auto-restarts on code changes. Drop it in production.

---

## Step 3 — Start the Voice Server

```bash
uvicorn src.voice.server:app --host 0.0.0.0 --port 8001 --reload
```

Verify:

```bash
curl http://localhost:8001/health
# → {"status": "ok"}
```

The voice server exposes:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/token` | GET | Issue LiveKit JWT for browser/mobile client |
| `/agent/start` | POST | Launch agent in a custom room (default room auto-starts at startup) |
| `/agent/stop` | POST | Cancel the active agent for a room |
| `/agent/status` | GET | List active rooms |
| `/health` | GET | Liveness probe |

**Startup behavior (Phase 3):** The RAG stack (PDF parse, Qdrant index, BGE reranker)
is pre-loaded in the background immediately after `uvicorn` starts (~15–30 s).  Once
ready, an agent is automatically started in the default room (`LIVEKIT_ROOM` or
`rag-voice`).  When the first user joins, the greeting fires immediately with no
per-connection initialization delay.

---

## Step 4 — Open the Frontend

The voice server (port 8001) also serves the frontend — no separate static server needed.

Once both servers are running, open:

```
http://localhost:8001
```

The voice server's `GET /` route returns `index.html`, and `GET /livekit-client.umd.js`
serves the vendored LiveKit JS SDK alongside it.

The frontend's Chat tab calls the Text API at `http://localhost:8000`.
The frontend's Voice tab connects to LiveKit via the token and agent endpoints on `http://localhost:8001`.

---

## Running Both Servers Together

Use two terminal windows or a process manager:

```bash
# Terminal 1
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Terminal 2
uvicorn src.voice.server:app --host 0.0.0.0 --port 8001
```
kill -9 $(lsof -ti:8000) && uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
Or with `make` / `tmux` / `honcho` — no Procfile is included yet.

---

## Logs

- **Structured logs**: printed to stdout by loguru (each query logs retrieve/rerank/generation timing)
- **Audit log**: every query appended to `logs/audit.jsonl` — never truncated, append-only

```bash
tail -f logs/audit.jsonl | python -m json.tool
```

---

## Running Evaluations

```bash
python evaluation/eval_runner.py
# or with a limit for quick smoke-test:
python evaluation/eval_runner.py --limit 5
```

Results are written to `evaluation/results/`. Requires `GROQ_API_KEY` in `.env`. Uses the golden dataset at `evaluation/golden_dataset.json`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `KeyError: GROQ_API_KEY` on startup | `.env` not found or key missing | Check `.env` file is in project root |
| `collection 'fintech_rag' not found` | Index not built | Run Step 1 |
| Reranker takes 3–5s per query | BGE cross-encoder is CPU-bound | Expected; use GPU or reduce `similarity_top_k` |
| Voice answer is vague / incomplete | Old 100-token chunks still in `storage/` | `rm -rf storage/ qdrant_storage/` and rebuild |
| TTS reads "dash dash" / "Note colon" | Voice prompt returning markdown | Check `_VOICE_QA_TEMPLATE` is selected for `mode="voice"` |
| Duplicate RAG queries in voice log | STT emitting trailing fragments | Dedup window active — check `pipeline.py` dedup logic |
| Port 8000/8001 already in use | Previous server still running | `lsof -i :8000` then `kill <PID>` |
