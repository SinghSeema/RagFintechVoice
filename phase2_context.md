# RagFintechVoice — Phase 2 Session Context
**Feed this file at the start of every Phase 3 session.**
**Phase 2 complete. Voice pipeline live: Pipecat + LiveKit + Deepgram TTS.**

---

## Stack (locked)

| Layer | Tool | Model / Config |
|-------|------|----------------|
| LLM (generation) | Groq | `llama-3.3-70b-versatile` |
| LLM (voice) | Groq | `llama-3.1-8b-instant` (fast, low latency) |
| LLM (PII/KYC) | Ollama | `mistral-7b` local — PII never leaves server |
| STT | Groq Whisper | `whisper-large-v3` via Groq API |
| TTS (active) | Deepgram | Aura `aura-asteria-en` — key in `.env`, 16kHz linear16 |
| TTS (primary, pending) | Cartesia | Sonic — credits available, swap by adding `CARTESIA_API_KEY` to `.env` |
| Embedding | HuggingFace | `BAAI/bge-small-en-v1.5` (384-dim, local) |
| Reranker | HuggingFace | `BAAI/bge-reranker-base` (local) |
| Vector store | Qdrant | Local file-backed (`qdrant_storage/`) |
| Keyword search | BM25 | In-memory (OpenSearch deferred to Phase 3) |
| Orchestration | Pipecat | `0.0.108` — VAD → STT → RAG → TTS processor chain |
| Transport | LiveKit | WebRTC — `wss://woka-qe4nlyjl.livekit.cloud` |
| Mobile client | React Native + Expo | Phase 3 — not yet built |

---

## Phase 2 Files Built

| Path | Purpose |
|------|---------|
| `src/voice/pipeline.py` | `build_voice_pipeline()` — full Pipecat voice pipeline |
| `src/voice/server.py` | FastAPI agent server — `/token`, `/agent/start`, `/agent/stop`, `/agent/status`, `/health` |
| `src/api/main.py` | FastAPI text gateway — `POST /query`, `POST /query/decompose` |
| `src/api/__init__.py` | Package init |
| `src/generation/subquery.py` | `build_sub_engine()` — SubQuestionQueryEngine (3 doc tools) |
| `tests/test_voice_pipeline.py` | 23 unit tests — helpers, RAGProcessor, TTS factory |
| `test_client.html` | Browser test client — connects to LiveKit, plays bot audio |
| `livekit-client.umd.js` | LiveKit JS SDK v1.15.13 (local — avoids CDN issues) |
| `logs/audit.jsonl` | Append-only audit log (ADR hard constraint) |

---

## Running the Voice Pipeline

```bash
# Terminal 1 — Agent server (loads RAG stack on startup, ~30s)
uvicorn src.voice.server:app --host 0.0.0.0 --port 8001

# Terminal 2 — HTML test client server
cd /home/ranjit/AI_Work/RagFintechVoice
python -m http.server 8080

# Terminal 3 — Start the RAG agent in the room
curl -X POST "http://localhost:8001/agent/start?room=rag-voice"

# Browser — open test client
http://localhost:8080/test_client.html
# Click "Connect + Start Agent" → allow mic → speak
```

## Running the Text API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Single-shot RAG
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What documents does an NRI need for KYC?"}'

# Multi-hop decomposition
curl -X POST http://localhost:8000/query/decompose \
     -H "Content-Type: application/json" \
     -d '{"question": "Compare simplified KYC with full CDD requirements"}'
```

---

## Phase 2 API Surface

### Voice server (`src/voice/server.py` — port 8001)

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness probe |
| `GET /token?room=X&identity=Y` | Issues 1-hour LiveKit JWT for browser/mobile client |
| `POST /agent/start?room=X` | Launches Pipecat RAG agent as participant in room |
| `POST /agent/stop?room=X` | Gracefully cancels the agent task |
| `GET /agent/status` | Lists active/stopped agent tasks |

### Text gateway (`src/api/main.py` — port 8000)

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness probe |
| `POST /query` | Single-shot RAG (text mode, Groq rate-limit retry) |
| `POST /query/decompose` | Multi-hop via SubQuestionQueryEngine |

---

## Phase 1 RAG API (unchanged)

```python
from src.generation.rag_chain import query, QueryResult
from src.retrieval.pipeline import build_pipeline

retriever, reranker = build_pipeline(index, leaf_nodes, storage_context)
result = query(
    question, index=index, nodes=leaf_nodes, storage_context=storage_context,
    mode="voice",  # or "text"
    retriever=retriever, reranker=reranker,
)
# result.answer : str  (inline [1] [2] citations stripped before TTS)
# result.sources: list[Source]  (rank, section_title, chapter, snippet)
# result.mode   : "voice" | "text"
```

---

## Key File Paths (Phase 1 + 2)

| Path | Purpose |
|------|---------|
| `src/generation/rag_chain.py` | `query()` — main RAG entry point |
| `src/generation/subquery.py` | `build_sub_engine()` — multi-hop decomposition |
| `src/retrieval/pipeline.py` | `build_pipeline()` — retriever + reranker |
| `src/retrieval/hybrid_retriever.py` | `configure_llm()` — must call before any query |
| `src/retrieval/vector_store.py` | `load_index()` — loads Qdrant from disk |
| `src/ingestion/chunker.py` | `chunk_document()` — load leaf_nodes + storage_context |
| `src/voice/pipeline.py` | Pipecat voice pipeline |
| `src/voice/server.py` | LiveKit agent server |
| `src/api/main.py` | Text query FastAPI gateway |
| `storage/` | Persisted docstore — gitignored |
| `qdrant_storage/` | Persisted Qdrant index — gitignored |
| `data/raw/` | Source PDFs (3 RBI documents) |
| `evaluation/eval_runner.py` | RAGAS + DeepEval harness |
| `logs/audit.jsonl` | Append-only audit log (ADR hard constraint) |

---

## Eval Baseline (partial — 2 questions, kyc_aml domain)

| Metric | Score | Threshold |
|--------|-------|-----------|
| faithfulness | 0.69 | 0.70 |
| answer_relevancy | 1.00 | 0.70 |
| context_recall | 0.57 | 0.70 |
| context_precision | 0.58 | 0.70 |

Run full baseline: `python -m evaluation.eval_runner --limit 20 --skip-deepeval`

---

## Groq Free Tier Limits (watch these)

| Resource | Limit | RAG cost | Headroom |
|----------|-------|----------|----------|
| TPM | 6,000 | ~1,500/call | ~4 concurrent |
| RPD (llama-3.3-70b) | 14,400 | 1/call | ~14,400 queries/day |
| RPD (whisper-large-v3) | 2,000 | 1/utterance | ~2,000 voice turns/day |

---

## Bot Audio Not Audible — Fixes Applied (Phase 2)

Root cause was a chain of 4 bugs, diagnosed via `DebugFrameLogger` at every pipeline stage.

| # | Bug | Fix |
|---|-----|-----|
| 1 | `DebugFrameLogger` didn't call `super().process_frame()` | Added `await super().process_frame(frame, direction)` first — without it `__started` stays False and `push_frame` silently drops every frame |
| 2 | `stop_secs=0.8` in Silero VAD too short | Increased to `stop_secs=1.5` — 0.8s cut sentences mid-word, triggering separate RAG queries per fragment |
| 3 | `double LLMFullResponseStartFrame` on error path | Removed duplicate push — error handler was pushing Start twice before the error text |
| 4 | Agent token `hidden=True` + wrong event name | Changed to `hidden=False` (hidden agents may not trigger `TrackSubscribed` on browser clients); fixed `on_participant_joined` → `on_first_participant_joined` (correct Pipecat event name) |

Pipeline frame flow (confirmed healthy via logs):
```
UserAudioRawFrame → [VAD/STT] → TranscriptionFrame (finalized=True)
→ RAGProcessor._answer() → LLMFullResponseStartFrame + TextFrame(s) + LLMFullResponseEndFrame
→ DeepgramTTSService → TTSAudioRawFrame (16kHz, linear16)
→ LiveKitOutputTransport → published to room as audio track
→ Browser: TrackSubscribed → track.attach() → room.startAudio() → audio plays
```

Browser autoplay: call `room.startAudio()` inside the Connect button click handler (user-gesture context). If blocked, show an "Unmute" button that calls `startAudio()` again.

---

## Known Issues / Open Items Entering Phase 3

| Issue | Status | Fix |
|-------|--------|-----|
| `context_recall` / `context_precision` LOW | Open | Pass full chunk text (not 120-char snippet) to RAGAS |
| Eval judge should be Mixtral-8x7B | Open | Revert when paid Groq key available |
| BGE-small → BGE-M3 for production | Phase 3 | Memory/speed trade-off |
| BM25 → OpenSearch | Phase 3 | Synonym expansion for CIBIL, PMLA, etc. |
| Nightly eval cron | Open | Add as CI/cron step |
| RAGAS eval broken on Groq free tier | Open | `'n': number must be at most 1` — Groq doesn't support `n>1` in async RAGAS job dispatch; needs custom LLM wrapper |
| React Native mobile client | Phase 3 | Not yet built |
| Audit log + Groq rate-limit retry not wired into voice pipeline | Open | Only in text API (`src/api/main.py`); voice turns bypass it |
| `llama-index-retrievers-bm25` pinned to `0.5.0` | Fixed | v0.7.1 broke `build_metadata_filter_fn` import on llama-index-core 0.12.x |
