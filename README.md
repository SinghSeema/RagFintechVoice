# RagFintechVoice

> A voice-enabled RAG (Retrieval-Augmented Generation) assistant for Indian fintech compliance — talk to RBI regulatory documents using your voice or chat interface.

---

## What It Does

RagFintechVoice lets you ask natural-language questions about RBI regulatory documents — KYC guidelines, housing loan rules, priority sector lending norms — and get accurate, cited answers in real-time. You can interact via:

- **Voice**: speak your question; a voice agent transcribes, retrieves, and speaks back the answer (sub-second first-word latency).
- **Chat**: type a question; the API returns a structured answer with source citations.

Built for compliance analysts, loan officers, and fintech developers who need fast, accurate answers from dense regulatory PDFs without reading them end-to-end.

---

## Architecture

```
Browser (index.html)
  ├── Chat tab  ──► Text API (port 8000)  ──► RAG Pipeline ──► Groq LLM
  └── Voice tab ──► Voice Server (port 8001)
                       ├── LiveKit WebRTC room
                       └── Pipecat pipeline
                             VAD → STT (Deepgram/Groq Whisper)
                                → RAG Processor
                                → TTS (Deepgram Aura / Cartesia Sonic)
```

### RAG Pipeline

| Layer | Technology |
|-------|-----------|
| PDF Parsing | pdfplumber |
| Chunking | 3-level hierarchy: leaf (256 tokens) → mid (400) → parent (1500) |
| Embedding | BGE-M3 (BAAI/bge-m3, 1024-dim, local) |
| Vector Store | Qdrant (local file-based or Docker) |
| Sparse Retrieval | BM25 (LlamaIndex) |
| Fusion | Reciprocal Rank Fusion (BM25 + dense) |
| Reranker | BGE-reranker-large (cross-encoder) |
| LLM | Groq — Llama-3.1-8B-Instant / Llama-3.3-70B |
| Answer Engine | LlamaIndex CitationQueryEngine + SubQuestionQueryEngine |

### Voice Pipeline

| Layer | Technology |
|-------|-----------|
| WebRTC | LiveKit Cloud |
| VAD | Silero VAD (via Pipecat) |
| STT | Deepgram Nova-2 / Groq Whisper-large-v3 |
| TTS | Deepgram Aura (default) · Cartesia Sonic (primary when key set) |
| Agent Framework | Pipecat |
| API Server | FastAPI + Uvicorn |

---

## Project Structure

```
RagFintechVoice/
├── src/
│   ├── api/
│   │   └── main.py              # Text query REST API (port 8000)
│   ├── ingestion/
│   │   ├── parser.py            # PDF → raw text
│   │   ├── chunker.py           # 3-level chunk hierarchy + metadata
│   │   ├── metadata.py          # Metadata enrichment
│   │   └── smart_map_builder.py # Document structure mapping
│   ├── retrieval/
│   │   ├── vector_store.py      # Qdrant index builder
│   │   ├── embedder.py          # BGE-M3 wrapper
│   │   ├── hybrid_retriever.py  # BM25 + dense fusion
│   │   ├── cascading_retriever.py
│   │   └── strategy_factory.py  # Retrieval strategy selector
│   ├── generation/
│   │   ├── rag_chain.py         # CitationQueryEngine
│   │   └── subquery.py          # SubQuestionQueryEngine (3 doc tools)
│   └── voice/
│       ├── pipeline.py          # Pipecat: VAD→STT→RAG→TTS
│       └── server.py            # LiveKit agent server (port 8001)
├── evaluation/
│   ├── eval_runner.py           # RAGAs + DeepEval evaluation runner
│   └── golden_dataset.json      # Hand-labelled Q&A pairs
├── data/
│   └── raw/                     # Place RBI PDFs here (not committed)
├── index.html                   # Frontend (Chat + Voice tabs)
├── requirements.txt
├── Makefile
├── RUNBOOK.md                   # Step-by-step run guide
└── .env.example                 # Environment variable template
```

---

## Quickstart

### 1. Prerequisites

- Python 3.10+
- LiveKit Cloud account (free tier works)
- API keys: Groq, Deepgram (or Cartesia), LiveKit

### 2. Install

```bash
git clone https://github.com/SinghSeema/RagFintechVoice.git
cd RagFintechVoice

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> First install downloads BGE-M3 (~2 GB) and BGE-reranker-large (~1 GB) from HuggingFace — cached at `~/.cache/huggingface/` after the first run.

### 3. Configure environment

Copy `.env.example` to `.env` and fill in your keys:

```bash
# LLM (required for both servers)
GROQ_API_KEY=gsk_...

# LiveKit (required for voice)
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=APIm...
LIVEKIT_API_SECRET=...

# TTS — choose one (Deepgram required as fallback)
DEEPGRAM_API_KEY=...
CARTESIA_API_KEY=...    # optional — if set, becomes primary TTS
```

### 4. Add RBI source PDFs

Place these in `data/raw/`:

```
data/raw/rbi_kyc_master_direction.pdf
data/raw/rbi_housing_loan_guidelines.pdf
data/raw/rbi_priority_sector_lending.pdf
```

### 5. Build the RAG index (one-time, ~15 min)

```bash
python -m src.ingestion.chunker
python -m src.retrieval.vector_store
```

### 6. Start the servers

```bash
# Terminal 1 — Text API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Voice server + frontend
uvicorn src.voice.server:app --host 0.0.0.0 --port 8001
```

Open **http://localhost:8001** in your browser.

---

## API Reference

### Text API (port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/query` | POST | Ask a question, get a cited answer |

**Example:**
```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What documents does an NRI need for KYC?"}'
```

### Voice Server (port 8001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/token` | GET | Issue LiveKit JWT for browser client |
| `/agent/start` | POST | Launch voice agent in a room |
| `/agent/stop` | POST | Stop active agent |
| `/agent/status` | GET | List active rooms |

---

## Evaluation

```bash
python evaluation/eval_runner.py          # full run
python evaluation/eval_runner.py --limit 5  # quick smoke test
```

Uses [RAGAs](https://github.com/explodinggradients/ragas) and [DeepEval](https://github.com/confident-ai/deepeval) against a hand-labelled golden dataset. Results written to `evaluation/results/`.

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Chunking | 3-level hierarchy (leaf/mid/parent) | AutoMerge retrieval — get full context, not just the matching fragment |
| Embedding | BGE-M3 (local) | Best multilingual + domain retrieval; no per-query embedding cost |
| Fusion | BM25 + dense | BM25 catches exact regulatory terms (e.g. "V-CIP"); dense catches semantics |
| TTS | Deepgram Aura / Cartesia Sonic | 90ms TTFT with byte-level streaming; swap via single env var |
| Voice agent | Pipecat + LiveKit | Production-grade VAD, STT dedup, sentence-level TTS streaming |
| LLM | Groq (Llama-3.1-8B) | Sub-100ms token latency on free tier; swap to 70B for accuracy |

Full ADR: [fintech_rag_adr_v1.2.md](fintech_rag_adr_v1.2.md)

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | — | Groq API key (LLM + Whisper STT) |
| `LIVEKIT_URL` | Voice only | — | LiveKit WebSocket URL |
| `LIVEKIT_API_KEY` | Voice only | — | LiveKit API key |
| `LIVEKIT_API_SECRET` | Voice only | — | LiveKit API secret |
| `DEEPGRAM_API_KEY` | Voice only | — | Deepgram key (STT or TTS) |
| `CARTESIA_API_KEY` | No | — | If set, Cartesia becomes primary TTS |
| `LIVEKIT_ROOM` | No | `rag-voice` | Default LiveKit room name |
| `QDRANT_URL` | No | local | Set to `http://localhost:6333` for Docker Qdrant |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `KeyError: GROQ_API_KEY` | Check `.env` is in project root with the correct key |
| `collection 'fintech_rag' not found` | Run `python -m src.ingestion.chunker` and `python -m src.retrieval.vector_store` |
| Reranker slow (3–5s) | Expected on CPU; reduce `similarity_top_k` or use GPU |
| TTS reads "dash dash" | LLM returned markdown; check voice prompt is selected |
| Port already in use | `lsof -i :8000` then `kill <PID>` |

See [RUNBOOK.md](RUNBOOK.md) for the full step-by-step guide.

---

## Tech Stack

- **Python 3.10+**
- [LlamaIndex](https://github.com/run-llama/llama_index) — RAG framework
- [Pipecat](https://github.com/pipecat-ai/pipecat) — voice agent framework
- [LiveKit](https://livekit.io) — WebRTC real-time transport
- [Qdrant](https://qdrant.tech) — vector database
- [Groq](https://groq.com) — ultra-fast LLM inference
- [Deepgram](https://deepgram.com) — STT + TTS
- [Cartesia](https://cartesia.ai) — streaming TTS (optional)
- [FastAPI](https://fastapi.tiangolo.com) — REST API server

---

## License

MIT
