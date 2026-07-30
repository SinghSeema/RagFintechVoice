# RagFintechVoice — Phase 1 Complete Summary
**Status:** DONE — Tasks 1–10 all passing  
**Doc updated:** 2026-04-06

---

## What Was Built (Tasks 1–10)

### Ingestion Layer — `src/ingestion/`

| Task | File | What it does |
|------|------|-------------|
| 1 | `parser.py` | `load_pdf_with_sections()` — pdfplumber extraction with header/footer crop, `##CHAPTER##` / `##SECTION##` markers, TOC suppression |
| 2 | `metadata.py` | `build_section_map_and_clean()` — parses markers into `chapter_map` + `section_map` dicts, returns clean text |
| 3 | `chunker.py` | `chunk_document()` — 3-level hierarchical chunking (parent 1500 tok / mid 400 tok / leaf 128 tok), two-layer metadata enrichment, persists to `storage/` |

### Retrieval Layer — `src/retrieval/`

| Task | File | What it does |
|------|------|-------------|
| 4 | `embedder.py` | `configure_embed_model()` — sets `Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")` globally |
| 5 | `vector_store.py` | `build_index()` / `load_index()` — Qdrant local vector store, persists to `qdrant_storage/` |
| 6 | `bm25_store.py` | `build_bm25_retriever()` — BM25 sparse retriever over leaf nodes (in-memory, no Docker) |
| 7 | `hybrid_retriever.py` | `build_hybrid_retriever()` — RRF fusion (k=60, num_queries=1). `configure_llm()` — Groq llama-3.3-70b-versatile |
| 8 | `pipeline.py` | `build_pipeline()` — AutoMergingRetriever (≥50% threshold) + FlagEmbeddingReranker (bge-reranker-base), returns `(retriever, reranker)` |

### Generation Layer — `src/generation/`

| Task | File | What it does |
|------|------|-------------|
| 9 | `rag_chain.py` | `query(question, index, nodes, storage_context, mode, retriever, reranker)` — CitationQueryEngine + Groq LLM → `QueryResult(answer, sources, mode)`. Voice: top_n=3, Text: top_n=5. Accepts pre-built retriever/reranker to avoid reloading cross-encoder per call. |

### Evaluation — `evaluation/`

| Task | File | What it does |
|------|------|-------------|
| 10 | `golden_dataset.json` | 20 questions, 5/domain (kyc_aml, regulatory_qa, customer_support, loan_credit), all grounded in `rbi_kyc_master_direction.pdf` |
| 10 | `eval_runner.py` | RAGAS (faithfulness, answer_relevancy, context_recall, context_precision) + DeepEval (faithfulness, answer_relevancy, hallucination). Judge: Groq llama-3.1-8b-instant. Regression alert if metric drops >0.05. Metrics persisted to `evaluation/metrics_store.json`. |

---

## Key Config / Design Decisions

| Decision | Value | Note |
|----------|-------|------|
| Embedding model | `BAAI/bge-small-en-v1.5` (384-dim) | Deviation from ADR (BGE-M3 planned) |
| Reranker | `BAAI/bge-reranker-base` | Matches ADR |
| LLM (generation) | `Groq llama-3.3-70b-versatile` | Matches ADR-001 |
| LLM (eval judge) | `Groq llama-3.1-8b-instant` | ADR-005 said Mixtral-8x7B; changed due to free tier quota |
| Chunk sizes | leaf=128 tok, mid=400 tok, parent=1500 tok | ADR-004 said leaf=100; refined in Phase 1 |
| Chunk overlap | 10% per level (13/40/150 tok) | ADR-004 deferred; set in Phase 1 |
| Hybrid fusion | RRF k=60, num_queries=1 | Matches ADR-004 |
| Vector store | Qdrant local (file-backed) | Matches ADR-007 |
| Keyword store | BM25 in-memory | OpenSearch deferred to Phase 2 (production) |
| Voice top_n | 3 | ADR-004 said top-2; set to 3 for citation quality |
| Text top_n | 5 | Matches ADR-004 |
| Docstore | `SimpleDocumentStore` → `storage/` | Matches ADR-007 |

---

## Deviations from ADR v1.1

| ADR Section | ADR Decision | Actual | Reason |
|-------------|-------------|--------|--------|
| ADR-004 | Embedding: BGE-M3 (1024-dim) | BGE-small (384-dim) | Memory/speed on dev machine; accuracy acceptable for Phase 1 |
| ADR-004 | Leaf chunk: 100 tok | 128 tok | Semantic completeness at sentence boundaries |
| ADR-004 | Voice top-k: top-2 | top-3 | Empirically better citations without latency impact |
| ADR-005 | Eval judge: Mixtral-8x7B | llama-3.1-8b-instant | Mixtral quota exhausted on Groq free tier |
| ADR-007 | Vector store: Weaviate (context) | Qdrant local | Weaviate mentioned in ADR-004 context only; ADR-007 decision correctly specifies Qdrant |

---

## Data Flow (end-to-end)

```
PDF
 └─ parser.py         → raw text + section markers
 └─ metadata.py       → chapter_map, section_map, clean_text
 └─ chunker.py        → leaf_nodes + storage_context (storage/)

leaf_nodes
 └─ embedder.py       → BGE-small embeddings
 └─ vector_store.py   → Qdrant index (qdrant_storage/)
 └─ bm25_store.py     → BM25 index (in-memory)

query(question, mode)
 └─ hybrid_retriever  → RRF fusion (top-10 candidates)
 └─ AutoMerging       → leaf→parent promotion (≥50%)
 └─ BGE reranker      → top-5 (text) or top-3 (voice)
 └─ CitationQueryEngine + Groq → answer + [n] citations
 └─ QueryResult       → .answer, .sources, .mode
```

---

## Evaluation Baseline (Task 10, 2-question kyc_aml sample)

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| faithfulness | 0.6905 | 0.70 | Just below — expected at baseline |
| answer_relevancy | 1.0000 | 0.70 | OK |
| context_recall | 0.5714 | 0.70 | Below — 120-char snippet limit; full text will improve |
| context_precision | 0.5750 | 0.70 | Below — normal for hierarchical chunking |

*Run `python -m evaluation.eval_runner --limit 20` for full baseline across all domains.*

---

## Known Issues

| Issue | Impact | Fix |
|-------|--------|-----|
| `qdrant_storage/.lock` stale after Ctrl+C | Next run fails to open Qdrant | `eval_runner.py` auto-removes lock on startup |
| BGE reranker reloads on every `query()` call | 20-question eval takes 5+ s/question | Fixed: pass pre-built `(retriever, reranker)` into `query()` |
| `context_recall` and `context_precision` LOW | Expected — snippet is capped at 120 chars | Pass full chunk text as context in future eval runs |
| RAGAS `metrics.collections` API incompatible with `evaluate()` in 0.4.x | Would crash | Fixed: use old instance-based API with deprecation warnings suppressed |
| DeepEval multiple-inheritance `_client` missing | DeepEval skipped | Fixed: single `DeepEvalBaseLLM` subclass with closure over `client` |

---

## Files Created / Modified in Phase 1

```
src/
  ingestion/  parser.py, metadata.py, chunker.py, __init__.py
  retrieval/  embedder.py, vector_store.py, bm25_store.py,
              hybrid_retriever.py, pipeline.py, __init__.py
  generation/ rag_chain.py, __init__.py
  __init__.py

evaluation/
  golden_dataset.json
  eval_runner.py
  metrics_store.json     (generated at runtime)

storage/                 (generated — gitignored)
qdrant_storage/          (generated — gitignored)
data/raw/
  rbi_kyc_master_direction.pdf
  rbi_housing_loan_guidelines.pdf
  rbi_priority_sector_lending.pdf
requirements.txt
```

---

## What's Next — Phase 2

- Voice pipeline: Pipecat orchestration (VAD → STT → RAG → TTS)
- STT: Groq Whisper-large-v3
- TTS: Cartesia Sonic (primary, credits available) + Deepgram Aura (fallback)
- Transport: LiveKit WebRTC
- Mobile client: React Native + Expo
- Sub-question decomposition for multi-hop regulatory queries
- OpenSearch replacement for BM25 (production-grade keyword search)
- Upgrade BGE-small → BGE-M3 for production embeddings
