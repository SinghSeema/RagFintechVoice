# RAG Pipeline Revamp — Architecture & Approach

**Date:** 2026-04-19  
**Project:** RagFintechVoice — RBI KYC Regulatory Compliance Chatbot  
**Status:** Pre-implementation design review  
**Scope:** RAG layer only — voice pipeline fixes deferred

---

## 1. Problem Statement (What Is Actually Broken)

The current pipeline has one root cause that explains most of the observed answer quality failures:

> **Fixed-size paragraph chunks do not carry enough semantic signal for the retrieval model to distinguish between topically related but semantically different regulatory sections.**

### Concrete example from logs

Query: *"What documents are needed for individual KYC?"*

Current retrieval returns **§38 (Periodic KYC Updation)** — scores -2.26 to -3.89. All nodes pass the `top_score - 3.0` and `-4.0` floor filters correctly. The LLM gets chunks about *updating* existing KYC records and correctly recognises they don't fully answer the question, producing a hedged or vague response.

Why §38 ranks above §16 (Initial CDD Procedure)?

A 256-token chunk from §38 contains the tokens: *individual, KYC, documents, procedure, Rule 9, customer* — same high-frequency terms as §16. The embedding model can't distinguish "update KYC documents" from "submit KYC documents initially" when both appear in similar-length paragraphs with the same vocabulary. BGE-M3 is doing its job; the input text is ambiguous.

---

## 2. What the New Approach Does Differently

Both new strategies attack the same root cause at **ingestion time**, not query time:

### Strategy A — Agentic Proposition Extraction

Transform each raw paragraph into standalone, self-contained propositions **before** embedding.

Before (raw §38 chunk, 256 tokens):
```
"They shall also ensure, in terms of sub-rule (14) of Rule 9, that the records
and documents are preserved for a period of five years after the business
relationship ends or the transaction is completed, whichever is later.
Periodic updation of KYC must be done for existing individual customers within
the prescribed timeframes..."
```

After (propositions — each becomes one TextNode):
```
"Regulated entities must preserve KYC records for five years after the business
 relationship with a customer ends. (Rule 9(14))"

"Regulated entities must preserve KYC records for five years after the relevant
 transaction is completed. (Rule 9(14))"

"Regulated entities must periodically update KYC records for existing individual
 customers within the prescribed timeframes. (Section 38)"
```

Now the embedding model gets unambiguous signal: "preserve/periodically update" vs "submit/obtain documents initially". The wrong section no longer retrieves for onboarding queries.

### Strategy B — Pre-Generated Q&A Library

Generate question-answer pairs offline and embed the **question text** for retrieval. User query matches stored questions directly — much higher cosine similarity than query-to-chunk.

```
Stored: "What documents are required for KYC of an individual customer?"
Query:  "What documents are needed for individual KYC?"
→ cosine similarity: ~0.92 (question-to-question match is direct)
→ returns the pre-written, complete, curated answer
```

This completely bypasses the retrieval-precision problem and the reranker.

---

## 3. Will This Actually Improve Accuracy? — Honest Assessment

### Where improvement is high-confidence

| Query pattern | Expected improvement | Reason |
|---|---|---|
| Initial KYC document list | **High** | Propositions explicitly say "onboarding"; §38 propositions say "update" — no confusion |
| Specific thresholds (INR amounts, time periods) | **High** | Propositions extract exact figures with explicit context; raw chunks dilute them |
| Known compliance Q&A (V-CIP timeline, PEP definition) | **High** | Q&A library returns complete pre-written answers, no synthesis errors |
| Modal verb precision (SHALL vs MAY) | **High** | Propositions preserve modal verbs; raw chunks hide them in paragraphs |

### Where improvement is moderate or uncertain

| Query pattern | Concern | Mitigation |
|---|---|---|
| Deep cross-references ("as per clause 16(b) above") | Llama-8B may flag `[XREF: resolve manually]` rather than resolve; knowledge gap | Fallback to sentence splitting for that batch; XREF flags visible in metadata |
| Complex nested conditionals ("Provided that, in case...") | Llama-8B may drop alternatives or exception clauses | Validation pass on first 20 batches before full ingestion |
| Questions requiring holistic section context | Over-splitting into atoms may scatter context across 10+ propositions; retrieval with `top_k=20` needed | Increase `similarity_top_k` for agentic path; BM25 on propositions helps with section-number matching |
| Novel phrasing of edge-case queries | Q&A library misses; falls through to agentic (intended behaviour) | This is the correct design — cascading handles it |

### Where this approach does NOT help

| Issue | Why not addressed here | What fixes it |
|---|---|---|
| Intent classifier FINANCE misclassification | Voice layer problem, unrelated to chunk format | Narrow tiebreaker rule in `_CLASSIFIER_PROMPT` |
| Race condition (stale RAG result pushes after interruption) | Async architecture in RAGProcessor | Cancel in-flight `Future` on new `TranscriptionFrame` |
| Reranker latency floor (~17s on CPU) | The CPU inference speed of `bge-reranker-base` is fixed | Q&A fast path bypasses reranker; agentic path still pays this cost |

### The real risk: Llama-8B proposition quality at scale

The RBI KYC document has ~200 paragraphs across 100+ pages. At batch size 3, that's ~70 API calls. Quality risks:

1. **Hallucinated propositions** — Llama-8B silently alters a threshold (e.g., "five years" → "3 years") or drops an exception clause. For a compliance chatbot, this is worse than the original noisy text — a wrong answer with high confidence.

2. **Amendment noise leakage** — "Substituted vide RBI/2023-24/..." paragraphs that the prompt should filter. If even 5% leak through as propositions, they create spurious knowledge.

3. **Drift after 30+ batches** — Even with the wrong-example anchor, Llama-8B degrades on pronoun resolution after extended batches. The first 10 batches are usually clean; batches 50-70 show regression.

**Mitigation plan:** Before full ingestion, run a validation pass on 20 random paragraphs from the densest cross-reference sections (Chapter VI, Annexure I) and manually verify proposition quality. If error rate > 10%, switch the proposition model to `llama-3.1-70b-versatile` or add a verification step.

**Net verdict:** The agentic approach will improve accuracy for the documented failure case and similar topical-confusion failures. It introduces a new risk class (hallucinated facts) that must be validated before production use. The Q&A library is lower risk (pre-written answers can be manually reviewed) and should deliver high-confidence improvement for predictable compliance queries.

---

## 4. Architecture: Complete Separation

### Design principle

The revamp runs as a **completely separate pipeline**. Not a refactor — a parallel system. Every new file lives in a `revamp/` namespace or has `_revamp` suffix. The current pipeline is untouched. Switching is one `.env` variable.

### File isolation map

```
Current (DO NOT TOUCH):                New (revamp path):
─────────────────────────────          ───────────────────────────────────────
src/ingestion/chunker.py               src/ingestion/agentic_chunker.py
src/ingestion/metadata.py              src/ingestion/qa_ingester.py
src/retrieval/pipeline.py              src/retrieval/revamp_pipeline.py
src/retrieval/hybrid_retriever.py      src/retrieval/cascading_retriever.py
src/retrieval/vector_store.py          src/retrieval/revamp_vector_store.py
src/generation/rag_chain.py            src/generation/revamp_rag_chain.py
                                       scripts/generate_qa_library.py
                                       scripts/ingest_revamp.py
                                       scripts/eval_revamp.py
config/settings.py                     config/revamp_settings.py
```

**The only shared file** is `src/ingestion/parser.py` (PDF loading) — read-only, no modification.

### `.env` switch

```env
# "legacy" → current HierarchicalNodeParser + AutoMergingRetriever pipeline
# "revamp"  → new agentic + Q&A pipeline (this document)
PIPELINE_MODE=legacy
```

The application entry point (`src/api/main.py`, `src/voice/pipeline.py`) reads `PIPELINE_MODE` and routes to the appropriate `query()` function. Both return the same `QueryResult` dataclass — no caller changes needed.

---

## 5. Data Flow Diagrams

### Current pipeline (legacy)

```
PDF
 │
 ▼
loader.py  ──────────────────────────────── pdfplumber text extraction
 │
 ▼
chunker.py ──────────────────────────────── HierarchicalNodeParser (256/400/1500 tok)
 │                                          SimpleDocumentStore (all nodes)
 │                                          metadata propagation (leaf → parent)
 ▼
vector_store.py ─────────────────────────── Qdrant collection: 'fintech_rag'
                                            BGE-M3 1024-dim embeddings
                                            Leaf nodes only (256-tok chunks)

── QUERY TIME ──────────────────────────────────────────────────────────────────

Question
 │
 ▼
hybrid_retriever.py ─────────────────────── QueryFusionRetriever (RRF k=60)
 │                                          BGE-M3 dense + BM25 sparse → top 15
 ▼
pipeline.py ─────────────────────────────── AutoMergingRetriever
 │                                          Leaf→parent promotion if ≥50% children hit
 ▼
pipeline.py ─────────────────────────────── FlagEmbeddingReranker (bge-reranker-base)
 │                                          batch_size=4, use_fp16=True, devices=cpu
 │                                          ~17s on CPU for 15 nodes
 ▼
rag_chain.py ────────────────────────────── Score filter (top_score-3.0 AND ≥-4.0)
 │                                          CitationQueryEngine (citation_chunk_size=256)
 ▼
QueryResult (answer, sources, mode)
```

### New pipeline (revamp)

```
PDF
 │
 ├──────────────────────────────────────────────────────────────────────────────
 │  PATH A: AGENTIC                       PATH B: Q&A LIBRARY
 │                                        (run once, offline)
 ▼                                        ▼
agentic_chunker.py                        generate_qa_library.py
 │  pdfplumber paragraphs                  │  Section-level Q&A generation
 │  Groq Llama-3.1-8b-instant              │  Groq Llama-3.1-8b-instant
 │  Batch size: 3 paragraphs               │  5-8 Q&A pairs/section
 │  Proposition extraction                 │  Stored in data/qa_library.json
 │  6 pathology rules                      │
 │  Dedup (cosine > 0.92)                  ▼
 │                                        qa_ingester.py
 ▼                                         │  Embed QUESTION text only
revamp_vector_store.py                     │  Store answer + section in metadata
 │  Qdrant: 'fintech_rag_agentic'          ▼
 │  BGE-M3 1024-dim                       revamp_vector_store.py
 │  Each proposition = 1 TextNode          │  Qdrant: 'fintech_rag_qa'
 │                                         │  BGE-M3 1024-dim
 │                                         │  Each Q&A pair = 1 TextNode
 │                                         │
 └─────────────────────────────────────────┘

── QUERY TIME ──────────────────────────────────────────────────────────────────

Question
 │
 ▼
revamp_pipeline.py ─────────────────────── BGE-M3 embed query
 │
 ├── CASCADING MODE (default) ─────────────────────────────────────────────────
 │    │
 │    ├── Step 1: Q&A collection search ── top-1 cosine similarity
 │    │    │
 │    │    ├── score ≥ CASCADE_CONFIDENCE_THRESHOLD (0.82)
 │    │    │    └── FAST PATH: return pre-written answer directly
 │    │    │         NO reranker, NO LLM synthesis, ~100ms
 │    │    │
 │    │    └── score < threshold
 │    │         └── FALLBACK PATH ──────────────────────────────────────────
 │    │              │
 │    │              ▼
 │    │         Agentic collection search ── BGE-M3 + BM25 hybrid, top-20
 │    │              │
 │    │              ▼
 │    │         FlagEmbeddingReranker ────── top-5, ~17s (known cost)
 │    │              │
 │    │              ▼
 │    │         Score filter + CitationQueryEngine
 │    │              │
 │    │              ▼
 │    │         LLM synthesis (Groq)
 │    │
 │    └── log: [cascading] query="..." score=X.XX → Q&A hit / agentic fallback
 │
 ├── FUSION MODE (phase 2 upgrade) ───────────────────────────────────────────
 │    │  Query both collections simultaneously
 │    │  RRF merge (k=60) → top-5 combined
 │    │  LLM gets both pre-written answer + propositions in context
 │    │
 └── ROLE-SEPARATED MODE (phase 3) ──────────────────────────────────────────
      Q&A answer → "answer candidate" in prompt
      Agentic propositions → "regulatory evidence" in prompt
      LLM cross-checks candidate against evidence before responding

▼
QueryResult (same schema as legacy — answer, sources, mode)
```

---

## 6. New Files — Responsibilities

### `src/ingestion/agentic_chunker.py`
- Receives pdfplumber paragraphs from `loader.py` (existing, unmodified)
- Filters noise: page headers, footers, amendment notices ("Substituted vide...")
- Carries section heading context per paragraph (from `parser.py`)
- Calls Groq in batches of 3 paragraphs per API call
- Extracts propositions with 6-rule prompt (see §7 below)
- Falls back to sentence splitting on JSON parse failure (never crashes ingestion)
- Deduplicates propositions using BGE-M3 cosine similarity > 0.92
- Creates `TextNode` per proposition with full metadata:
  ```python
  {
      "source": "rbi_kyc_master_direction",
      "strategy": "agentic",
      "section": "<heading>",
      "chapter": "<chapter label>",
      "original_paragraph_idx": int,
      "proposition_idx": int,
      "char_count": int,
      "has_xref": bool,   # True if [XREF: resolve manually] present
  }
  ```
- Progress bar via `tqdm`; estimated runtime ~8-12 min for full document

### `src/ingestion/qa_ingester.py`
- Loads `data/qa_library.json`
- Creates one `TextNode` per Q&A pair
- Embeds the **question** text (not the answer) for retrieval
- Stores full answer + metadata in node
- Validates JSON schema before ingestion; raises clear error if file missing

### `src/retrieval/revamp_vector_store.py`
- Mirrors `vector_store.py` but supports multiple named collections
- `build_revamp_index(nodes, collection_name)` — creates or overwrites collection
- `load_revamp_index(collection_name)` — loads existing collection
- Uses same BGE-M3 embedder and Qdrant singleton client
- Collections: `fintech_rag_agentic`, `fintech_rag_qa`
- `--force-recreate` flag for ingestion scripts

### `src/retrieval/cascading_retriever.py`
- `CascadingRetriever` — Q&A first, agentic fallback, threshold-controlled
- `FusionRetriever` — parallel queries, RRF merge (k=60)
- Logging format: `[cascading] query="..." score=X.XX → Q&A hit | agentic fallback`
- `CascadingRetriever` does NOT use `AutoMergingRetriever` (no parent/child hierarchy)
- Uses straight Qdrant dense + BM25 hybrid for the agentic path (same fusion logic, different nodes)

### `src/retrieval/revamp_pipeline.py`
- `build_revamp_pipeline(agentic_index, qa_index, nodes)` → `CascadingRetriever`
- Wraps `CascadingRetriever` or `FusionRetriever` based on `RETRIEVAL_MODE`
- No `AutoMergingRetriever` — propositions are flat, no hierarchy to merge

### `src/generation/revamp_rag_chain.py`
- `query_revamp(question, ...) -> QueryResult` — drop-in replacement for `rag_chain.query()`
- Same `QueryResult` dataclass output (importable from `rag_chain.py`)
- Q&A fast path: score above threshold → return pre-written answer as-is, source = the Q&A pair
- Agentic path: same CitationQueryEngine flow as legacy, but with proposition nodes
- No reranker on Q&A path; reranker applied only on agentic fallback path

### `scripts/generate_qa_library.py`
- Standalone offline script (not part of live pipeline)
- `python scripts/generate_qa_library.py --input data/rbi_kyc_master.pdf --output data/qa_library.json`
- Runs once; commit `qa_library.json` to repo
- Sections split by heading detection: `r'^(PART|CHAPTER|SECTION|\d+\.)'`
- 8 Q&A pairs per section (configurable via `QA_PAIRS_PER_SECTION`)
- Retry logic: 3 attempts with exponential backoff (2s, 4s, 8s)

### `scripts/ingest_revamp.py`
- Runs agentic chunking and/or Q&A ingestion based on `CHUNKING_STRATEGY`
- `python scripts/ingest_revamp.py --strategy agentic|qa_library|both`
- `--force-recreate` to wipe and rebuild collections
- Progress bar; logs batch failures without crashing

### `scripts/eval_revamp.py`
- Runs same 10 eval queries against legacy, agentic, and cascading modes
- Prints comparison table with top-3 preview, latency, strategy served
- Manual relevance scoring column (0/1/2) for user to fill in
- This is the gate before any production switch — **do not switch `PIPELINE_MODE=revamp` without running this**

### `config/revamp_settings.py`
- Reads all new `.env` keys; validates at startup
- Raises `ValueError` for invalid `CHUNKING_STRATEGY` or `RETRIEVAL_MODE`

---

## 7. Prompt Architecture (Summary)

Full prompts are defined in the task spec. Key design decisions captured here:

### Proposition extraction prompt (Llama-3.1-8b-instant)
- 6 explicit rules matching 6 RBI KYC pathologies (pronoun resolution, cross-refs, compound sentences, amendment noise, modal precision, numerical precision)
- Wrong example is **load-bearing** — anchors model against pronoun retention after 30+ batches
- Strict JSON output: `[` first, `]` last, nothing else
- Returns `[]` for pure amendment paragraphs (correct — don't create empty propositions)
- `temperature=0.0` — deterministic for citations and thresholds

### Q&A generation prompt (Llama-3.1-8b-instant)
- 5 question types forced: obligation, threshold, exception, definition, process
- Specific > generic: *"What is the periodic updation period for low-risk customers?"* not *"What are the KYC requirements?"*
- Answers are self-contained: no "see above", include regulatory basis and exact numbers
- `temperature=0.1` — slight creativity acceptable for question variety

---

## 8. New `.env` Keys

```env
# ── Pipeline mode ──────────────────────────────────────────────────────────
PIPELINE_MODE=legacy          # "legacy" | "revamp"

# ── Chunking strategy (used by ingest_revamp.py) ───────────────────────────
CHUNKING_STRATEGY=both        # "agentic" | "qa_library" | "both"

# ── Retrieval mode (used by revamp_rag_chain.py) ───────────────────────────
RETRIEVAL_MODE=cascading      # "agentic" | "qa_library" | "cascading" | "fusion"
CASCADE_CONFIDENCE_THRESHOLD=0.82   # Start here; tune down if too many fall to agentic

# ── Agentic chunking ───────────────────────────────────────────────────────
AGENTIC_CHUNK_BATCH_SIZE=3
AGENTIC_PROPOSITION_MODEL=llama-3.1-8b-instant
AGENTIC_DEDUP_THRESHOLD=0.92

# ── Q&A library ────────────────────────────────────────────────────────────
QA_LIBRARY_PATH=data/qa_library.json
QA_GENERATION_MODEL=llama-3.1-8b-instant
QA_PAIRS_PER_SECTION=8

# ── Qdrant collections (never mix with legacy 'fintech_rag') ───────────────
QDRANT_COLLECTION_AGENTIC=fintech_rag_agentic
QDRANT_COLLECTION_QA=fintech_rag_qa
```

---

## 9. Integration Point (How It Wires to Existing App)

The only code that needs a small change is the **dispatcher** — one `if/else` on `PIPELINE_MODE`. Everything else is new files.

```python
# src/api/main.py  (or wherever query() is called)
# ADD these 3 lines — nothing else changes

from config.revamp_settings import revamp_settings

if revamp_settings.PIPELINE_MODE == "revamp":
    from src.generation.revamp_rag_chain import query_revamp as query_fn
else:
    from src.generation.rag_chain import query as query_fn

result = query_fn(question, ...)  # QueryResult — same schema either way
```

For the voice pipeline, the same dispatch applies. Since `QueryResult` is the same dataclass, the voice layer never needs to know which pipeline ran.

---

## 10. Implementation Sequence

Run in this order — each step is independently testable:

```
Step 1 — Validate proposition quality (BEFORE writing production code)
  Run agentic_chunker on 20 paragraphs from Chapter VI and Annexure I manually.
  Check: pronoun resolution, cross-ref flags, modal verb preservation, no hallucinated thresholds.
  Gate: error rate < 10% on this sample before proceeding.

Step 2 — Build agentic ingestion
  agentic_chunker.py + revamp_vector_store.py
  Run: python scripts/ingest_revamp.py --strategy agentic
  Verify: correct proposition count, metadata, section headings

Step 3 — Build Q&A library
  scripts/generate_qa_library.py → data/qa_library.json
  Manual review: check 20 random Q&A pairs for correctness and specificity
  qa_ingester.py → fintech_rag_qa collection

Step 4 — Build retrieval layer
  cascading_retriever.py + revamp_pipeline.py
  Unit test: cascading returns Q&A hit for known query, agentic fallback for novel query

Step 5 — Build generation layer
  revamp_rag_chain.py
  Verify: QueryResult schema matches legacy; voice mode doesn't get citation numbers

Step 6 — Evaluate
  scripts/eval_revamp.py
  Run all 10 eval queries, score manually, compare legacy vs revamp
  Gate: revamp score ≥ legacy on ≥ 7 of 10 queries before flipping PIPELINE_MODE

Step 7 — Flip switch
  .env: PIPELINE_MODE=revamp
  Legacy pipeline untouched, revertable by setting PIPELINE_MODE=legacy
```

---

## 11. Rollback Plan

Since the legacy pipeline is completely untouched:

```
Rollback: set PIPELINE_MODE=legacy in .env → restart app → done.
```

No database migration. No code change. The `fintech_rag` Qdrant collection is never touched by the revamp path.

The two revamp Qdrant collections (`fintech_rag_agentic`, `fintech_rag_qa`) can be dropped independently with `qdrant_client.delete_collection(name)` if storage is a concern.

---

## 12. Evaluation Queries (Gate Before Production)

```python
EVAL_QUERIES = [
    # The documented failure case — must improve
    "What documents are required for KYC of an individual customer?",
    # Section-number exact match — BM25 advantage test
    "FATCA CRS reporting requirements section 57",
    # Cross-reference heavy — tests proposition quality
    "What is the periodic KYC updation period for low-risk customers?",
    # Threshold extraction — tests numerical precision in propositions
    "What is the threshold for enhanced due diligence?",
    # Multi-part query — tests proposition recall breadth
    "What are the obligations for Politically Exposed Persons?",
    # Exception/edge case — tests Q&A coverage
    "What are simplified KYC norms for small accounts?",
    # Process query — tests step-by-step coverage
    "What is the step-by-step procedure for V-CIP?",
    # NRI-specific — tests metadata section routing
    "What documents does an NRI need for KYC?",
    # Off-topic (should be blocked before RAG — tests that nothing leaks)
    "How can I improve my daily productivity routine?",
    # Novel edge case (tests agentic fallback from Q&A)
    "Does the KYC direction apply to OCI cardholders?",
]
```

Score each result 0/1/2 manually (0=irrelevant, 1=partial, 2=correct). Production gate: revamp ≥ legacy on 7 of 10 queries, with the documented failure case (#1) showing improvement.

---

## 13. Open Questions Before Implementation

1. **Cascade threshold calibration**: 0.82 is a starting point. After ingesting both collections, run the eval queries through the Q&A path only and observe the distribution of scores. If most known queries land at 0.80-0.85, the threshold needs lowering. If most are 0.90+, 0.82 is safe.

2. **Proposition count estimate**: Need to run agentic chunker on the first 10 pages and extrapolate. If the document generates > 3,000 propositions, increase `similarity_top_k` to 20 (more candidates needed to ensure all relevant propositions are in the reranker pool).

3. **Q&A library size**: At 8 pairs per section, with ~40 sections in the document, expect ~320 Q&A pairs. Enough for the fast path to be meaningful. If sections are sparse, reduce to 5 pairs.

4. **BM25 on propositions**: Short 1-2 sentence propositions change BM25 term frequency distributions. Section numbers and regulatory codes (Rule 9(14), Section 16(b)) become high-weight terms. This is a **feature** — exact section queries will match strongly. No change needed to `bm25_store.py`.

5. **Voice mode latency budget**: The Q&A fast path (~100ms + LLM-free) is ideal for voice. The agentic fallback still hits the reranker (~17s). Consider whether voice mode should use a stricter Q&A threshold (0.78) to maximise fast-path hits, at the cost of some accuracy on edge queries.
