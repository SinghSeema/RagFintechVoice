# Phase 1 — Implementation Refinements

Living log of decisions and refinements made during Phase 1 implementation
that extend or clarify `fintech_rag_adr_v1.1.pdf`. These will be folded into
ADR v1.2 at the end of Phase 1.

---

## Task 1 — PDF loading (pdfplumber)

**Status:** Complete (2026-04-04)

- Used `page.crop((0, 40, width, height-40))` to strip headers/footers.
- Empty/scanned pages skipped silently (no exception).
- Verified against `data/raw/rbi_kyc_master_direction.pdf`: 232,054 chars
  extracted, no page numbers or "RBI | Page X of Y" headers in output.
- Two additional test PDFs available: `rbi_housing_loan_guidelines.pdf`,
  `rbi_priority_sector_lending.pdf`.

---

## ADR-003 — TTS Confirmation

**Status:** Confirmed (2026-04-05)

- **Primary:** Cartesia Sonic — credits available. 90ms TTFT, byte-level streaming, native Pipecat processor.
- **Fallback:** Deepgram Aura — key available. Swap via single Pipecat config flag, no code change.
- ADR-003 decision stands exactly as written.

---

## Task 2 — Chunking (HierarchicalNodeParser)

**Status:** In progress (2026-04-05)

### Refinement 1 — Chunk overlap set to 10% per level

ADR Section 12 (Open Items) defers exact overlap % to Phase 1 with "10% as
starting point, tune after first eval run." Applying 10% per level:

| Level | Chunk size | Overlap |
|-------|------------|---------|
| Parent | 1500 tok | 150 tok |
| Mid | 400 tok | 40 tok |
| Leaf | 100 tok | 10 tok |

LlamaIndex's default is a flat 20-tok overlap, which would give 20% at leaf
and ~1.3% at parent — explicitly overriding avoids that imbalance.

### Refinement 2 — StorageContext persistence

Use `StorageContext.from_defaults(docstore=SimpleDocumentStore())` and
`storage_context.persist(persist_dir="./storage")` after chunking. On
subsequent runs, load from disk via `StorageContext.from_defaults(persist_dir=...)`
and skip re-parsing entirely.

**Why:** Chunking a multi-hundred-page regulatory PDF is slow. Persisting
the docstore makes Tasks 3–9 iteration fast (load from disk in ms, not
re-chunk every run).

**Gitignore:** `storage/` added to `.gitignore`. It can reach hundreds of
MB for large document sets and must be regenerated from raw PDFs, not
committed.

### Refinement 3 — Function signature

ADR specifies: `chunk_document(doc) -> (leaf_nodes, docstore)`

**Refined to:** `chunk_document(text: str, persist_dir: str = "./storage") -> (leaf_nodes, storage_context)`

Two changes:
1. **Input is `str` not `Document`** — keeps Document wrapping internal to
   the chunker. Callers pass raw text from `load_pdf()` directly; no
   LlamaIndex knowledge leaks to ingestion callers.
2. **Return `storage_context` not `docstore`** — `storage_context` is the
   canonical handle used by `VectorStoreIndex` (Task 5),
   `AutoMergingRetriever` (Task 8), and `.persist()`. Downstream tasks
   would otherwise have to reconstruct a StorageContext from the docstore.
   The docstore remains accessible via `storage_context.docstore`.

### Refinement 4 — All nodes stored, only leaves returned for indexing

Per ADR-007 CAUTION box: only leaf nodes go into the vector store;
parent and mid nodes live in the docstore for AutoMergingRetriever to
fetch at query time. `chunk_document` stores all three levels in the
docstore but returns only `get_leaf_nodes(all_nodes)` for indexing.

### Refinement 5 — Two-layer metadata (added during Task 3)

**Layer 1 (document-level):** `extract_metadata(filename, text) -> dict`
Set on `Document` before chunking — LlamaIndex propagates automatically
to all child nodes (leaf, mid, parent).
Fields: `jurisdiction`, `doc_type`, `effective_date`, `is_stale`,
`clearance_level`, `citation_priority`, `source_file`.

**Layer 2 (chunk-level):** `enrich_nodes(leaf_nodes, clean_text, section_map, chapter_map)`
Applied after chunking. Uses global `str.find()` offset lookup (not
`start_char_idx` which is parent-local) + binary search on offset maps.
Fields: `chunk_index`, `section_title`, `chapter`.

### Refinement 6 — Chapter metadata field

Added `chapter` as a third chunk-level field alongside `section_title`.

**Why:** RBI Master Directions and Priority Sector Lending directions use
Roman-numeral chapter structure (`CHAPTER I — PRELIMINARY`). Chapter
label provides meaningful citation context for Task 9 responses.

**Detection:** Single-pass with `load_pdf_with_sections()`. Fully-bold
lines matching Roman numeral pattern (`CHAPTER I`, `CHAPTER – III`) are
detected. The immediately-following bold line is the chapter title,
joined as `"Chapter I — PRELIMINARY"`. TOC entries are suppressed by
blanking the TOC marker and re-emitting at the real body position.

**Embed visibility:** `chapter` is excluded from embedding
(`excluded_embed_metadata_keys`) — citation display only. Only
`jurisdiction`, `doc_type`, `section_title` enter the embedding vector.

**Document coverage:**
- `rbi_kyc_master_direction.pdf` — 11 chapters (I–XI), 1,896/1,958 leaves enriched
- `rbi_priority_sector_lending.pdf` — 4 chapters (I–IV), full coverage
- `rbi_housing_loan_guidelines.pdf` — 0 chapters (different format); `chapter=None`

### Refinement 7 — excluded_llm_metadata_keys on Document

Without this, LlamaIndex prepends all document metadata (51 tokens) to
every chunk during LLM context injection, shrinking 100-token leaves to
<50 tokens of actual content. Fixed by setting
`excluded_llm_metadata_keys=list(doc_metadata.keys())` on the Document
object before chunking.

### Refinement 8 — Final chunk_document signature

```python
chunk_document(
    text: str,
    filename: str,
    section_map: list | None = None,
    chapter_map: list | None = None,
    clean_text: str | None = None,
    persist_dir: str = "./storage",
) -> (leaf_nodes, storage_context)
```

### Task 3 verification results (KYC Master Direction)

| Metric | Value |
|--------|-------|
| Total nodes in docstore | 2,248 |
| Leaf nodes | 1,958 |
| Leaves with chapter | 1,896 / 1,958 (97%) |
| Leaves with section_title | 1,895 / 1,958 (97%) |
| First chapter assigned | `CHAPTER I — PRELIMINARY` at chunk_index=62 |
| Leaf contained in parent | True |

---

## Tasks 4–8 — Retrieval Layer

**Status:** Complete (2026-04-05/06)

All tasks matched ADR-004 / ADR-007 decisions with the refinements below.

### Refinement 9 — Embedding model downgrade (Task 4)

ADR-004 specifies BGE-M3 (1024-dim). Implemented with `BAAI/bge-small-en-v1.5` (384-dim).

**Why:** BGE-M3 (~570 MB) OOM on dev machine during repeated eval iterations. BGE-small (~90 MB) is sufficient for Phase 1 quality bar and already cached from the reranker pipeline. Will upgrade for production (Phase 3).

**Impact:** Lower embedding dimensionality → slightly reduced semantic discrimination on long regulatory sentences. Acceptable for Phase 1 baseline.

### Refinement 10 — Pre-built pipeline for batch queries (Task 8 → Task 9)

`build_pipeline()` instantiates `FlagEmbeddingReranker` which downloads and loads the cross-encoder tokenizer. Called per-query, this added ~4–5 s per question during evaluation (20 downloads).

**Fix:** `query()` in `rag_chain.py` accepts optional `retriever` and `reranker` parameters. When supplied, `build_pipeline()` is bypassed entirely. `eval_runner.py` builds the pipeline once and passes it into every `query()` call.

### Refinement 11 — Voice top_n set to 3 (Task 8)

ADR-004 specified voice top-k as `top-2 + auto-merge`. Implemented as `top_n=3` in the reranker.

**Why:** top-2 frequently merged into a single parent (auto-merge ≥50% threshold), leaving only 1 source for CitationQueryEngine. With top-3, 2–3 distinct citations are reliably returned for voice answers without impacting latency materially.

---

## Task 9 — Generation (CitationQueryEngine)

**Status:** Complete (2026-04-06)

No deviations from ADR-007. `StaticRetriever` pattern used to decouple reranking from CitationQueryEngine's internal retrieval.

---

## Task 10 — Evaluation

**Status:** Complete (2026-04-06)

### Refinement 12 — Judge LLM changed from Mixtral-8x7B to llama-3.1-8b-instant

ADR-005 specified Groq Mixtral-8x7B as judge. Changed to `llama-3.1-8b-instant`.

**Why:** Mixtral-8x7B quota exhausted on Groq free tier during development runs. llama-3.1-8b-instant is a different model family from the generation LLM (llama-3.3-70b-versatile), preserving the ADR-005 anti-self-evaluation-bias constraint.

### Refinement 13 — RAGAS API compatibility (0.4.x)

`ragas.metrics.collections` (the documented new API) is rejected by `ragas.evaluate()` in 0.4.x — `evaluate()` checks `isinstance(m, Metric)` which only the old instance-based metrics pass. Solution: use old instance-based API (`from ragas.metrics import faithfulness, ...`) with `warnings.catch_warnings()` to suppress deprecation noise.

### Refinement 14 — context_recall/context_precision LOW due to snippet truncation

`QueryResult.sources` stores 120-char snippets per source. RAGAS `context_recall` and `context_precision` compare retrieved contexts against ground truth — short snippets hurt both metrics.

**Planned fix (Phase 2):** Pass full chunk text (not truncated snippets) into the RAGAS dataset. Requires `Source.full_text` field in `rag_chain.py`.

### Evaluation baseline (2-question kyc_aml sample)

| Metric | Score |
|--------|-------|
| faithfulness | 0.6905 |
| answer_relevancy | 1.0000 |
| context_recall | 0.5714 |
| context_precision | 0.5750 |
