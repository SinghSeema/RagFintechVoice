# RAG Pipeline — Parsing, Chunking & Retrieval Techniques

Ground source of truth for the RagFintechVoice project. Covers both the **legacy pipeline** (production) and the **revamp pipeline** (in evaluation). All code references are relative to the project root.

---

## 1. Document Parsing

**File:** `src/ingestion/parser.py`  
**Entry point:** `load_pdf_with_sections(path) -> str`

### Approach
Uses `pdfplumber` to extract words with font attributes (name, size, position) from each page. Injects structural markers `<<CHAPTER:...>>` and `<<SECTION:...>>` into the text stream based on bold-text heuristics.

### Rules
| Signal | Detection |
|--------|-----------|
| Chapter | Fully-bold line matching `Chapter [IVX]+` regex — peek at next bold line for title |
| Section | Fully-bold line starting with `\d+\.` — may span multiple lines (buffered) |
| TOC suppression | Chapter headings before first non-bold body text are skipped; body occurrence wins |
| Header/footer | Words above `HEADER_H` px or below `page.height - FOOTER_H` px are excluded |

### Output
Raw text with markers injected, e.g.:
```
<<CHAPTER:Chapter VI — Customer Due Diligence (CDD) Procedure>>
<<SECTION:8. Compliance of KYC policy>>
8. Compliance of KYC policy
(a) REs shall ensure compliance with KYC Policy through:
...
```

### Known Limitation
Sections 9–23 (V-CIP, small accounts, etc.) in the RBI KYC PDF are **not detected** because their section numbers are not rendered fully-bold in the PDF. All content from sections 9–23 falls under the section 8 metadata bucket. This is a structural PDF issue, not a code bug.

**`build_section_map_and_clean(text_with_markers)`** strips the markers and builds `chapter_map` and `section_map` as `(char_offset, title)` lists used downstream for metadata assignment.

---

## 2. Chunking

### 2a. Legacy — HierarchicalNodeParser (3-level hierarchy)

**File:** `src/ingestion/chunker.py`  
**Entry point:** `chunk_document(text, filename, section_map, ...) -> (leaf_nodes, storage_context)`

#### Hierarchy
```
Parent  1500 tokens  150 tok overlap   ← auto-merged context
Mid      400 tokens   40 tok overlap   ← balanced context
Leaf     256 tokens   25 tok overlap   ← indexed in Qdrant (fintech_rag)
```

- LlamaIndex `HierarchicalNodeParser` builds the tree.
- Only **leaf nodes** are stored in the Qdrant vector index.
- Parent and mid nodes are stored in `SimpleDocumentStore` (disk, `./storage/`).
- Metadata: `jurisdiction`, `doc_type`, `effective_date`, `is_stale`, `clearance_level`, `section_title`, `chunk_index`.

#### Persistence
On first run, the hierarchy is built and saved to `./storage/`. Subsequent runs load from disk — no re-parsing or re-chunking. Delete `./storage/` to force rebuild.

---

### 2b. Revamp — Agentic Proposition Chunker

**File:** `src/ingestion/agentic_chunker.py`  
**Entry point:** `chunk_to_propositions(chapter_map, section_map, clean_text, api_key) -> list[TextNode]`  
**Collection:** `fintech_rag_agentic`

#### Goal
Decompose regulatory paragraphs into **atomic, self-contained propositions**. Each proposition is one standalone fact that resolves all pronouns, preserves citations, and splits compound obligations.

#### Pipeline
1. **Paragraph splitting** — double-newline boundaries within each section, max 800 chars per paragraph.
2. **Noise filtering** — drop page numbers, amendment notices, blank lines, paragraphs < 30 chars.
3. **Groq API batching** — `AGENTIC_CHUNK_BATCH_SIZE` paragraphs per call (default 1), model `llama-3.1-8b-instant`, `max_tokens=1024`.
4. **JSON repair** — Model output is a JSON array of strings. On truncation (TPD/TPM limits), `_repair_json_array()` regex-extracts complete quoted strings from partial JSON.
5. **Deduplication** — BGE-M3 cosine similarity; pairs above `AGENTIC_DEDUP_THRESHOLD=0.92` are deduplicated.
6. **Section prefix** — Each proposition is prepended with `[{section_name}]\n` so BM25 can match section-number queries (e.g. "section 57").
7. **Sliding-window summary nodes** (Fix C) — One additional `TextNode` per 1800-char window of each section's original text, with 900-char stride (50% overlap). These restore sequential/procedural context lost by proposition atomization.

#### Prompt
System prompt instructs the model to: replace all pronouns with explicit referents, keep section/rule citations, split compound obligations (one SHALL per proposition), preserve exact numbers, and strip amendment notices. Output is a JSON array — nothing else.

#### Resilience to TPD Exhaustion
When Groq's 500K daily token limit is hit mid-ingest, all batches beyond that point fall through to `_sentence_fallback()` (splits on `.` boundaries). Summary nodes are always built from the original `filtered` paragraph list — never from Groq output — so section summaries exist for the entire document regardless of where the TPD cutoff occurred.

#### TextNode metadata
```python
{
  "source":   "rbi_kyc_master_direction",
  "strategy": "agentic",          # or "summary"
  "chapter":  "Chapter VI — ...",
  "section":  "8. Compliance of KYC policy",
  "original_paragraph_idx": int,
  "proposition_idx": int,
  "char_count": int,
  "has_xref": bool,               # True if [XREF: ...] flag present
}
```

---

### 2c. Revamp — Q&A Library Chunker

**File:** `src/ingestion/qa_ingester.py`  
**Entry point:** `create_qa_nodes(qa_pairs) -> list[TextNode]`  
**Collection:** `fintech_rag_qa`

#### Goal
Pre-written (LLM-generated) question–answer pairs. At retrieval time, **the question is embedded** (not the answer). Question-to-question similarity is far more precise than query-to-paragraph similarity for known compliance questions.

#### Source
`data/qa_library.json` — generated once by `scripts/generate_qa_library.py` using `llama-3.1-8b-instant`. Contains 1137 pairs across all sections of the RBI KYC Master Direction. Format:
```json
{
  "question": "What are the simplified KYC norms for small accounts?",
  "answer":   "Under paragraph 23 ...",
  "section":  "23",
  "question_type": "process"
}
```

#### TextNode structure
- `node.text` = question (embedded for retrieval)
- `node.metadata["answer"]` = full pre-written answer (retrieved at query time)
- `node.metadata["strategy"]` = `"qa_library"`

#### _node_content serialisation
Qdrant payload must include a `_node_content` field containing the full JSON-serialised `TextNode` (including `metadata`) for LlamaIndex to correctly reconstruct the node on retrieval. Missing this causes `node.metadata` to be empty and `answer` to fall back to `node.text` (the question string).

---

## 3. Embeddings

**Model:** `BAAI/bge-m3` (FlagEmbedding)  
**Dimension:** 1024  
**Used for:** Both legacy (via LlamaIndex Qdrant integration) and revamp (explicit `BGEM3FlagModel.encode()`)  
**Precision:** `use_fp16=True` for CPU efficiency  
**Max sequence length:** 512 tokens

BGE-M3 is a multilingual multi-function model supporting dense, sparse, and ColBERT retrieval. Only **dense** vectors are used here.

---

## 4. Vector Store

**Backend:** Qdrant (Docker, `http://localhost:6333`)  
**Collections:**

| Collection | Pipeline | Contents | Points |
|------------|----------|----------|--------|
| `fintech_rag` | Legacy | Leaf nodes (256-tok chunks) | ~460 |
| `fintech_rag_agentic` | Revamp | Proposition nodes + sliding-window summary nodes | ~1360 |
| `fintech_rag_qa` | Revamp | Q&A question nodes | 1137 |

**Node sidecars** (for BM25 — Qdrant doesn't support BM25 natively):  
`qdrant_storage/revamp_nodes_{collection}.json` — list of `{node_id, text, metadata}` dicts loaded into memory to build the BM25 index on startup.

---

## 5. Retrieval

### 5a. Legacy — Hybrid + AutoMerging + Reranker

**Files:** `src/retrieval/hybrid_retriever.py`, `src/retrieval/pipeline.py`

#### Stage 1 — Hybrid Retrieval (RRF)
`QueryFusionRetriever` combines:
- **Qdrant dense retriever** (BGE-M3 cosine similarity, top-15)
- **BM25 sparse retriever** (`BM25Retriever` on leaf nodes, top-15)

Fusion: **Reciprocal Rank Fusion** with `k=60`.  
Score = `1/(k + rank)`. Uses only rank position — avoids BM25/cosine scale mismatch. Documents in both lists get a double boost.

No query expansion (`num_queries=1`) — preserves precision for regulatory queries where paraphrasing can drift.

#### Stage 2 — AutoMergingRetriever
If ≥ 50% of a parent chunk's leaf children appear in the candidate set, children are replaced by the parent node (up to 400-tok mid or 1500-tok parent). Gives the LLM richer connected context.

#### Stage 3 — FlagEmbeddingReranker (BGE cross-encoder)
`BAAI/bge-reranker-base` cross-encoder reads `(query, chunk)` pairs together — far more accurate than dot-product similarity alone.  
- Text mode: top-5  
- Voice mode: top-3 (latency budget)

#### Noise filter
After reranking: drop nodes with score < -4.0 or more than 3.0 points below the top score.

---

### 5b. Revamp — Cascading Retriever

**File:** `src/retrieval/cascading_retriever.py`  
**Entry point:** `CascadingRetriever.retrieve(query) -> (list[NodeWithScore], path)`

#### Logic
```
Query
  │
  ▼
Q&A collection (BGE-M3 cosine)
  │
  ├─ top-1 score ≥ CASCADE_CONFIDENCE_THRESHOLD (0.90)?
  │     YES → return Q&A node directly (path="qa")
  │           latency ~250ms, no LLM synthesis needed
  │
  └─ NO → agentic fallback (path="agentic")
          BGE-M3 + BM25 hybrid (top-20) over fintech_rag_agentic
          → FlagEmbeddingReranker (top-5)
          → noise filter
```

**Threshold 0.90:** Raised from 0.82 after Q3 ("periodic KYC updation for low-risk") returned a high-risk answer at 0.82 (false Q&A match). 0.90 requires near-exact question paraphrase.

**Why Q&A first:**  
For well-known compliance questions, question-to-question cosine similarity is extremely precise (scores near 1.0 for paraphrases). The pre-written answer is always correct and comprehensive. Agentic propositions can miss procedural completeness.

---

### 5c. Revamp — Fusion Retriever (alternative mode)

**File:** `src/retrieval/cascading_retriever.py` — `FusionRetriever`

Queries both `fintech_rag_qa` and `fintech_rag_agentic` in parallel (ThreadPoolExecutor). Merges results using RRF (k=60). Returns combined top-K for LLM synthesis. Used when `RETRIEVAL_MODE=fusion`.

---

## 6. Generation

**LLM:** Groq `llama-3.1-8b-instant`  
**Temperature:** 0.0  
**System prompt:** Finova persona — limits answers strictly to provided source passages. Refuses stock/crypto/tax questions.

### Legacy (`src/generation/rag_chain.py`)
LlamaIndex `RetrieverQueryEngine` with custom text/voice QA templates. Sources include section title + chapter from leaf node metadata.

### Revamp (`src/generation/revamp_rag_chain.py`)
- **Q&A fast path:** Returns `node.metadata["answer"]` directly — no LLM call. Sub-500ms end-to-end.
- **Agentic path:** LlamaIndex `RetrieverQueryEngine` synthesis from top-5 proposition/summary nodes.

---

## 7. Configuration

**File:** `config/revamp_settings.py` (reads from `.env`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `PIPELINE_MODE` | `legacy` | `legacy` or `revamp` |
| `CHUNKING_STRATEGY` | `both` | `agentic`, `qa_library`, `both` |
| `RETRIEVAL_MODE` | `cascading` | `agentic`, `qa_library`, `cascading`, `fusion` |
| `CASCADE_CONFIDENCE_THRESHOLD` | `0.90` | Min Q&A cosine score to accept Q&A answer |
| `AGENTIC_CHUNK_BATCH_SIZE` | `1` | Paragraphs per Groq API call |
| `AGENTIC_PROPOSITION_MODEL` | `llama-3.1-8b-instant` | Model for proposition extraction |
| `AGENTIC_DEDUP_THRESHOLD` | `0.92` | BGE-M3 cosine threshold for deduplication |
| `QA_PAIRS_PER_SECTION` | `8` | Q&A pairs generated per section |
| `QDRANT_COLLECTION_AGENTIC` | `fintech_rag_agentic` | Never overwrite `fintech_rag` (legacy) |
| `QDRANT_COLLECTION_QA` | `fintech_rag_qa` | Q&A collection name |

---

## 8. Key Design Decisions & Tradeoffs

### Why Propositions?
Atomic propositions improve retrieval precision — each chunk is about exactly one fact, so cosine similarity is less diluted. Downside: procedural sequences (V-CIP steps, small accounts rules) get fragmented. Mitigated by sliding-window summary nodes.

### Why Sliding-Window Summary Nodes?
The 900-char stride with 1800-char windows ensures that content deep inside large sections (sections 9–23 mislabeled as section 8, spanning 32K chars) gets represented. Without this, propositions alone miss sequential context.

### Why Q&A First in Cascading?
For a compliance assistant, the most common queries are repeatable (document requirements, thresholds, obligations). Pre-written answers are always authoritative and complete. Sub-second latency for these is a major UX win for the voice pipeline.

### Why BGE-M3 + BM25 Hybrid?
- BGE-M3 alone: misses exact regulation codes, section numbers ("section 57"), acronyms ("V-CIP").
- BM25 alone: misses semantic paraphrases ("what documents are needed" ≠ "customer identification").
- RRF fusion: double-boosts chunks that both agree on. No calibration needed between score scales.

### Why Cross-Encoder Reranker?
Bi-encoder (dot product) scores used for fast ANN retrieval are imprecise. Cross-encoder reads query+chunk together — semantic interactions matter for compliance text where context changes meaning. ~200–400ms on CPU for top-5 candidates is acceptable.

### Token Limits (Groq)
- TPM (tokens per minute): 6,000 — limits batching speed.
- TPD (tokens per day): 500,000 — limits full ingest to 1 run per day.
- JSON truncation: model hits output token limit mid-array at batch ~105. `_repair_json_array()` regex-extracts complete strings from partial JSON before falling back to sentence splitting.

---

## 9. Evaluation

**Script:** `scripts/eval_revamp.py`

```
python scripts/eval_revamp.py --modes legacy cascading --queries 1 2 6 7
```

- `--queries N ...`: 1-based indices into `EVAL_QUERIES` list (10 total)
- `--quick`: first 3 only
- `--modes`: subset of `legacy`, `agentic`, `cascading`

**Production gate:** revamp ≥ legacy on ≥ 7/10 queries AND query #1 (KYC documents) must improve.

**Token cost per eval run:** ~4 queries × 3 modes × ~2K tokens ≈ 24K tokens (safe within 500K TPD).

### Targeted eval queries (critical path)
| # | Query | Tests |
|---|-------|-------|
| 1 | What documents are required for KYC of an individual customer? | Must-improve gate |
| 2 | FATCA CRS reporting requirements section 57 | Section-number BM25 match |
| 6 | What are the simplified KYC norms for small accounts? | Q&A fast path |
| 7 | What is the step-by-step procedure for V-CIP? | Sliding-window summary coverage |

---

## 10. Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/ingest_revamp.py` | Ingest agentic + Q&A collections (runs Groq) |
| `scripts/generate_qa_library.py` | Generate `data/qa_library.json` via Groq |
| `scripts/rebuild_summary_nodes.py` | Rebuild only the sliding-window summary nodes (no Groq — reads PDF directly). Run after changing `_MAX_SUMMARY_CHARS` or `_SUMMARY_STRIDE`. |
| `scripts/eval_revamp.py` | Compare legacy vs revamp pipelines |

---

## 11. File Map

```
src/
  ingestion/
    parser.py              — pdfplumber PDF → marked text
    metadata.py            — section/chapter map, node enrichment
    chunker.py             — legacy HierarchicalNodeParser (3-level)
    agentic_chunker.py     — proposition extraction + summary nodes
    qa_ingester.py         — Q&A library → TextNodes
  retrieval/
    vector_store.py        — legacy Qdrant index load/build
    revamp_vector_store.py — revamp Qdrant index + node sidecar load
    hybrid_retriever.py    — RRF fusion (BGE-M3 + BM25) + LLM config
    bm25_store.py          — BM25Retriever builder from node list
    pipeline.py            — legacy AutoMergingRetriever + reranker
    revamp_pipeline.py     — revamp pipeline builder (cascading/fusion)
    cascading_retriever.py — CascadingRetriever + FusionRetriever
  generation/
    rag_chain.py           — legacy query + source formatting
    revamp_rag_chain.py    — revamp query (Q&A fast path + agentic synthesis)
config/
  revamp_settings.py       — dataclass settings from .env
data/
  qa_library.json          — 1137 Q&A pairs (source of truth for Q&A collection)
  raw/
    rbi_kyc_master_direction.pdf
qdrant_storage/
  revamp_nodes_fintech_rag_agentic.json   — node sidecar (1360 nodes)
  revamp_nodes_fintech_rag_qa.json        — node sidecar (1137 nodes)
storage/                   — legacy LlamaIndex StorageContext (disk)
scripts/
  ingest_revamp.py
  generate_qa_library.py
  rebuild_summary_nodes.py
  eval_revamp.py
```
