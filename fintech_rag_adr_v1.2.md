# FINTECH MULTI-AGENT RAG + VOICE
## Architecture Decision Record — v1.2
**Addendum to v1.1 PDF · Phase 1 Implementation Decisions · 2026-04-06**

> This document records decisions made *during* Phase 1 engineering that extend,
> correct, or confirm decisions in `fintech_rag_adr_v1.1.pdf`. Read v1.1 first.
> Items marked **DEVIATION** differ from v1.1. Items marked **CONFIRMED** match v1.1 exactly.

---

### ADR-004 — Chunking Strategy (Phase 1 update)

| Parameter | v1.1 Decision | v1.2 Actual | Status |
|-----------|--------------|-------------|--------|
| Leaf chunk size | 100 tok | 128 tok | DEVIATION |
| Mid chunk size | 400 tok | 400 tok | CONFIRMED |
| Parent chunk size | 1,500 tok | 1,500 tok | CONFIRMED |
| Chunk overlap | "10% as starting point" (open item) | 10% per level (13/40/150 tok) | RESOLVED |
| Voice top-k | top-2 + auto-merge | top-3 (reranker_top_n=3) | DEVIATION |
| Text top-k | top-5 | top-5 | CONFIRMED |
| Embedding model | BGE-M3 (1,024-dim) | BGE-small-en-v1.5 (384-dim) | DEVIATION |
| Vector store | Qdrant | Qdrant local (file-backed) | CONFIRMED |
| Keyword search | OpenSearch (dev: BM25) | BM25 in-memory (dev) | CONFIRMED |
| Fusion | RRF k=60 | RRF k=60, num_queries=1 | CONFIRMED |
| Auto-merge threshold | ≥50% children | ≥50% children | CONFIRMED |

**DEVIATION — Leaf chunk 128 tok (was 100):**
Sentence-aware splitting at 100 tok frequently split mid-sentence on dense regulatory text. 128 tok gives one additional sentence buffer while staying well below the 400-tok mid level. Keeps leaf→LLM context compact.

**DEVIATION — Embedding: BGE-small (was BGE-M3):**
BGE-M3 (570 MB) caused OOM on the development machine during repeated reloads in the evaluation loop. BGE-small-en-v1.5 (90 MB) fits in RAM alongside the cross-encoder reranker. Phase 1 accuracy bar is met. Upgrade to BGE-M3 is scheduled for Phase 3 (production hardening).

**DEVIATION — Voice top_n=3 (was top-2):**
top-2 triggered auto-merge ≥50% threshold too frequently, collapsing two retrieved leaves into a single parent and leaving CitationQueryEngine with only one source. top-3 consistently produces 2–3 distinct citations without measurable latency increase at Groq speeds.

---

### ADR-005 — Evaluation Framework (Phase 1 update)

| Parameter | v1.1 Decision | v1.2 Actual | Status |
|-----------|--------------|-------------|--------|
| RAGAS metrics | faithfulness, answer_relevancy, context_recall, context_precision | Same | CONFIRMED |
| DeepEval metrics | hallucination, answer_correctness | faithfulness, answer_relevancy, hallucination | DEVIATION |
| Judge LLM | Groq Mixtral-8x7B | Groq llama-3.1-8b-instant | DEVIATION |
| Golden dataset | 20 questions, 5/domain | 20 questions, 5/domain | CONFIRMED |
| Domains | regulatory_qa, customer_support, loan_credit, kyc_aml | Same | CONFIRMED |
| Regression alert | drop >0.05 | drop >0.05 | CONFIRMED |
| Run cadence | nightly | manual + `--limit N` for dev | OPEN |

**DEVIATION — Judge LLM: llama-3.1-8b-instant (was Mixtral-8x7B):**
Mixtral-8x7B quota (14,400 RPD free tier) exhausted during Phase 1 development runs. `llama-3.1-8b-instant` is a different model family from the generation LLM (`llama-3.3-70b-versatile`), preserving the ADR-005 anti-self-evaluation-bias constraint. Mixtral-8x7B remains the intended production judge; revert when a paid key is available.

**DEVIATION — DeepEval `answer_correctness` → `answer_relevancy`:**
`AnswerCorrectnessMetric` in the installed DeepEval version requires a separate NLI model download not available offline. Replaced with `AnswerRelevancyMetric` which uses the same judge LLM and is a valid proxy for Phase 1.

**OPEN — Nightly cadence:**
`eval_runner.py` is a CLI script. Nightly scheduling deferred to Phase 2 when CI/CD pipeline is established. Add as a cron job or GitHub Actions step at that point.

---

### ADR-007 — RAG Orchestration (Phase 1 update)

All ADR-007 decisions confirmed as implemented. Key clarifications:

**Metadata enrichment (not in v1.1):**
Two-layer metadata was added during implementation:
- **Layer 1 (document-level):** `jurisdiction`, `doc_type`, `effective_date`, `is_stale`, `clearance_level`, `citation_priority`, `source_file` — set on `Document` before chunking, propagated to all child nodes automatically.
- **Layer 2 (chunk-level):** `chunk_index`, `section_title`, `chapter` — applied post-chunking via global offset lookup + binary search on section/chapter maps.
- `chapter` excluded from embedding (`excluded_embed_metadata_keys`) — citation display only.
- All document-level metadata excluded from LLM context injection (`excluded_llm_metadata_keys`) — prevents 51-token metadata prefix shrinking 128-tok leaves to <50 tokens of actual content.

**Pre-built pipeline pattern (not in v1.1):**
`query()` in `rag_chain.py` accepts optional `retriever` and `reranker` parameters. When supplied, `build_pipeline()` is skipped. Required for batch evaluation — reranker model loads once, not once per question.

---

### Phase 1 Baseline Metrics

Recorded in `evaluation/metrics_store.json`. Partial run (2 questions, kyc_aml domain):

| Metric | Score | Floor (alert if below) |
|--------|-------|------------------------|
| ragas_faithfulness | 0.6905 | 0.6405 |
| ragas_answer_relevancy | 1.0000 | 0.9500 |
| ragas_context_recall | 0.5714 | 0.5214 |
| ragas_context_precision | 0.5750 | 0.5250 |

Full 20-question baseline to be recorded before Phase 2 begins.

---

### Open Items Carried into Phase 2

| Item | ADR ref | Decision |
|------|---------|----------|
| Upgrade BGE-small → BGE-M3 | ADR-004 | Phase 3 (production hardening) |
| OpenSearch replace BM25 | ADR-007 | Phase 2 (when synonym expansion needed) |
| Nightly eval cron | ADR-005 | Phase 2 (with CI/CD) |
| Ollama Mistral-7B for PII queries | ADR-001 | Phase 2 (KYC/AML workflow) |
| Pass full chunk text to RAGAS (not 120-char snippets) | ADR-005 | Phase 2 (eval improvement) |
| Sub-question decomposition for multi-hop queries | ADR-007 | Phase 2 |
