# RagFintechVoice — Phase 3 Context: Improvements & Refinements

**Status:** Planning  
**Created:** 2026-04-07  
**Builds on:** Phase 2 complete (voice pipeline live, text API live, frontend redesigned)

---

## Current State Baseline

| Component | Current Config | Status |
|-----------|---------------|--------|
| Embedding | BGE-M3 (BAAI/bge-m3, 1024-dim, local) | ✅ Good |
| Leaf chunk size | 100 tokens (~75 words) | ⚠ Too small |
| Mid chunk size | 400 tokens | ✅ Reasonable |
| Parent chunk size | 1500 tokens | ✅ Good |
| Leaf overlap | 10 tokens | ⚠ Too small |
| BM25 + Qdrant fusion | 10 candidates each before fusion | ⚠ Borderline |
| Reranker | BGE-reranker-base | ⚠ Lightweight — upgrade candidate |
| Reranker top-n | 5 (text + voice, fixed 2026-04-07) | ✅ Fixed |
| LLM | Groq llama-3.1-8b-instant | ✅ Fast |
| citation_chunk_size | 512 tokens (CitationQueryEngine) | ⚠ Mismatched |
| Voice sentence chunker | Split on `[.!?]\s+` only | ❌ Broken for lists |
| Voice answer prompt | Generic CitationQueryEngine prompt | ❌ Not TTS-friendly |
| STT deduplication | None | ❌ Spurious fragments |

---

## Observed Failures (from live testing 2026-04-07)

### Issue 1 — Voice: `chunks=1` — TTS streaming disabled de facto

**Log evidence:**
```
RAGProcessor | sources=5 answer_len=496 chunks=1
```

**Root cause:**  
`_sentence_chunks()` in `pipeline.py:290` splits only on `(?<=[.!?])\s+`.  
The RAG LLM returns markdown bullet lists:
```
- Officially Valid Documents (OVDs)
- Proof of possession of Aadhaar [2, 3]
- PAN details verified from the database...
Note: The term "V-CIP"...
```
Bullet items don't end with `.!?` so the entire 496-char answer becomes **one TextFrame**.  
TTS cannot start streaming until the whole answer is assembled — eliminating the TTFT benefit
of sentence-level streaming.

**Fix needed:** Rewrite `_sentence_chunks` to split on `\n` boundaries too, and strip list
markers (`-`, `•`, `*`) before pushing to TTS.

---

### Issue 2 — Voice: Spurious fragment TranscriptionFrames

**Log evidence:**
```
RAGProcessor | question: 'What are the official valid document accepted...'  ← real
RAGProcessor | question: 'under RBI norms.'                                  ← fragment
```

**Root cause:**  
Groq Whisper STT occasionally emits multiple `TranscriptionFrame`s for one utterance —
the main transcription and a trailing fragment. Both have `finalized=True`.  
The fragment `'under RBI norms.'` triggers a second full RAG query (classifier + retrieval).

**Fix needed:**  
- Deduplicate: ignore any TranscriptionFrame whose text is a substring of the previous
  question handled within the last 5 seconds.
- Or: accumulate frames with a short debounce window (300ms) before dispatching to RAG.

---

### Issue 3 — Voice: Answer not TTS-friendly (markdown in spoken output)

**Log evidence:**
```
TTS [Based on the provided sources, the official valid documents include:
- Officially Valid Documents (OVDs)
- Equivalent e-document of OVDs...
Note: The term "V-CIP" in source 5 is not explicitly defined...]
```

**Root cause:**  
`CitationQueryEngine` uses a generic prompt that encourages structured markdown output.
Deepgram TTS reads "dash, dash" for bullet markers and "Note colon" literally.  
The "V-CIP not explicitly defined" hedge is a hallucination hedge — inappropriate for voice.

**Fix needed:**  
Add a voice-mode answer rewrite step: after RAG returns, make one fast Groq call:
> "Rewrite the following answer as natural spoken English in 3–5 sentences.
>  No bullet points, no markdown, no 'Note:' hedges. State what is known directly."

Alternatively, add a voice-specific system prompt override to CitationQueryEngine.

---

### Issue 4 — Retrieval: 100-token leaf is too granular for list-type answers

**Root cause:**  
The RBI KYC Master Direction lists OVDs as a numbered enumeration spanning ~200 words.
At 100 tokens per leaf (~75 words), this list is **split across 3 leaf nodes**.
AutoMerging can reunite them under their 400-token mid-parent — but only if all 3 leaves
score in the top-10 candidates. With BM25+vector fusion at k=10, borderline leaves can
drop out before reranking.

**Evidence:** Chat (CitationQueryEngine, full answer) lists 4 specific OVDs by name.
Voice (same reranker top-n=5 after fix) gives a vague "OVDs including..." response — the
specific OVD enumeration chunk scored below the cutoff.

**Fix needed:**
- Increase leaf size: `100 → 256 tokens` — reduces fragmentation of lists and tables.
- Increase leaf overlap: `10 → 25 tokens` — bridges cross-chunk list continuity.
- Increase `similarity_top_k`: `10 → 15` per retriever — more candidates reach the reranker.

---

### Issue 5 — Reranker: bge-reranker-base under-ranks specific enumeration chunks

**Root cause:**  
`bge-reranker-base` (cross-encoder, ~278M params) is the lightweight variant.
For list-retrieval queries ("What are the OVDs?"), it tends to favour chunks with
high query-term density over chunks that contain the actual enumeration.
`bge-reranker-large` (~560M params) is significantly more accurate on precise retrieval.

**Fix needed:**  
Upgrade reranker model from `BAAI/bge-reranker-base` → `BAAI/bge-reranker-large`.  
Trade-off: ~2x memory, ~1.5x latency per rerank batch.  
Impact: voice latency increases ~200–400ms per query (acceptable).

---

### Issue 6 — citation_chunk_size mismatch

**Root cause:**  
`CitationQueryEngine(citation_chunk_size=512)` re-chunks the retrieved nodes into
512-token citation blocks. But leaf nodes are 100 tokens and mid nodes are 400 tokens.
Re-chunking at 512 tokens can **splice two unrelated mid-nodes** together into one
citation source, diluting precision and confusing the LLM about source boundaries.

**Fix needed:**  
Set `citation_chunk_size` to match the chunk hierarchy:
- If using leaf nodes (100 tok): set `citation_chunk_size=100, citation_chunk_overlap=0`
- If using auto-merged nodes (400 tok): set `citation_chunk_size=400`
- Best approach: disable re-chunking by setting `citation_chunk_size` large enough
  that the retrieved nodes are never split further: `citation_chunk_size=1600`.

---

## Phase 3 Improvement Plan

### P0 — Voice pipeline fixes (no re-ingestion needed)

| Task | File | Change | Status |
|------|------|--------|--------|
| P0-1 | `src/voice/pipeline.py` | Rewrite `_sentence_chunks` to split on newlines + strip list markers | ✅ Done 2026-04-07 |
| P0-2 | `src/generation/rag_chain.py` | Custom QA prompts per mode (text + voice) | ⚠ Partial — see notes |
| P0-3 | `src/voice/pipeline.py` | Add TranscriptionFrame deduplication (5s window, substring check) | ✅ Done 2026-04-08 |

**P0-2 implementation notes (2026-04-07):**
- Replaced default `CitationQueryEngine` prompt (verbatim, hedged, bureaucratic) with two custom `PromptTemplate`s selected by `mode`.
- `_TEXT_QA_TEMPLATE`: plain English, no preamble, inline citations, short paragraphs, for chat/API consumers.
- `_VOICE_QA_TEMPLATE`: 3–4 natural spoken sentences, no bullets, no markdown, no citation numbers, no hedging — sent when `mode="voice"`.
- Zero added latency (prompt swap at construction time, not a second LLM call).
- Also fixed `_CITATION_RE` in `pipeline.py` — now strips `[1, 3]` / `[1,2,3]` multi-number citations that were leaking into TTS as spoken "one comma three".
- Observed from live test: previous answer opened with "According to the provided sources..." and contained bullet points — both now explicitly prohibited by prompt rules.
- **Root cause identified (2026-04-07):** Heavy custom voice prompts ("no hedging", "state facts directly") caused llama-3.1-8b-instant to say "I don't have sufficient information" on partial-content leaf chunks. The model is small — it responds poorly to long negative instruction lists and ignores explicit "do not say X" rules. Reverted to minimal prompts: voice = default CitationQueryEngine phrasing + light format rules (no bullets, no markdown, no citation numbers). Text = default + plain English / no bullet points.
- **Score filter added (2026-04-07):** Nodes with reranker score < 0 are dropped before CitationQueryEngine. In tests, slots [4] and [5] consistently score -0.5 to -1.0 for OVD queries. Keeps at least top-1 as safety floor.
- **Real fix needed: P2 (larger chunks).** The "insufficient information" is not a prompt failure — the 100-token leaf nodes genuinely don't contain the OVD enumeration text. Auto-merged=0 in all tests. Until P2 (256-token leaves) is done, voice answers for list-type queries will be incomplete.

**P0-1 implementation notes (2026-04-07):**
- `_sentence_chunks` now splits on `\n` first, strips list markers (`-`, `•`, `*`, `1.`, `1)`), then splits on `[.!?]` within each line.
- Test: bullet-list answer (496 chars) → **5 chunks** (was 1). Plain sentences unchanged.
- Added TTFT timing log in `RAGProcessor._answer`: logs ms from TranscriptionFrame received → first TextFrame pushed (covers classifier + RAG + chunking).
- Added per-stage timing in `src/generation/rag_chain.py`: retrieve / rerank / generation ms + total.
- Added auto-merge detection: counts nodes where `NodeRelationship.CHILD` present (= promoted from leaf). Logged as `auto-merged=N` on every query.
- Added reranked node detail log per slot: score, merged flag, chapter, section, content length.
- `AutoMergingRetriever` set to `verbose=True` — its internal merge decisions (`> Merging X nodes into parent`) now appear in stdout alongside loguru output.

### P1 — Retrieval tuning (no re-ingestion needed)

| Task | File | Change | Status |
|------|------|--------|--------|
| P1-1 | `src/retrieval/pipeline.py` | ~~Upgrade reranker: `bge-reranker-base` → `bge-reranker-large`~~ **DEFERRED** — too much voice latency | ❌ Deferred |
| P1-2 | `src/retrieval/hybrid_retriever.py` | Increase `similarity_top_k` from `10` → `15` | ✅ Already done (default=15 in pipeline.py:37,116) |
| P1-3 | `src/generation/rag_chain.py` | Fix `citation_chunk_size` mismatch (set to 1600 or match leaf size) | ✅ Done 2026-04-08 (set to 256, same as P2-3) |

### P2 — Re-ingestion: better chunking (requires delete `storage/` + rebuild)

| Task | File | Change |
|------|------|--------|
| P2-1 | `src/ingestion/chunker.py` | Leaf: `100 → 256 tokens`, overlap `10 → 25 tokens` | ✅ Done 2026-04-08 |
| P2-2 | `src/ingestion/chunker.py` | Mid: `400 → 600 tokens`, overlap `40 → 60 tokens` | ✅ Done 2026-04-08 |
| P2-3 | `src/generation/rag_chain.py` | Align `citation_chunk_size=256` after leaf resize | ✅ Done 2026-04-08 |
| P2-4 | (eval) | Re-run `evaluation/` golden dataset to measure MRR / hit-rate improvement |

### P3 — Voice UX (lower priority)

| Task | File | Change |
|------|------|--------|
| P3-1 | `src/voice/pipeline.py` | Voice-mode answer length cap: 4 sentences max |
| P3-2 | `src/voice/pipeline.py` | Add "Finova is thinking..." audio cue during RAG latency window |
| P3-3 | `index.html` | Show voice transcript in real-time in the Voice tab log feed |

---

## Expected Outcome After P0 + P1

| Metric | Before | Expected After |
|--------|--------|---------------|
| Voice chunks per answer | 1 (broken) | 4–8 sentence chunks |
| TTS first-audio latency | Full answer wait (~5s) | ~1s (first sentence streams) |
| OVD list answer completeness (voice) | Vague / incomplete | Full 6-item list |
| Spurious duplicate queries | 1 extra per utterance | 0 |
| Reranker recall (list queries) | ~60% | ~80% (estimated) |

---

## Re-ingestion Checklist (P2, when ready)

```bash
# 1. Delete persisted chunks (forces slow-path rebuild)
rm -rf storage/

# 2. Clear Qdrant vectors
rm -rf qdrant_storage/

# 3. Re-ingest with new chunk sizes
python -m src.ingestion.chunker

# 4. Rebuild vector index
python -m src.retrieval.vector_store

# 5. Run evaluation against golden dataset
python evaluation/eval_runner.py

# 6. Compare metrics vs baseline in evaluation/results/
```

---

## Files To Change Summary

```
src/voice/pipeline.py          P0-1, P0-2, P0-3, P3-1, P3-2
src/retrieval/pipeline.py      P1-1
src/retrieval/hybrid_retriever.py  P1-2
src/generation/rag_chain.py    P1-3, P2-3
src/ingestion/chunker.py       P2-1, P2-2
index.html                     P3-3
```

---

## Live Test Observations (2026-04-07)

Query: *"what are the official valid documents accepted for individual KYC under RBI norms?"*

| Stage | Timing | Notes |
|-------|--------|-------|
| Classifier | 426ms | FINANCE — correct |
| Retrieve (AutoMerging) | 1697ms | 10 nodes, **auto-merged=0** — all leaf-level |
| Rerank (bge-reranker-base) | 3161ms | **Main latency bottleneck** |
| Generation (Groq) | 619ms | — |
| Total / TTFT | 5904ms | Dominated by reranker |

**Retrieval quality:** All 5 reranked nodes are leaf-level (100 tokens, ~75 words).
Scores: 2.47, 2.27, 1.34, 0.99, 0.76 — steep drop-off.
Sections: Definitions (ch I) + KYC Compliance (ch VI). No auto-merging.
Answer lists 4 OVDs — correct but incomplete (full list has 6+ items).

**Root cause of incomplete answer:** OVD enumeration spans ~200 words across 3 leaf nodes.
Auto-merging never fires because all 3 siblings don't rank in top-10 together. Confirms P2 need.

**Reranker latency:** 3161ms for 10 candidates. Upgrading to `bge-reranker-large` deferred (too slow).
Alternative: reduce voice `similarity_top_k` 10→6 to cut candidates into the reranker.

---

---

## Eval Run 2026-04-08 — Findings & Fixes

### Eval run config
- Dataset: golden_dataset.json, `--limit 5` (kyc_aml domain, 5 questions)
- Judge: Groq `llama-3.1-8b-instant`
- Pipeline: BGE-reranker-base, similarity_top_k=10, reranker_top_n=5

### Results

| Metric | Value | Status |
|--------|-------|--------|
| RAGAS faithfulness | nan | BROKEN (n>1 error) |
| RAGAS answer_relevancy | nan | BROKEN (n>1 error) |
| RAGAS context_recall | 0.30 | LOW (regression −0.16 vs prev 0.46) |
| RAGAS context_precision | 0.30 | LOW |
| DeepEval faithfulness | 0.63 | LOW |
| DeepEval answer_relevancy | 0.79 | OK |
| DeepEval hallucination | 0.64 | HIGH |
| avg latency | 10,324 ms | SLOW |
| p95 latency | 17,086 ms | VERY SLOW |

### Latency breakdown (worst-case query)

| Stage | Time |
|-------|------|
| Retrieve (AutoMerging) | 468 ms |
| Rerank (BGE-reranker-base, 5 nodes) | 5,275 ms |
| Generation (Groq, large context) | 11,342 ms |
| Total | 17,086 ms |

### Root causes found

| # | Issue | Root cause | Severity |
|---|-------|-----------|----------|
| E1 | RAGAS nan (faithfulness, answer_relevancy) | `LangchainLLMWrapper` bypasses `_generate`/`_agenerate` overrides in some async paths in RAGAS 0.4.x. Groq rejects `n>1` → all samples fail silently | P0 |
| E2 | context_recall regression 0.46→0.30 | `run_queries` passed `s.snippet` (120 chars) to RAGAS/DeepEval instead of full chunk text. Judge couldn't find ground truth in 120-char excerpts | P0 (critical eval bug) |
| E3 | Merged nodes show `chapter='?'` | `enrich_nodes()` only enriches leaf nodes; parent/mid nodes never get `chapter`/`section_title`. AutoMerged nodes have no metadata | P1 |
| E4 | 64% hallucination + 11s generation | Node 5 (score=−5.48, **5664 chars**) + node 4 (score=−2.37, 1106 chars) included in LLM prompt — ~7000 chars of irrelevant merged context inflates prompt and creates noise the LLM hallucinates around | P1 |
| E5 | Reranker 5.2s for 5 nodes | BGE cross-encoder encoding the 5664-char merged node is the bottleneck; also applies to generation | P2 |

### Fixes applied 2026-04-08

| Fix | File | Change |
|-----|------|--------|
| E1: RAGAS n>1 | `evaluation/eval_runner.py` | Replaced `_generate`/`_agenerate` subclass override with direct monkeypatching of `llm.client.create` and `llm.async_client.create` — intercepts at the raw OpenAI SDK call level before Groq sees `n>1`. Sequential n×1 calls, merged choices. |
| E2: Context truncation | `evaluation/eval_runner.py:173` | Changed `contexts = [s.snippet ...]` → `[s.content ...]`. Added `content: str` field to `Source` dataclass (full chunk text). `snippet` kept as 120-char display field. |
| E3: Merged metadata | `src/ingestion/chunker.py` | After `enrich_nodes(leaf_nodes)`, walk each leaf's parent ancestry and fill `chapter`/`section_title` on parent/mid nodes if blank. **Requires `rm -rf storage/` to rebuild.** |
| E4: Noise node filter | `src/generation/rag_chain.py:204` | Changed filter from `score >= 0` to `score >= top_score − 5.0`. Removes node 4 (−2.37) and node 5 (−5.48) while keeping borderline node 3 (−0.29). Fallback: always keep at least top-1. |
| E2b: Source.content | `src/generation/rag_chain.py:101` | Added `content: str = ""` to `Source` dataclass; populated with `node.get_content()` (full text). |

### Expected outcome after fixes

| Metric | Before | Expected |
|--------|--------|---------|
| RAGAS faithfulness | nan | computable (>0.5 target) |
| RAGAS answer_relevancy | nan | computable (>0.6 target) |
| RAGAS context_recall | 0.30 | ~0.45+ (full context restores judge accuracy) |
| DeepEval hallucination | 0.64 | <0.40 (noise nodes removed) |
| avg latency | 10,324 ms | ~5,000 ms (no 5664-char node in prompt) |
| p95 latency | 17,086 ms | ~8,000 ms |
| Merged node metadata | chapter='?' | chapter/section visible |

---

## Guardrail & Prompt Review (2026-04-08)

### Issues found

| # | Component | Issue | Severity |
|---|-----------|-------|----------|
| G1 | `pipeline.py` — `_CLASSIFIER_PROMPT` | CHAT definition too broad — "makes sense", "perfect", "that's helpful" can be follow-up phrases before a finance question; classifier routed them to CHAT instead of FINANCE | High |
| G2 | `pipeline.py` — `_CLASSIFIER_PROMPT` | No tiebreaker rule — ambiguous short messages defaulted unpredictably | Medium |
| G3 | `pipeline.py` — `_CHAT_PERSONA_PROMPT` | No guardrail inside the CHAT persona — model could answer finance questions if one slipped through classification | Medium |
| G4 | `pipeline.py` — `_ETHICS_KEYWORDS` | Only 13 keywords; missed insider trading, benami, shell companies, document forgery, terror financing, smuggling | Medium |
| G5 | `api/main.py` | **Text API had zero guardrails** — every question went straight to RAG regardless of topic or ethics | Critical |

### Fixes applied 2026-04-08

| Fix | File | Change |
|-----|------|--------|
| G1: CHAT definition | `src/voice/pipeline.py` | Narrowed CHAT to pure single-word/phrase greetings only. Added explicit counter-examples ("thanks, what about NRI docs?" → NOT CHAT). |
| G2: Tiebreaker | `src/voice/pipeline.py` | Added rule: "When uncertain between FINANCE and CHAT, always choose FINANCE. Any question mark, topical noun, or info request → FINANCE." |
| G2: Follow-ups → FINANCE | `src/voice/pipeline.py` | Explicitly listed follow-up phrases ("tell me more", "elaborate", "what else?", "can you explain?") as FINANCE examples in classifier prompt. |
| G3: CHAT persona guardrail | `src/voice/pipeline.py` | Rewrote `_CHAT_PERSONA_PROMPT` with strict rules: must NOT answer any finance question; if finance content detected → redirect with "Go ahead and ask your compliance question." |
| G4: Ethics keywords | `src/voice/pipeline.py` | Expanded from 13 → 30 keywords: insider trading, benami, shell company, round tripping, market manipulation, fake KYC/documents, terror/terrorist financing, smuggling, drug money. |
| G5: Text API guardrails | `src/api/main.py` | Added `_guard_question()` async function: ethics keyword check → 3-way LLM classifier → FINANCE/CHAT/BLOCKED routing. Wired into both `/query` and `/query/decompose`. CHAT/BLOCKED paths return immediately without touching RAG. Classifier constants identical to voice pipeline. |

### Guardrail architecture (both voice + text, post-fix)

```
User input
  │
  ├─ Ethics keyword match? → BLOCKED (instant, no LLM call)
  │
  ├─ Groq 8b classifier (~150 ms)
  │    ├─ FINANCE → RAG pipeline
  │    ├─ CHAT    → canned greeting (voice: _conversational_reply; text: _CHAT_REPLY)
  │    └─ BLOCKED → _OUT_OF_SCOPE_REPLY
  │
  └─ [FINANCE only] → retrieve → rerank → CitationQueryEngine → answer
```

---

## Decision Log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-04-07 | Fixed `reranker_top_n=3→5` in voice pipeline | Voice was missing OVD list chunks that text API found |
| 2026-04-07 | Added CORS middleware to text API | Browser fetch blocked without it |
| 2026-04-07 | Frontend redesigned: white + sea-green theme, colorful coordinated UI | Production-ready look for demo |
| 2026-04-07 | Phase 3 plan created | Voice answer quality still poor after top-n fix; deeper issues in chunking, TTS formatting, and STT dedup identified |
| 2026-04-08 | Score filter changed `>=0` → `>=top_score−5.0` | Old filter dropped valid borderline nodes; new gap filter targets only extreme outliers (node 5 at −5.48 was 5664 chars of noise) |
| 2026-04-08 | RAGAS n>1 fix via client-level monkeypatch | Previous subclass override of `_generate`/`_agenerate` missed async code paths in RAGAS 0.4.x LangchainLLMWrapper |
| 2026-04-08 | `Source.content` field added | Eval harness was silently truncating context to 120 chars; needed full text for faithful RAGAS/DeepEval scoring |
| 2026-04-08 | Parent/mid metadata propagation added to chunker | AutoMerged parent nodes had no chapter/section_title; metadata was leaf-only |
| 2026-04-08 | Guardrail review — classifier prompt tightened | CHAT was too broad; follow-ups were not reaching RAG; tiebreaker added (default to FINANCE) |
| 2026-04-08 | Text API guardrails added (`api/main.py`) | Text API had no classifier gate; all questions hit RAG including blocked topics |
| 2026-04-08 | Ethics keywords expanded 13→30 | Insider trading, benami, shell company, fake documents, terror financing not previously covered |
