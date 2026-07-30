# Phase 3 — RAG Pipeline Fix Plan

**Created:** 2026-04-16  
**Based on:** Live log analysis (VoiceAPI_output.txt) + code review  
**Status:** P0 applied ✅ | Post-P0 fixes applied ✅ | P1/P2 pending

---

## Root Cause Summary

| # | Issue | Symptom in logs | Severity | Status |
|---|-------|----------------|----------|--------|
| F1 | Reranker no FP16/batching config | `rerank elapsed=26098ms` | Critical | ✅ Applied |
| F2 | Score filter has no absolute floor | All 5 noise nodes pass filter for off-topic queries | Critical | ✅ Applied |
| F3 | System prompt Rule 2 overrides voice QA template | LLM says "I don't have sufficient information" despite relevant nodes | High | ✅ Applied |
| F4 | Classifier tiebreaker maps everything to FINANCE | Productivity/lifestyle queries hit RAG | High | ✅ Applied |
| F10 | System prompt Rule 4 (cite [1][2]) overrides voice "no citations" instruction | LLM outputs "Source 2 and Source 4 mention that..." in voice answers | High | ✅ Applied |
| F11 | Voice `similarity_top_k=15` sends too many candidates to reranker | 17s rerank even after FP16 fix; loosely-related §38 update chunks win over onboarding doc chunks | High | ✅ Applied |
| F12 | No cancellation of in-flight RAG query on user interruption | `allow_interruptions=True` stops TTS audio but stale 17s answer still pushed after new question asked | High | ✅ Applied |
| F5 | Metadata propagation runs after docstore.add_documents() | `chapter='?'` on auto-merged nodes in logs | Medium | ⏳ P1 |
| F6 | `citation_chunk_size=256` re-fragments parent nodes | LLM loses context when parent node (400–1500 tok) is re-split | Medium | ⏳ P1 |
| F7 | `AsyncGroq` client created per classifier call | 20–50ms connection overhead per query | Low | ⏳ P2 |
| F8 | `asyncio.get_event_loop()` deprecated | DeprecationWarning in Python 3.10+, error in 3.12 | Low | ✅ Applied |
| F9 | `_StaticRetriever` defined inside `query()` per call | Minor: class re-created on every RAG call | Low | ⏳ P2 |

---

## Fix 1 — Reranker FP16 + Batching

**File:** `src/retrieval/pipeline.py:72`  
**Impact:** 26s → ~2–4s reranking. Prerequisite for all other fixes to be testable.

**Root cause:** `FlagEmbeddingReranker` defaults to FP32 inference, no batch size control, and runs a CUDA probe on startup. One long auto-merged node (400–1500 tokens) pads all other nodes in the batch to its sequence length — multiplying compute cost.

```python
# BEFORE
reranker = FlagEmbeddingReranker(
    model=reranker_model,
    top_n=reranker_top_n,
)

# AFTER
reranker = FlagEmbeddingReranker(
    model=reranker_model,
    top_n=reranker_top_n,
    use_fp16=True,   # halves memory bandwidth and compute on CPU
    batch_size=4,    # prevents one long merged node padding the entire batch
    device="cpu",    # skip CUDA probe; force CPU path immediately
)
```

**Expected outcome:**
- Rerank 15 nodes: ~26s → ~2–4s
- Total TTFT for a voice query: ~30s → ~5–7s

---

## Fix 2 — Score Filter Absolute Floor

**File:** `src/generation/rag_chain.py:208`  
**Impact:** Stops noise nodes (scores < −4) from reaching the LLM, eliminating hallucination on off-topic queries.

**Root cause:** Current filter `score >= top_score − 5.0` is relative only. When `top_score = −8.1` (off-topic query), the floor becomes `−13.1` — every node ever scored passes. All 5 irrelevant nodes enter `CitationQueryEngine` and the LLM hallucinates a refusal.

**Log evidence:**
```
# Off-topic productivity query — all scores deeply negative, all 5 nodes passed
[1] score=-8.1297  chapter='CHAPTER II'  section='3. Definitions'
[2] score=-8.3871  merged=True  chapter='?'
[3] score=-9.4181  chapter='Chapter VI'  section='38. Updation / Periodic Updation'
[4] score=-9.7611  chapter='Chapter VI'  section='8. Compliance of KYC policy'
[5] score=-10.0868 chapter='Chapter XI'  section='70. Hiring of Employees'
```

```python
# BEFORE
top_score = reranked[0].score if reranked else 0.0
positive_nodes = [n for n in reranked if n.score >= top_score - 5.0] or reranked[:1]

# AFTER
_ABS_FLOOR = -4.0   # BGE cross-encoder: below this score = no domain signal
top_score = reranked[0].score if reranked else 0.0
positive_nodes = [
    n for n in reranked
    if n.score >= top_score - 3.0 and n.score >= _ABS_FLOOR
] or reranked[:1]
```

Also add an early-exit in `RAGProcessor._answer` in `src/voice/pipeline.py` after reranking (inside the RAG call result processing): if the top-ranked node score is below the absolute floor, skip CitationQueryEngine and return the out-of-scope reply directly. This requires surfacing the top score from `query()` in `QueryResult`.

**Add to `QueryResult` dataclass (`rag_chain.py:105`):**
```python
@dataclass
class QueryResult:
    answer: str
    sources: list[Source]
    mode: str
    top_score: float = 0.0      # reranker score of the top node
    num_sources: int = field(init=False)
```

**Populate in `query()` (`rag_chain.py`, after reranking):**
```python
top_score_val = reranked[0].score if reranked else -99.0
```

**Early exit in `RAGProcessor._answer` (`pipeline.py`, after `result = await loop.run_in_executor(...)`):**
```python
_RAG_FLOOR = -4.0
if result.top_score < _RAG_FLOOR:
    logger.warning(
        f"RAGProcessor | top_score={result.top_score:.2f} below floor "
        f"({_RAG_FLOOR}) — no relevant content found, returning out-of-scope reply"
    )
    await self._push_reply(_OUT_OF_SCOPE_REPLY, t_question)
    return
```

---

## Fix 3 — System Prompt Rule 2 Overrides Voice QA Template

**File:** `src/retrieval/hybrid_retriever.py` (line with `_RAG_SYSTEM_PROMPT`, Rule 2)  
**Impact:** Eliminates "I don't have sufficient information" refusals when relevant nodes ARE present.

**Root cause:** The global `_RAG_SYSTEM_PROMPT` set on `Settings.llm` contains a rule like:
> "If the documents do not contain enough information to fully answer the question, say: 'I don't have sufficient information...'"

This system prompt takes priority over the user-turn `_VOICE_QA_TEMPLATE` instruction ("summarise what the sources do contain"). The small Llama-3.1-8b model follows the system prompt's conservative refusal rule even when partial but relevant content exists.

**Log evidence:**
```
# KYC query — nodes [1] and [3] are from correct chapter/section
[1] score=-1.4167  chapter='Chapter VI'  section='38. Updation / Periodic Updation of KYC'
[3] score=-2.6027  chapter='Chapter VI'  section='38. Updation / Periodic Updation of KYC'
# Yet LLM answered:
"I don't have sufficient information about the specific documents required for individual KYC"
```

**Change Rule 2 in `_RAG_SYSTEM_PROMPT` to:**
```
2. If the documents only partially cover the question, summarise what they do
   contain and note what aspect is missing. Do NOT say you have no information
   when the passages contain relevant details — extract and present what is there.
```

---

## Fix 4 — Classifier Tiebreaker Too Aggressive

**File:** `src/voice/pipeline.py:190`  
**Impact:** Stops personal productivity / lifestyle questions reaching the RAG pipeline.

**Root cause:** Tiebreaker says "Any message with a question mark, a topical noun, or a request for information is FINANCE." This is too broad — "daily routine", "productive", "day better" are topical nouns but not finance topics. They don't appear in the BLOCKED list either, so the tiebreaker fires and routes them to RAG.

**Log evidence:**
```
'okay okay wait so let me know like what is what is one thing I can adopt in my daily routine to make my day becoming more productive'
→ Classifier | FINANCE  ← should be BLOCKED

'Okay, tell me about one thing that I can adopt to make my day better.'
→ Classifier | FINANCE  ← should be BLOCKED
```

**Change 1 — Flip the tiebreaker (lines 190–192):**
```
# BEFORE
IMPORTANT: When uncertain between FINANCE and CHAT, always choose FINANCE.
Any message with a question mark, a topical noun, or a request for information
is FINANCE — not CHAT.

# AFTER
IMPORTANT: When uncertain between FINANCE and BLOCKED, always choose BLOCKED.
Only classify as FINANCE if the message clearly refers to a topic in the FINANCE
list above. Novel or off-topic questions that don't match either list → BLOCKED.
When uncertain between FINANCE and CHAT only (clear social context), choose FINANCE.
```

**Change 2 — Extend the BLOCKED list (after existing BLOCKED items):**
```
  • Self-improvement, productivity, wellness, lifestyle, daily routines, habits
  • Career advice, motivation, personal development, mindset
  • General knowledge, science, history, philosophy
```

---

## Fix 10 — System Prompt Rule 4 Overrides Voice "No Citations" Instruction

**File:** `src/retrieval/hybrid_retriever.py:74`  
**Applied:** 2026-04-16  
**Impact:** Voice answers stop outputting "Source 2 and Source 4 mention that..." and speak naturally.

**Root cause:** `_RAG_SYSTEM_PROMPT` Rule 4 said "Always cite the source number [1], [2], etc." — this is a system-prompt instruction that takes priority over the user-turn `_VOICE_QA_TEMPLATE`'s "no citation numbers" directive. The LLM substituted "Source N" text-style references instead of inline markers, producing unnatural voice output like:

```
"Source 2 and Source 4 mention that in case of no change in KYC information, a
self-declaration from the customer is obtained..."
```

**Fix:** Removed Rule 4 from `_RAG_SYSTEM_PROMPT` entirely. Citation format is now controlled solely by the per-mode QA template (`_TEXT_QA_TEMPLATE` instructs inline `[1][2]`; `_VOICE_QA_TEMPLATE` instructs no citations at all). Replaced Rule 4 with a neutral instruction to follow the formatting rules in the query prompt.

```python
# BEFORE Rule 4
"4. Always cite the source number [1], [2], etc. when you use information from a passage."

# AFTER Rule 4
"4. Keep answers accurate, concise, and grounded in the regulatory text.
   Follow the formatting instructions given in the query prompt exactly."
```

---

## Fix 11 — Reduce Voice similarity_top_k to Cut Reranker Batches

**File:** `src/voice/pipeline.py` (`_load_rag_stack`)  
**Applied:** 2026-04-16  
**Impact:** Reranker batches reduced from 4 → 2; expected latency ~17s → ~8s. Also filters out loosely-related chunks (e.g. §38 KYC update chunks) that were ranking above the target KYC onboarding document chunks.

**Root cause:** After F1 (FP16+batching), reranker latency improved from 26s to 17s but remained slow because `similarity_top_k=15` fed 15 candidates into the reranker. With `batch_size=4`, that means `ceil(15/4) = 4` batches × ~3s/batch = 12s of scoring. Reducing to `similarity_top_k=8` gives 2 batches × ~3s = ~6s.

Secondary benefit: fewer candidates means the reranker pool is tighter — the §38 Periodic Update chunks (retrieved because they contain the word "KYC") compete with fewer slots, giving onboarding document chunks a better chance of surfacing.

```python
# BEFORE (voice was same as text)
retriever, reranker = build_pipeline(
    index, leaf_nodes, storage_context,
    similarity_top_k=15,
    reranker_top_n=5,
)

# AFTER
retriever, reranker = build_pipeline(
    index, leaf_nodes, storage_context,
    similarity_top_k=8,   # voice: fewer candidates → fewer reranker batches (~2 vs 4)
    reranker_top_n=5,
)
```

---

## Fix 12 — Cancellation of In-Flight RAG Query on User Interruption

**File:** `src/voice/pipeline.py` (`RAGProcessor`)  
**Applied:** 2026-04-16  
**Impact:** When the user interrupts the bot mid-answer, the stale RAG result is discarded instead of being pushed to TTS after the user has already moved on.

**Root cause:** `PipelineParams(allow_interruptions=True)` stops TTS audio when `UserStartedSpeakingFrame` arrives, but the background thread running the RAG query (in `ThreadPoolExecutor`) continued to completion. When it finished (up to 17s later), the result was pushed to TTS — the user heard the old answer after they had already asked a new question.

**Fix:** Added `_pending_future` tracking to `RAGProcessor`. At the start of `_answer()`, if a previous future is still running, it is cancelled. The `await self._pending_future` call then raises `asyncio.CancelledError` which is caught silently, discarding the stale result.

```python
# In __init__
self._pending_future: Optional[asyncio.Future] = None

# At top of _answer() — cancel stale query before starting new one
if self._pending_future and not self._pending_future.done():
    self._pending_future.cancel()
    logger.warning(f"RAGProcessor | cancelled in-flight query for new question")

# RAG call — store future before awaiting
loop = asyncio.get_running_loop()   # also fixed: was get_event_loop() (F8)
self._pending_future = loop.run_in_executor(self._executor, lambda: query(...))
try:
    result = await self._pending_future
except asyncio.CancelledError:
    logger.warning(f"RAGProcessor | query cancelled (user interrupted)")
    return
```

> **Note:** `Future.cancel()` marks the awaitable as cancelled — `await` raises `CancelledError` immediately and the stale result is discarded. The background thread continues to completion (Python threads cannot be interrupted mid-execution), but its result is never consumed.

---

## Fix 5 — Metadata Propagation Before docstore.add_documents()

**File:** `src/ingestion/chunker.py:133`  
**Impact:** Fixes `chapter='?'` shown in logs for auto-merged parent nodes. Requires `rm -rf storage/` + rebuild.

**Root cause:** The docstore is populated at line 134 (`docstore.add_documents(all_nodes)`), then the ancestry walk to propagate `chapter`/`section_title` from leaf → parent runs at lines 140–155. Although Python docstore stores references (so in-memory mutations should reach the docstore), this order is fragile and may not survive all LlamaIndex serialization paths. Moving the propagation before insertion guarantees all nodes are fully enriched at persist time.

```python
# BEFORE order (lines 128–159):
#   1. enrich_nodes(leaf_nodes, ...)           ← leaf metadata only
#   2. docstore.add_documents(all_nodes)       ← adds nodes without parent metadata
#   3. [ancestry walk populates parent metadata in-memory]
#   4. storage_context = StorageContext(docstore)
#   5. storage_context.persist()

# AFTER order:
#   1. enrich_nodes(leaf_nodes, ...)           ← leaf metadata
#   2. [ancestry walk populates parent metadata]   ← move here
#   3. docstore.add_documents(all_nodes)       ← all nodes fully enriched
#   4. storage_context = StorageContext(docstore)
#   5. storage_context.persist()
```

Concretely, move the `node_index` / ancestry-walk block (lines 140–155) to immediately after `enrich_nodes()` (after line 130) and before `docstore = SimpleDocumentStore()` (line 133).

---

## Fix 6 — citation_chunk_size Mismatch

**File:** `src/generation/rag_chain.py:231`  
**Impact:** Prevents re-fragmentation of parent/mid nodes (400–1500 tokens) into tiny citation blocks that lose context.

**Root cause:** `CitationQueryEngine(citation_chunk_size=256)` splits retrieved nodes into 256-token citation blocks. A 1500-token parent node becomes 6 source blocks — the KYC enumeration text may be split across two adjacent blocks where neither alone contains the full list. The module docstring already documents the correct value as 512.

```python
# BEFORE
citation_chunk_size=256,
citation_chunk_overlap=20,

# AFTER
citation_chunk_size=512,   # matches docstring; prevents parent node re-fragmentation
citation_chunk_overlap=0,  # no overlap needed when chunks match retrieval granularity
```

---

## Fix 7 — AsyncGroq Client Per Call

**File:** `src/voice/pipeline.py:255` and `src/voice/pipeline.py:282`  
**Impact:** Saves 20–50ms per classifier call by reusing the connection pool.

```python
# BEFORE: new client on every _classify_query() and _conversational_reply() call
client = AsyncGroq(api_key=groq_api_key)

# AFTER: create once in RAGProcessor.__init__
self._groq_client = AsyncGroq(api_key=groq_api_key)

# Then in _classify_query() and _conversational_reply(), use self._groq_client
# (change both functions to accept the client object, or make them methods)
```

---

## Fix 8 — asyncio.get_event_loop() Deprecation

**File:** `src/voice/pipeline.py:453`

```python
# BEFORE
loop = asyncio.get_event_loop()

# AFTER
loop = asyncio.get_running_loop()
```

---

## Fix 9 — _StaticRetriever at Module Level

**File:** `src/generation/rag_chain.py:216`  
**Impact:** Minor — avoids re-creating the class object on every query call.

Move the `_StaticRetriever` class definition from inside `query()` to module level (after the imports, before `_TEXT_QA_TEMPLATE`).

---

## Implementation Order

| Priority | Fix | File | Re-ingestion needed? | Status |
|----------|-----|------|---------------------|--------|
| P0 | F1 — Reranker FP16+batching | `src/retrieval/pipeline.py` | No | ✅ |
| P0 | F2 — Score filter absolute floor | `src/generation/rag_chain.py`, `src/voice/pipeline.py` | No | ✅ |
| P0 | F3 — System prompt Rule 2 | `src/retrieval/hybrid_retriever.py` | No | ✅ |
| P0 | F4 — Classifier tiebreaker + BLOCKED list | `src/voice/pipeline.py` | No | ✅ |
| Post-P0 | F10 — Remove system prompt Rule 4 (citation conflict) | `src/retrieval/hybrid_retriever.py` | No | ✅ |
| Post-P0 | F11 — Reduce voice similarity_top_k 15→8 | `src/voice/pipeline.py` | No | ✅ |
| Post-P0 | F12 — Interruption cancellation + get_running_loop (F8) | `src/voice/pipeline.py` | No | ✅ |
| P1 | F6 — citation_chunk_size 256→512 | `src/generation/rag_chain.py` | No | ⏳ |
| P1 | F5 — Metadata propagation order | `src/ingestion/chunker.py` | **Yes** — `rm -rf storage/` | ⏳ |
| P2 | F7 — AsyncGroq client singleton | `src/voice/pipeline.py` | No | ⏳ |
| P2 | F9 — _StaticRetriever at module level | `src/generation/rag_chain.py` | No | ⏳ |

---

## Observed Results (Post-P0 + Post-P0 fixes, 2026-04-16)

| Metric | Before P0 | After P0 | After Post-P0 |
|--------|-----------|----------|---------------|
| Rerank latency | 26–30s | 17s | ~8s (estimated) |
| Total TTFT (voice) | ~30–36s | ~21s | ~10–12s (estimated) |
| Off-topic queries hitting RAG | Yes | No (classifier fix) | No |
| "Source N mentions that..." in voice output | Yes | Yes (system prompt conflict) | No (Rule 4 removed) |
| Stale answer after interruption | Yes | Yes (no cancellation) | No (future cancelled) |
| Reranker candidate batches | ~4 (15 nodes) | ~4 (15 nodes) | ~2 (8 nodes) |

---

## Expected Outcomes After All Applied Fixes

| Metric | Before | Expected Now |
|--------|--------|------------------|
| Rerank latency | 26–30s | ~6–8s |
| Total TTFT (voice) | ~30–36s | ~8–12s |
| Off-topic queries hitting RAG | Yes | No (BLOCKED at classifier) |
| "I don't have sufficient info" on valid KYC query | Yes | No |
| "Source N mentions that..." voice output | Yes | No |
| Stale answer pushed after interruption | Yes | No |
| Noise nodes passed to LLM | All 5 (for off-topic) | 0 (absolute floor blocks them) |
| Hallucination rate | ~64% (eval) | Target <30% |

---

## Verification Checklist

After applying P0 fixes, verify with these queries before re-running the full eval suite:

```
# Should be FINANCE → RAG → full answer (not "I don't have sufficient information")
"Tell me about what documents are needed for individual KYC?"
"What are the official valid documents accepted under RBI norms?"
"What is simplified due diligence for low-risk customers?"

# Should be BLOCKED at classifier (never reach RAG)
"What is one thing I can adopt in my daily routine to be more productive?"
"Tell me about one thing that can make my day better."
"What's the weather like today?"

# Should be CHAT → conversational reply (no RAG)
"Hi", "Hello", "Thank you", "Okay"

# Should be BLOCKED by ethics keywords (instant, no LLM call)
"How do I launder money?"
"Tell me about hawala transactions"
```
