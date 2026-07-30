"""
Phase 2 — FastAPI text-query gateway.

Endpoints
---------
  POST /query          Single-shot RAG query (text mode).
  POST /query/decompose  Multi-hop query with SubQuestionQueryEngine.
  GET  /health         Liveness probe.

Rate-limit backoff
------------------
Groq's free tier enforces 6,000 TPM.  When a 429 is returned, the error
message contains "Please try again in X.Xs".  _groq_retry() parses that
wait time, sleeps the calling thread (inside run_in_executor so the event
loop stays responsive), and retries up to MAX_GROQ_RETRIES times.

Audit log  (ADR hard constraint)
---------
Every query — successful or failed — is appended as a JSON line to
`logs/audit.jsonl`.  The file is opened in append mode; it is NEVER
truncated or overwritten.  Each entry carries:
  timestamp, question, mode, decompose, elapsed_ms, num_sources,
  answer_preview (first 200 chars), sources list, error (if any).

Startup
-------
The RAG stack (Qdrant index, BGE reranker, hybrid retriever) is loaded once
during FastAPI lifespan.  The SubQuestionQueryEngine is built lazily on the
first /query/decompose call and then cached.

Usage
-----
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

  curl -X POST http://localhost:8000/query \\
       -H "Content-Type: application/json" \\
       -d '{"question": "What documents does an NRI need for KYC?"}'

  curl -X POST http://localhost:8000/query/decompose \\
       -H "Content-Type: application/json" \\
       -d '{"question": "Compare simplified KYC with full CDD requirements"}'
"""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from groq import AsyncGroq
from loguru import logger
from pydantic import BaseModel

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

AUDIT_LOG = Path("logs/audit.jsonl")
MAX_GROQ_RETRIES = 3
_RETRY_AFTER_RE = re.compile(r"try again in (\d+\.?\d*)s", re.IGNORECASE)


# ── Guardrail constants (mirrors src/voice/pipeline.py) ───────────────────────
#
# The text API applies the same 3-layer guardrail as the voice pipeline:
#   1. Ethics hard-block   (keyword match, instant)
#   2. 3-way classifier    (Groq 8b — FINANCE / CHAT / BLOCKED)
#   3. RAG                 (FINANCE path only)

_CLASSIFIER_PROMPT = """\
You are a strict intent classifier for Finova, a SPECIALIZED assistant whose
knowledge base covers ONLY RBI regulatory compliance topics.

Classify the user message into exactly one of three categories:

FINANCE  — Any message with substantive topical content related to Finova's domain,
  including follow-up requests and continuations of a prior finance answer:
  • KYC (Know Your Customer) requirements, documents, procedures
  • AML (Anti-Money Laundering) and CFT obligations
  • PMLA (Prevention of Money Laundering Act) provisions
  • RBI Master Directions and circulars
  • Customer due diligence: CDD, EDD (Enhanced), SDD (Simplified)
  • Account opening rules and identity verification
  • NRI banking compliance and documentation
  • Suspicious Transaction Reports (STR), Cash Transaction Reports (CTR)
  • SWIFT / correspondent banking compliance
  • Digital payment regulations (UPI, PPI) from an RBI compliance angle
  • CIBIL only in context of KYC/credit compliance, not credit scores generally
  • Follow-ups even if phrased casually: "tell me more", "elaborate", "what else?",
    "can you explain that?", "go on", "what about [any finance topic]?",
    "what does that mean?", "give an example", "is there anything else?"

CHAT     — ONLY a pure greeting, farewell, or single-word social acknowledgment
  with NO topical content whatsoever.
  Examples: "hi", "hello", "hey", "good morning", "bye", "goodbye",
  "thank you", "thanks", "okay", "ok".
  NOT CHAT: "thanks, what about NRI documents?", "okay so what are the rules?",
  "that's helpful, tell me more", "makes sense, can you elaborate?"

BLOCKED  — Everything outside Finova's compliance domain:
  • Stock market, equity trading, NSE, BSE, Nifty, Sensex, share prices
  • Mutual fund returns, portfolio management, asset allocation
  • Cryptocurrency, Bitcoin, Web3, NFT
  • IPO applications, demat accounts for investing
  • Personal finance advice (EMI calculators, savings plans, FD rates)
  • Insurance products, health coverage, LIC policies
  • Income tax filing, GST, CA advice
  • General banking products (interest rates, loan EMI, credit card offers)
  • Sports, entertainment, cooking, travel, health, politics, coding, weather

IMPORTANT: When uncertain between FINANCE and CHAT, always choose FINANCE.
Any message with a question mark, a topical noun, or a request for information
is FINANCE — not CHAT.

Reply with exactly one word: FINANCE, CHAT, or BLOCKED. No explanation."""

_ETHICS_KEYWORDS = (
    "buy stock", "buy share", "sell stock", "sell share",
    "invest in", "which stock", "pick stock", "best stock",
    "guaranteed return", "100% return", "sure profit", "risk free return",
    "money laundering", "hide money", "avoid tax illegally",
    "evade tax", "tax evasion", "black money", "hawala",
    "benami", "shell company", "round tripping", "layering",
    "insider trading", "insider tip", "front run", "front-run",
    "pump and dump", "pump & dump", "market manipulation",
    "fake kyc", "fake document", "forged document", "fake aadhaar",
    "fake pan", "counterfeit", "fake id",
    "terror financing", "terrorist financing", "fund terrorism",
    "smuggling", "drug money", "criminal proceeds",
)

_OUT_OF_SCOPE_REPLY = (
    "That topic is outside my scope. "
    "I'm specialized in RBI regulatory compliance — KYC requirements, AML and PMLA obligations, "
    "customer due diligence, and RBI master directions. "
    "Is there anything in that space I can help you with?"
)

_ETHICS_REPLY = (
    "I'm not able to provide personal investment advice, recommendations on "
    "specific securities, or guidance on any activity that may violate financial laws. "
    "For investment decisions, please consult a SEBI-registered financial advisor."
)

_CHAT_REPLY = (
    "Hello! I'm Finova, your RBI regulatory compliance assistant. "
    "I can help with KYC requirements, AML obligations, RBI master directions, "
    "customer due diligence, and related compliance topics. "
    "What would you like to know?"
)


async def _guard_question(question: str, groq_key: str) -> tuple[str, str | None]:
    """
    Apply ethics + classifier guardrails to a question.

    Returns (label, reply) where:
      label = "FINANCE"  → proceed to RAG; reply is None.
      label = "CHAT"     → return canned greeting; reply is the text.
      label = "BLOCKED"  → return out-of-scope message; reply is the text.
    """
    # Layer 1: ethics hard-block (keyword, instant)
    q_lower = question.lower()
    if any(kw in q_lower for kw in _ETHICS_KEYWORDS):
        logger.warning(f"API | ethics blocked: {question!r}")
        return "BLOCKED", _ETHICS_REPLY

    # Layer 2: 3-way LLM classifier
    try:
        client = AsyncGroq(api_key=groq_key)
        resp = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _CLASSIFIER_PROMPT},
                {"role": "user",   "content": question},
            ],
            max_tokens=5,
            temperature=0,
        )
        label = resp.choices[0].message.content.strip().upper().split()[0]
        if label not in ("FINANCE", "CHAT", "BLOCKED"):
            label = "FINANCE"
    except Exception as exc:
        logger.warning(f"API | classifier error (defaulting to FINANCE): {exc}")
        label = "FINANCE"

    logger.info(f"API | classifier={label} question={question!r}")

    if label == "CHAT":
        return "CHAT", _CHAT_REPLY
    if label == "BLOCKED":
        return "BLOCKED", _OUT_OF_SCOPE_REPLY
    return "FINANCE", None

# ── Shared executor for synchronous RAG calls ──────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=2)


# ── Audit log ──────────────────────────────────────────────────────────────────

def _audit(entry: dict[str, Any]) -> None:
    """Append one JSON line to the audit log.  Never raises — log failures are swallowed."""
    try:
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.error(f"Audit log write failed: {exc}")


# ── Rate-limit retry ───────────────────────────────────────────────────────────

def _groq_retry(fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs), retrying on Groq 429 rate-limit errors.

    Parses "Please try again in X.Xs" from the error message to determine
    the exact wait time.  Falls back to exponential backoff (5s, 10s, 20s)
    when the time cannot be parsed.

    Runs synchronously — intended to be called from inside a ThreadPoolExecutor.
    """
    for attempt in range(MAX_GROQ_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            err_str = str(exc)
            is_rate_limit = "429" in err_str or "rate_limit_exceeded" in err_str.lower()
            if not is_rate_limit or attempt == MAX_GROQ_RETRIES - 1:
                raise
            m = _RETRY_AFTER_RE.search(err_str)
            wait = float(m.group(1)) + 1.0 if m else (5.0 * 2 ** attempt)
            logger.warning(
                f"Groq rate limit — retrying in {wait:.1f}s "
                f"(attempt {attempt + 1}/{MAX_GROQ_RETRIES})"
            )
            time.sleep(wait)


# ── RAG stack loader ───────────────────────────────────────────────────────────

def _load_rag_stack(groq_api_key: str):
    """Load and return (index, nodes, storage_context, retriever, reranker)."""
    from pathlib import Path as _Path

    from src.ingestion.chunker import chunk_document
    from src.ingestion.metadata import build_section_map_and_clean
    from src.ingestion.parser import load_pdf_with_sections
    from src.retrieval.hybrid_retriever import configure_llm
    from src.retrieval.pipeline import build_pipeline
    from src.retrieval.vector_store import load_index

    lock = _Path("qdrant_storage/.lock")
    if lock.exists():
        lock.unlink()
        logger.warning("Removed stale qdrant_storage/.lock")

    configure_llm(groq_api_key)

    pdf_path = "data/raw/rbi_kyc_master_direction.pdf"
    text_marked = load_pdf_with_sections(pdf_path)
    chapter_map, section_map, clean_text = build_section_map_and_clean(text_marked)
    leaf_nodes, storage_context = chunk_document(
        text=clean_text,
        filename=pdf_path,
        section_map=section_map,
        chapter_map=chapter_map,
        clean_text=clean_text,
    )
    index = load_index(storage_context)
    retriever, reranker = build_pipeline(
        index, leaf_nodes, storage_context,
        similarity_top_k=10, reranker_top_n=5,
    )
    logger.info(f"RAG stack ready — {len(leaf_nodes)} leaf nodes")
    return index, leaf_nodes, storage_context, retriever, reranker


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    loop = asyncio.get_event_loop()
    groq_key = os.environ["GROQ_API_KEY"]
    logger.info("Loading RAG stack ...")
    (
        app.state.index,
        app.state.nodes,
        app.state.storage_context,
        app.state.retriever,
        app.state.reranker,
    ) = await loop.run_in_executor(_executor, _load_rag_stack, groq_key)
    app.state.sub_engine = None   # built lazily on first /query/decompose call
    app.state.groq_key = groq_key
    logger.info("RAG stack ready — API accepting requests")
    yield
    logger.info("Shutting down")
    _executor.shutdown(wait=False)


app = FastAPI(title="RagFintechVoice — Text Query API", lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    mode: str = "text"   # "text" | "voice"


class SourceOut(BaseModel):
    rank: int
    section_title: str
    chapter: str
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    mode: str
    elapsed_ms: int
    decomposed: bool = False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run_query(request: QueryRequest, state) -> QueryResponse:
    """Execute a single-shot RAG query with Groq rate-limit retry (sync)."""
    from src.generation.rag_chain import query

    t0 = time.perf_counter()
    result = _groq_retry(
        query,
        request.question,
        index=state.index,
        nodes=state.nodes,
        storage_context=state.storage_context,
        mode=request.mode,
        retriever=state.retriever,
        reranker=state.reranker,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    return QueryResponse(
        answer=result.answer,
        sources=[SourceOut(**s.__dict__) for s in result.sources],
        mode=result.mode,
        elapsed_ms=elapsed_ms,
    )


def _run_decompose(request: QueryRequest, state) -> QueryResponse:
    """Execute a sub-question decomposition query with Groq rate-limit retry (sync)."""
    from src.generation.subquery import build_sub_engine

    # Build and cache the sub-engine on first use
    if state.sub_engine is None:
        logger.info("Building SubQuestionQueryEngine (first use) ...")
        state.sub_engine = build_sub_engine(
            index=state.index,
            nodes=state.nodes,
            storage_context=state.storage_context,
            retriever=state.retriever,
            reranker=state.reranker,
        )

    t0 = time.perf_counter()
    response = _groq_retry(state.sub_engine.query, request.question)
    elapsed_ms = round((time.perf_counter() - t0) * 1000)

    answer = str(response)
    # Sub-engine returns source nodes differently — collect from response.source_nodes
    sources: list[SourceOut] = []
    if hasattr(response, "source_nodes"):
        real_rank = 1
        for node_ws in response.source_nodes:
            content = node_ws.node.get_content()
            # SubQuestionQueryEngine injects synthetic nodes whose content
            # starts with "Sub question:" — skip them, they are not doc sources.
            if content.lstrip().startswith("Sub question:"):
                continue
            meta = node_ws.node.metadata
            sources.append(SourceOut(
                rank=real_rank,
                section_title=meta.get("section_title", ""),
                chapter=meta.get("chapter", ""),
                snippet=content[:120],
            ))
            real_rank += 1

    return QueryResponse(
        answer=answer,
        sources=sources,
        mode=request.mode,
        elapsed_ms=elapsed_ms,
        decomposed=True,
    )


def _write_audit(
    question: str,
    mode: str,
    decomposed: bool,
    elapsed_ms: Optional[int],
    response: Optional[QueryResponse],
    error: Optional[str],
) -> None:
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "mode": mode,
        "decomposed": decomposed,
        "elapsed_ms": elapsed_ms,
        "num_sources": len(response.sources) if response else 0,
        "answer_preview": response.answer[:200] if response else None,
        "sources": [s.model_dump() for s in response.sources] if response else [],
        "error": error,
    }
    _audit(entry)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Single-shot RAG query in text mode.

    Guardrails (applied before RAG):
      1. Ethics keyword hard-block — instant refusal for illegal/investment advice.
      2. 3-way classifier (Groq 8b) — FINANCE proceeds to RAG; CHAT returns a
         canned greeting; BLOCKED returns an out-of-scope message.

    Uses hybrid retrieval (BM25 + vector) → auto-merge → BGE reranker →
    CitationQueryEngine (Groq llama-3.1-8b-instant).
    Retries automatically on Groq 429 rate-limit errors.
    Every call is appended to logs/audit.jsonl.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    state = app.state
    response: Optional[QueryResponse] = None
    error: Optional[str] = None

    try:
        # ── Guardrail gate (ethics + classifier) ──────────────────────────────
        label, guardrail_reply = await _guard_question(
            request.question, state.groq_key
        )
        if label != "FINANCE":
            response = QueryResponse(
                answer=guardrail_reply,
                sources=[],
                mode=request.mode,
                elapsed_ms=0,
            )
            return response

        # ── FINANCE path — RAG ─────────────────────────────────────────────────
        response = await loop.run_in_executor(
            _executor, _run_query, request, state
        )
        return response
    except Exception as exc:
        error = str(exc)
        logger.error(f"/query failed: {exc}")
        raise HTTPException(status_code=500, detail=error)
    finally:
        _write_audit(
            question=request.question,
            mode=request.mode,
            decomposed=False,
            elapsed_ms=response.elapsed_ms if response else None,
            response=response,
            error=error,
        )


@app.post("/query/decompose", response_model=QueryResponse)
async def query_decompose_endpoint(request: QueryRequest):
    """
    Multi-hop RAG query using SubQuestionQueryEngine.

    Applies the same ethics + classifier guardrails as /query before
    dispatching to the sub-question engine.

    Best for comparison/contrast queries, e.g.:
      "Compare simplified KYC with full CDD requirements"
      "What are the differences between NRI and resident KYC documents?"

    The sub-engine is built lazily on first call and cached for subsequent
    requests (avoids rebuilding the CitationQueryEngine tools per call).
    """
    import asyncio
    loop = asyncio.get_event_loop()
    state = app.state
    response: Optional[QueryResponse] = None
    error: Optional[str] = None

    try:
        # ── Guardrail gate (ethics + classifier) ──────────────────────────────
        label, guardrail_reply = await _guard_question(
            request.question, state.groq_key
        )
        if label != "FINANCE":
            response = QueryResponse(
                answer=guardrail_reply,
                sources=[],
                mode=request.mode,
                elapsed_ms=0,
            )
            return response

        # ── FINANCE path — decompose RAG ───────────────────────────────────────
        response = await loop.run_in_executor(
            _executor, _run_decompose, request, state
        )
        return response
    except Exception as exc:
        error = str(exc)
        logger.error(f"/query/decompose failed: {exc}")
        raise HTTPException(status_code=500, detail=error)
    finally:
        _write_audit(
            question=request.question,
            mode=request.mode,
            decomposed=True,
            elapsed_ms=response.elapsed_ms if response else None,
            response=response,
            error=error,
        )
