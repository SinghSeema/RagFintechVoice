"""
Revamp RAG chain — drop-in replacement for rag_chain.query().

query_revamp(question, agentic_index, qa_index, agentic_nodes, mode)
    -> QueryResult   (same dataclass as rag_chain.py)

Two execution paths based on which retriever serves the query:

  Q&A path (cascading threshold met):
    Returns the pre-written answer stored in node.metadata["answer"].
    No reranker, no LLM synthesis, no citation engine.
    ~100ms response time.

  Agentic path (cascading threshold missed, fusion, or agentic-only mode):
    Hybrid BGE-M3 + BM25 retrieval → FlagEmbeddingReranker → score filter
    → CitationQueryEngine (same as legacy).
    ~17s on CPU due to reranker; Q&A cache layer reduces how often this runs.

Output schema: same QueryResult / Source dataclasses as rag_chain.py.
Import QueryResult from rag_chain so no schema divergence is possible.
"""

from __future__ import annotations

import time
from typing import Literal

from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.schema import NodeWithScore, TextNode
from loguru import logger

from src.generation.rag_chain import QueryResult, Source
from src.retrieval.revamp_pipeline import build_revamp_pipeline


# ── Answer templates (same design as rag_chain.py, no citation conflict) ─────

_TEXT_QA_TEMPLATE = PromptTemplate(
    """\
Please provide an answer based solely on the provided sources. \
When referencing information from a source, cite it with its corresponding \
number [1], [2], etc. Write in plain English — short paragraphs, no bullet lists. \
If no source is relevant, say so clearly.

{context_str}

Question: {query_str}
Answer:"""
)

_VOICE_QA_TEMPLATE = PromptTemplate(
    """\
You are a friendly bank compliance assistant speaking to a customer over a voice call.
Answer using only the information in the sources below.

STRICT RULES — follow every one:
1. Paraphrase everything in plain everyday English. NEVER copy regulatory text word-for-word.
2. Convert legal phrasing: "Officially Valid Documents" → "approved ID documents", \
"regulated entity" → "your bank", "Proof of possession of Aadhaar" → "Aadhaar card".
3. Do NOT mention source numbers, clause labels, or sub-clause codes like (a), (ab), (iii).
4. Maximum 3 short sentences. Voice responses must be brief.
5. If listing documents or steps, name only the 2-3 most common ones, \
then say "and similar documents" — do not read out every item.
6. Close with one short reference if helpful, e.g. "as per RBI KYC guidelines" — nothing longer.
7. If the sources do not contain a clear answer, say "I don't have that detail right now, \
please check with your branch."

{context_str}

Question: {query_str}
Answer:"""
)


# ── Static retriever wrapper (same pattern as rag_chain.py) ──────────────────

from llama_index.core.retrievers import BaseRetriever


class _StaticRetriever(BaseRetriever):
    """Returns a fixed node list regardless of query."""

    def __init__(self, nodes: list[NodeWithScore]) -> None:
        super().__init__()
        self._nodes = nodes

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return self._nodes


# ── Q&A fast-path answer builder ──────────────────────────────────────────────

def _build_qa_result(qa_nodes: list[NodeWithScore], question: str, mode: str) -> QueryResult:
    """Construct QueryResult from a Q&A library hit (no LLM synthesis)."""
    node  = qa_nodes[0].node
    score = qa_nodes[0].score or 0.0

    answer  = node.metadata.get("answer", node.text)
    section = node.metadata.get("section", "")
    question_stored = node.metadata.get("question", node.text)

    logger.info(
        f"RAG-revamp | Q&A fast path: score={score:.3f} "
        f"section={section!r} answer_len={len(answer)}"
    )

    source = Source(
        rank=1,
        section_title=section,
        chapter="",
        snippet=question_stored[:120],
        content=answer,
    )
    return QueryResult(
        answer=answer,
        sources=[source],
        mode=mode,
        top_score=score,
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def query_revamp(
    question: str,
    agentic_index: VectorStoreIndex,
    qa_index: VectorStoreIndex,
    agentic_nodes: list[TextNode],
    mode: Literal["text", "voice"] = "text",
    retrieval_mode: str | None = None,
    retriever=None,
) -> QueryResult:
    """
    Run the revamp RAG pipeline and return a structured QueryResult.

    Args:
        question:       User question (natural language).
        agentic_index:  VectorStoreIndex for 'fintech_rag_agentic'.
        qa_index:       VectorStoreIndex for 'fintech_rag_qa'.
        agentic_nodes:  TextNodes loaded from agentic sidecar (for BM25).
        mode:           "text" or "voice" — controls answer formatting.
        retrieval_mode: Overrides RETRIEVAL_MODE env var if set.
        retriever:      Pre-built CascadingRetriever/FusionRetriever.
                        Pass this to avoid rebuilding the reranker on every call
                        when running batch evaluations.

    Returns:
        QueryResult with .answer, .sources, .mode, .top_score.
        Schema identical to rag_chain.QueryResult.
    """
    reranker_top_n  = 3 if mode == "voice" else 5
    similarity_top_k = 15 if mode == "voice" else 20

    if retriever is None:
        retriever = build_revamp_pipeline(
            agentic_index=agentic_index,
            qa_index=qa_index,
            agentic_nodes=agentic_nodes,
            mode=retrieval_mode,
            reranker_top_n=reranker_top_n,
            similarity_top_k=similarity_top_k,
        )

    t0 = time.perf_counter()
    nodes, path = retriever.retrieve(question)
    t_retrieved = time.perf_counter()

    logger.info(
        f"RAG-revamp | retrieve path={path!r} nodes={len(nodes)} "
        f"elapsed={1000*(t_retrieved-t0):.0f}ms"
    )

    # ── Q&A fast path: return pre-written answer ──────────────────────────
    if path == "qa":
        return _build_qa_result(nodes, question, mode)

    # ── Agentic/fusion path: CitationQueryEngine synthesis ────────────────
    if not nodes:
        return QueryResult(
            answer="I could not find relevant information to answer this question.",
            sources=[],
            mode=mode,
            top_score=0.0,
        )

    top_score = nodes[0].score if nodes else 0.0

    if mode == "voice":
        # Bypass CitationQueryEngine for voice — it creates "Source N: (ab)..."
        # labels that the LLM echoes verbatim, producing robotic output.
        passages = "\n\n".join(n.node.get_content()[:600] for n in nodes[:3])
        prompt = _VOICE_QA_TEMPLATE.format(
            context_str=passages,
            query_str=question,
        )
        from llama_index.core import Settings
        answer = str(Settings.llm.complete(prompt)).strip()
    else:
        engine = CitationQueryEngine.from_args(
            agentic_index,
            retriever=_StaticRetriever(nodes),
            citation_chunk_size=256,
            citation_chunk_overlap=20,
            text_qa_template=_TEXT_QA_TEMPLATE,
            verbose=False,
        )
        answer = str(engine.query(question)).strip()

    t_generated = time.perf_counter()
    logger.info(
        f"RAG-revamp | generation elapsed={1000*(t_generated-t_retrieved):.0f}ms"
    )

    sources: list[Source] = []
    for i, src_node in enumerate(nodes, 1):
        meta    = src_node.node.metadata
        snippet = src_node.node.get_content()[:120]
        sources.append(Source(
            rank=i,
            section_title=meta.get("section", meta.get("section_title", "")),
            chapter=meta.get("chapter", ""),
            snippet=snippet,
            content=src_node.node.get_content(),
        ))

    return QueryResult(
        answer=answer,
        sources=sources,
        mode=mode,
        top_score=top_score,
    )
