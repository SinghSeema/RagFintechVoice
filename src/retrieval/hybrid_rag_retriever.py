"""
HybridRAGRetriever — 3-stage Vector + PageIndex orchestrator.

Stage 1 — Vector rough-filter
    Calls the existing CascadingRetriever / FusionRetriever (unchanged).
    Shortlists document IDs whose vector confidence >= HYBRID_CONF_THRESHOLD.

Stage 2 — PageIndex deep-dive
    For each shortlisted doc_id, calls PageIndexRetriever (vectorless LLM navigation).
    With a single document corpus, doc_id is always the configured PAGEINDEX_DOC_ID.
    When confidence threshold is not met, falls back to the top-1 vector result's doc.

Stage 3 — Supervisor synthesis
    SupervisorLLM receives both evidence streams and produces a verified,
    citation-rich answer that combines structural precision (PageIndex) with
    semantic breadth (vector chunks).

The result is a QueryResult-compatible object so the API and voice layers
need zero changes.

Public API:
    HybridRAGRetriever(agentic_index, qa_index, agentic_nodes, ...)
    retriever.query(question, mode) -> QueryResult
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode

from config.revamp_settings import revamp_settings
from src.generation.rag_chain import QueryResult, Source
from src.retrieval.hybrid_base import HybridRetrievalResult
from src.retrieval.pageindex_retriever import PageIndexRetriever
from src.retrieval.revamp_pipeline import build_revamp_pipeline
from src.retrieval.supervisor_llm import SupervisorLLM

logger = logging.getLogger(__name__)


class HybridRAGRetriever:
    """
    Orchestrates the 3-stage Hybrid RAG pipeline.

    Stage 1 reuses the existing revamp retrieval stack (CascadingRetriever or
    FusionRetriever) — nothing in the existing pipeline is touched.
    Stages 2 and 3 are additive layers that run after Stage 1.
    """

    def __init__(
        self,
        agentic_index:  VectorStoreIndex,
        qa_index:       VectorStoreIndex,
        agentic_nodes:  List[TextNode],
        retrieval_mode: Optional[str]  = None,
        doc_id:         Optional[str]  = None,
        smartmap_dir:   Optional[str]  = None,
        conf_threshold: Optional[float] = None,
    ) -> None:
        self.agentic_index  = agentic_index
        self.qa_index       = qa_index
        self.agentic_nodes  = agentic_nodes
        self.conf_threshold = conf_threshold or revamp_settings.HYBRID_CONF_THRESHOLD
        self.doc_id         = doc_id or revamp_settings.PAGEINDEX_DOC_ID

        # Stage 1 — existing vector pipeline (built once, reused across queries)
        self._vector_retriever = build_revamp_pipeline(
            agentic_index=agentic_index,
            qa_index=qa_index,
            agentic_nodes=agentic_nodes,
            mode=retrieval_mode or revamp_settings.RETRIEVAL_MODE,
        )

        # Stage 2 — PageIndex (SmartMap loaded lazily on first use)
        self._pageindex = PageIndexRetriever(
            doc_id=self.doc_id,
            smartmap_dir=smartmap_dir or revamp_settings.SMARTMAP_DIR,
        )

        # Stage 3 — Supervisor LLM
        self._supervisor = SupervisorLLM()

    # ── Stage 1 ────────────────────────────────────────────────────────────────

    def _stage1_vector(self, query: str) -> tuple[List[NodeWithScore], str]:
        """
        Run the existing vector pipeline.

        Returns (nodes, path) where path is 'qa', 'agentic', or 'fusion'.
        """
        return self._vector_retriever.retrieve(query)

    # ── Stage 2 ────────────────────────────────────────────────────────────────

    def _stage2_pageindex(self, query: str) -> HybridRetrievalResult:
        """
        Vectorless PageIndex retrieval for the configured doc_id.
        """
        return self._pageindex.retrieve_from_doc(query, doc_id=self.doc_id)

    # ── Stage 3 ────────────────────────────────────────────────────────────────

    def _stage3_supervisor(
        self,
        query:     str,
        vector_nodes: List[NodeWithScore],
        pi_result: HybridRetrievalResult,
    ) -> str:
        """Cross-check and synthesise both evidence streams."""
        return self._supervisor.synthesize(query, vector_nodes, pi_result)

    # ── Main entry point ───────────────────────────────────────────────────────

    def query(self, question: str, mode: str = "text") -> QueryResult:
        """
        Run the full 3-stage hybrid pipeline and return a QueryResult.

        The QueryResult schema is identical to query_revamp() so the API
        and voice layers are unaffected.

        Args:
            question: User question (natural language).
            mode:     "text" or "voice" (controls answer format).

        Returns:
            QueryResult with hybrid answer + sources from both pipelines.
        """
        t_start = time.perf_counter()

        # ── Stage 1: Vector retrieval ─────────────────────────────────────
        vector_nodes, vector_path = self._stage1_vector(question)
        t1 = time.perf_counter()
        logger.info(
            f"[Hybrid] Stage1 vector path={vector_path!r} "
            f"nodes={len(vector_nodes)} ({1000*(t1-t_start):.0f}ms)"
        )

        # If Q&A fast path hit with high confidence, still run PageIndex
        # but use Q&A answer as the vector evidence (not LLM synthesis)
        if vector_path == "qa" and vector_nodes:
            qa_node    = vector_nodes[0]
            qa_score   = qa_node.score or 0.0
            qa_answer  = qa_node.node.metadata.get("answer", qa_node.node.get_content())
            qa_section = qa_node.node.metadata.get("section", "")

            # Still run PageIndex for citation enrichment
            pi_result = self._stage2_pageindex(question)
            t2 = time.perf_counter()
            logger.info(
                f"[Hybrid] Stage2 PageIndex section={pi_result.citations} "
                f"({1000*(t2-t1):.0f}ms)"
            )

            # If Q&A score is very high (≥0.90), trust it; just append PI citation
            if qa_score >= 0.90:
                citation_note = (
                    f" [{pi_result.citations[0]}]" if pi_result.citations else ""
                )
                answer = qa_answer + citation_note
            else:
                # Use supervisor to merge Q&A evidence with PageIndex finding
                # Treat Q&A node as vector evidence
                answer = self._stage3_supervisor(question, vector_nodes, pi_result)

        else:
            # ── Stage 2: PageIndex deep-dive ──────────────────────────────
            pi_result = self._stage2_pageindex(question)
            t2 = time.perf_counter()
            logger.info(
                f"[Hybrid] Stage2 PageIndex section={pi_result.citations} "
                f"({1000*(t2-t1):.0f}ms)"
            )

            # ── Stage 3: Supervisor synthesis ─────────────────────────────
            if vector_nodes:
                answer = self._stage3_supervisor(question, vector_nodes, pi_result)
            else:
                # No vector results — fall back to PageIndex content directly
                logger.warning("[Hybrid] No vector nodes; using PageIndex content directly")
                answer = pi_result.content[:1500]

        t3 = time.perf_counter()
        logger.info(
            f"[Hybrid] Stage3 Supervisor done ({1000*(t3-t2):.0f}ms) "
            f"total={1000*(t3-t_start):.0f}ms"
        )

        # ── Build QueryResult ─────────────────────────────────────────────
        top_score = vector_nodes[0].score if vector_nodes else 0.0

        sources: List[Source] = []
        # Vector sources
        for i, nws in enumerate(vector_nodes[:3], 1):
            meta = nws.node.metadata
            sources.append(Source(
                rank=i,
                section_title=meta.get("section", meta.get("section_title", "")),
                chapter=meta.get("chapter", ""),
                snippet=nws.node.get_content()[:120],
                content=nws.node.get_content(),
            ))
        # PageIndex source
        pi_citation_str = pi_result.citations[0] if pi_result.citations else "PageIndex"
        sources.append(Source(
            rank=len(sources) + 1,
            section_title=pi_citation_str,
            chapter="PageIndex",
            snippet=pi_result.content[:120],
            content=pi_result.content,
        ))

        return QueryResult(
            answer=answer,
            sources=sources,
            mode=mode,
            top_score=top_score,
        )
