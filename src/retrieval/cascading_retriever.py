"""
Cascading and Fusion retrievers for the revamp RAG pipeline.

CascadingRetriever
  Query the Q&A collection first. If the top-1 cosine similarity score
  meets or exceeds CASCADE_CONFIDENCE_THRESHOLD, return that Q&A node
  directly — no reranking, no LLM synthesis needed.
  Otherwise, fall through to agentic retrieval (hybrid BGE-M3 + BM25
  → FlagEmbeddingReranker).

  Log line per query (one line):
    [cascading] query="..." score=X.XX → Q&A hit | agentic fallback

FusionRetriever
  Query both collections in parallel. Merge results using Reciprocal
  Rank Fusion (k=60). Return combined top-K for LLM synthesis.

Both retrievers expose a .retrieve(query) -> list[NodeWithScore] interface.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode

from config.revamp_settings import revamp_settings
from src.retrieval.bm25_store import build_bm25_retriever
from src.retrieval.hybrid_retriever import build_hybrid_retriever

logger = logging.getLogger(__name__)


# ── Score filter (same logic as rag_chain.py) ──────────────────────────────

_ABS_FLOOR   = -4.0
_REL_GAP     = 3.0


def _filter_noise(nodes: list[NodeWithScore]) -> list[NodeWithScore]:
    """Drop nodes below -4.0 or more than 3 points below the top score."""
    if not nodes:
        return nodes
    top_score = nodes[0].score or 0.0
    kept = [
        n for n in nodes
        if (n.score or 0.0) >= top_score - _REL_GAP
        and (n.score or 0.0) >= _ABS_FLOOR
    ]
    return kept or nodes[:1]


# ── Agentic retrieval helper (hybrid + reranker) ──────────────────────────

def _agentic_retrieve(
    query: str,
    agentic_index: VectorStoreIndex,
    agentic_nodes: list[TextNode],
    reranker,
    similarity_top_k: int = 20,
) -> list[NodeWithScore]:
    """
    Hybrid BGE-M3 + BM25 retrieval over the agentic proposition collection,
    followed by FlagEmbeddingReranker and noise filtering.
    """
    hybrid = build_hybrid_retriever(
        agentic_index, agentic_nodes, similarity_top_k=similarity_top_k
    )
    raw_nodes = hybrid.retrieve(query)

    query_bundle = QueryBundle(query_str=query)
    reranked = reranker.postprocess_nodes(raw_nodes, query_bundle)

    filtered = _filter_noise(reranked)
    return filtered


# ── CascadingRetriever ────────────────────────────────────────────────────────

class CascadingRetriever:
    """
    Q&A first, agentic fallback.

    The QA path is reranker-free (~100ms including Qdrant lookup).
    The agentic path uses hybrid retrieval + FlagEmbeddingReranker.
    """

    def __init__(
        self,
        qa_index: VectorStoreIndex,
        agentic_index: VectorStoreIndex,
        agentic_nodes: list[TextNode],
        reranker,
        threshold: float | None = None,
        reranker_top_n: int = 5,
        similarity_top_k: int = 20,
    ) -> None:
        self.qa_index        = qa_index
        self.agentic_index   = agentic_index
        self.agentic_nodes   = agentic_nodes
        self.reranker        = reranker
        self.threshold       = threshold if threshold is not None else revamp_settings.CASCADE_CONFIDENCE_THRESHOLD
        self.reranker_top_n  = reranker_top_n
        self.similarity_top_k = similarity_top_k

    def retrieve(self, query: str) -> tuple[list[NodeWithScore], str]:
        """
        Returns (nodes, path) where path is "qa" or "agentic".

        The caller checks path to decide whether to use the pre-written
        answer (qa) or synthesise from propositions (agentic).
        """
        t0 = time.perf_counter()

        # ── Q&A fast path ─────────────────────────────────────────────────
        qa_retriever = self.qa_index.as_retriever(similarity_top_k=1)
        qa_results   = qa_retriever.retrieve(query)

        qa_score = qa_results[0].score if qa_results else 0.0
        qa_ms    = (time.perf_counter() - t0) * 1000

        if qa_results and qa_score >= self.threshold:
            logger.info(
                f'[cascading] query={query!r:.60} score={qa_score:.3f} '
                f'≥ threshold={self.threshold} → Q&A hit ({qa_ms:.0f}ms)'
            )
            return qa_results, "qa"

        # ── Agentic fallback ──────────────────────────────────────────────
        logger.info(
            f'[cascading] query={query!r:.60} score={qa_score:.3f} '
            f'< threshold={self.threshold} → agentic fallback'
        )
        agentic_nodes = _agentic_retrieve(
            query,
            self.agentic_index,
            self.agentic_nodes,
            self.reranker,
            self.similarity_top_k,
        )
        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f'[cascading] agentic path returned {len(agentic_nodes)} nodes '
            f'(total={total_ms:.0f}ms)'
        )
        return agentic_nodes, "agentic"


# ── FusionRetriever ────────────────────────────────────────────────────────────

class FusionRetriever:
    """
    Queries both collections simultaneously. Merges via Reciprocal Rank Fusion.

    RRF formula: score(d) = Σ 1 / (k + rank(d))  where k=60.
    Higher fused score = better combined rank across both collections.

    The LLM receives both a Q&A answer candidate and supporting propositions,
    enabling cross-checking before synthesis.
    """

    def __init__(
        self,
        qa_index: VectorStoreIndex,
        agentic_index: VectorStoreIndex,
        agentic_nodes: list[TextNode],
        reranker,
        top_k: int = 5,
        rrf_k: int = 60,
        similarity_top_k: int = 20,
    ) -> None:
        self.qa_index         = qa_index
        self.agentic_index    = agentic_index
        self.agentic_nodes    = agentic_nodes
        self.reranker         = reranker
        self.top_k            = top_k
        self.rrf_k            = rrf_k
        self.similarity_top_k = similarity_top_k

    def retrieve(self, query: str) -> tuple[list[NodeWithScore], str]:
        """Returns (fused_nodes, 'fusion')."""
        t0 = time.perf_counter()

        with ThreadPoolExecutor(max_workers=2) as pool:
            qa_future      = pool.submit(self.qa_index.as_retriever(similarity_top_k=5).retrieve, query)
            agentic_future = pool.submit(
                _agentic_retrieve,
                query, self.agentic_index, self.agentic_nodes, self.reranker, self.similarity_top_k
            )
            qa_results      = qa_future.result()
            agentic_results = agentic_future.result()

        fused = self._rrf_merge([qa_results, agentic_results])
        elapsed_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            f'[fusion] query={query!r:.60} '
            f'qa={len(qa_results)} agentic={len(agentic_results)} '
            f'→ fused top-{len(fused)} ({elapsed_ms:.0f}ms)'
        )
        return fused, "fusion"

    def _rrf_merge(
        self, result_lists: list[list[NodeWithScore]]
    ) -> list[NodeWithScore]:
        rrf_scores: dict[str, float] = {}
        node_map:   dict[str, NodeWithScore] = {}

        for results in result_lists:
            for rank, nws in enumerate(results):
                nid = nws.node.node_id
                rrf_scores[nid] = rrf_scores.get(nid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                if nid not in node_map:
                    node_map[nid] = nws

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        fused = []
        for nid in sorted_ids[: self.top_k]:
            nws = node_map[nid]
            nws.score = rrf_scores[nid]
            fused.append(nws)

        return fused
