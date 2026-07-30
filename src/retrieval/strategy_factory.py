"""
RetrievalStrategyFactory — reads RETRIEVAL_STRATEGY from .env and returns
the correct retriever instance.

Supported strategies (set RETRIEVAL_STRATEGY in .env):
  "vector"    — existing revamp pipeline (CascadingRetriever / FusionRetriever),
                no change to any existing behaviour. Default.
  "pageindex" — pure vectorless PageIndex; LLM navigates document TOC directly,
                no Qdrant or BM25 involved.
  "hybrid"    — Stage1 vector rough-filter + Stage2 PageIndex deep-dive
                + Stage3 Supervisor LLM cross-check synthesis.

The factory is the ONLY place that reads RETRIEVAL_STRATEGY. Everything else
in the pipeline just calls retriever.query() or retriever.retrieve().

Usage (in the API or eval harness):
    from src.retrieval.strategy_factory import get_hybrid_retriever
    retriever = get_hybrid_retriever(agentic_index, qa_index, agentic_nodes)
    result = retriever.query(question)          # for hybrid / pageindex
    # or call query_revamp() directly for vector mode (unchanged path)
"""

from __future__ import annotations

import logging
from typing import List, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode

from config.revamp_settings import revamp_settings

logger = logging.getLogger(__name__)


def get_hybrid_retriever(
    agentic_index:  VectorStoreIndex,
    qa_index:       VectorStoreIndex,
    agentic_nodes:  List[TextNode],
    strategy:       Optional[str] = None,
):
    """
    Factory: return the retriever for the configured RETRIEVAL_STRATEGY.

    For "vector" strategy, returns None — callers should use query_revamp()
    directly so the existing path is completely unchanged.

    For "pageindex" and "hybrid", returns a HybridRAGRetriever instance.

    Args:
        agentic_index: VectorStoreIndex for agentic proposition collection.
        qa_index:      VectorStoreIndex for Q&A library collection.
        agentic_nodes: Sidecar nodes for BM25 (used by Stage 1).
        strategy:      Override RETRIEVAL_STRATEGY env var (for testing).

    Returns:
        HybridRAGRetriever instance, or None if strategy == "vector".
    """
    strategy = strategy or revamp_settings.RETRIEVAL_STRATEGY

    if strategy == "vector":
        logger.info("[StrategyFactory] strategy=vector → existing pipeline (no hybrid layer)")
        return None

    if strategy in ("pageindex", "hybrid"):
        from src.retrieval.hybrid_rag_retriever import HybridRAGRetriever
        logger.info(f"[StrategyFactory] strategy={strategy} → HybridRAGRetriever")
        return HybridRAGRetriever(
            agentic_index=agentic_index,
            qa_index=qa_index,
            agentic_nodes=agentic_nodes,
        )

    raise ValueError(
        f"Unknown RETRIEVAL_STRATEGY={strategy!r}. "
        f"Must be one of: vector, pageindex, hybrid"
    )


def query_with_strategy(
    question:      str,
    agentic_index: VectorStoreIndex,
    qa_index:      VectorStoreIndex,
    agentic_nodes: List[TextNode],
    mode:          str = "text",
    strategy:      Optional[str] = None,
):
    """
    Convenience wrapper: run query using the configured strategy.

    Dispatches to either query_revamp() (vector) or HybridRAGRetriever.query()
    (pageindex / hybrid). Returns a QueryResult either way.

    This is the single call-site change needed to plug the hybrid layer into
    existing API or voice handlers.
    """
    strategy = strategy or revamp_settings.RETRIEVAL_STRATEGY

    if strategy == "vector":
        from src.generation.revamp_rag_chain import query_revamp
        return query_revamp(
            question=question,
            agentic_index=agentic_index,
            qa_index=qa_index,
            agentic_nodes=agentic_nodes,
            mode=mode,
        )

    retriever = get_hybrid_retriever(agentic_index, qa_index, agentic_nodes, strategy)
    return retriever.query(question, mode=mode)
