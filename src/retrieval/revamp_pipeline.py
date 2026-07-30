"""
Revamp retrieval pipeline factory.

build_revamp_pipeline(agentic_index, qa_index, agentic_nodes, mode)
    -> CascadingRetriever | FusionRetriever

The reranker (FlagEmbeddingReranker) is constructed here and shared
across both retriever types to avoid reloading the ~278MB model.
"""

from __future__ import annotations

import logging

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

from config.revamp_settings import revamp_settings
from src.retrieval.cascading_retriever import CascadingRetriever, FusionRetriever

logger = logging.getLogger(__name__)

_RERANKER_MODEL = "BAAI/bge-reranker-base"


def _build_reranker(top_n: int = 5) -> FlagEmbeddingReranker:
    """
    Build the BGE cross-encoder reranker, patched to use CPU with FP16
    and a small batch size to avoid memory pressure on long merged nodes.
    """
    from FlagEmbedding import FlagReranker

    reranker = FlagEmbeddingReranker(model=_RERANKER_MODEL, top_n=top_n, use_fp16=True)
    reranker._model = FlagReranker(
        _RERANKER_MODEL,
        use_fp16=True,
        devices="cpu",
        batch_size=4,
    )
    logger.info(f"[revamp_pipeline] Reranker loaded: {_RERANKER_MODEL} (CPU, fp16, batch=4)")
    return reranker


def build_revamp_pipeline(
    agentic_index: VectorStoreIndex,
    qa_index: VectorStoreIndex,
    agentic_nodes: list[TextNode],
    mode: str | None = None,
    reranker_top_n: int = 5,
    similarity_top_k: int = 20,
) -> CascadingRetriever | FusionRetriever:
    """
    Build the revamp retrieval pipeline based on RETRIEVAL_MODE.

    Args:
        agentic_index:    VectorStoreIndex for 'fintech_rag_agentic' collection.
        qa_index:         VectorStoreIndex for 'fintech_rag_qa' collection.
        agentic_nodes:    TextNodes from the agentic sidecar (for BM25).
        mode:             "cascading" | "fusion" | "agentic" | "qa_library".
                          Defaults to revamp_settings.RETRIEVAL_MODE.
        reranker_top_n:   Final nodes returned by reranker (agentic path only).
        similarity_top_k: Candidates per retriever before fusion.

    Returns:
        CascadingRetriever or FusionRetriever instance.
    """
    mode = mode or revamp_settings.RETRIEVAL_MODE

    reranker = _build_reranker(top_n=reranker_top_n)

    if mode == "cascading":
        logger.info(
            f"[revamp_pipeline] Mode=cascading "
            f"(threshold={revamp_settings.CASCADE_CONFIDENCE_THRESHOLD})"
        )
        return CascadingRetriever(
            qa_index=qa_index,
            agentic_index=agentic_index,
            agentic_nodes=agentic_nodes,
            reranker=reranker,
            threshold=revamp_settings.CASCADE_CONFIDENCE_THRESHOLD,
            reranker_top_n=reranker_top_n,
            similarity_top_k=similarity_top_k,
        )

    if mode == "fusion":
        logger.info("[revamp_pipeline] Mode=fusion (RRF k=60)")
        return FusionRetriever(
            qa_index=qa_index,
            agentic_index=agentic_index,
            agentic_nodes=agentic_nodes,
            reranker=reranker,
            top_k=reranker_top_n,
            similarity_top_k=similarity_top_k,
        )

    if mode == "agentic":
        logger.info("[revamp_pipeline] Mode=agentic only (no Q&A lookup)")
        # Wrap as a CascadingRetriever with threshold=1.1 so Q&A is never hit
        return CascadingRetriever(
            qa_index=qa_index,
            agentic_index=agentic_index,
            agentic_nodes=agentic_nodes,
            reranker=reranker,
            threshold=1.1,   # impossible threshold → always falls to agentic
            reranker_top_n=reranker_top_n,
            similarity_top_k=similarity_top_k,
        )

    if mode == "qa_library":
        logger.info("[revamp_pipeline] Mode=qa_library only (no agentic fallback)")
        # Threshold=0.0 → Q&A always wins regardless of score
        return CascadingRetriever(
            qa_index=qa_index,
            agentic_index=agentic_index,
            agentic_nodes=agentic_nodes,
            reranker=reranker,
            threshold=0.0,
            reranker_top_n=reranker_top_n,
            similarity_top_k=similarity_top_k,
        )

    raise ValueError(
        f"Unknown RETRIEVAL_MODE='{mode}'. "
        f"Must be one of: cascading, fusion, agentic, qa_library"
    )
