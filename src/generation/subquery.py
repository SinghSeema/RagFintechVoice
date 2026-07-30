"""
Phase 2 — Sub-question decomposition for multi-hop regulatory queries.

Why
---
Complex regulatory questions often span multiple provisions or time periods, e.g.:
  "Compare simplified KYC with full KYC requirements"
  "What are the exemptions for low-risk customers vs high-risk customers?"

A direct single-shot RAG query retrieves a small window of context (top-3 chunks in
voice mode, top-5 in text mode) which may not cover both halves of a comparison.

SubQuestionQueryEngine (LlamaIndex built-in) solves this by:
  1. Sending the question to the LLM to decompose into independent sub-questions.
  2. Answering each sub-question with its own full RAG retrieval pass.
  3. Synthesising a final answer from all sub-answers.

Step 2 runs every sub-question through the full hybrid-retrieval + auto-merge +
BGE-rerank pipeline, so each sub-question gets fresh context.

Design
------
- One QueryEngineTool per source document (3 PDFs).  The tool description tells
  the LLM what each document covers so it can route sub-questions correctly.
- Each tool is backed by its own CitationQueryEngine with a FilteredRetriever
  that restricts the Qdrant query to nodes whose `source_file` metadata matches
  the document.
- `build_sub_engine()` accepts the already-loaded index / nodes / storage_context
  and the pre-built (retriever, reranker) pair — same pattern as rag_chain.query().

Usage
-----
  sub_engine = build_sub_engine(index, nodes, storage_context, retriever, reranker)
  response   = sub_engine.query("Compare simplified vs full KYC requirements")
  print(str(response))
"""

from __future__ import annotations

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.query_engine import CitationQueryEngine, SubQuestionQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.tools import QueryEngineTool, ToolMetadata


# ── Document registry ──────────────────────────────────────────────────────────

# Each entry: (source_file fragment to match, tool name, tool description)
_DOCUMENT_TOOLS: list[tuple[str, str, str]] = [
    (
        "rbi_kyc_master_direction",
        "kyc_aml",
        (
            "RBI KYC Master Direction. Covers: Know Your Customer (KYC) procedures, "
            "Anti-Money Laundering (AML) obligations, customer due diligence (CDD), "
            "enhanced due diligence (EDD), simplified due diligence for low-risk accounts, "
            "periodic KYC updates, Video-based Customer Identification Process (V-CIP), "
            "NRI KYC documents, politically exposed persons (PEPs), and KYC exemptions."
        ),
    ),
    (
        "rbi_housing_loan",
        "housing_loans",
        (
            "RBI Housing Loan Guidelines. Covers: home loan eligibility, LTV ratios, "
            "loan-to-value limits by loan size, risk weights for housing loans, "
            "priority sector classification for housing, affordable housing definitions, "
            "and NHB / co-operative housing finance rules."
        ),
    ),
    (
        "rbi_priority_sector",
        "priority_sector",
        (
            "RBI Priority Sector Lending guidelines. Covers: agriculture loans, MSME credit, "
            "export credit, education loans, social infrastructure, renewable energy, "
            "weaker section targets, priority sector lending certificates (PSLCs), "
            "and adjusted net bank credit (ANBC) calculation."
        ),
    ),
]


# ── Filtered retriever ─────────────────────────────────────────────────────────

class _FilteredRetriever(BaseRetriever):
    """
    Wraps the pre-built AutoMergingRetriever + BGE reranker but only returns
    nodes whose `source_file` metadata contains `file_fragment`.

    This lets each document tool answer only from its own document while reusing
    the shared hybrid index (no per-document Qdrant collection needed).
    """

    def __init__(
        self,
        retriever,
        reranker,
        file_fragment: str,
        reranker_top_n: int = 5,
    ) -> None:
        super().__init__()
        self._retriever = retriever
        self._reranker = reranker
        self._file_fragment = file_fragment
        self._reranker_top_n = reranker_top_n

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        raw = self._retriever.retrieve(query_bundle.query_str)
        # Filter to this document
        filtered = [
            n for n in raw
            if self._file_fragment in n.node.metadata.get("source_file", "")
        ]
        if not filtered:
            return []
        reranked = self._reranker.postprocess_nodes(filtered, query_bundle)
        return reranked[: self._reranker_top_n]


# ── Public factory ─────────────────────────────────────────────────────────────

def build_sub_engine(
    index: VectorStoreIndex,
    nodes: list,
    storage_context: StorageContext,
    retriever,
    reranker,
    verbose: bool = False,
) -> SubQuestionQueryEngine:
    """
    Build a SubQuestionQueryEngine with one tool per source document.

    Args:
        index:            Shared VectorStoreIndex (Qdrant-backed).
        nodes:            Leaf nodes (unused here but kept for API consistency).
        storage_context:  StorageContext with docstore.
        retriever:        Pre-built AutoMergingRetriever.
        reranker:         Pre-built FlagEmbeddingReranker.
        verbose:          If True, SubQuestionQueryEngine logs sub-questions.

    Returns:
        SubQuestionQueryEngine ready for .query() calls.
    """
    tools: list[QueryEngineTool] = []

    for file_fragment, tool_name, description in _DOCUMENT_TOOLS:
        filtered_retriever = _FilteredRetriever(
            retriever=retriever,
            reranker=reranker,
            file_fragment=file_fragment,
            reranker_top_n=5,
        )
        engine = CitationQueryEngine.from_args(
            index,
            retriever=filtered_retriever,
            citation_chunk_size=512,
            citation_chunk_overlap=20,
            verbose=False,
        )
        tools.append(
            QueryEngineTool(
                query_engine=engine,
                metadata=ToolMetadata(name=tool_name, description=description),
            )
        )

    return SubQuestionQueryEngine.from_defaults(
        query_engine_tools=tools,
        verbose=verbose,
        use_async=False,   # sync — runs in thread executor in the API layer
    )
