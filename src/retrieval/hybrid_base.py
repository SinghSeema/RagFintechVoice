"""
Base types for the Hybrid RAG layer (Vector + PageIndex).

HybridRetrievalResult  — structured result returned by all three strategies
HybridBaseRetriever    — abstract interface; all strategies implement retrieve()

Naming:
  "hybrid_base" avoids collision with llama_index.core.retrievers.BaseRetriever
  which is already imported throughout the codebase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class HybridRetrievalResult:
    """
    Unified result container for all three retrieval strategies.

    Fields:
        doc_id       — source document identifier (PDF stem name)
        content      — retrieved passage text
        page_numbers — list of 1-based page numbers covered by this result
        confidence   — retrieval confidence [0.0, 1.0]
        source       — which retriever produced this: 'vector' | 'pageindex' | 'hybrid'
        citations    — human-readable citation strings, e.g.
                       ['rbi_kyc_master_direction | Pages 12-15 | Section 16']
        section_id   — SmartMap section_id if from PageIndex, else empty string
        metadata     — pass-through for any extra fields
    """
    doc_id:       str
    content:      str
    page_numbers: List[int]
    confidence:   float
    source:       str               # 'vector' | 'pageindex' | 'hybrid'
    citations:    List[str]
    section_id:   str               = ""
    metadata:     dict              = field(default_factory=dict)


class HybridBaseRetriever(ABC):
    """
    Abstract retriever for the Hybrid RAG layer.

    All three strategies (VectorStrategy, PageIndexStrategy, HybridStrategy)
    implement this contract so the eval harness and strategy factory
    call a single retrieve() method regardless of active mode.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[HybridRetrievalResult]:
        """
        Retrieve relevant content for the given query.

        Args:
            query: Natural language question.
            top_k: Maximum number of results to return.

        Returns:
            List of HybridRetrievalResult ordered by descending confidence.
        """
        ...
