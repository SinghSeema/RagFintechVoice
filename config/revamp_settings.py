"""
Revamp pipeline settings — reads from .env, validates at startup.

All new keys used by the revamp RAG path live here. Legacy pipeline
settings (GROQ_API_KEY, QDRANT_URL, etc.) are read directly with
os.environ where needed, same as the legacy code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


_VALID_CHUNKING  = {"agentic", "qa_library", "both"}
_VALID_RETRIEVAL = {"agentic", "qa_library", "cascading", "fusion"}
_VALID_PIPELINE  = {"legacy", "revamp"}


@dataclass
class RevampSettings:
    # ── Top-level switch ──────────────────────────────────────────────────
    PIPELINE_MODE: str = field(default_factory=lambda: os.getenv("PIPELINE_MODE", "legacy"))

    # ── Ingestion ─────────────────────────────────────────────────────────
    CHUNKING_STRATEGY: str       = field(default_factory=lambda: os.getenv("CHUNKING_STRATEGY", "both"))

    # ── Retrieval ─────────────────────────────────────────────────────────
    RETRIEVAL_MODE: str          = field(default_factory=lambda: os.getenv("RETRIEVAL_MODE", "cascading"))
    CASCADE_CONFIDENCE_THRESHOLD: float = field(
        default_factory=lambda: float(os.getenv("CASCADE_CONFIDENCE_THRESHOLD", "0.82"))
    )

    # ── Agentic chunking ──────────────────────────────────────────────────
    AGENTIC_CHUNK_BATCH_SIZE: int    = field(default_factory=lambda: int(os.getenv("AGENTIC_CHUNK_BATCH_SIZE", "3")))
    AGENTIC_PROPOSITION_MODEL: str   = field(default_factory=lambda: os.getenv("AGENTIC_PROPOSITION_MODEL", "llama-3.1-8b-instant"))
    AGENTIC_DEDUP_THRESHOLD: float   = field(default_factory=lambda: float(os.getenv("AGENTIC_DEDUP_THRESHOLD", "0.92")))

    # ── Q&A library ───────────────────────────────────────────────────────
    QA_LIBRARY_PATH: str         = field(default_factory=lambda: os.getenv("QA_LIBRARY_PATH", "data/qa_library.json"))
    QA_GENERATION_MODEL: str     = field(default_factory=lambda: os.getenv("QA_GENERATION_MODEL", "llama-3.1-8b-instant"))
    QA_PAIRS_PER_SECTION: int    = field(default_factory=lambda: int(os.getenv("QA_PAIRS_PER_SECTION", "8")))

    # ── Qdrant collections ────────────────────────────────────────────────
    QDRANT_COLLECTION_AGENTIC: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION_AGENTIC", "fintech_rag_agentic"))
    QDRANT_COLLECTION_QA: str      = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION_QA", "fintech_rag_qa"))

    # ── Hybrid RAG: Vector + PageIndex layer (hooked via RETRIEVAL_STRATEGY) ─
    # "vector"    → existing pipeline, no change
    # "pageindex" → pure vectorless LLM TOC navigation only
    # "hybrid"    → Stage1: vector rough-filter → Stage2: PageIndex deep-dive
    #               → Stage3: Supervisor LLM cross-check synthesis
    RETRIEVAL_STRATEGY: str      = field(default_factory=lambda: os.getenv("RETRIEVAL_STRATEGY", "vector"))
    HYBRID_VECTOR_TOP_K: int     = field(default_factory=lambda: int(os.getenv("HYBRID_VECTOR_TOP_K", "3")))
    HYBRID_CONF_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("HYBRID_CONF_THRESHOLD", "0.85")))
    SUPERVISOR_MODEL: str        = field(default_factory=lambda: os.getenv("SUPERVISOR_MODEL", "llama-3.1-8b-instant"))
    SUPERVISOR_MAX_TOKENS: int   = field(default_factory=lambda: int(os.getenv("SUPERVISOR_MAX_TOKENS", "512")))
    PAGEINDEX_NAV_MODEL: str     = field(default_factory=lambda: os.getenv("PAGEINDEX_NAV_MODEL", "llama-3.1-8b-instant"))
    PAGEINDEX_DOC_ID: str        = field(default_factory=lambda: os.getenv("PAGEINDEX_DOC_ID", "rbi_kyc_master_direction"))
    SMARTMAP_DIR: str            = field(default_factory=lambda: os.getenv("SMARTMAP_DIR", "qdrant_storage"))

    # ── Groq ──────────────────────────────────────────────────────────────
    GROQ_API_KEY: str = field(default_factory=lambda: os.environ["GROQ_API_KEY"])

    def __post_init__(self) -> None:
        if self.PIPELINE_MODE not in _VALID_PIPELINE:
            raise ValueError(
                f"PIPELINE_MODE='{self.PIPELINE_MODE}' invalid. "
                f"Must be one of: {sorted(_VALID_PIPELINE)}"
            )
        if self.CHUNKING_STRATEGY not in _VALID_CHUNKING:
            raise ValueError(
                f"CHUNKING_STRATEGY='{self.CHUNKING_STRATEGY}' invalid. "
                f"Must be one of: {sorted(_VALID_CHUNKING)}"
            )
        if self.RETRIEVAL_MODE not in _VALID_RETRIEVAL:
            raise ValueError(
                f"RETRIEVAL_MODE='{self.RETRIEVAL_MODE}' invalid. "
                f"Must be one of: {sorted(_VALID_RETRIEVAL)}"
            )
        _VALID_STRATEGY = {"vector", "pageindex", "hybrid"}
        if self.RETRIEVAL_STRATEGY not in _VALID_STRATEGY:
            raise ValueError(
                f"RETRIEVAL_STRATEGY='{self.RETRIEVAL_STRATEGY}' invalid. "
                f"Must be one of: {sorted(_VALID_STRATEGY)}"
            )


# Process-wide singleton — import this everywhere instead of constructing fresh
revamp_settings = RevampSettings()
