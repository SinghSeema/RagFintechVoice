"""
Revamp ingestion runner — populates Qdrant collections for the revamp pipeline.

Usage:
    # Agentic propositions only
    python scripts/ingest_revamp.py --strategy agentic

    # Q&A library only (generate_qa_library.py must have run first)
    python scripts/ingest_revamp.py --strategy qa_library

    # Both in one pass (default)
    python scripts/ingest_revamp.py --strategy both

    # Force-recreate collections (wipes existing data)
    python scripts/ingest_revamp.py --strategy both --force-recreate

Collections created:
    fintech_rag_agentic  — proposition TextNodes (CHUNKING_STRATEGY=agentic|both)
    fintech_rag_qa       — Q&A TextNodes         (CHUNKING_STRATEGY=qa_library|both)

Legacy collection 'fintech_rag' is NEVER touched.

Sidecars written to qdrant_storage/:
    revamp_nodes_fintech_rag_agentic.json
    revamp_nodes_fintech_rag_qa.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _ingest_agentic(force_recreate: bool) -> None:
    from config.revamp_settings import revamp_settings
    from src.ingestion.parser import load_pdf_with_sections
    from src.ingestion.metadata import build_section_map_and_clean
    from src.ingestion.agentic_chunker import chunk_to_propositions
    from src.retrieval.revamp_vector_store import build_revamp_index

    pdf_path = "data/raw/rbi_kyc_master_direction.pdf"
    collection = revamp_settings.QDRANT_COLLECTION_AGENTIC

    logger.info(f"=== Agentic ingestion → '{collection}' ===")
    logger.info(f"Loading PDF: {pdf_path}")

    text_marked = load_pdf_with_sections(pdf_path)
    chapter_map, section_map, clean_text = build_section_map_and_clean(text_marked)
    logger.info(
        f"PDF loaded: {len(clean_text):,} chars | "
        f"{len(chapter_map)} chapters | {len(section_map)} sections"
    )

    logger.info("Extracting propositions (this takes ~8-12 minutes on first run)...")
    nodes = chunk_to_propositions(
        chapter_map=chapter_map,
        section_map=section_map,
        clean_text=clean_text,
    )
    logger.info(f"Extracted {len(nodes)} proposition nodes")

    # Validate sample — print first 5 propositions
    logger.info("Sample propositions (first 5):")
    for n in nodes[:5]:
        logger.info(
            f"  section={n.metadata.get('section','?')!r} "
            f"has_xref={n.metadata.get('has_xref', False)}"
        )
        logger.info(f"  text: {n.text[:120]!r}")

    xref_count = sum(1 for n in nodes if n.metadata.get("has_xref"))
    if xref_count:
        logger.warning(
            f"{xref_count}/{len(nodes)} propositions have unresolved cross-references "
            f"[XREF: resolve manually]. Review before production use."
        )

    logger.info(f"Indexing {len(nodes)} nodes into Qdrant '{collection}'...")
    build_revamp_index(nodes, collection, force_recreate=force_recreate)
    logger.info(f"=== Agentic ingestion complete ===\n")


def _ingest_qa(force_recreate: bool) -> None:
    from config.revamp_settings import revamp_settings
    from src.ingestion.qa_ingester import load_qa_library, create_qa_nodes
    from src.retrieval.revamp_vector_store import build_revamp_index

    collection = revamp_settings.QDRANT_COLLECTION_QA
    qa_path    = revamp_settings.QA_LIBRARY_PATH

    logger.info(f"=== Q&A ingestion → '{collection}' ===")

    qa_pairs = load_qa_library(qa_path)
    logger.info(f"Loaded {len(qa_pairs)} Q&A pairs from {qa_path}")

    # Show question type distribution
    from collections import Counter
    types = Counter(p.get("question_type", "unknown") for p in qa_pairs)
    logger.info(f"Question types: {dict(types)}")

    nodes = create_qa_nodes(qa_pairs)
    logger.info(f"Created {len(nodes)} Q&A nodes")

    logger.info(f"Indexing {len(nodes)} nodes into Qdrant '{collection}'...")
    build_revamp_index(nodes, collection, force_recreate=force_recreate)
    logger.info(f"=== Q&A ingestion complete ===\n")


def main() -> None:
    from config.revamp_settings import revamp_settings

    parser = argparse.ArgumentParser(description="Revamp pipeline ingestion")
    parser.add_argument(
        "--strategy",
        choices=["agentic", "qa_library", "both"],
        default=revamp_settings.CHUNKING_STRATEGY,
        help="Which collections to populate (default from CHUNKING_STRATEGY env var)",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Drop and recreate Qdrant collections (WARNING: deletes existing data)",
    )
    args = parser.parse_args()

    if args.force_recreate:
        logger.warning(
            "⚠  --force-recreate: existing revamp collections will be DELETED. "
            "Legacy 'fintech_rag' collection is unaffected."
        )

    if args.strategy in ("agentic", "both"):
        _ingest_agentic(force_recreate=args.force_recreate)

    if args.strategy in ("qa_library", "both"):
        _ingest_qa(force_recreate=args.force_recreate)

    logger.info("Ingestion complete. Collections ready for eval_revamp.py")


if __name__ == "__main__":
    main()
