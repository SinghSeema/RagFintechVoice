"""
Q&A library ingester — loads qa_library.json into a Qdrant collection.

Each Q&A pair becomes one TextNode where:
  - node.text   = the question  (this is what gets embedded for retrieval)
  - node.metadata["answer"] = the full pre-written answer

At query time, cosine similarity between user query and stored question
text drives retrieval — question-to-question matching is far more precise
than query-to-paragraph matching.

Entry points:
  load_qa_library(path)         -> list[dict]
  create_qa_nodes(qa_pairs)     -> list[TextNode]
"""

from __future__ import annotations

import json
import logging
import uuid

from llama_index.core.schema import TextNode

from config.revamp_settings import revamp_settings

logger = logging.getLogger(__name__)


def load_qa_library(path: str | None = None) -> list[dict]:
    """
    Load qa_library.json from disk.

    Args:
        path: Path to json file. Defaults to revamp_settings.QA_LIBRARY_PATH.

    Returns:
        List of Q&A pair dicts with keys: question, answer, section, question_type.

    Raises:
        FileNotFoundError: If the file does not exist (run generate_qa_library.py first).
        ValueError:        If the JSON structure is invalid.
    """
    path = path or revamp_settings.QA_LIBRARY_PATH
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Q&A library not found at '{path}'.\n"
            f"Run: python scripts/generate_qa_library.py "
            f"--input data/raw/rbi_kyc_master_direction.pdf --output {path}"
        )

    if isinstance(data, dict) and "qa_pairs" in data:
        pairs = data["qa_pairs"]
    elif isinstance(data, list):
        pairs = data
    else:
        raise ValueError(
            f"Invalid qa_library.json structure. Expected a list or "
            f"a dict with 'qa_pairs' key. Got: {type(data)}"
        )

    if not pairs:
        raise ValueError(f"Q&A library at '{path}' is empty.")

    logger.info(f"[qa_ingester] Loaded {len(pairs)} Q&A pairs from '{path}'")
    return pairs


def create_qa_nodes(qa_pairs: list[dict]) -> list[TextNode]:
    """
    Convert Q&A pairs into TextNodes ready for Qdrant ingestion.

    The QUESTION text is stored as node.text so that BGE-M3 embeds the
    question for retrieval. The full answer is stored in metadata and
    returned to the caller at query time.

    Args:
        qa_pairs: List of dicts with keys: question, answer, section, question_type.

    Returns:
        List of TextNode objects — one per Q&A pair.
    """
    nodes: list[TextNode] = []
    skipped = 0

    for idx, pair in enumerate(qa_pairs):
        question      = (pair.get("question") or "").strip()
        answer        = (pair.get("answer")   or "").strip()
        section       = (pair.get("section")  or "").strip()
        question_type = (pair.get("question_type") or "unknown").strip()

        if not question or not answer:
            logger.warning(f"[qa_ingester] Skipping pair idx={idx} — missing question or answer")
            skipped += 1
            continue

        node = TextNode(
            node_id=str(uuid.uuid4()),
            text=question,   # embed the question for retrieval
            metadata={
                "source":        "rbi_kyc_master_direction",
                "strategy":      "qa_library",
                "section":       section,
                "question":      question,
                "answer":        answer,
                "question_type": question_type,
                "char_count":    len(answer),
            },
        )
        # Keep only 'source' and 'section' in the embedding context —
        # the full answer text must NOT influence the question embedding.
        node.excluded_embed_metadata_keys = [
            "strategy", "question", "answer", "question_type", "char_count",
        ]
        nodes.append(node)

    if skipped:
        logger.warning(f"[qa_ingester] Skipped {skipped} invalid pairs")

    logger.info(f"[qa_ingester] Created {len(nodes)} Q&A TextNodes")
    return nodes
