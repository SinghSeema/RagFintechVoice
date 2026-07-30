"""
Revamp pipeline evaluation harness.

Runs 10 representative compliance queries against three configurations:
  legacy      — current HierarchicalNodeParser + AutoMergingRetriever pipeline
  agentic     — revamp pipeline (agentic-only mode)
  cascading   — revamp pipeline (Q&A first, agentic fallback)

Prints a comparison table with top-1 answer preview, latency, and a
blank relevance score column for manual annotation.

Usage:
    python scripts/eval_revamp.py

    # Run only specific modes
    python scripts/eval_revamp.py --modes legacy cascading

    # Run a quick sanity check (first 3 queries only)
    python scripts/eval_revamp.py --quick

Prerequisites:
    - Legacy 'fintech_rag' Qdrant collection must exist (legacy pipeline ingested)
    - 'fintech_rag_agentic' and/or 'fintech_rag_qa' must exist (ingest_revamp.py run)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING)   # suppress INFO noise during eval
logger = logging.getLogger(__name__)


# ── Eval query set ────────────────────────────────────────────────────────────

EVAL_QUERIES = [
    # 1 — Documented failure case: must improve over legacy
    "What documents are required for KYC of an individual customer?",
    # 2 — Section-number exact match: BM25 strength test
    "FATCA CRS reporting requirements section 57",
    # 3 — Cross-reference heavy: tests proposition quality
    "What is the periodic KYC updation period for low-risk customers?",
    # 4 — Threshold extraction: tests numerical precision
    "What is the threshold for enhanced due diligence?",
    # 5 — Multi-obligation: tests proposition recall breadth
    "What are the obligations for Politically Exposed Persons?",
    # 6 — Exception/edge: tests Q&A coverage
    "What are the simplified KYC norms for small accounts?",
    # 7 — Process query: tests step coverage
    "What is the step-by-step procedure for V-CIP?",
    # 8 — NRI-specific: tests metadata section routing
    "What documents does an NRI need for KYC?",
    # 9 — Off-topic: should be filtered before RAG, but tests score floor
    "How can I improve my daily productivity routine?",
    # 10 — Novel edge: tests agentic fallback from Q&A
    "Does the KYC direction apply to OCI cardholders?",
]


# ── Legacy pipeline loader ────────────────────────────────────────────────────

def _load_legacy():
    from src.ingestion.parser import load_pdf_with_sections
    from src.ingestion.metadata import build_section_map_and_clean
    from src.ingestion.chunker import chunk_document
    from src.retrieval.vector_store import load_index
    from src.retrieval.hybrid_retriever import configure_llm
    from src.retrieval.pipeline import build_pipeline

    configure_llm(os.environ["GROQ_API_KEY"])

    pdf_path = "data/raw/rbi_kyc_master_direction.pdf"
    text_marked = load_pdf_with_sections(pdf_path)
    chapter_map, section_map, clean_text = build_section_map_and_clean(text_marked)
    leaf_nodes, storage_context = chunk_document(
        text=clean_text, filename=pdf_path,
        section_map=section_map, chapter_map=chapter_map, clean_text=clean_text,
    )
    index = load_index(storage_context)
    retriever, reranker = build_pipeline(index, leaf_nodes, storage_context, reranker_top_n=5)
    return index, leaf_nodes, storage_context, retriever, reranker


def _run_legacy(question: str, index, nodes, storage_context, retriever, reranker):
    from src.generation.rag_chain import query
    return query(
        question, index, nodes, storage_context,
        mode="text", retriever=retriever, reranker=reranker,
    )


# ── Revamp pipeline loader ────────────────────────────────────────────────────

def _load_revamp(mode: str):
    from config.revamp_settings import revamp_settings
    from src.retrieval.revamp_vector_store import load_revamp_index, load_node_sidecar
    from src.retrieval.hybrid_retriever import configure_llm
    from src.retrieval.revamp_pipeline import build_revamp_pipeline

    configure_llm(os.environ["GROQ_API_KEY"])

    agentic_col = revamp_settings.QDRANT_COLLECTION_AGENTIC
    qa_col      = revamp_settings.QDRANT_COLLECTION_QA

    agentic_index = load_revamp_index(agentic_col)
    agentic_nodes = load_node_sidecar(agentic_col)
    qa_index      = load_revamp_index(qa_col)

    retriever = build_revamp_pipeline(
        agentic_index=agentic_index,
        qa_index=qa_index,
        agentic_nodes=agentic_nodes,
        mode=mode,
    )
    return agentic_index, qa_index, agentic_nodes, retriever


def _run_revamp(question: str, agentic_index, qa_index, agentic_nodes, retriever):
    from src.generation.revamp_rag_chain import query_revamp
    return query_revamp(
        question, agentic_index, qa_index, agentic_nodes,
        mode="text", retriever=retriever,
    )


# ── Table formatting ──────────────────────────────────────────────────────────

def _preview(text: str, width: int = 80) -> str:
    text = text.replace("\n", " ").strip()
    return text[:width] + "…" if len(text) > width else text


def _print_table(rows: list[dict]) -> None:
    col_w = {"#": 3, "Query": 45, "Mode": 11, "Path": 8, "Latency": 9, "Answer preview": 82, "Score": 7}
    sep  = "─" * (sum(col_w.values()) + len(col_w) * 3 + 1)
    hdr  = " | ".join(k.ljust(col_w[k]) for k in col_w)

    print()
    print(sep)
    print(f" {hdr}")
    print(sep)

    prev_q = None
    for row in rows:
        qnum    = str(row["qnum"]).ljust(col_w["#"])
        q_short = (row["query"][:43] + "…" if len(row["query"]) > 43 else row["query"]).ljust(col_w["Query"])
        mode    = row["mode"].ljust(col_w["Mode"])
        path    = row.get("path", "—").ljust(col_w["Path"])
        latency = f"{row['latency_ms']:.0f}ms".ljust(col_w["Latency"])
        preview = _preview(row["answer"], col_w["Answer preview"]).ljust(col_w["Answer preview"])
        score   = "[   ]".ljust(col_w["Score"])

        # Blank query column for 2nd and 3rd row of same question
        if row["query"] == prev_q:
            q_short = "".ljust(col_w["Query"])
            qnum    = "".ljust(col_w["#"])
        else:
            prev_q = row["query"]
            print(sep)

        print(f" {qnum} | {q_short} | {mode} | {path} | {latency} | {preview} | {score}")

    print(sep)
    print("\nFill Score column: 0=irrelevant, 1=partial, 2=correct")
    print("Production gate: revamp ≥ legacy on ≥ 7 of 10 queries, query #1 must improve.\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Eval revamp vs legacy pipeline")
    parser.add_argument(
        "--modes", nargs="+",
        choices=["legacy", "agentic", "cascading"],
        default=["legacy", "agentic", "cascading"],
    )
    parser.add_argument("--quick", action="store_true", help="First 3 queries only")
    parser.add_argument(
        "--queries", nargs="+", type=int, metavar="N",
        help="1-based query indices to run (e.g. --queries 1 2 6 7)"
    )
    args = parser.parse_args()

    if args.queries:
        queries = [EVAL_QUERIES[i - 1] for i in args.queries if 1 <= i <= len(EVAL_QUERIES)]
    elif args.quick:
        queries = EVAL_QUERIES[:3]
    else:
        queries = EVAL_QUERIES

    # ── Load pipelines ────────────────────────────────────────────────────
    pipelines = {}

    if "legacy" in args.modes:
        print("Loading legacy pipeline...")
        try:
            pipelines["legacy"] = _load_legacy()
            print("  Legacy pipeline ready")
        except Exception as e:
            print(f"  WARN: Could not load legacy pipeline: {e}")

    if "agentic" in args.modes or "cascading" in args.modes:
        for rev_mode in ("agentic", "cascading"):
            if rev_mode in args.modes:
                print(f"Loading revamp pipeline (mode={rev_mode})...")
                try:
                    pipelines[rev_mode] = _load_revamp(rev_mode)
                    print(f"  Revamp/{rev_mode} pipeline ready")
                except Exception as e:
                    print(f"  WARN: Could not load revamp/{rev_mode} pipeline: {e}")

    print()

    # ── Run queries ───────────────────────────────────────────────────────
    query_indices = args.queries if args.queries else list(range(1, len(queries) + 1))

    rows: list[dict] = []
    for run_idx, (qnum, question) in enumerate(zip(query_indices, queries), 1):
        print(f"[{run_idx}/{len(queries)}] Q{qnum}: {question[:65]}")

        for mode_name, pipeline in pipelines.items():
            t0 = time.perf_counter()
            path = "—"

            try:
                if mode_name == "legacy":
                    index, nodes, sc, retriever, reranker = pipeline
                    result = _run_legacy(question, index, nodes, sc, retriever, reranker)
                else:
                    agentic_index, qa_index, agentic_nodes, retriever = pipeline
                    result = _run_revamp(question, agentic_index, qa_index, agentic_nodes, retriever)
                    # Detect Q&A path from source metadata
                    if result.sources and result.sources[0].section_title:
                        first_src = result.sources[0]
                        strategy = getattr(first_src, "content", "")
                        path = "qa" if len(result.sources) == 1 and result.top_score > 0.7 else "agentic"
                answer = result.answer
            except Exception as e:
                answer = f"ERROR: {e}"
                logger.exception(f"Query failed for mode={mode_name}")

            latency_ms = (time.perf_counter() - t0) * 1000
            print(f"  [{mode_name}] {latency_ms:.0f}ms — {_preview(answer, 60)}")

            rows.append({
                "qnum":       qnum,
                "query":      question,
                "mode":       mode_name,
                "path":       path,
                "latency_ms": latency_ms,
                "answer":     answer,
            })

    # ── Print comparison table ────────────────────────────────────────────
    _print_table(rows)


if __name__ == "__main__":
    main()
