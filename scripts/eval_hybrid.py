"""
Hybrid RAG evaluation harness — Vector pipeline vs Hybrid (Vector + PageIndex) pipeline.

Runs 4 questions from golden_dataset.json through both pipelines and
produces a side-by-side comparison table showing:
  - Answer preview
  - Latency
  - Which retrieval path was used (qa / agentic / pageindex / hybrid)
  - Page/section citations (PageIndex only)

The 4 questions are chosen to cover all difficulty levels and doc sections:
  Q1  (easy)  : OVD documents for individual KYC
  Q3  (medium): V-CIP requirements
  Q7  (hard)  : PEPs special measures
  Q11 (hard)  : Customer refuses KYC — bank obligations

Usage:
    python scripts/eval_hybrid.py
    python scripts/eval_hybrid.py --modes vector hybrid
    python scripts/eval_hybrid.py --modes pageindex   # pure PageIndex only
    python scripts/eval_hybrid.py --questions 1 3     # subset

Prerequisites:
    - Revamp collections must exist (run ingest_revamp.py first)
    - SmartMap sidecar must exist (run ingest_smartmap.py first)
    - GROQ_API_KEY set in .env
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Load 4 golden questions (indices 0, 2, 6, 10 = Q1, Q3, Q7, Q11) ──────────

_GOLDEN_PATH = Path("evaluation/golden_dataset.json")
_QUESTION_INDICES = [0, 2, 6, 10]   # 0-based indices in golden_dataset.json


def _load_golden_questions(indices: list[int]) -> list[dict]:
    with open(_GOLDEN_PATH, encoding="utf-8") as f:
        dataset = json.load(f)
    return [dataset[i] for i in indices if i < len(dataset)]


# ── Pipeline loaders ──────────────────────────────────────────────────────────

def _load_revamp_stack():
    """Load Qdrant indexes, nodes, and configure LLM. Shared by all modes."""
    from config.revamp_settings import revamp_settings
    from src.retrieval.revamp_vector_store import load_revamp_index, load_node_sidecar
    from src.retrieval.hybrid_retriever import configure_llm

    configure_llm(os.environ["GROQ_API_KEY"])

    agentic_col   = revamp_settings.QDRANT_COLLECTION_AGENTIC
    qa_col        = revamp_settings.QDRANT_COLLECTION_QA
    agentic_index = load_revamp_index(agentic_col)
    agentic_nodes = load_node_sidecar(agentic_col)
    qa_index      = load_revamp_index(qa_col)
    return agentic_index, qa_index, agentic_nodes


def _run_vector(question: str, agentic_index, qa_index, agentic_nodes) -> dict:
    """Run existing cascading vector pipeline."""
    from src.generation.revamp_rag_chain import query_revamp

    t0     = time.perf_counter()
    result = query_revamp(
        question, agentic_index, qa_index, agentic_nodes, mode="text"
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    path = "qa" if (result.top_score > 0.85 and len(result.sources) == 1) else "agentic"
    return {
        "answer":     result.answer,
        "latency_ms": elapsed_ms,
        "path":       path,
        "citations":  [s.section_title for s in result.sources[:2]],
        "top_score":  result.top_score,
    }


def _run_pageindex(question: str) -> dict:
    """Run pure PageIndex (vectorless) — no Qdrant involved."""
    from src.retrieval.pageindex_retriever import PageIndexRetriever

    retriever = PageIndexRetriever()
    t0        = time.perf_counter()
    results   = retriever.retrieve(question)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    r = results[0]
    return {
        "answer":     r.content[:800],   # raw section text as answer
        "latency_ms": elapsed_ms,
        "path":       "pageindex",
        "citations":  r.citations,
        "top_score":  r.confidence,
    }


def _run_hybrid(question: str, agentic_index, qa_index, agentic_nodes) -> dict:
    """Run full 3-stage hybrid pipeline."""
    from src.retrieval.hybrid_rag_retriever import HybridRAGRetriever

    retriever  = HybridRAGRetriever(agentic_index, qa_index, agentic_nodes)
    t0         = time.perf_counter()
    result     = retriever.query(question, mode="text")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Identify PageIndex source from sources list
    pi_citations = [
        s.section_title for s in result.sources
        if s.chapter == "PageIndex"
    ]
    vec_citations = [
        s.section_title for s in result.sources
        if s.chapter != "PageIndex"
    ][:2]

    return {
        "answer":     result.answer,
        "latency_ms": elapsed_ms,
        "path":       "hybrid",
        "citations":  pi_citations + vec_citations,
        "top_score":  result.top_score,
    }


# ── Output formatting ─────────────────────────────────────────────────────────

def _truncate(text: str, width: int) -> str:
    text = text.replace("\n", " ").strip()
    return (text[:width - 1] + "…") if len(text) > width else text


def _print_results(rows: list[dict]) -> None:
    sep = "─" * 130
    print()
    print(sep)
    print(f" {'#':>3}  {'Mode':<10}  {'Path':<10}  {'ms':>7}  {'Answer (first 90 chars)':<90}  {'Score':>6}")
    print(sep)

    prev_q_idx = None
    for row in rows:
        q_idx = row["q_idx"]
        if q_idx != prev_q_idx:
            print()
            q_short = _truncate(row["question"], 110)
            print(f"  Q{q_idx+1}  {q_short}")
            gt = row.get("ground_truth", "")
            print(f"      Ground truth: {_truncate(gt, 105)}")
            prev_q_idx = q_idx

        mode_str  = row["mode"].ljust(10)
        path_str  = row["path"].ljust(10)
        ms_str    = f"{row['latency_ms']:.0f}ms".rjust(7)
        ans_str   = _truncate(row["answer"], 90).ljust(90)
        score_str = f"{row['top_score']:.3f}".rjust(6)
        print(f"       {mode_str}  {path_str}  {ms_str}  {ans_str}  {score_str}")

        if row["citations"]:
            cit_str = " | ".join(row["citations"][:2])
            print(f"       {'citations:':<12}  {_truncate(cit_str, 100)}")

    print()
    print(sep)
    print()


def _print_summary(rows: list[dict]) -> None:
    """Print per-mode average latency."""
    modes: dict[str, list[float]] = {}
    for row in rows:
        modes.setdefault(row["mode"], []).append(row["latency_ms"])

    print("Latency summary:")
    for mode, latencies in modes.items():
        avg = sum(latencies) / len(latencies)
        print(f"  {mode:<12}  avg {avg:.0f}ms over {len(latencies)} queries")
    print()
    print("Manual scoring guide: 0 = wrong/irrelevant  1 = partially correct  2 = fully correct")
    print("Hybrid target: PageIndex citations present + answer >= vector quality\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid RAG eval: vector vs pageindex vs hybrid")
    parser.add_argument(
        "--modes", nargs="+",
        choices=["vector", "pageindex", "hybrid"],
        default=["vector", "hybrid"],
    )
    parser.add_argument(
        "--questions", nargs="+", type=int,
        metavar="N",
        help="1-based question numbers from QUESTION_INDICES list (1-4)",
    )
    args = parser.parse_args()

    # ── Load questions ────────────────────────────────────────────────────
    indices = _QUESTION_INDICES
    if args.questions:
        indices = [_QUESTION_INDICES[i - 1] for i in args.questions if 1 <= i <= len(_QUESTION_INDICES)]

    golden = _load_golden_questions(indices)
    if not golden:
        print("ERROR: Could not load golden questions. Check evaluation/golden_dataset.json")
        sys.exit(1)

    print(f"\nEvaluating {len(golden)} questions × {len(args.modes)} modes "
          f"= {len(golden)*len(args.modes)} total queries\n")
    for i, q in enumerate(golden, 1):
        print(f"  Q{i}: [{q['difficulty']}] {q['question'][:80]}")
    print()

    # ── Load shared pipeline stack ────────────────────────────────────────
    needs_revamp = any(m in args.modes for m in ("vector", "hybrid"))
    revamp_stack = None
    if needs_revamp:
        print("Loading revamp pipeline (Qdrant + BGE-M3 + reranker)...")
        try:
            revamp_stack = _load_revamp_stack()
            print("  Revamp stack ready\n")
        except Exception as e:
            print(f"  ERROR loading revamp stack: {e}")
            if "vector" in args.modes or "hybrid" in args.modes:
                print("  Cannot run vector or hybrid modes without revamp stack.")
                args.modes = [m for m in args.modes if m == "pageindex"]
                if not args.modes:
                    sys.exit(1)

    # ── Run queries ───────────────────────────────────────────────────────
    rows: list[dict] = []

    for q_num, item in enumerate(golden):
        question   = item["question"]
        ground_truth = item["ground_truth"]
        q_idx      = indices[q_num]

        print(f"[{q_num+1}/{len(golden)}] Q{q_idx+1}: {question[:70]}")

        for mode in args.modes:
            t0 = time.perf_counter()
            try:
                if mode == "vector" and revamp_stack:
                    ai, qi, an = revamp_stack
                    row_data = _run_vector(question, ai, qi, an)
                elif mode == "pageindex":
                    row_data = _run_pageindex(question)
                elif mode == "hybrid" and revamp_stack:
                    ai, qi, an = revamp_stack
                    row_data = _run_hybrid(question, ai, qi, an)
                else:
                    continue

                print(f"  [{mode}] {row_data['latency_ms']:.0f}ms  path={row_data['path']}  "
                      f"ans={_truncate(row_data['answer'], 55)}")

                rows.append({
                    "q_idx":       q_idx,
                    "question":    question,
                    "ground_truth": ground_truth,
                    "mode":        mode,
                    **row_data,
                })

            except Exception as exc:
                print(f"  [{mode}] ERROR: {exc}")
                logger.exception(f"Query failed mode={mode}")
                rows.append({
                    "q_idx":       q_idx,
                    "question":    question,
                    "ground_truth": ground_truth,
                    "mode":        mode,
                    "answer":      f"ERROR: {exc}",
                    "latency_ms":  (time.perf_counter() - t0) * 1000,
                    "path":        "error",
                    "citations":   [],
                    "top_score":   0.0,
                })

        print()

    # ── Print results ─────────────────────────────────────────────────────
    _print_results(rows)
    _print_summary(rows)

    # ── Save JSON results ─────────────────────────────────────────────────
    out_path = Path("evaluation/hybrid_eval_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Full results saved → {out_path}")


if __name__ == "__main__":
    main()
