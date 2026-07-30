"""
Task 10 — Evaluation harness: RAGAS + DeepEval + golden dataset.

ADR-005 decisions implemented here:

  Metrics
  -------
  RAGAS:    faithfulness, answer_relevancy, context_recall, context_precision
  DeepEval: faithfulness, answer_relevancy, hallucination

  Judge LLM
  ---------
  Groq Mixtral-8x7B via OpenAI-compatible endpoint.
  CAUTION (ADR-005): generation LLM is llama-3.3-70b-versatile. The judge
  MUST be a different model to avoid self-evaluation bias.

  Golden dataset
  --------------
  20 questions, 5 per domain (kyc_aml, regulatory_qa, customer_support,
  loan_credit). All grounded in rbi_kyc_master_direction.pdf (public RBI doc).
  Format per entry: question, ground_truth, reference_doc, reference_section,
  domain, difficulty.

  Regression alerting
  -------------------
  Metrics are persisted to evaluation/metrics_store.json after every run.
  If any metric drops more than ALERT_THRESHOLD (0.05) vs the previous run,
  a REGRESSION alert is printed to stdout and the process exits non-zero.

Usage
-----
  # Full run (loads pipeline from disk):
  python -m evaluation.eval_runner

  # Custom dataset path:
  python -m evaluation.eval_runner --dataset evaluation/golden_dataset.json

  # Skip DeepEval (faster; RAGAS only):
  python -m evaluation.eval_runner --skip-deepeval

  # Dry-run (generates answers, skips LLM-judge eval):
  python -m evaluation.eval_runner --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────

EVAL_DIR = Path(__file__).parent
DATASET_PATH = EVAL_DIR / "golden_dataset.json"
METRICS_STORE = EVAL_DIR / "metrics_store.json"
ALERT_THRESHOLD = 0.05

# Groq Mixtral model used as judge (must differ from generation LLM)
JUDGE_MODEL = "llama-3.1-8b-instant"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


# ── Golden dataset ─────────────────────────────────────────────────────────────

def load_golden_dataset(path: Path = DATASET_PATH) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} questions across "
          f"{len({d['domain'] for d in data})} domains")
    return data


# ── RAG pipeline ───────────────────────────────────────────────────────────────

def load_pipeline():
    """Load the full RAG pipeline from persisted storage (fast — no re-chunking)."""
    from src.ingestion.chunker import chunk_document
    from src.ingestion.metadata import build_section_map_and_clean
    from src.ingestion.parser import load_pdf_with_sections
    from src.retrieval.hybrid_retriever import configure_llm
    from src.retrieval.vector_store import load_index

    # Remove stale Qdrant lock left by a previously interrupted process.
    # Qdrant local mode is single-process; a Ctrl+C or crash leaves .lock behind.
    qdrant_lock = Path("qdrant_storage/.lock")
    if qdrant_lock.exists():
        qdrant_lock.unlink()
        print("  Removed stale qdrant_storage/.lock")

    groq_api_key = os.environ["GROQ_API_KEY"]
    configure_llm(groq_api_key)

    pdf_path = "data/raw/rbi_kyc_master_direction.pdf"
    print(f"  Parsing {pdf_path} ...")
    text_marked = load_pdf_with_sections(pdf_path)
    chapter_map, section_map, clean_text = build_section_map_and_clean(text_marked)
    leaf_nodes, storage_context = chunk_document(
        text=clean_text,
        filename=pdf_path,
        section_map=section_map,
        chapter_map=chapter_map,
        clean_text=clean_text,
    )
    print(f"  {len(leaf_nodes)} leaf nodes loaded from storage")

    try:
        index = load_index(storage_context)
        print("  Qdrant index loaded")
    except ValueError:
        from src.retrieval.vector_store import build_index
        print("  Collection missing — building Qdrant index (one-time embed) ...")
        index = build_index(leaf_nodes, storage_context)
        print("  Qdrant index built")

    return index, leaf_nodes, storage_context, groq_api_key


# ── Query generation ───────────────────────────────────────────────────────────

def run_queries(
    dataset: list[dict],
    index,
    nodes: list,
    storage_context,
) -> list[dict]:
    """
    Run each golden question through the full RAG pipeline and collect:
    - answer: generated text
    - contexts: list of retrieved snippet strings (used by RAGAS/DeepEval)
    - elapsed_ms: end-to-end latency

    The retrieval pipeline (AutoMergingRetriever + FlagEmbeddingReranker) is
    built ONCE here and reused for every question.  Building it per-query would
    re-download / re-load the ~278 MB BGE cross-encoder model 20 times.
    """
    from src.generation.rag_chain import query
    from src.retrieval.pipeline import build_pipeline

    print("  Building retrieval pipeline (loads BGE reranker once) ...")
    retriever, reranker = build_pipeline(
        index,
        nodes,
        storage_context,
        similarity_top_k=10,
        reranker_top_n=5,   # text mode
    )
    print("  Pipeline ready\n")

    results: list[dict] = []
    for i, item in enumerate(dataset, 1):
        q_short = item["question"][:55] + ("..." if len(item["question"]) > 55 else "")
        print(f"  [{i:02d}/{len(dataset)}] {q_short}")
        t0 = time.perf_counter()
        result = query(
            item["question"],
            index=index,
            nodes=nodes,
            storage_context=storage_context,
            mode="text",
            retriever=retriever,
            reranker=reranker,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000)

        # Collect full chunk text for context (Source.content is the full text;
        # Source.snippet is only 120 chars and would cause artificially low
        # context_recall / faithfulness scores in RAGAS and DeepEval)
        contexts = [s.content for s in result.sources]

        results.append({
            **item,
            "answer": result.answer,
            "contexts": contexts,
            "num_sources": result.num_sources,
            "elapsed_ms": elapsed_ms,
        })
        print(f"         → {elapsed_ms} ms | sources={result.num_sources}")

    return results


# ── RAGAS evaluation ───────────────────────────────────────────────────────────

def run_ragas(results: list[dict], groq_api_key: str) -> dict[str, float]:
    """
    RAGAS evaluation using the configured JUDGE_MODEL.

    Metrics:
      faithfulness      — are answer claims grounded in retrieved contexts?
      answer_relevancy  — does the answer address the question?
      context_recall    — did retrieval surface enough info to answer?
      context_precision — are the retrieved contexts actually relevant?

    RAGAS 0.4.x compatibility note
    --------------------------------
    ragas.metrics.collections (the "new" API) is NOT accepted by ragas.evaluate()
    in 0.4.x — evaluate() checks isinstance(m, Metric) which only the OLD
    instance-based metrics satisfy.  We use the old API and suppress the
    deprecation warnings that come with it.  Both APIs are deprecated in some
    direction; the old one still works, the new one doesn't yet.
    """
    import warnings
    from openai import OpenAI
    from datasets import Dataset
    from ragas import evaluate
    from ragas.llms import llm_factory
    from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings
    from ragas.run_config import RunConfig

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (     # noqa: PLC0415
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )

    # llm_factory requires openai.OpenAI — use Groq's OpenAI-compatible endpoint
    openai_client = OpenAI(api_key=groq_api_key, base_url=GROQ_BASE_URL)
    judge_llm = llm_factory(JUDGE_MODEL, client=openai_client)

    # BGE-M3 local embeddings — no OpenAI needed
    judge_emb = RagasHFEmbeddings(model="BAAI/bge-m3")

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    ragas_dataset = Dataset.from_list([
        {
            "question":     r["question"],
            "answer":       r["answer"],
            "contexts":     r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in results
    ])

    # max_workers=1 — Groq's free tier is 6000 TPM; parallel jobs exceed it fast.
    run_config = RunConfig(max_workers=1, max_retries=3, timeout=60)

    scores = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=run_config,
    )

    def _agg(key: str) -> float:
        val = scores[key]
        if isinstance(val, list):
            valid = [v for v in val if v is not None]
            return round(sum(valid) / len(valid), 4) if valid else 0.0
        return round(float(val), 4)

    return {
        "ragas_faithfulness":      _agg("faithfulness"),
        "ragas_answer_relevancy":  _agg("answer_relevancy"),
        "ragas_context_recall":    _agg("context_recall"),
        "ragas_context_precision": _agg("context_precision"),
    }


# ── DeepEval evaluation ────────────────────────────────────────────────────────

def _build_deepeval_judge(api_key: str):
    """
    Build a DeepEval-compatible judge backed by Groq's OpenAI-compatible endpoint.

    Single DeepEvalBaseLLM subclass — avoids multiple-inheritance MRO issues that
    cause _client to be missing when DeepEvalBaseLLM.__init__ runs after the mixin.
    JSON mode is requested when DeepEval passes a schema for structured extraction.
    """
    from openai import OpenAI
    from deepeval.models import DeepEvalBaseLLM

    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    class _Judge(DeepEvalBaseLLM):
        def load_model(self):
            return client

        def _call(self, prompt: str, json_mode: bool = False) -> str:
            kwargs: dict = dict(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2048,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""

        def generate(self, prompt: str, schema=None):
            content = self._call(prompt, json_mode=(schema is not None))
            if schema is None:
                return content
            try:
                import json as _json
                return schema(**_json.loads(content))
            except Exception:
                return content

        async def a_generate(self, prompt: str, schema=None):
            return self.generate(prompt, schema)

        def get_model_name(self) -> str:
            return f"Groq/{JUDGE_MODEL}"

    return _Judge()


def run_deepeval(results: list[dict], groq_api_key: str) -> dict[str, float]:
    """
    DeepEval evaluation using Mixtral-8x7B as judge.

    Metrics:
      faithfulness      — is every answer claim supported by the retrieved context?
      answer_relevancy  — does the answer stay on-topic?
      hallucination     — does the answer contradict the retrieved context?
    """
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        HallucinationMetric,
    )

    judge = _build_deepeval_judge(groq_api_key)

    faith_metric = FaithfulnessMetric(model=judge, threshold=0.7, verbose_mode=False)
    relev_metric = AnswerRelevancyMetric(model=judge, threshold=0.7, verbose_mode=False)
    hallu_metric = HallucinationMetric(model=judge, threshold=0.5, verbose_mode=False)

    faith_scores, relev_scores, hallu_scores = [], [], []

    for i, r in enumerate(results, 1):
        tc = LLMTestCase(
            input=r["question"],
            actual_output=r["answer"],
            expected_output=r["ground_truth"],
            retrieval_context=r["contexts"],  # for faithfulness + answer_relevancy
            context=r["contexts"],            # for hallucination
        )
        faith_metric.measure(tc)
        relev_metric.measure(tc)
        hallu_metric.measure(tc)

        faith_scores.append(faith_metric.score)
        relev_scores.append(relev_metric.score)
        hallu_scores.append(hallu_metric.score)

        print(f"  [{i:02d}/{len(results)}] faith={faith_metric.score:.3f} "
              f"relev={relev_metric.score:.3f} hallu={hallu_metric.score:.3f}")

    def _mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    return {
        "deepeval_faithfulness":    _mean(faith_scores),
        "deepeval_answer_relevancy": _mean(relev_scores),
        "deepeval_hallucination":   _mean(hallu_scores),
    }


# ── Regression detection ───────────────────────────────────────────────────────

def check_regression(
    current: dict[str, float],
    store_path: Path = METRICS_STORE,
    threshold: float = ALERT_THRESHOLD,
) -> list[str]:
    """Compare current metrics against the most recent stored run."""
    if not store_path.exists():
        return []
    with open(store_path) as f:
        history = json.load(f)
    if not history:
        return []

    prev = history[-1]["metrics"]
    alerts = []
    for key, val in current.items():
        prev_val = prev.get(key)
        if prev_val is None:
            continue
        drop = prev_val - val
        if drop > threshold:
            alerts.append(
                f"REGRESSION  {key:<35}  {prev_val:.4f} → {val:.4f}  "
                f"(Δ = -{drop:.4f}, threshold = {threshold})"
            )
    return alerts


def save_metrics(metrics: dict[str, float], store_path: Path = METRICS_STORE) -> None:
    history: list[dict] = []
    if store_path.exists():
        with open(store_path) as f:
            history = json.load(f)
    history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    })
    with open(store_path, "w") as f:
        json.dump(history, f, indent=2)


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(
    ragas_scores: dict[str, float],
    deepeval_scores: dict[str, float],
    results: list[dict],
    alerts: list[str],
) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print("EVALUATION REPORT")
    print(sep)

    print("\nRAPGAS Metrics  (judge: Groq Mixtral-8x7B)")
    for k, v in ragas_scores.items():
        bar = "OK" if v >= 0.7 else "LOW"
        label = k.replace("ragas_", "")
        print(f"  {label:<28}  {v:.4f}  [{bar}]")

    print("\nDeepEval Metrics  (judge: Groq Mixtral-8x7B)")
    for k, v in deepeval_scores.items():
        bar = "OK" if v >= 0.7 else "LOW"
        label = k.replace("deepeval_", "")
        print(f"  {label:<28}  {v:.4f}  [{bar}]")

    latencies = [r["elapsed_ms"] for r in results]
    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]
    p95 = latencies_sorted[min(int(len(latencies_sorted) * 0.95), len(latencies_sorted) - 1)]
    print(f"\nLatency over {len(results)} questions")
    print(f"  avg  {sum(latencies) / len(latencies):.0f} ms")
    print(f"  p50  {p50} ms")
    print(f"  p95  {p95} ms")

    print(f"\nPer-domain summary")
    domains: dict[str, list[dict]] = {}
    for r in results:
        domains.setdefault(r["domain"], []).append(r)
    for domain, items in sorted(domains.items()):
        avg_lat = sum(i["elapsed_ms"] for i in items) / len(items)
        print(f"  {domain:<22}  {len(items)} questions  avg {avg_lat:.0f} ms")

    if alerts:
        print(f"\n{'!' * 65}")
        print("REGRESSION ALERTS")
        for alert in alerts:
            print(f"  {alert}")
        print(f"{'!' * 65}")
    else:
        print("\nNo regressions detected (metrics stable vs previous run).")

    print()


# ── Entry-point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG evaluation harness — Task 10")
    p.add_argument("--dataset", default=str(DATASET_PATH),
                   help="Path to golden dataset JSON")
    p.add_argument("--skip-deepeval", action="store_true",
                   help="Skip DeepEval (faster; RAGAS metrics only)")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate answers but skip all LLM-judge evaluation")
    p.add_argument("--alert-threshold", type=float, default=ALERT_THRESHOLD,
                   help="Metric drop that triggers a REGRESSION alert")
    p.add_argument("--limit", type=int, default=None,
                   help="Only evaluate the first N questions (e.g. --limit 5 for a quick smoke-check)")
    return p.parse_args()


def main() -> None:
    load_dotenv()

    args = parse_args()

    if "GROQ_API_KEY" not in os.environ:
        print("ERROR: GROQ_API_KEY not set in environment / .env", file=sys.stderr)
        sys.exit(1)

    # ── 1. Load dataset ──────────────────────────────────────────────────────
    print("\nStep 1/5 — Loading golden dataset ...")
    dataset = load_golden_dataset(Path(args.dataset))
    if args.limit:
        dataset = dataset[: args.limit]
        print(f"  --limit {args.limit}: evaluating {len(dataset)} questions")

    # ── 2. Load pipeline ─────────────────────────────────────────────────────
    print("\nStep 2/5 — Loading RAG pipeline ...")
    index, nodes, storage_context, groq_api_key = load_pipeline()

    # ── 3. Generate answers ──────────────────────────────────────────────────
    print("\nStep 3/5 — Generating answers ...")
    results = run_queries(dataset, index, nodes, storage_context)

    if args.dry_run:
        print("\nDry-run mode: skipping LLM-judge evaluation.")
        print(f"Generated {len(results)} answers successfully.")
        return

    # ── 4. RAGAS ─────────────────────────────────────────────────────────────
    print("\nStep 4/5 — RAGAS evaluation (judge: Groq Mixtral-8x7B) ...")
    ragas_scores = run_ragas(results, groq_api_key)

    # ── 5. DeepEval ──────────────────────────────────────────────────────────
    deepeval_scores: dict[str, float] = {}
    if not args.skip_deepeval:
        print("\nStep 5/5 — DeepEval evaluation (judge: Groq Mixtral-8x7B) ...")
        try:
            deepeval_scores = run_deepeval(results, groq_api_key)
        except Exception as exc:
            print(f"  WARNING: DeepEval failed ({exc}). Continuing without it.")
    else:
        print("\nStep 5/5 — DeepEval skipped (--skip-deepeval).")

    # ── 6. Regression check + persist ────────────────────────────────────────
    all_metrics = {**ragas_scores, **deepeval_scores}
    alerts = check_regression(all_metrics, threshold=args.alert_threshold)
    save_metrics(all_metrics)

    # ── 7. Report ─────────────────────────────────────────────────────────────
    print_report(ragas_scores, deepeval_scores, results, alerts)

    if alerts:
        sys.exit(1)  # non-zero exit for CI gating


if __name__ == "__main__":
    main()
