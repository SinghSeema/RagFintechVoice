"""
Task 9 — Generation: CitationQueryEngine + Groq LLM.

Wires the full RAG pipeline into a single query() entry-point:

  Pipeline
  --------
  query(question, mode)
    → build_pipeline()          (hybrid → AutoMerging → BGE reranker)
    → CitationQueryEngine       (LlamaIndex built-in; appends [1] [2] ... citations)
    → structured QueryResult    (answer_text, sources list)

  Mode
  ----
  "text"  → reranker_top_n=5  (richer context, more citations)
  "voice" → reranker_top_n=3  (fewer chunks → faster generation per ADR-004)

  CitationQueryEngine behaviour
  ------------------------------
  - Wraps each retrieved chunk in a numbered source block before sending to LLM
  - LLM is instructed to cite inline with [1], [2] etc.
  - citation_chunk_size=512  (larger than leaf=256; CitationQueryEngine passes chunks through unsplit)
  - citation_chunk_overlap=20

  Output schema
  -------------
  QueryResult.answer      : str   — LLM answer with inline [n] citation markers
  QueryResult.sources     : list  — [{rank, section_title, chapter, snippet}]
  QueryResult.mode        : str   — "text" or "voice"
  QueryResult.num_sources : int
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Literal

from dotenv import load_dotenv
from llama_index.core import QueryBundle, StorageContext, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.schema import NodeRelationship
from loguru import logger

from src.retrieval.hybrid_retriever import configure_llm
from src.retrieval.pipeline import build_pipeline


# ── Answer prompts ──────────────────────────────────────────────────────────
#
# CitationQueryEngine wraps retrieved chunks into numbered Source blocks and
# appends them before the query. Our prompt replaces the default, which
# produces verbose, hedged, bureaucratic output ("Based on the provided
# sources, the official valid documents include...").
#
# Two modes:
#   text  — clear prose, cite sources inline, no markdown lists
#   voice — short spoken English, no citations, 3–4 sentences max
#
# Both prompts share the same {context_str} / {query_str} placeholders that
# CitationQueryEngine requires.

_TEXT_QA_TEMPLATE = PromptTemplate(
    """\
Please provide an answer based solely on the provided sources. \
When referencing information from a source, cite the appropriate source(s) \
using their corresponding numbers. Every answer should include at least one \
source citation. Write in plain English — no bullet points, use short paragraphs. \
If none of the sources are helpful, indicate that clearly.

{context_str}

Question: {query_str}
Answer:"""
)

_VOICE_QA_TEMPLATE = PromptTemplate(
    """\
You are a friendly bank compliance assistant speaking to a customer over a voice call.
Answer using only the information in the sources below.

STRICT RULES — follow every one:
1. Paraphrase everything in plain everyday English. NEVER copy regulatory text word-for-word.
2. Convert legal phrasing: "Officially Valid Documents" → "approved ID documents", \
"regulated entity" → "your bank", "Proof of possession of Aadhaar" → "Aadhaar card".
3. Do NOT mention source numbers, clause labels, or sub-clause codes like (a), (ab), (iii).
4. Maximum 3 short sentences. Voice responses must be brief.
5. If listing documents or steps, name only the 2-3 most common ones, \
then say "and similar documents" — do not read out every item.
6. Close with one short reference if helpful, e.g. "as per RBI KYC guidelines" — nothing longer.
7. If the sources do not contain a clear answer, say "I don't have that detail right now, \
please check with your branch."

{context_str}

Question: {query_str}
Answer:"""
)


# ── Output dataclass ────────────────────────────────────────────────────────

@dataclass
class Source:
    rank: int
    section_title: str
    chapter: str
    snippet: str          # first 120 chars of the chunk (display)
    content: str = ""     # full chunk text (for evaluation / faithfulness)


@dataclass
class QueryResult:
    answer: str
    sources: list[Source]
    mode: str
    top_score: float = 0.0    # reranker score of the top node; used for early-exit floor check
    num_sources: int = field(init=False)

    def __post_init__(self) -> None:
        self.num_sources = len(self.sources)

    def __str__(self) -> str:
        lines = [f"Answer ({self.mode}):", self.answer, "", "Sources:"]
        for s in self.sources:
            lines.append(
                f"  [{s.rank}] chapter={s.chapter!r} | "
                f"section={s.section_title!r}"
            )
            lines.append(f"       {s.snippet!r}")
        return "\n".join(lines)


# ── Main entry-point ────────────────────────────────────────────────────────

def query(
    question: str,
    index: VectorStoreIndex,
    nodes: list,
    storage_context: StorageContext,
    mode: Literal["text", "voice"] = "text",
    similarity_top_k: int = 10,
    reranker_model: str = "BAAI/bge-reranker-base",
    retriever=None,
    reranker=None,
) -> QueryResult:
    """
    Run the full RAG pipeline and return a structured QueryResult.

    Args:
        question:         User question (natural language).
        index:            VectorStoreIndex (Qdrant-backed).
        nodes:            Leaf nodes list (for BM25).
        storage_context:  StorageContext with docstore (for AutoMerging parent lookup).
        mode:             "text" → top_n=5; "voice" → top_n=3.
        similarity_top_k: Candidates per retriever before fusion (default 10).
        reranker_model:   BGE cross-encoder model name.
        retriever:        Pre-built AutoMergingRetriever. If provided together with
                          reranker, build_pipeline() is skipped — avoids re-loading
                          the cross-encoder model on every call (critical for batch
                          evaluation over many questions).
        reranker:         Pre-built FlagEmbeddingReranker. See `retriever` above.

    Returns:
        QueryResult with .answer (str) and .sources (list[Source]).
    """
    if retriever is None or reranker is None:
        reranker_top_n = 5   # same for both modes — quality > marginal latency saving
        retriever, reranker = build_pipeline(
            index,
            nodes,
            storage_context,
            similarity_top_k=similarity_top_k,
            reranker_top_n=reranker_top_n,
            reranker_model=reranker_model,
        )

    # Rerank manually first so we control the final node list
    query_bundle = QueryBundle(query_str=question)

    t0 = time.perf_counter()
    raw_nodes = retriever.retrieve(question)
    t_retrieved = time.perf_counter()

    # Detect auto-merging: a node that has CHILD relationships was promoted
    # from leaf → mid/parent by AutoMergingRetriever.
    merged_count = sum(
        1 for n in raw_nodes
        if NodeRelationship.CHILD in n.node.relationships
    )
    logger.info(
        f"RAG | retrieve: {len(raw_nodes)} nodes "
        f"(auto-merged={merged_count}, leaf={len(raw_nodes)-merged_count}) "
        f"elapsed={1000*(t_retrieved-t0):.0f}ms"
    )

    reranked = reranker.postprocess_nodes(raw_nodes, query_bundle)
    t_reranked = time.perf_counter()
    logger.info(
        f"RAG | rerank: top-{len(reranked)} of {len(raw_nodes)} "
        f"elapsed={1000*(t_reranked-t_retrieved):.0f}ms"
    )
    for i, n in enumerate(reranked, 1):
        meta = n.node.metadata
        is_merged = NodeRelationship.CHILD in n.node.relationships
        logger.info(
            f"RAG |   [{i}] score={n.score:.4f} merged={is_merged} "
            f"chapter={meta.get('chapter','?')!r} "
            f"section={meta.get('section_title','?')!r} "
            f"len={len(n.node.get_content())}"
        )

    # Filter noise nodes with two guards:
    #   1. Relative gap:  drop nodes more than 3 points below the top score.
    #   2. Absolute floor: drop nodes below -4.0 regardless of relative gap.
    #      BGE cross-encoder scores below -4.0 indicate no domain signal — the
    #      chunk is unrelated to the query.  Without this floor, off-topic queries
    #      (top_score=-8) produce a floor of -13, keeping all 5 noise nodes and
    #      causing the LLM to hallucinate a refusal ("I don't have sufficient info").
    # Always keep at least the top node as a fallback (caller checks top_score).
    _ABS_FLOOR = -4.0
    top_score = reranked[0].score if reranked else 0.0
    positive_nodes = [
        n for n in reranked
        if n.score >= top_score - 3.0 and n.score >= _ABS_FLOOR
    ] or reranked[:1]

    # CitationQueryEngine expects a plain retriever; feed it a pre-built
    # StaticRetriever wrapping the already-reranked nodes
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore

    class _StaticRetriever(BaseRetriever):
        """Returns a fixed node list regardless of query."""
        def __init__(self, scored_nodes: list[NodeWithScore]) -> None:
            super().__init__()
            self._nodes = scored_nodes

        def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
            return self._nodes

    static_retriever = _StaticRetriever(positive_nodes)

    if mode == "voice":
        # For voice, bypass CitationQueryEngine — it re-chunks content into
        # "Source N: (ab) clause..." labels that the LLM echoes verbatim.
        # Instead: assemble clean passage text and call the LLM directly.
        passages = "\n\n".join(
            n.node.get_content()[:600] for n in positive_nodes[:3]
        )
        prompt = _VOICE_QA_TEMPLATE.format(
            context_str=passages,
            query_str=question,
        )
        from llama_index.core import Settings
        response_obj = Settings.llm.complete(prompt)
        answer = str(response_obj).strip()
    else:
        engine = CitationQueryEngine.from_args(
            index,
            retriever=static_retriever,
            citation_chunk_size=256,
            citation_chunk_overlap=20,
            text_qa_template=_TEXT_QA_TEMPLATE,
            verbose=False,
        )
        answer = str(engine.query(question))

    t_generated = time.perf_counter()
    logger.info(
        f"RAG | generation elapsed={1000*(t_generated-t_reranked):.0f}ms "
        f"total={1000*(t_generated-t0):.0f}ms"
    )

    # ── Build structured sources list (only the nodes the LLM actually saw) ──
    sources: list[Source] = []
    for rank, node_with_score in enumerate(positive_nodes, start=1):
        meta = node_with_score.node.metadata
        full_text = node_with_score.node.get_content()
        sources.append(
            Source(
                rank=rank,
                section_title=meta.get("section_title", ""),
                chapter=meta.get("chapter", ""),
                snippet=full_text[:120],
                content=full_text,
            )
        )

    return QueryResult(
        answer=answer,
        sources=sources,
        mode=mode,
        top_score=top_score,
    )


# ── CLI smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    from src.ingestion.chunker import chunk_document
    from src.ingestion.metadata import build_section_map_and_clean
    from src.ingestion.parser import load_pdf_with_sections
    from src.retrieval.vector_store import load_index

    load_dotenv()
    configure_llm(os.environ["GROQ_API_KEY"])

    pdf_path = "data/raw/rbi_kyc_master_direction.pdf"

    print("Loading chunks ...")
    text_marked = load_pdf_with_sections(pdf_path)
    chapter_map, section_map, clean_text = build_section_map_and_clean(text_marked)
    leaf_nodes, storage_context = chunk_document(
        text=clean_text,
        filename=pdf_path,
        section_map=section_map,
        chapter_map=chapter_map,
        clean_text=clean_text,
    )
    print(f"  {len(leaf_nodes)} leaf nodes loaded")

    print("Loading Qdrant index ...")
    index = load_index(storage_context)
    print("  Index loaded\n")

    test_cases = [
        ("What documents does an NRI need for KYC?", "text"),
        ("Simplified due diligence for low risk customers", "voice"),
    ]

    for question, mode in test_cases:
        print(f"{'='*65}")
        print(f"Question [{mode}]: {question!r}")
        t0 = time.perf_counter()
        result = query(
            question,
            index=index,
            nodes=leaf_nodes,
            storage_context=storage_context,
            mode=mode,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"Elapsed: {elapsed:.0f} ms  |  sources={result.num_sources}")
        print()
        print(result)
        print()

    # ── Assertions ───────────────────────────────────────────────────────────
    # Re-run text query for assertion
    r = query(
        "What documents does an NRI need for KYC?",
        index=index,
        nodes=leaf_nodes,
        storage_context=storage_context,
        mode="text",
    )
    assert r.answer, "FAIL: empty answer"
    assert 1 <= r.num_sources <= 5, f"FAIL: num_sources={r.num_sources}"

    r_voice = query(
        "What documents does an NRI need for KYC?",
        index=index,
        nodes=leaf_nodes,
        storage_context=storage_context,
        mode="voice",
    )
    assert r_voice.num_sources <= 3, f"FAIL: voice mode returned {r_voice.num_sources} sources (max 3)"

    print("PASS")
