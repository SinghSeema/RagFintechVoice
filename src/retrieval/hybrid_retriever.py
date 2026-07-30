"""
Task 7 — Hybrid retrieval: QueryFusionRetriever with RRF (k=60).

Combines Qdrant dense retriever (semantic) + BM25 sparse retriever (keyword)
using Reciprocal Rank Fusion.

RRF score = 1/(k + rank) where k=60 (standard).
  - Uses only rank positions, not raw scores — avoids BM25/cosine scale mismatch
  - Documents appearing in both lists get double boost
  - No normalisation needed

Why hybrid beats either alone:
  - Qdrant finds semantically related chunks even without exact keyword match
  - BM25 nails exact regulation codes, section numbers, acronyms
  - RRF fusion gives high scores to chunks that both retrievers agree on

Output: build_hybrid_retriever(index, nodes) -> QueryFusionRetriever
"""

from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core import VectorStoreIndex, Settings
from llama_index.retrievers.bm25 import BM25Retriever

from src.retrieval.bm25_store import build_bm25_retriever


def build_hybrid_retriever(
    index: VectorStoreIndex,
    nodes: list,
    similarity_top_k: int = 15,
) -> QueryFusionRetriever:
    """
    Build a hybrid retriever combining Qdrant dense + BM25 sparse via RRF.

    Args:
        index:            VectorStoreIndex from build_index() or load_index().
        nodes:            Leaf nodes — same list passed to build_index().
        similarity_top_k: Candidates per retriever before fusion.
                          ADR-004 voice: top-2, text: top-3/4.
                          Default 10 gives fusion enough candidates to rerank.

    Returns:
        QueryFusionRetriever configured with RECIPROCAL_RANK mode, k=60.
    """
    qdrant_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    bm25_retriever   = build_bm25_retriever(nodes, similarity_top_k=similarity_top_k)

    return QueryFusionRetriever(
        retrievers=[qdrant_retriever, bm25_retriever],
        similarity_top_k=similarity_top_k,
        num_queries=1,          # no query expansion — use original query only
        mode="reciprocal_rerank",
        use_async=False,
        llm=Settings.llm,       # must be set before calling; use Groq via configure_llm()
    )


_RAG_SYSTEM_PROMPT = """\
You are Finova, a specialized assistant for RBI regulatory compliance.
Your knowledge is LIMITED to what is contained in the provided source documents,
which cover RBI Master Directions on KYC, AML/CFT obligations, PMLA provisions,
customer due diligence, and related banking compliance topics.

Rules you must follow without exception:
1. ONLY answer from the information present in the numbered source passages below.
2. Do NOT use your pretrained knowledge to answer. If the source passages only
   partially cover the question, summarise what they do contain and note what is
   missing. Do NOT say you have no information when the passages contain relevant
   details — extract and present what is there. Only say you lack information when
   the passages are entirely unrelated to the question.
3. Do NOT answer questions about stock markets, trading, crypto, investment advice,
   insurance products, tax filing, or general personal finance — even if you know the
   answer from training. Politely say these are outside your scope.
4. Keep answers accurate, concise, and grounded in the regulatory text.
   Follow the formatting instructions given in the query prompt exactly."""


def configure_llm(api_key: str, model: str = "llama-3.1-8b-instant") -> None:
    """Set global LLM to Groq with a grounding system prompt. Call once at startup."""
    from llama_index.llms.groq import Groq
    Settings.llm = Groq(model=model, api_key=api_key, system_prompt=_RAG_SYSTEM_PROMPT)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from src.ingestion.parser import load_pdf_with_sections
    from src.ingestion.metadata import build_section_map_and_clean
    from src.ingestion.chunker import chunk_document
    from src.retrieval.vector_store import load_index

    load_dotenv()
    api_key = os.environ["GROQ_API_KEY"]
    configure_llm(api_key)

    pdf_path = "data/raw/rbi_kyc_master_direction.pdf"

    print("Loading chunks from storage/ ...")
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

    print("Loading Qdrant index from qdrant_storage/ ...")
    index = load_index(storage_context)
    print("  Index loaded\n")

    hybrid = build_hybrid_retriever(index, leaf_nodes, similarity_top_k=10)

    # ── Test queries ────────────────────────────────────────────────────────
    # Q1: BM25 misses this (no exact token match), Qdrant finds it semantically
    q1 = "What documents does an NRI need for KYC?"

    # Q2: BM25 wins this (exact section number + acronym)
    q2 = "FATCA CRS reporting requirements section 57"

    for q in [q1, q2]:
        qdrant_res = index.as_retriever(similarity_top_k=5).retrieve(q)
        bm25_res   = build_bm25_retriever(leaf_nodes, similarity_top_k=5).retrieve(q)
        hybrid_res = hybrid.retrieve(q)

        print(f"{'='*65}")
        print(f"Query: {q!r}")
        print(f"{'─'*65}")
        print(f"  QDRANT top-3")
        for r in qdrant_res[:3]:
            print(f"    [{r.score:.4f}] {r.node.get_content()[:60]!r}")
        print()
        print(f"  BM25 top-3")
        for r in bm25_res[:3]:
            print(f"    [{r.score:.4f}] {r.node.get_content()[:60]!r}")
        print()
        print(f"  HYBRID (RRF) top-3")
        for r in hybrid_res[:3]:
            print(f"    [{r.score:.4f}] {r.node.get_content()[:60]!r}")
        print()

    # ── Verify: for Q2 fused result has Section 57 ─────────────────────────
    q2_hybrid = build_hybrid_retriever(index, leaf_nodes, similarity_top_k=10).retrieve(q2)
    q2_sections = [r.node.metadata.get("section_title", "") for r in q2_hybrid[:3]]
    has_57 = any("57" in s for s in q2_sections)
    print(f"Hybrid top-3 contains Section 57 for FATCA query: {has_57}")
    assert has_57, f"FAIL — sections returned: {q2_sections}"
    print("PASS")
