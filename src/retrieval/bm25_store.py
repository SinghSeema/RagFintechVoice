"""
Task 6 — BM25 keyword retrieval via LlamaIndex BM25Retriever.

BM25 scores documents by term frequency weighted by document length.
It finds exact matches that semantic search misses:
  - Regulation codes: 'Section 4.2', 'PMLA', 'FATF'
  - Section numbers: '56.', '64.'
  - Acronyms: 'NRI', 'KYC', 'CDD', 'CKYCR'

Phase 1 uses LlamaIndex's built-in BM25Retriever (rank-bm25, pure Python,
no Docker). Phase 2 replaces with OpenSearch via identical
QueryFusionRetriever interface — one line swap.

Output: build_bm25_retriever(nodes) -> BM25Retriever
"""

from llama_index.retrievers.bm25 import BM25Retriever


def build_bm25_retriever(nodes: list, similarity_top_k: int = 5) -> BM25Retriever:
    """
    Build a BM25 retriever over the given leaf nodes.

    Args:
        nodes:            Leaf nodes from chunk_document() — 100-token chunks.
        similarity_top_k: Number of results to return per query.

    Returns:
        BM25Retriever ready for querying.
    """
    return BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=similarity_top_k,
    )


if __name__ == "__main__":
    from src.ingestion.parser import load_pdf_with_sections
    from src.ingestion.metadata import build_section_map_and_clean
    from src.ingestion.chunker import chunk_document

    pdf_path = "data/raw/rbi_kyc_master_direction.pdf"

    # Load chunks from storage (fast path)
    print(f"Loading chunks from storage/ ...")
    text_marked = load_pdf_with_sections(pdf_path)
    chapter_map, section_map, clean_text = build_section_map_and_clean(text_marked)
    leaf_nodes, storage_context = chunk_document(
        text=clean_text,
        filename=pdf_path,
        section_map=section_map,
        chapter_map=chapter_map,
        clean_text=clean_text,
    )
    print(f"  {len(leaf_nodes)} leaf nodes\n")

    # Build BM25 retriever
    bm25 = build_bm25_retriever(leaf_nodes, similarity_top_k=5)

    # ── Query 1: exact acronym — BM25 strength ────────────────────────────
    # Semantic search may return related but not exact; BM25 nails FATCA/CRS
    q1 = "FATCA CRS reporting requirements"
    print(f"Query 1 (exact acronym): {q1!r}")
    bm25_results1 = bm25.retrieve(q1)
    print(f"BM25 top 3:")
    for i, r in enumerate(bm25_results1[:3]):
        print(f"  [{i+1}] score={r.score:.4f}  section={r.node.metadata.get('section_title')!r}")
        print(f"       {r.node.get_content()[:120]!r}")
    print()

    # ── Query 2: section number reference — BM25 advantage ────────────────
    # 'Section 56' scores 0 in Qdrant (no semantic meaning); high in BM25
    q2 = "Section 56 CDD procedure sharing"
    print(f"Query 2 (section reference): {q2!r}")
    bm25_results2 = bm25.retrieve(q2)
    print(f"BM25 top 3:")
    for i, r in enumerate(bm25_results2[:3]):
        print(f"  [{i+1}] score={r.score:.4f}  section={r.node.metadata.get('section_title')!r}")
        print(f"       {r.node.get_content()[:120]!r}")
    print()

    # ── Verify BM25 finds exact section references ─────────────────────────
    q2_top_section = bm25_results2[0].node.metadata.get("section_title", "")
    assert "56" in q2_top_section, f"FAIL: expected section 56, got {q2_top_section!r}"
    print(f"BM25 correctly finds Section 56 for section-number query: PASS")
    print()

    # ── Note on Qdrant comparison ──────────────────────────────────────────
    # Qdrant comparison omitted here to avoid re-embedding (12 min on CPU).
    # Task 7 (hybrid_retriever.py) demonstrates the difference in-context
    # with both retrievers pre-built and fused via RRF.
