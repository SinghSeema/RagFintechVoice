"""
Task 8 — Retrieval pipeline: AutoMergingRetriever + BGE cross-encoder reranker.

Two-stage pipeline on top of hybrid retrieval (Task 7):

Stage 1 — AutoMergingRetriever
  Wraps the hybrid retriever. After retrieval, if >= 50% of a parent chunk's
  leaf children appear in the candidate set, the children are replaced by the
  parent (up to 400-token mid or 1500-token parent). This gives the LLM richer
  context without retrieving redundant overlapping leaves.

Stage 2 — FlagEmbeddingReranker (BGE cross-encoder)
  Model: BAAI/bge-reranker-base (~278 MB, CPU ~200-400 ms for 10 candidates)
  Reads (query, chunk) pair together — far more accurate than dot-product
  similarity alone. Reorders the merged candidates and returns top_n.

  top_n:
    - voice: 3  (ADR-004 target <500ms total; fewer chunks = faster generation)
    - text:  5  (richer citations acceptable)

Output: build_pipeline(index, nodes, storage_context) -> (retriever, reranker)
  Caller applies reranker as a postprocessor:
    nodes = reranker.postprocess_nodes(retriever.retrieve(query), query_bundle)
"""

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

from src.retrieval.hybrid_retriever import build_hybrid_retriever



def build_pipeline(
    index: VectorStoreIndex,
    nodes: list,
    storage_context: StorageContext,
    similarity_top_k: int = 15,
    reranker_top_n: int = 5,
    reranker_model: str = "BAAI/bge-reranker-base",
) -> tuple[AutoMergingRetriever, FlagEmbeddingReranker]:
    """
    Build the full retrieval pipeline.

    Args:
        index:             VectorStoreIndex (Qdrant-backed).
        nodes:             Leaf nodes list (for BM25).
        storage_context:   StorageContext with docstore (for AutoMerging parent lookup).
        similarity_top_k:  Candidates per retriever before fusion (default 10).
        reranker_top_n:    Final chunks returned after reranking.
                           Use 3 for voice, 5 for text.
        reranker_model:    BGE cross-encoder model name.

    Returns:
        (auto_merging_retriever, reranker) tuple.
        Usage:
            retriever, reranker = build_pipeline(...)
            query_bundle = QueryBundle(query_str=query)
            raw_nodes = retriever.retrieve(query)
            reranked  = reranker.postprocess_nodes(raw_nodes, query_bundle)
    """
    hybrid_retriever = build_hybrid_retriever(
        index, nodes, similarity_top_k=similarity_top_k
    )

    # AutoMergingRetriever needs the docstore to look up parent nodes
    auto_merging_retriever = AutoMergingRetriever(
        hybrid_retriever,
        storage_context,
        verbose=True,   # logs merge decisions: "> Merging X nodes into parent ..."
    )

    reranker = FlagEmbeddingReranker(
        model=reranker_model,
        top_n=reranker_top_n,
        use_fp16=True,
    )
    # Patch _model to use CPU device and smaller batch_size for CPU performance.
    # FlagEmbeddingReranker (v0.5.0) doesn't expose these args, but FlagReranker does.
    from FlagEmbedding import FlagReranker
    reranker._model = FlagReranker(
        reranker_model,
        use_fp16=True,
        devices="cpu",   # skip CUDA probe; force CPU path immediately
        batch_size=4,    # prevents one long merged node from padding the entire batch
    )

    return auto_merging_retriever, reranker


if __name__ == "__main__":
    import os
    import time
    from dotenv import load_dotenv
    from llama_index.core import QueryBundle

    from src.ingestion.parser import load_pdf_with_sections
    from src.ingestion.metadata import build_section_map_and_clean
    from src.ingestion.chunker import chunk_document
    from src.retrieval.vector_store import load_index
    from src.retrieval.hybrid_retriever import configure_llm

    load_dotenv()
    configure_llm(os.environ["GROQ_API_KEY"])

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

    print("Building pipeline (downloading reranker model on first run) ...")
    retriever, reranker = build_pipeline(
        index, leaf_nodes, storage_context,
        similarity_top_k=15,
        reranker_top_n=5,
    )
    print("  Pipeline ready\n")

    queries = [
        "What documents does an NRI need for KYC?",
        "FATCA CRS reporting requirements section 57",
        "Simplified due diligence low risk customers",
    ]

    for q in queries:
        query_bundle = QueryBundle(query_str=q)

        t0 = time.perf_counter()
        raw_nodes = retriever.retrieve(q)
        reranked  = reranker.postprocess_nodes(raw_nodes, query_bundle)
        elapsed   = (time.perf_counter() - t0) * 1000

        print(f"{'='*65}")
        print(f"Query : {q!r}")
        print(f"Merged: {len(raw_nodes)} nodes  →  Reranked top-{len(reranked)}  ({elapsed:.0f} ms)")
        print(f"{'─'*65}")
        for i, r in enumerate(reranked, 1):
            meta = r.node.metadata
            print(
                f"  [{i}] score={r.score:.4f} | "
                f"chapter={meta.get('chapter','?')!r} | "
                f"section={meta.get('section_title','?')!r}"
            )
            print(f"       {r.node.get_content()[:80]!r}")
        print()

    # ── Assert: reranked list is non-empty and within top_n ─────────────────
    assert len(reranked) <= 5, f"FAIL: got {len(reranked)} nodes, expected <= 5"
    assert len(reranked) >= 1, "FAIL: no nodes returned"
    print("PASS")
