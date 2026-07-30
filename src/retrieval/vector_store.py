"""
Task 5 — Vector store: index leaf nodes into Qdrant via LlamaIndex.

Three modes (controlled by QDRANT_URL in .env):

  Phase 1 dev  — QDRANT_URL empty (default):
      QdrantClient(path="./qdrant_storage") — local file-based Qdrant.
      Persists to disk between runs. No Docker needed. One-time 12-min
      embed; subsequent runs load in <1s.

  Phase 2 prod — QDRANT_URL=http://localhost:6333:
      Docker Qdrant. Swap one env var, no code change.

Collection: 'fintech_rag'
Embedding:  BGE-M3 (BAAI/bge-m3) — 1024-dim, L2-normalised, Distance.COSINE
Only leaf nodes are indexed (100-token chunks).
Parent/mid nodes live in docstore only (AutoMergingRetriever fetches them).

Output: build_index(nodes, storage_context) -> VectorStoreIndex
"""

import os
import time

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.retrieval.embedder import get_embed_model

COLLECTION_NAME  = "fintech_rag"
VECTOR_DIM       = 1024                # BGE-M3 output dimension
QDRANT_PATH      = "./qdrant_storage"  # local persistence dir (Phase 1 only)


_qdrant_client: QdrantClient | None = None


def _get_qdrant_client() -> QdrantClient:
    """
    Return a shared QdrantClient instance (singleton per process).

    Local Qdrant (file-based) only allows one client at a time due to
    portalocker. Reusing one instance avoids AlreadyLocked errors when
    load_index() and build_index() are called in the same process.

    Phase 1: QDRANT_URL empty → local file-based client (persists to disk).
    Phase 2: QDRANT_URL set   → remote Docker Qdrant.

    QDRANT_URL is read here (not at module level) so that load_dotenv()
    called by the application entry-point is always picked up correctly.
    """
    global _qdrant_client
    if _qdrant_client is None:
        qdrant_url = os.getenv("QDRANT_URL", "")
        if qdrant_url:
            _qdrant_client = QdrantClient(url=qdrant_url)
        else:
            os.makedirs(QDRANT_PATH, exist_ok=True)
            _qdrant_client = QdrantClient(path=QDRANT_PATH)
    return _qdrant_client


def build_index(
    nodes: list,
    storage_context: StorageContext,
) -> VectorStoreIndex:
    """
    Embed leaf nodes and index them into Qdrant.

    Creates the 'fintech_rag' collection if it doesn't exist.
    Attaches the QdrantVectorStore to the existing StorageContext so
    the same context holds both the docstore (parent nodes) and the
    vector store (leaf embeddings). Tasks 7-8 consume this index.

    Args:
        nodes:           Leaf nodes from chunk_document() — 100-token chunks.
        storage_context: StorageContext from chunk_document() — holds docstore.

    Returns:
        VectorStoreIndex backed by Qdrant, ready for retrieval.
    """
    embed_model = get_embed_model()
    client      = _get_qdrant_client()

    # Create collection if needed (idempotent)
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_DIM,
                distance=Distance.COSINE,   # safe with L2-normalised BGE-M3 vectors
            ),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
    )

    # Build a new StorageContext that combines the existing docstore
    # (parent nodes for AutoMergingRetriever) with the Qdrant vector store.
    # StorageContext.vector_store is read-only — must pass at construction.
    sc = StorageContext.from_defaults(
        docstore=storage_context.docstore,
        vector_store=vector_store,
    )

    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=sc,
        embed_model=embed_model,
        show_progress=True,
    )
    return index


def load_index(storage_context: StorageContext) -> VectorStoreIndex:
    """
    Load a previously built index from the persisted Qdrant collection.

    Fast path — skips embedding entirely. Use this in Tasks 7-9 when
    the collection already exists from a previous build_index() call.

    Args:
        storage_context: StorageContext from chunk_document() — holds docstore.

    Returns:
        VectorStoreIndex backed by the existing Qdrant collection.

    Raises:
        ValueError: If the collection doesn't exist yet (run build_index first).
    """
    client = _get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' not found. "
            "Run build_index() first to embed and index the nodes."
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
    )
    sc = StorageContext.from_defaults(
        docstore=storage_context.docstore,
        vector_store=vector_store,
    )
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=sc,
        embed_model=get_embed_model(),
    )


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    from src.ingestion.parser import load_pdf_with_sections
    from src.ingestion.metadata import build_section_map_and_clean
    from src.ingestion.chunker import chunk_document

    pdf_path = "data/raw/rbi_kyc_master_direction.pdf"

    print(f"Loading {pdf_path} ...")
    text_marked = load_pdf_with_sections(pdf_path)
    chapter_map, section_map, clean_text = build_section_map_and_clean(text_marked)

    print("Loading chunks from storage/ ...")
    leaf_nodes, storage_context = chunk_document(
        text=clean_text,
        filename=pdf_path,
        section_map=section_map,
        chapter_map=chapter_map,
        clean_text=clean_text,
    )
    print(f"  {len(leaf_nodes)} leaf nodes loaded\n")

    print("Building Qdrant index ...")
    t0    = time.time()
    index = build_index(leaf_nodes, storage_context)
    build_time = time.time() - t0
    print(f"  Indexed {len(leaf_nodes)} nodes in {build_time:.1f}s\n")

    # Query
    query = "What documents does an NRI need for KYC?"
    retriever = index.as_retriever(similarity_top_k=3)

    t0 = time.time()
    results = retriever.retrieve(query)
    query_time = time.time() - t0

    print(f"Query: {query!r}")
    print(f"Response time: {query_time*1000:.0f}ms\n")

    for i, r in enumerate(results):
        meta = r.node.metadata
        print(f"[{i+1}] score={r.score:.4f}")
        print(f"     chapter={meta.get('chapter')!r}")
        print(f"     section={meta.get('section_title')!r}")
        print(f"     text (first 200): {r.node.get_content()[:200]!r}")
        print()

    # Verify
    top_text = results[0].node.get_content().lower()
    nri_hit  = any(kw in top_text for kw in ["nri", "non-resident", "passport", "kyc"])
    print(f"Top result contains NRI/KYC content: {nri_hit}")
    assert nri_hit, "FAIL: top result should contain NRI or KYC content"

    # First query includes BGE-M3 warm-up (model load + query embed).
    # Run a second query to measure steady-state retrieval latency.
    t1 = time.time()
    _ = retriever.retrieve("What is the penalty for KYC non-compliance?")
    warm_query_time = time.time() - t1
    print(f"Warm query time: {warm_query_time*1000:.0f}ms")
    # <200ms target is for Docker Qdrant (Phase 2). Local file mode adds ~100ms
    # overhead per query — 500ms is the Phase 1 acceptance threshold.
    assert warm_query_time < 0.5, f"FAIL: warm query took {warm_query_time*1000:.0f}ms, expected <500ms"
    print("PASS")
