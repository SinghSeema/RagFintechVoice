"""
Revamp vector store — multi-collection Qdrant support.

Mirrors vector_store.py but handles two named collections:
  fintech_rag_agentic  — proposition TextNodes
  fintech_rag_qa       — Q&A pair TextNodes (embedded on question text)

Node sidecar JSON:
  Every build_revamp_index() call writes a sidecar file alongside the
  Qdrant storage so the proposition/QA nodes can be reloaded for BM25
  without re-embedding.  Stored at:
    qdrant_storage/revamp_nodes_{collection_name}.json

Functions:
  build_revamp_index(nodes, collection_name, force_recreate) -> VectorStoreIndex
  load_revamp_index(collection_name)                          -> VectorStoreIndex
  save_node_sidecar(nodes, collection_name)                   -> None
  load_node_sidecar(collection_name)                          -> list[TextNode]
  collection_exists(collection_name)                          -> bool
"""

from __future__ import annotations

import json
import os
import time

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.retrieval.embedder import get_embed_model

VECTOR_DIM   = 1024
QDRANT_PATH  = "./qdrant_storage"
_SIDECAR_DIR = "./qdrant_storage"


# ── Qdrant client (separate from legacy singleton to avoid interference) ──────

_revamp_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _revamp_client
    if _revamp_client is None:
        qdrant_url = os.getenv("QDRANT_URL", "")
        if qdrant_url:
            _revamp_client = QdrantClient(url=qdrant_url)
        else:
            os.makedirs(QDRANT_PATH, exist_ok=True)
            _revamp_client = QdrantClient(path=QDRANT_PATH)
    return _revamp_client


# ── Collection helpers ────────────────────────────────────────────────────────

def collection_exists(collection_name: str) -> bool:
    client = _get_client()
    return collection_name in [c.name for c in client.get_collections().collections]


def _ensure_collection(collection_name: str, force_recreate: bool = False) -> None:
    client = _get_client()
    existing = collection_exists(collection_name)
    if existing and force_recreate:
        client.delete_collection(collection_name)
        existing = False
    if not existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


# ── Node sidecar (plain JSON — no LlamaIndex serialisation machinery) ─────────

def _sidecar_path(collection_name: str) -> str:
    os.makedirs(_SIDECAR_DIR, exist_ok=True)
    return os.path.join(_SIDECAR_DIR, f"revamp_nodes_{collection_name}.json")


def save_node_sidecar(nodes: list[TextNode], collection_name: str) -> None:
    """Persist node text + metadata so BM25 can be rebuilt without re-embedding."""
    data = [
        {"node_id": n.node_id, "text": n.text, "metadata": n.metadata}
        for n in nodes
    ]
    path = _sidecar_path(collection_name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_node_sidecar(collection_name: str) -> list[TextNode]:
    """Reload TextNodes from the sidecar JSON (text + metadata, no vectors)."""
    path = _sidecar_path(collection_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Node sidecar not found: {path}\n"
            f"Run ingest_revamp.py --strategy agentic (or qa_library) first."
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        TextNode(node_id=item["node_id"], text=item["text"], metadata=item["metadata"])
        for item in data
    ]


# ── Index build / load ────────────────────────────────────────────────────────

def build_revamp_index(
    nodes: list[TextNode],
    collection_name: str,
    force_recreate: bool = False,
) -> VectorStoreIndex:
    """
    Embed nodes and ingest them into a named Qdrant collection.

    Also writes a node sidecar JSON so BM25 can be rebuilt at query time.
    If force_recreate=False and the collection already exists, skips embedding
    and just returns load_revamp_index(collection_name).
    """
    if collection_exists(collection_name) and not force_recreate:
        print(f"[revamp_vector_store] Collection '{collection_name}' exists — loading (use --force-recreate to rebuild).")
        return load_revamp_index(collection_name)

    _ensure_collection(collection_name, force_recreate=force_recreate)
    embed_model = get_embed_model()
    client      = _get_client()

    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    sc = StorageContext.from_defaults(vector_store=vector_store)

    t0 = time.time()
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=sc,
        embed_model=embed_model,
        show_progress=True,
    )
    elapsed = time.time() - t0
    print(f"[revamp_vector_store] Indexed {len(nodes)} nodes into '{collection_name}' in {elapsed:.1f}s")

    save_node_sidecar(nodes, collection_name)
    return index


def load_revamp_index(collection_name: str) -> VectorStoreIndex:
    """Load a previously built revamp index from Qdrant (fast path, no re-embedding)."""
    if not collection_exists(collection_name):
        raise ValueError(
            f"Collection '{collection_name}' not found. "
            f"Run ingest_revamp.py first."
        )
    client      = _get_client()
    embed_model = get_embed_model()

    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
    sc = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=sc,
        embed_model=embed_model,
    )
