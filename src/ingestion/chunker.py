"""
Task 2 + 3 — Hierarchical chunking with LlamaIndex HierarchicalNodeParser
             + two-layer metadata enrichment.

Three-level hierarchy per ADR-004:
    parent = 1500 tokens  (auto-merged context, Task 8)
    mid    =  400 tokens  (balanced context for text queries)
    leaf   =  256 tokens  (indexed in vector store; fits OVD-style enumerations)

10% overlap per level (ADR Section 12 starting point):
    parent: 150 tok  |  mid: 40 tok  |  leaf: 25 tok

Metadata — two layers:
    Layer 1 (document-level): jurisdiction, doc_type, effective_date,
        is_stale, clearance_level, citation_priority, source_file.
        Set on Document before chunking — LlamaIndex propagates to all
        child nodes automatically.
    Layer 2 (chunk-level): chunk_index, section_title.
        Added to leaf nodes after chunking via enrich_nodes().

Persistence:
    StorageContext persists to `persist_dir` on first run (with fully
    enriched node metadata). Subsequent runs load from disk — no re-parsing,
    no re-chunking. Delete `storage/` to force a full rebuild.

Output: chunk_document(text, filename, section_map, clean_text, persist_dir)
            -> (leaf_nodes, storage_context)
"""

import os
from typing import Optional

from llama_index.core import Document, StorageContext
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore

from src.ingestion.metadata import extract_metadata, enrich_nodes


# ADR-004: chunk sizes largest → smallest (required by HierarchicalNodeParser)
_CHUNK_SIZES    = [1500, 400, 256]
# 10% overlap per level
_CHUNK_OVERLAPS = [150,  40,  25]


def chunk_document(
    text: str,
    filename: str,
    section_map: Optional[list] = None,
    chapter_map: Optional[list] = None,
    clean_text: Optional[str] = None,
    persist_dir: str = "./storage",
) -> tuple:
    """
    Chunk a parsed document into a 3-level hierarchy, enrich metadata,
    and persist the storage context to disk.

    Fast path (persist_dir already populated):
        Loads StorageContext from disk. Enriched metadata (including
        section_title) was persisted on the first run — no re-parsing
        or re-chunking needed. Returns immediately.

    Slow path (no persist_dir):
        1. Wraps text in a LlamaIndex Document with Layer 1 metadata.
        2. Runs HierarchicalNodeParser (leaf=256, mid=400, parent=1500 tok).
        3. Calls enrich_nodes() to attach chunk_index and section_title.
           section_title uses global text-search against clean_text
           (same as `text` unless explicitly overridden).
        4. Stores ALL nodes (leaf + mid + parent) in SimpleDocumentStore.
           Only leaf nodes are returned for vector indexing (Task 5).
           Parent/mid nodes live in docstore for AutoMergingRetriever (Task 8).
        5. Persists to persist_dir and returns.

    Args:
        text:        Clean plain text (markers already stripped).
        filename:    Source PDF path/name — used by extract_metadata().
        section_map: Section (offset, title) pairs from build_section_map_and_clean().
                     Pass None to skip section_title enrichment.
        chapter_map: Chapter (offset, label) pairs from build_section_map_and_clean().
                     Pass None to skip chapter enrichment (will be None on leaf nodes).
        clean_text:  Full text used for offset lookup.
                     Defaults to `text` if not provided.
        persist_dir: Directory for LlamaIndex StorageContext persistence.
                     Listed in .gitignore.

    Returns:
        (leaf_nodes, storage_context)
        leaf_nodes:      256-token TextNodes ready for vector indexing (Task 5).
        storage_context: Holds the docstore with all 3 node levels.
    """
    # ── Fast path ─────────────────────────────────────────────────────────────
    if os.path.isdir(persist_dir) and os.listdir(persist_dir):
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        all_nodes = list(storage_context.docstore.docs.values())
        leaf_nodes = get_leaf_nodes(all_nodes)
        return leaf_nodes, storage_context

    # ── Slow path — build, enrich, persist ────────────────────────────────────
    full_text = clean_text if clean_text is not None else text

    # Layer 1: document-level metadata (propagates to all child nodes)
    doc_metadata = extract_metadata(filename, full_text)

    # Exclude metadata from LLM context injection to protect the leaf
    # token budget (metadata is 51 tokens; without exclusion LlamaIndex
    # prepends it to every chunk, shrinking 100-token leaves below 50 tokens)
    document = Document(
        text=full_text,
        metadata=doc_metadata,
        excluded_llm_metadata_keys=list(doc_metadata.keys()),
    )

    # HierarchicalNodeParser — one SentenceSplitter per chunk level
    parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=_CHUNK_SIZES,
        chunk_overlap=_CHUNK_OVERLAPS[-1],  # base; overridden per-level below
    )
    # Override chunk_overlap on each per-level SentenceSplitter.
    # node_parser_map values are SentenceSplitter instances at runtime;
    # Pylance sees the NodeParser base class which lacks chunk_overlap.
    for splitter_id, overlap in zip(parser.node_parser_ids, _CHUNK_OVERLAPS):
        parser.node_parser_map[splitter_id].chunk_overlap = overlap  # type: ignore[attr-defined]

    all_nodes = parser.get_nodes_from_documents([document])
    leaf_nodes = get_leaf_nodes(all_nodes)

    # Layer 2: chunk-level metadata on leaf nodes (in-place)
    if section_map is not None:
        enrich_nodes(leaf_nodes, full_text, section_map, chapter_map)

    # Store ALL nodes — AutoMergingRetriever needs parents at query time
    docstore = SimpleDocumentStore()
    docstore.add_documents(all_nodes)

    # Propagate chapter + section_title from leaf → parent/mid nodes.
    # enrich_nodes() only touches leaf nodes; when AutoMergingRetriever
    # promotes a parent, its metadata has chapter=None / section_title=None
    # (showing as '?' in logs).  Walk each leaf's ancestry and fill blanks.
    node_index: dict = {n.node_id: n for n in all_nodes}
    for leaf in leaf_nodes:
        chapter = leaf.metadata.get("chapter")
        section = leaf.metadata.get("section_title")
        if not chapter and not section:
            continue
        parent_ref = leaf.parent_node
        while parent_ref:
            parent = node_index.get(parent_ref.node_id)
            if parent is None:
                break
            if not parent.metadata.get("chapter"):
                parent.metadata["chapter"] = chapter
            if not parent.metadata.get("section_title"):
                parent.metadata["section_title"] = section
            parent_ref = parent.parent_node
    storage_context = StorageContext.from_defaults(docstore=docstore)

    os.makedirs(persist_dir, exist_ok=True)
    storage_context.persist(persist_dir=persist_dir)

    return leaf_nodes, storage_context


if __name__ == "__main__":
    import sys
    from src.ingestion.parser import load_pdf_with_sections
    from src.ingestion.metadata import build_section_map_and_clean

    pdf_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/raw/rbi_kyc_master_direction.pdf"
    )

    print(f"Loading and extracting sections from {pdf_path}...")
    text_with_markers = load_pdf_with_sections(pdf_path)
    chapter_map, section_map, clean_text = build_section_map_and_clean(text_with_markers)
    print(f"  {len(clean_text):,} chars  |  {len(chapter_map)} chapters  |  {len(section_map)} sections\n")

    print("Chunking (or loading from storage/)...")
    leaf_nodes, storage_context = chunk_document(
        text=clean_text,
        filename=pdf_path,
        section_map=section_map,
        chapter_map=chapter_map,
        clean_text=clean_text,
    )

    all_nodes = list(storage_context.docstore.docs.values())
    print(f"Total nodes in docstore : {len(all_nodes)}")
    print(f"Leaf nodes (indexable)  : {len(leaf_nodes)}")
    print(f"Non-leaf (docstore only): {len(all_nodes) - len(leaf_nodes)}\n")

    # Verify Layer 1 — document-level metadata
    leaf = leaf_nodes[0]
    print("--- Layer 1 metadata (document-level) ---")
    for k in ("jurisdiction", "doc_type", "effective_date", "is_stale",
              "clearance_level", "citation_priority", "source_file"):
        print(f"  {k:20s}: {leaf.metadata.get(k)}")
    print()

    # Verify Layer 2 — chapter + section on first 5 leaves with values
    with_chapter = [n for n in leaf_nodes if n.metadata.get("chapter")]
    with_section = [n for n in leaf_nodes if n.metadata.get("section_title")]
    print(f"Leaves with chapter      : {len(with_chapter)} / {len(leaf_nodes)}")
    print(f"Leaves with section_title: {len(with_section)} / {len(leaf_nodes)}")
    print()
    print("--- First 5 leaves with chapter + section_title ---")
    shown = 0
    for n in leaf_nodes:
        if n.metadata.get("chapter") and n.metadata.get("section_title"):
            print(
                f"  [{n.metadata.get('chunk_index'):>4}]  "
                f"chapter={n.metadata.get('chapter')!r}\n"
                f"          section={n.metadata.get('section_title')!r}"
            )
            shown += 1
            if shown >= 5:
                break
    print()

    # Verify parent relationship
    parent_info = leaf.parent_node
    if parent_info:
        parent_node = storage_context.docstore.get_node(parent_info.node_id)
        # get_node returns BaseNode; cast to TextNode for .text access
        parent = parent_node  # type: ignore[assignment]
        leaf_text: str = leaf.get_content()
        parent_text: str = parent.get_content()  # type: ignore[attr-defined]
        print(f"leaf ({len(leaf_text)} chars): {leaf_text[:120]!r}")
        print(f"parent ({len(parent_text)} chars, first 200): {parent_text[:200]!r}")
        print(f"Leaf contained in parent: {leaf.text.strip() in parent_text}")  # type: ignore[attr-defined]
