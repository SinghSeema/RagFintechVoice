"""
Rebuild section summary nodes in fintech_rag_agentic without re-running Groq.

Steps:
  1. Delete all existing summary-strategy points from Qdrant.
  2. Generate fresh sliding-window summary nodes from the PDF text.
  3. Embed them with BGE-M3 and upsert into Qdrant.
  4. Update the local sidecar JSON.
"""
from __future__ import annotations

import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from collections import defaultdict

from llama_index.core.schema import TextNode
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct

from src.ingestion.parser import load_pdf_with_sections
from src.ingestion.metadata import build_section_map_and_clean
from src.ingestion.agentic_chunker import _is_noise_paragraph, _strip_amendment_lines, _resolve_from_map
from src.retrieval.revamp_vector_store import load_node_sidecar


COLLECTION = "fintech_rag_agentic"
SIDECAR    = f"qdrant_storage/revamp_nodes_{COLLECTION}.json"

_MAX_SUMMARY_CHARS = 1800
_SUMMARY_STRIDE    = 900


def _embed(texts: list[str]) -> list[list[float]]:
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    out = model.encode(texts, batch_size=12, max_length=512)
    return out["dense_vecs"].tolist()


def build_summary_nodes(chapter_map, section_map, clean_text) -> list[TextNode]:
    c_offsets = [o for o, _ in chapter_map]
    c_titles  = [t for _, t in chapter_map]

    # Build paragraphs (same logic as agentic_chunker)
    _MAX_PARA_CHARS = 800
    para_with_offsets: list[tuple[str, int]] = []
    for sec_start, sec_end in zip(
        [o for o, _ in section_map],
        [o for o, _ in section_map[1:]] + [len(clean_text)],
    ):
        block = clean_text[sec_start:sec_end]
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        chunk_lines, chunk_len, chunk_start = [], 0, sec_start
        for line in lines:
            if chunk_len + len(line) + 1 > _MAX_PARA_CHARS and chunk_lines:
                para_with_offsets.append(("\n".join(chunk_lines), chunk_start))
                approx = clean_text.find(line, chunk_start)
                chunk_start = approx if approx != -1 else chunk_start
                chunk_lines, chunk_len = [line], len(line)
            else:
                if not chunk_lines:
                    approx = clean_text.find(line, sec_start)
                    chunk_start = approx if approx != -1 else sec_start
                chunk_lines.append(line)
                chunk_len += len(line) + 1
        if chunk_lines:
            para_with_offsets.append(("\n".join(chunk_lines), chunk_start))

    # Also handle text before first section marker
    if section_map:
        pre = clean_text[:section_map[0][0]]
        lines = [ln.strip() for ln in pre.split("\n") if ln.strip()]
        if lines:
            para_with_offsets.insert(0, ("\n".join(lines), 0))

    # Filter noise
    filtered: list[tuple[str, int, str, str]] = []
    for para, offset in para_with_offsets:
        if _is_noise_paragraph(para):
            continue
        clean_para = _strip_amendment_lines(para)
        if not clean_para or len(clean_para) < 30:
            continue
        chapter = _resolve_from_map(offset, c_offsets, c_titles) or ""
        section = _resolve_from_map(
            offset,
            [o for o, _ in section_map],
            [t for _, t in section_map],
        ) or ""
        filtered.append((clean_para, offset, chapter, section))

    # Group by (section, chapter) and slide window
    section_para_groups: dict[tuple, list[str]] = defaultdict(list)
    for para_text, _offset, chapter, section in filtered:
        key = (section or chapter or "document", chapter)
        section_para_groups[key].append(para_text)

    nodes: list[TextNode] = []
    for (section, chapter), paras in section_para_groups.items():
        combined = "\n\n".join(paras)
        if len(combined) < 80:
            continue
        label = f"[{section}]\n" if section else ""
        start = 0
        while start < len(combined):
            chunk = combined[start : start + _MAX_SUMMARY_CHARS]
            snode = TextNode(
                node_id=str(uuid.uuid4()),
                text=label + chunk,
                metadata={
                    "source":     "rbi_kyc_master_direction",
                    "strategy":   "summary",
                    "chapter":    chapter,
                    "section":    section,
                    "char_count": len(chunk),
                    "has_xref":   False,
                },
            )
            snode.excluded_embed_metadata_keys = [
                "strategy", "chapter", "char_count", "has_xref",
            ]
            nodes.append(snode)
            if start + _MAX_SUMMARY_CHARS >= len(combined):
                break
            start += _SUMMARY_STRIDE
    return nodes


def main() -> None:
    # 1. Parse PDF
    print("Parsing PDF...")
    pdf_path = "data/raw/rbi_kyc_master_direction.pdf"
    text_marked = load_pdf_with_sections(pdf_path)
    chapter_map, section_map, clean_text = build_section_map_and_clean(text_marked)
    print(f"  {len(section_map)} sections, {len(chapter_map)} chapters")

    # 2. Build new summary nodes
    print("Building sliding-window summary nodes...")
    new_nodes = build_summary_nodes(chapter_map, section_map, clean_text)
    print(f"  {len(new_nodes)} summary nodes")

    # 3. Embed
    print("Embedding summary nodes...")
    texts = [n.text for n in new_nodes]
    vectors = _embed(texts)
    print(f"  Embedded {len(vectors)} nodes")

    # 4. Connect to Qdrant
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

    # 5. Delete old summary points
    print("Deleting old summary nodes from Qdrant...")
    client.delete(
        collection_name=COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="strategy", match=MatchValue(value="summary"))]
        ),
    )
    print("  Deleted")

    # 6. Upsert new points
    print("Upserting new summary nodes...")
    points = []
    for node, vec in zip(new_nodes, vectors):
        points.append(PointStruct(
            id=node.node_id,
            vector=vec,
            payload={
                "source":     node.metadata["source"],
                "strategy":   node.metadata["strategy"],
                "chapter":    node.metadata["chapter"],
                "section":    node.metadata["section"],
                "char_count": node.metadata["char_count"],
                "has_xref":   node.metadata["has_xref"],
                "_node_content": json.dumps({"text": node.text, "id_": node.node_id}),
            },
        ))
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"  Upserted {len(points)} points")

    # 7. Update sidecar
    print("Updating sidecar...")
    with open(SIDECAR) as f:
        existing = json.load(f)
    # Keep non-summary nodes
    kept = [n for n in existing if n.get("metadata", {}).get("strategy") != "summary"]
    # Append new summary nodes
    for node in new_nodes:
        kept.append({"node_id": node.node_id, "text": node.text, "metadata": node.metadata})
    with open(SIDECAR, "w") as f:
        json.dump(kept, f)
    print(f"  Sidecar: {len(kept)} total nodes ({len(new_nodes)} summary + {len(kept)-len(new_nodes)} proposition)")

    # Count by section
    from collections import Counter
    sec_counts = Counter(n.metadata.get("section","?") for n in new_nodes)
    print("\nSummary nodes by section (top 10 by count):")
    for sec, cnt in sec_counts.most_common(10):
        print(f"  {cnt}x  {sec[:70]}")


if __name__ == "__main__":
    main()
