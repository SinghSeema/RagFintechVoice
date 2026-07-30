"""
Task 3 — Two-layer metadata for every chunk.

Layer 1 — Document-level (set before chunking, propagates to all nodes):
    extract_metadata(filename, text) -> dict
    Fields: jurisdiction, doc_type, effective_date, is_stale,
            clearance_level, citation_priority, source_file

Layer 2 — Chunk-level (applied to leaf nodes after chunking):
    enrich_nodes(leaf_nodes, section_map) -> None
    Fields: chunk_index, section_title
    section_title resolved via binary search on char offsets.

Helper:
    build_section_map_and_clean(text_with_markers) -> (section_map, clean_text)
    Parses <<SECTION:title>> markers from load_pdf_with_sections() output,
    computes their character offsets relative to the clean (stripped) text,
    returns both. The clean_text is safe to pass to LlamaIndex.

Embed visibility:
    Only jurisdiction, doc_type, section_title go into the embedding vector.
    Operational fields (chunk_index, is_stale, clearance_level, etc.) are
    excluded via excluded_embed_metadata_keys.
"""

import os
import re


# ── Layer 1 helpers ────────────────────────────────────────────────────────────

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

_JURISDICTION_PREFIXES = {
    "rbi": "RBI",
    "sebi": "SEBI",
    "fema": "FEMA",
}

# Ordered so longer/more-specific keywords match before shorter ones
_DOC_TYPE_KEYWORDS: list[tuple[str, str]] = [
    ("master_direction",  "master_direction"),
    ("master_directions", "master_direction"),
    ("circular",          "circular"),
    ("guidelines",        "guideline"),
    ("guideline",         "guideline"),
    ("notification",      "notification"),
    ("regulations",       "regulation"),
    ("regulation",        "regulation"),
    ("directions",        "direction"),
    ("direction",         "direction"),
    ("lending",           "guideline"),   # e.g. priority_sector_lending
]

_MONTHS_PAT = (
    r"january|february|march|april|may|june|"
    r"july|august|september|october|november|december"
)


def _parse_jurisdiction(basename: str) -> str:
    prefix = basename.lower().split("_")[0]
    return _JURISDICTION_PREFIXES.get(prefix, "UNKNOWN")


def _parse_doc_type(basename: str) -> str:
    lower = basename.lower()
    for keyword, doc_type in _DOC_TYPE_KEYWORDS:
        if keyword in lower:
            return doc_type
    return "unknown"


def _parse_effective_date(text: str) -> str | None:
    """
    Return ISO 'YYYY-MM-DD' for the document's original issue date, or None.

    Search order (preference for explicitly-labelled dates first):
    1. 'dated DD Month YYYY' / 'issued on DD Month YYYY'
    2. 'Month DD, YYYY'   (common in RBI cover pages)
    3. 'DD Month YYYY'
    All patterns searched within the first 3,000 characters only.
    """
    head = text[:3000]

    # Pattern 1: "dated 25 February 2016" or "issued on 25 February 2016"
    m = re.search(
        rf"(?:dated|issued\s+on)\s+(\d{{1,2}})\s+({_MONTHS_PAT})\s+(\d{{4}})",
        head, re.IGNORECASE,
    )
    if m:
        day, mon, year = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        return f"{year:04d}-{_MONTH_MAP[mon]:02d}-{day:02d}"

    # Pattern 2: "February 25, 2016"
    m = re.search(
        rf"({_MONTHS_PAT})\s+(\d{{1,2}}),?\s+(\d{{4}})",
        head, re.IGNORECASE,
    )
    if m:
        mon, day, year = m.group(1).lower(), int(m.group(2)), int(m.group(3))
        return f"{year:04d}-{_MONTH_MAP[mon]:02d}-{day:02d}"

    # Pattern 3: "25 February 2016"
    m = re.search(
        rf"(\d{{1,2}})\s+({_MONTHS_PAT})\s+(\d{{4}})",
        head, re.IGNORECASE,
    )
    if m:
        day, mon, year = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        return f"{year:04d}-{_MONTH_MAP[mon]:02d}-{day:02d}"

    return None


def extract_metadata(filename: str, text: str) -> dict:
    """
    Extract document-level metadata from filename pattern and text content.

    Set this dict as Document.metadata before chunking — LlamaIndex
    propagates it to every child node (leaf, mid, parent) automatically.

    Args:
        filename: Path or basename of the source PDF.
        text:     Plain text content (from load_pdf or clean_text).

    Returns:
        Dict with keys: jurisdiction, doc_type, effective_date, is_stale,
        clearance_level, citation_priority, source_file.
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    return {
        "jurisdiction":      _parse_jurisdiction(basename),
        "doc_type":          _parse_doc_type(basename),
        "effective_date":    _parse_effective_date(text),
        "is_stale":          False,
        "clearance_level":   1,
        "citation_priority": 1,
        "source_file":       os.path.basename(filename),
    }


# ── Marker parsing ────────────────────────────────────────────────────────────

_ALL_MARKERS_RE = re.compile(r"<<(CHAPTER|SECTION):([^>]+)>>\n?")


def build_section_map_and_clean(text_with_markers: str) -> tuple[list, list, str]:
    """
    Parse <<CHAPTER:...>> and <<SECTION:...>> markers from load_pdf_with_sections().

    Computes each marker's character offset relative to the clean
    (marker-stripped) text. Offsets are used in enrich_nodes() to assign
    chapter and section_title to each leaf node via text-position lookup.

    Args:
        text_with_markers: Output of load_pdf_with_sections().

    Returns:
        (chapter_map, section_map, clean_text)

        chapter_map:  list of (char_offset_in_clean_text, chapter_label)
                      e.g. [(7200, 'Chapter VI — Customer Due Diligence')]
        section_map:  list of (char_offset_in_clean_text, section_title)
                      e.g. [(7760, '1. Short Title and Commencement.')]
        clean_text:   text with all markers stripped — safe for LlamaIndex.
    """
    chapter_map: list[tuple[int, str]] = []
    section_map: list[tuple[int, str]] = []
    offset_adj = 0

    for m in _ALL_MARKERS_RE.finditer(text_with_markers):
        clean_offset = m.start() - offset_adj
        kind, title  = m.group(1), m.group(2)
        if kind == "CHAPTER":
            chapter_map.append((clean_offset, title))
        else:
            section_map.append((clean_offset, title))
        offset_adj += len(m.group(0))

    clean_text = _ALL_MARKERS_RE.sub("", text_with_markers)
    return chapter_map, section_map, clean_text


# ── Layer 2 — chunk-level enrichment ──────────────────────────────────────────

def _resolve_from_map(global_pos: int, offsets: list, titles: list) -> str | None:
    """Binary search: return the title whose offset is last <= global_pos."""
    lo, hi, result = 0, len(offsets) - 1, None
    while lo <= hi:
        mid_idx = (lo + hi) // 2
        if offsets[mid_idx] <= global_pos:
            result = titles[mid_idx]
            lo = mid_idx + 1
        else:
            hi = mid_idx - 1
    return result


def enrich_nodes(
    leaf_nodes: list,
    clean_text: str,
    section_map: list,
    chapter_map: list | None = None,
) -> None:
    """
    Add chunk_index, section_title, and chapter to each leaf node in-place.

    Offset strategy:
        HierarchicalNodeParser's start_char_idx is local to the parent chunk,
        not a global document offset. We locate each leaf's text in the full
        clean_text via str.find() to get a reliable global offset, then
        binary-search both maps.

    chapter field:
        Citation-display only — excluded from embedding to avoid diluting
        the semantic signal. May be None for documents without chapters
        (e.g. rbi_housing_loan_guidelines.pdf).

    Args:
        leaf_nodes:  List of LlamaIndex TextNode leaf objects.
        clean_text:  Full marker-stripped document text.
        section_map: (offset, title) pairs from build_section_map_and_clean().
        chapter_map: (offset, label) pairs from build_section_map_and_clean().
                     Pass None to skip chapter enrichment.
    """
    s_offsets = [off for off, _ in section_map]
    s_titles  = [t   for _, t  in section_map]
    c_offsets = [off for off, _ in (chapter_map or [])]
    c_titles  = [t   for _, t  in (chapter_map or [])]
    search_from = 0

    for i, node in enumerate(leaf_nodes):
        node.metadata["chunk_index"] = i

        # Locate leaf in full document (global offset)
        node_text  = node.text.strip()
        global_pos = clean_text.find(node_text, search_from)
        if global_pos == -1:
            global_pos = clean_text.find(node_text)   # overlap fallback

        if global_pos == -1:
            node.metadata["section_title"] = None
            node.metadata["chapter"]       = None
        else:
            search_from = global_pos
            node.metadata["section_title"] = (
                _resolve_from_map(global_pos, s_offsets, s_titles)
                if s_offsets else None
            )
            node.metadata["chapter"] = (
                _resolve_from_map(global_pos, c_offsets, c_titles)
                if c_offsets else None
            )

        # Embedding visibility:
        #   IN  embedding : jurisdiction, doc_type, section_title
        #   OUT of embedding: everything else (chapter is citation-only)
        node.excluded_embed_metadata_keys = [
            "chunk_index",
            "chapter",           # citation display only — not semantic
            "is_stale",
            "clearance_level",
            "citation_priority",
            "effective_date",
            "source_file",
        ]


if __name__ == "__main__":
    from src.ingestion.parser import load_pdf_with_sections

    # --- Test 1: Real KYC PDF ---
    path = "data/raw/rbi_kyc_master_direction.pdf"
    text_marked = load_pdf_with_sections(path)
    chapter_map, section_map, clean_text = build_section_map_and_clean(text_marked)
    meta = extract_metadata(path, clean_text)

    print("=== KYC Master Direction ===")
    print(f"jurisdiction     : {meta['jurisdiction']}")
    print(f"doc_type         : {meta['doc_type']}")
    print(f"effective_date   : {meta['effective_date']}")
    print(f"is_stale         : {meta['is_stale']}")
    print(f"clearance_level  : {meta['clearance_level']}")
    print(f"citation_priority: {meta['citation_priority']}")
    print(f"sections found   : {len(section_map)}")
    print(f"first 3 sections : {[t for _, t in section_map[:3]]}")
    print()

    # --- Test 2: Synthetic SEBI circular filename ---
    sebi_meta = extract_metadata("sebi_investor_protection_circular.pdf", "dummy text")
    print("=== Synthetic SEBI circular ===")
    print(f"jurisdiction: {sebi_meta['jurisdiction']}")
    print(f"doc_type    : {sebi_meta['doc_type']}")
    print()

    # --- Test 3: Housing loan guidelines ---
    path2 = "data/raw/rbi_housing_loan_guidelines.pdf"
    text_marked2 = load_pdf_with_sections(path2)
    _, __, clean2 = build_section_map_and_clean(text_marked2)
    meta2 = extract_metadata(path2, clean2)
    print("=== Housing Loan Guidelines ===")
    print(f"jurisdiction: {meta2['jurisdiction']}")
    print(f"doc_type    : {meta2['doc_type']}")
    print(f"effective_date: {meta2['effective_date']}")
