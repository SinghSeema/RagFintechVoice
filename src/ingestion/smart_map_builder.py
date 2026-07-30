"""
SmartMapBuilder — builds a page-aware structural index of a PDF document.

This is the ingestion component for PageIndex RAG (vectorless retrieval).
No embeddings are produced here. The output is a SmartMap: a hierarchical
map of the document's sections with exact page boundaries and full section
text, serialised to a JSON sidecar.

At query time, PageIndexRetriever loads this map and uses an LLM to navigate
directly to the most relevant section — bypassing cosine similarity entirely.

Public API:
    build_smart_map(doc_id, pdf_path) -> SmartMap
    save_smart_map(smart_map, sidecar_dir)
    load_smart_map(doc_id, sidecar_dir) -> SmartMap
    smart_map_exists(doc_id, sidecar_dir) -> bool
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pdfplumber

logger = logging.getLogger(__name__)

# ── Constants (reuse same crop dimensions as parser.py) ───────────────────────
_HEADER_H = 40
_FOOTER_H = 40
_BOLD_RE   = re.compile(r"bold|(-b)|w-700|black", re.IGNORECASE)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class TOCEntry:
    section_id: str
    title: str
    level: int        # 1 = chapter, 2 = numbered section
    start_page: int
    end_page: int


@dataclass
class SmartMap:
    doc_id: str
    toc: List[TOCEntry]
    sections: Dict[str, str]          # section_id -> full section text
    toc_by_id: Dict[str, TOCEntry] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.toc_by_id:
            self.toc_by_id = {e.section_id: e for e in self.toc}


# ── Font helpers (same logic as parser.py) ────────────────────────────────────

def _is_fully_bold(word_list: list) -> bool:
    content = [w for w in word_list if w["text"].strip()]
    if not content:
        return False
    return all(_BOLD_RE.search(w.get("fontname", "")) for w in content)


def _is_bare_number(text: str) -> bool:
    return bool(re.fullmatch(r"(\d+\.)+\d*\.?\s*", text.strip()))


def _has_real_title(text: str) -> bool:
    remainder = re.sub(r"^(\d+\.)+\s*", "", text.strip())
    return bool(re.search(r"[a-zA-Z]{2,}", remainder))


def _is_numbered_section(line_text: str, line_is_bold: bool) -> bool:
    """True if this line looks like a numbered section heading."""
    return (
        line_is_bold
        and bool(re.match(r"^(\d+\.)+", line_text))
        and not _is_bare_number(line_text)
        and _has_real_title(line_text)
    )


# ── Core builder ──────────────────────────────────────────────────────────────

def build_smart_map(doc_id: str, pdf_path: str) -> SmartMap:
    """
    Parse a PDF page-by-page and build a SmartMap with:
      - A TOC entry per detected section (numbered bold headings)
      - Exact start_page / end_page for each section
      - Full section text (all lines from heading until next heading)

    Detection uses the same bold-font heuristic as parser.py. Chapter headings
    are also captured as level-1 TOC entries for completeness, but numbered
    sections (level-2) are the primary navigation targets for PageIndex.

    Args:
        doc_id:   Identifier for this document (typically the PDF stem name).
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        SmartMap with toc, sections, and toc_by_id populated.
    """
    toc: List[TOCEntry]      = []
    sections: Dict[str, str] = {}

    # Current section accumulator
    cur_id:     Optional[str]       = None
    cur_title:  str                 = ""
    cur_level:  int                 = 2
    cur_start:  int                 = 1
    cur_lines:  List[str]           = []
    sec_counter = 0

    # Chapter-level detection state
    _CHAPTER_RE = re.compile(
        r"^chapter\s*[-–—]?\s*(I{1,3}|IV|VI{0,3}|IX|XI{0,3}|XIV|XV|XVI{0,3}|XIX|XX)\s*$",
        re.IGNORECASE,
    )
    pending_chapter: Optional[str] = None

    def _close_section(end_page: int) -> None:
        nonlocal cur_id
        if cur_id is None:
            return
        toc.append(TOCEntry(
            section_id=cur_id,
            title=cur_title,
            level=cur_level,
            start_page=cur_start,
            end_page=end_page,
        ))
        sections[cur_id] = "\n".join(cur_lines).strip()
        cur_id = None

    def _open_section(title: str, level: int, page_num: int) -> None:
        nonlocal cur_id, cur_title, cur_level, cur_start, cur_lines, sec_counter
        sec_counter += 1
        cur_id    = f"sec_{sec_counter:03d}"
        cur_title = title
        cur_level = level
        cur_start = page_num
        cur_lines = [title]

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages, 1):
            # Crop headers/footers
            words = page.extract_words(extra_attrs=["fontname", "size"])
            words = [
                w for w in words
                if w["top"] >= _HEADER_H
                and w["bottom"] <= page.height - _FOOTER_H
            ]
            if not words:
                continue

            # Group words into visual lines by y-position
            lines: dict[float, list] = {}
            for w in words:
                lines.setdefault(round(w["top"], 0), []).append(w)

            for top in sorted(lines):
                line_words = lines[top]
                line_text  = " ".join(w["text"] for w in line_words).strip()
                if not line_text:
                    continue

                line_is_bold    = _is_fully_bold(line_words)
                is_chapter_line = line_is_bold and bool(_CHAPTER_RE.match(line_text))

                # ── Resolve pending chapter title ──────────────────────────
                if pending_chapter is not None:
                    if line_is_bold and not is_chapter_line and not re.match(r"^(\d+\.)+", line_text):
                        chapter_label = f"{pending_chapter} — {line_text}"
                        _close_section(page_num - 1 if page_num > 1 else 1)
                        _open_section(chapter_label, level=1, page_num=page_num)
                        pending_chapter = None
                        continue
                    else:
                        _close_section(page_num - 1 if page_num > 1 else 1)
                        _open_section(pending_chapter, level=1, page_num=page_num)
                        pending_chapter = None

                # ── Chapter number line ────────────────────────────────────
                if is_chapter_line:
                    pending_chapter = line_text
                    continue

                # ── Numbered section heading ───────────────────────────────
                if _is_numbered_section(line_text, line_is_bold):
                    _close_section(page_num - 1 if page_num > cur_start else page_num)
                    _open_section(line_text, level=2, page_num=page_num)
                    continue

                # ── Body text — accumulate into current section ────────────
                if cur_id is not None:
                    cur_lines.append(line_text)
                else:
                    # Pre-section preamble: create a header section on first text
                    if line_text and not line_is_bold:
                        _open_section("Document Preamble", level=2, page_num=page_num)
                        cur_lines.append(line_text)

        # Close the last open section
        _close_section(total_pages)

    # Edge case: if nothing was found, create a single catch-all section
    if not toc:
        logger.warning(f"[SmartMap] No sections detected in {pdf_path}; creating single section")
        raw_text = _extract_plain_text(pdf_path)
        toc.append(TOCEntry("sec_001", "Full Document", 2, 1, 1))
        sections["sec_001"] = raw_text[:8000]

    smart_map = SmartMap(doc_id=doc_id, toc=toc, sections=sections)
    logger.info(
        f"[SmartMap] Built for {doc_id!r}: "
        f"{len(toc)} sections, {sum(len(v) for v in sections.values())} chars total"
    )
    return smart_map


def _extract_plain_text(pdf_path: str) -> str:
    """Fallback: plain text extraction without structure detection."""
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            cropped = page.crop((0, _HEADER_H, page.width, page.height - _FOOTER_H))
            text = cropped.extract_text()
            if text:
                parts.append(text.strip())
    return "\n\n".join(parts)


# ── Sidecar persistence ───────────────────────────────────────────────────────

def _sidecar_path(doc_id: str, sidecar_dir: str) -> Path:
    return Path(sidecar_dir) / f"smartmap_{doc_id}.json"


def save_smart_map(smart_map: SmartMap, sidecar_dir: str = "qdrant_storage") -> Path:
    """Serialise SmartMap to JSON sidecar. Returns the path written."""
    path = _sidecar_path(smart_map.doc_id, sidecar_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "doc_id":   smart_map.doc_id,
        "toc":      [asdict(e) for e in smart_map.toc],
        "sections": smart_map.sections,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info(f"[SmartMap] Saved to {path}")
    return path


def load_smart_map(doc_id: str, sidecar_dir: str = "qdrant_storage") -> SmartMap:
    """Load SmartMap from JSON sidecar. Raises FileNotFoundError if missing."""
    path = _sidecar_path(doc_id, sidecar_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"SmartMap sidecar not found: {path}\n"
            f"Run: python scripts/ingest_smartmap.py --doc-id {doc_id}"
        )
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    toc = [TOCEntry(**e) for e in payload["toc"]]
    return SmartMap(
        doc_id=payload["doc_id"],
        toc=toc,
        sections=payload["sections"],
    )


def smart_map_exists(doc_id: str, sidecar_dir: str = "qdrant_storage") -> bool:
    return _sidecar_path(doc_id, sidecar_dir).exists()
