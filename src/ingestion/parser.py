"""
Task 1 — PDF loading with pdfplumber.

Two public functions:

  load_pdf(path) -> str
      Plain text extraction. Strips page headers (top 40px) and footers
      (bottom 40px) via page.crop(). Skips empty/scanned pages.

  load_pdf_with_sections(path) -> str
      Single-pass extraction with bold-heading detection.
      Detects two structural levels and injects markers into the text:

      Chapter markers  <<CHAPTER:Chapter I — PRELIMINARY>>
          Detected when a fully-bold line matches the Roman-numeral chapter
          pattern (e.g. 'CHAPTER I', 'CHAPTER – III', 'Chapter V').
          The title on the immediately-following bold line is joined with
          a dash: 'Chapter I — PRELIMINARY'.
          Handles two dash variants: no-dash ('CHAPTER I') and
          em/en-dash ('CHAPTER – I').

      Section markers  <<SECTION:1. Short Title and Commencement.>>
          Detected when a fully-bold line starts with a digit+dot pattern
          (e.g. '1.', '3.2', '8.1 Farm Credit'). Multi-line bold headings
          are accumulated and flushed as one section title.

      Both marker types are stripped by metadata.build_section_map_and_clean()
      which returns clean text safe for LlamaIndex + offset maps for both
      chapter and section lookups.

      Heading text is kept in the body text as well as the marker so that
      embeddings retain heading context.
"""

import re
import pdfplumber


_HEADER_H = 40   # pixels to crop from top of every page
_FOOTER_H = 40   # pixels to crop from bottom of every page

# ── Bold-heading helpers (ported from rbipdftest_5.py) ────────────────────────

_BOLD_RE    = re.compile(r'bold|(-b)|w-700|black', re.IGNORECASE)

# Matches 'CHAPTER I', 'Chapter VI', 'CHAPTER – III', 'CHAPTER - IV'
# Roman numerals up to 39 (XXXIX) cover any realistic document.
_CHAPTER_RE = re.compile(
    r'^chapter\s*[-–—]?\s*(I{1,3}|IV|VI{0,3}|IX|XI{0,3}|XIV|XV|XVI{0,3}|XIX|XX)\s*$',
    re.IGNORECASE,
)


def _is_fully_bold(word_list: list) -> bool:
    """True only if every non-whitespace word carries a bold font tag."""
    if not word_list:
        return False
    content = [w for w in word_list if w["text"].strip()]
    if not content:
        return False
    return all(_BOLD_RE.search(w.get("fontname", "")) for w in content)


def _is_bare_number(text: str) -> bool:
    """True if text is a lone section number with no descriptive title."""
    return bool(re.fullmatch(r"(\d+\.)+\d*\.?\s*", text.strip()))


def _has_real_title(text: str) -> bool:
    """True if text contains real alphabetic words after any leading number."""
    remainder = re.sub(r"^(\d+\.)+\s*", "", text.strip())
    return bool(re.search(r"[a-zA-Z]{2,}", remainder))


def _flush_section(header_buffer: list) -> str | None:
    """
    Finalise accumulated bold lines into a section title.
    Returns None if the buffer is a bare number or has no real words.
    """
    if not header_buffer:
        return None
    joined = " ".join(header_buffer)
    if _is_bare_number(joined):
        return None
    if not _has_real_title(joined):
        return None
    return joined


# ── Core extraction ────────────────────────────────────────────────────────────

def load_pdf(path: str) -> str:
    """
    Extract plain text from a PDF, stripping page headers and footers.
    Empty or scanned pages are silently skipped.
    Pages are joined with a double newline.
    """
    pages_text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            cropped = page.crop(
                (0, _HEADER_H, page.width, page.height - _FOOTER_H)
            )
            text = cropped.extract_text()
            if text and text.strip():
                pages_text.append(text.strip())
    return "\n\n".join(pages_text)


def load_pdf_with_sections(path: str) -> str:
    """
    Single-pass PDF extraction with chapter and section heading detection.

    Opens the PDF once, crops headers/footers, extracts words with font
    metadata, groups into lines, then classifies each bold line as:

      - A chapter heading  ('CHAPTER I', 'CHAPTER – III' etc.)
      - A section heading  ('1. Short Title', '8.1 Farm Credit' etc.)
      - A bold body line   (bold but neither of the above — kept as text)

    Chapter detection:
        When a line matches _CHAPTER_RE, we peek at the next bold line
        to grab the chapter title (e.g. 'PRELIMINARY'). The combined
        label 'Chapter I — PRELIMINARY' is injected as <<CHAPTER:...>>.
        The chapter number line and title line are also kept in the body
        text for embedding context.

    Section detection:
        Fully-bold lines starting with digit+dot are accumulated into a
        buffer (handles multi-line headings). When the buffer is flushed
        a <<SECTION:...>> marker is injected.

    TOC suppression:
        Chapter headings appearing before the first non-bold body text
        are treated as table-of-contents entries and skipped. Only the
        second (real) occurrence of each chapter heading is used.

    Args:
        path: Path to PDF file.

    Returns:
        Text with <<CHAPTER:...>> and <<SECTION:...>> markers injected.
    """
    parts: list[str] = []
    section_buffer: list[str] = []

    # Chapter state machine
    # pending_chapter: set when chapter line seen, waiting for title line
    pending_chapter: str | None = None

    # Dedup by normalised chapter NUMBER (not full label).
    # Maps norm_num -> index in `parts` where the <<CHAPTER:...>> marker sits.
    # When a titled version arrives after an untitled one, we REPLACE the
    # existing marker in-place so the final output always uses the titled form.
    seen_chapter_nums: dict = {}      # norm_num -> parts index

    body_started: bool = False        # True after first non-bold text line

    def _normalize_chapter_num(text: str) -> str:
        """Canonical form for dedup: strip dashes, collapse spaces, uppercase."""
        return re.sub(r'\s+', ' ', re.sub(r'\s*[-–—]\s*', ' ', text).strip()).upper()

    def _flush_section_buffer(buf: list[str]) -> None:
        """Inject <<SECTION:...>> marker + keep heading lines in body."""
        title = _flush_section(buf)
        if title:
            parts.append(f"<<SECTION:{title}>>")
        for line in buf:
            parts.append(line)

    def _emit_chapter(chapter_num: str, chapter_title: str) -> None:
        """
        Inject <<CHAPTER:...>> marker + keep heading lines in body.

        Dedup by chapter number (normalised). Rules:
        - First occurrence (TOC, usually untitled): emit and record parts index.
        - Later occurrence with title (real body): REPLACE the existing marker
          with the titled version; don't emit a second marker.
        - Later occurrence also untitled (true duplicate): skip silently.
        """
        label    = f"{chapter_num} — {chapter_title}" if chapter_title else chapter_num
        norm_num = _normalize_chapter_num(chapter_num)

        if norm_num in seen_chapter_nums:
            existing_idx = seen_chapter_nums[norm_num]
            # Body occurrence: blank the old marker (removes TOC position)
            # and append a fresh one at the current (body) position.
            # This ensures chapter offsets in the final text sit at the
            # real section boundary, not the TOC entry.
            parts[existing_idx] = ""   # blank TOC marker — harmless empty line
            seen_chapter_nums[norm_num] = len(parts)
            parts.append(f"<<CHAPTER:{label}>>")
        else:
            seen_chapter_nums[norm_num] = len(parts)
            parts.append(f"<<CHAPTER:{label}>>")

        # Always keep heading lines in body text for embedding context
        parts.append(chapter_num)
        if chapter_title:
            parts.append(chapter_title)

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(extra_attrs=["fontname", "size"])
            words = [
                w for w in words
                if w["top"] >= _HEADER_H
                and w["bottom"] <= page.height - _FOOTER_H
            ]
            if not words:
                continue

            lines: dict[float, list] = {}
            for w in words:
                lines.setdefault(round(w["top"], 0), []).append(w)

            for top in sorted(lines):
                line_words = lines[top]
                line_text = " ".join(w["text"] for w in line_words).strip()
                if not line_text:
                    continue

                line_is_bold    = _is_fully_bold(line_words)
                is_chapter_line = line_is_bold and bool(_CHAPTER_RE.match(line_text))
                starts_with_num = bool(re.match(r"^(\d+\.)+", line_text))

                # ── Chapter title line (follows a pending chapter number) ──
                if pending_chapter is not None:
                    if line_is_bold and not is_chapter_line and not starts_with_num:
                        # This bold line is the chapter title
                        if body_started:
                            _emit_chapter(pending_chapter, line_text)
                        pending_chapter = None
                        continue
                    else:
                        # No title line found — emit chapter with no title
                        if body_started:
                            _emit_chapter(pending_chapter, "")
                        pending_chapter = None
                        # Fall through to process current line normally

                # ── Chapter number line ────────────────────────────────────
                if is_chapter_line:
                    if section_buffer:
                        _flush_section_buffer(section_buffer)
                        section_buffer = []
                    pending_chapter = line_text
                    continue

                # ── Numbered section heading ───────────────────────────────
                if line_is_bold and starts_with_num:
                    if section_buffer:
                        _flush_section_buffer(section_buffer)
                    section_buffer = [line_text]
                    continue

                # ── Bold continuation of a multi-line section heading ──────
                if line_is_bold and section_buffer:
                    section_buffer.append(line_text)
                    continue

                # ── Body text (bold with no special role, or non-bold) ─────
                if section_buffer:
                    _flush_section_buffer(section_buffer)
                    section_buffer = []
                if not line_is_bold:
                    body_started = True
                parts.append(line_text)

    # Final flushes
    if pending_chapter is not None:
        if body_started:
            _emit_chapter(pending_chapter, "")
    if section_buffer:
        _flush_section_buffer(section_buffer)

    return "\n".join(parts)


if __name__ == "__main__":
    for pdf_path in [
        "data/raw/rbi_kyc_master_direction.pdf",
        "data/raw/rbi_priority_sector_lending.pdf",
        "data/raw/rbi_housing_loan_guidelines.pdf",
    ]:
        print(f"\n{'='*60}")
        print(f"  {pdf_path}")
        print('='*60)
        text_marked = load_pdf_with_sections(pdf_path)

        chapters = re.findall(r"<<CHAPTER:([^>]+)>>", text_marked)
        sections = re.findall(r"<<SECTION:([^>]+)>>", text_marked)

        print(f"  Chapters : {len(chapters)}")
        for c in chapters[:6]:
            print(f"    {c!r}")

        print(f"  Sections : {len(sections)}")
        for s in sections[:6]:
            print(f"    {s!r}")
