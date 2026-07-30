"""
PageIndexRetriever — vectorless LLM-guided TOC navigation.

This retriever does NOT use embeddings or cosine similarity.
Instead it:
  1. Loads the pre-built SmartMap (TOC + section text) from the JSON sidecar
  2. Sends the TOC as a compact JSON list to Groq 8B
  3. LLM reasons about which section best answers the query and returns section_id
  4. Full section text + exact page range is returned as a HybridRetrievalResult

Why this outperforms vector search for regulatory text:
  - RBI KYC document has numbered sections (16, 16.1, 56...) that vector search
    may semantically conflate
  - LLM navigation picks the structurally correct section (e.g., Section 16 for
    simplified KYC, not Section 15 which also mentions low-risk customers)
  - Page numbers are exact — never split across chunk boundaries

Public API:
    PageIndexRetriever(doc_id, smartmap_dir, nav_model, groq_api_key)
    retriever.retrieve(query, top_k=1) -> List[HybridRetrievalResult]
    retriever.retrieve_from_doc(query, doc_id) -> HybridRetrievalResult
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import List, Optional

from config.revamp_settings import revamp_settings
from src.ingestion.smart_map_builder import SmartMap, TOCEntry, load_smart_map
from src.retrieval.hybrid_base import HybridBaseRetriever, HybridRetrievalResult

logger = logging.getLogger(__name__)

# ── Navigation prompt ──────────────────────────────────────────────────────────

_NAVIGATION_PROMPT = """\
You are navigating a regulatory document's Table of Contents to find the section \
most likely to contain a direct answer to a user query.

Each entry has: section_id, title, pages, and a short preview of its content.

TABLE OF CONTENTS (JSON):
{toc_json}

USER QUERY:
{query}

Instructions:
- Read the title AND the preview of each entry before deciding.
- Prefer specific numbered sections (e.g. "3. Definitions", "41. Accounts of PEPs") \
  over broad chapter headings (e.g. "Chapter VI").
- If the query asks for a definition or list (e.g. "what are OVDs", "what documents"), \
  prefer the Definitions section or the section specifically about that procedure.
- Return ONLY valid JSON on a single line: \
  {{"section_id": "sec_NNN", "reasoning": "one sentence explaining why this section"}}
- Do NOT answer the query. Only identify the section_id.
"""

# Fallback: if the LLM returns something unparseable, pick the first section
_FALLBACK_SECTION = "sec_001"


# ── Retriever ──────────────────────────────────────────────────────────────────

class PageIndexRetriever(HybridBaseRetriever):
    """
    Vectorless PageIndex retriever.

    Loads SmartMap from sidecar, uses Groq LLM to navigate the TOC,
    returns full section text with page citations.
    """

    def __init__(
        self,
        doc_id:       Optional[str] = None,
        smartmap_dir: Optional[str] = None,
        nav_model:    Optional[str] = None,
        groq_api_key: Optional[str] = None,
    ) -> None:
        self.doc_id       = doc_id       or revamp_settings.PAGEINDEX_DOC_ID
        self.smartmap_dir = smartmap_dir or revamp_settings.SMARTMAP_DIR
        self.nav_model    = nav_model    or revamp_settings.PAGEINDEX_NAV_MODEL
        self.api_key      = groq_api_key or revamp_settings.GROQ_API_KEY

        # Lazy-load SmartMap on first use (avoid loading at import time)
        self._smart_map: Optional[SmartMap] = None

        # Lazy-load Groq LLM client
        self._llm = None

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_smart_map(self) -> SmartMap:
        if self._smart_map is None:
            self._smart_map = load_smart_map(self.doc_id, self.smartmap_dir)
            logger.info(
                f"[PageIndex] SmartMap loaded: doc={self.doc_id!r} "
                f"sections={len(self._smart_map.toc)}"
            )
        return self._smart_map

    def _get_llm(self):
        if self._llm is None:
            from llama_index.llms.groq import Groq
            self._llm = Groq(
                model=self.nav_model,
                api_key=self.api_key,
                temperature=0.0,
                max_tokens=150,
            )
        return self._llm

    def _build_toc_json(self, smart_map: SmartMap) -> str:
        """
        Compact TOC representation sent to the navigation LLM.

        Filters out noise entries so the LLM navigates real content:
          - Skips entries from the document's own TOC table (page 1-3, short content)
          - Skips entries with < 200 chars (heading-only, no body text)
          - Adds a 120-char content preview so the LLM can confirm relevance
        """
        _MIN_CONTENT_CHARS = 200
        _TOC_PAGE_CUTOFF   = 3    # pages 1-3 are typically the document's TOC table

        entries = []
        for e in smart_map.toc:
            content = smart_map.sections.get(e.section_id, "")
            # Skip TOC-table duplicates: short content AND on first 3 pages
            if e.start_page <= _TOC_PAGE_CUTOFF and len(content) < _MIN_CONTENT_CHARS:
                continue
            # Skip heading-only stubs with no body
            if len(content) < _MIN_CONTENT_CHARS:
                continue
            # Build a one-line preview from the body text (strip the heading line)
            body_lines = [ln for ln in content.splitlines() if ln.strip()]
            preview = " ".join(body_lines[1:3])[:120] if len(body_lines) > 1 else ""
            entries.append({
                "section_id": e.section_id,
                "title":      e.title,
                "pages":      f"{e.start_page}–{e.end_page}",
                "preview":    preview,
            })
        return json.dumps(entries, ensure_ascii=False)

    def _navigate(self, query: str, smart_map: SmartMap) -> tuple[str, str]:
        """
        Call LLM to select the best section_id for the query.

        Returns (section_id, reasoning).
        Falls back to sec_001 if parse fails.
        """
        toc_json = self._build_toc_json(smart_map)
        prompt   = _NAVIGATION_PROMPT.format(toc_json=toc_json, query=query)

        llm = self._get_llm()
        try:
            response = llm.complete(prompt)
            raw      = str(response).strip()

            # Extract JSON even if the LLM wraps it in text
            json_match = re.search(r'\{[^}]+\}', raw)
            if not json_match:
                raise ValueError(f"No JSON found in LLM response: {raw!r}")

            parsed    = json.loads(json_match.group())
            sec_id    = parsed.get("section_id", _FALLBACK_SECTION)
            reasoning = parsed.get("reasoning", "")

            # Validate section_id exists in map
            if sec_id not in smart_map.toc_by_id:
                logger.warning(
                    f"[PageIndex] LLM returned unknown section_id={sec_id!r}, "
                    f"using fallback"
                )
                sec_id = _FALLBACK_SECTION

            return sec_id, reasoning

        except Exception as exc:
            logger.warning(f"[PageIndex] Navigation parse failed ({exc}); using fallback")
            return _FALLBACK_SECTION, "fallback"

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve_from_doc(
        self,
        query:   str,
        doc_id:  Optional[str] = None,
    ) -> HybridRetrievalResult:
        """
        Navigate the SmartMap for doc_id and return the best section.

        Args:
            query:  User query.
            doc_id: Override the retriever's default doc_id.

        Returns:
            Single HybridRetrievalResult from the PageIndex path.
        """
        t0       = time.perf_counter()
        doc_id   = doc_id or self.doc_id
        sm       = self._get_smart_map()

        sec_id, reasoning = self._navigate(query, sm)
        entry   = sm.toc_by_id[sec_id]
        content = sm.sections.get(sec_id, "")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[PageIndex] doc={doc_id!r} section={entry.title!r} "
            f"pages={entry.start_page}-{entry.end_page} "
            f"({elapsed_ms:.0f}ms) reasoning={reasoning!r}"
        )

        return HybridRetrievalResult(
            doc_id       = doc_id,
            content      = content,
            page_numbers = list(range(entry.start_page, entry.end_page + 1)),
            confidence   = 0.92,      # structural navigation is high-confidence
            source       = "pageindex",
            citations    = [
                f"{doc_id} | Pages {entry.start_page}–{entry.end_page} | {entry.title}"
            ],
            section_id   = sec_id,
            metadata     = {"reasoning": reasoning, "level": entry.level},
        )

    def retrieve(self, query: str, top_k: int = 1) -> List[HybridRetrievalResult]:
        """HybridBaseRetriever interface — wraps retrieve_from_doc."""
        result = self.retrieve_from_doc(query)
        return [result]
