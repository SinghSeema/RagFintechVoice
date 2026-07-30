"""
Agentic chunking — proposition extraction from RBI KYC regulatory text.

Pipeline:
  1. Split clean_text into paragraphs (double-newline boundaries).
  2. Filter noise: page numbers, amendment notices, blank lines.
  3. Assign section/chapter to each paragraph via offset lookup.
  4. Batch 3 paragraphs per Groq API call (Llama-3.1-8b-instant).
  5. Extract standalone propositions — each resolves pronouns, preserves
     citations, splits compound obligations, removes amendment noise.
  6. Fallback to sentence splitting on JSON parse failure (never crashes).
  7. Deduplicate propositions using BGE-M3 cosine similarity > 0.92.
  8. Return one TextNode per proposition with full metadata.

Entry point:
  chunk_to_propositions(chapter_map, section_map, clean_text, api_key)
      -> list[TextNode]
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Optional

from llama_index.core.schema import TextNode

from config.revamp_settings import revamp_settings

logger = logging.getLogger(__name__)


# ── Proposition extraction system prompt (Llama-3.1-8b) ──────────────────────

PROPOSITION_SYSTEM_PROMPT = """You are a regulatory document analyst. Extract factual propositions from RBI KYC regulatory text.

WHAT YOU MUST DO:
1. Replace every pronoun with its explicit referent
   - "they" or "them" → "regulated entities" (default)
   - "it" → name the specific thing (e.g. "the KYC document", "the account")
   - If context makes a more specific referent clear (e.g. "banks", "NBFCs", "walk-in customers"), use that instead

2. Keep every Rule/Section/Clause reference
   - "(Rule 9(14))" must appear in the proposition that mentions it
   - Append as a suffix: "...five years after the relationship ends. (Rule 9(14))"

3. Split compound sentences into single-fact propositions
   - One SHALL or MUST per proposition
   - Conditions get their own proposition: "X applies unless Y" → "X applies." + "X does not apply when Y."
   - Alternatives get their own proposition: "whichever is later" → two propositions, one per event

4. Preserve exact numbers — never paraphrase
   - Keep "INR 50,000", "five years", "60 days" exactly as written
   - "large amount" is not acceptable if the source says "INR 10,00,000"

5. Remove amendment notices entirely
   - Lines starting with "Substituted vide", "Inserted vide", "Deleted vide" → skip
   - Extract only the substantive regulatory content that follows

6. Flag unresolvable cross-references
   - "as referred to above" with no clear referent → add [XREF: resolve manually] at end

WHAT YOU MUST NOT DO:
- Do not leave any pronoun (they/them/their/it) in output
- Do not combine two obligations into one proposition
- Do not add information not present in the input paragraph
- Do not output markdown, code fences, explanations, or preamble of any kind

OUTPUT FORMAT:
Return only a JSON array of strings.
Start your response with [ and end with ]
Nothing before [. Nothing after ].
If a paragraph has no extractable content (pure amendment notice, page header, blank), return []

EXAMPLE INPUT:
"They shall also ensure, in terms of sub-rule (14) of Rule 9, that the records
and documents are preserved for a period of five years after the business
relationship ends or the transaction is completed, whichever is later."

CORRECT OUTPUT:
["Regulated entities must preserve KYC records for five years after the business relationship ends. (Rule 9(14))", "Regulated entities must preserve KYC records for five years after the transaction is completed. (Rule 9(14))", "The five-year retention period applies from whichever event occurs later — end of business relationship or completion of transaction. (Rule 9(14))"]

WRONG OUTPUT (do NOT produce this):
["They shall preserve records for five years."]
Reason this is wrong: pronoun not resolved, Rule citation dropped, compound condition not split into separate propositions."""


# ── Noise filters ─────────────────────────────────────────────────────────────

_PAGE_NUM_RE   = re.compile(r"^\s*\d{1,4}\s*$")
_AMENDMENT_RE  = re.compile(r"^(Substituted|Inserted|Deleted|Omitted)\s+vide", re.IGNORECASE)
_SHORT_LINE_RE = re.compile(r"^.{1,25}$")  # likely header/footer noise


def _is_noise_paragraph(para: str) -> bool:
    """Return True if the whole paragraph is noise that shouldn't be sent to Groq."""
    lines = [l.strip() for l in para.strip().splitlines() if l.strip()]
    if not lines:
        return True
    # Pure page number
    if len(lines) == 1 and _PAGE_NUM_RE.match(lines[0]):
        return True
    # All lines are amendment notices → skip entirely
    if all(_AMENDMENT_RE.match(l) for l in lines):
        return True
    return False


def _strip_amendment_lines(para: str) -> str:
    """Remove individual amendment-notice lines within a paragraph."""
    clean = [l for l in para.splitlines() if not _AMENDMENT_RE.match(l.strip())]
    return "\n".join(clean).strip()


# ── Section / chapter lookup (mirrors metadata.enrich_nodes) ─────────────────

def _resolve_from_map(global_pos: int, offsets: list[int], titles: list[str]) -> str | None:
    lo, hi, result = 0, len(offsets) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if offsets[mid] <= global_pos:
            result = titles[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return result


def _assign_context(
    para: str,
    para_offset: int,
    chapter_map: list,
    section_map: list,
) -> tuple[str, str]:
    """Return (chapter_label, section_title) for a paragraph at para_offset."""
    c_offsets = [o for o, _ in chapter_map]
    c_titles  = [t for _, t in chapter_map]
    s_offsets = [o for o, _ in section_map]
    s_titles  = [t for _, t in section_map]

    chapter = _resolve_from_map(para_offset, c_offsets, c_titles) or ""
    section = _resolve_from_map(para_offset, s_offsets, s_titles) or ""
    return chapter, section


# ── Groq proposition extraction ───────────────────────────────────────────────

def _build_user_message(paragraphs: list[str], section_heading: str) -> str:
    para_block = "\n\n---\n\n".join(
        f"[Paragraph {i + 1}]\n{p.strip()}" for i, p in enumerate(paragraphs)
    )
    return (
        f"Section context: {section_heading}\n\n"
        f"Process these {len(paragraphs)} paragraph(s) and return a single "
        f"JSON array containing all propositions from all paragraphs combined.\n\n"
        f"{para_block}"
    )


def _call_groq(
    paragraphs: list[str],
    section_heading: str,
    model: str,
    api_key: str,
    batch_idx: int,
    retries: int = 3,
) -> list[str]:
    """Call Groq and return a list of proposition strings. Falls back on failure."""
    from groq import Groq

    client = Groq(api_key=api_key)
    user_msg = _build_user_message(paragraphs, section_heading)

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": PROPOSITION_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
            )
            raw = resp.choices[0].message.content.strip()
            finish_reason = resp.choices[0].finish_reason

            # Find the JSON array boundaries robustly
            start = raw.find("[")
            end   = raw.rfind("]")

            if start == -1 or end == -1:
                # No closing bracket — output severely truncated, try repair immediately
                repaired = _repair_json_array(raw)
                if repaired:
                    logger.warning(
                        f"[agentic_chunker] batch={batch_idx} JSON severely truncated "
                        f"(no closing ]) — repaired {len(repaired)} props"
                    )
                    return repaired
                raise ValueError(f"No JSON array found in response: {raw[:200]!r}")

            # Try full parse; on any JSON error attempt repair before retrying
            try:
                propositions = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                repaired = _repair_json_array(raw)
                if repaired:
                    logger.warning(
                        f"[agentic_chunker] batch={batch_idx} JSON truncated "
                        f"(finish_reason={finish_reason}) — repaired {len(repaired)} props"
                    )
                    return repaired
                raise  # re-raise to trigger retry (repair found nothing)

            if not isinstance(propositions, list):
                raise ValueError("Response is not a JSON array")
            return [p for p in propositions if isinstance(p, str) and p.strip()]

        except Exception as exc:
            wait = 2 ** attempt
            logger.warning(
                f"[agentic_chunker] batch={batch_idx} attempt={attempt+1} failed: {exc} "
                f"— retrying in {wait}s"
            )
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                logger.error(
                    f"[agentic_chunker] batch={batch_idx} all retries exhausted — "
                    f"falling back to sentence splitting"
                )
                return _sentence_fallback(paragraphs)

    return []


def _repair_json_array(raw: str) -> list[str]:
    """
    Recover complete proposition strings from a truncated JSON array.
    When the model hits its output token limit mid-array, the closing ] and
    possibly the closing " of the last string are missing. We extract all
    properly-closed JSON strings using regex, ignoring the trailing fragment.
    """
    start = raw.find("[")
    if start == -1:
        return []
    content = raw[start:]
    # Match complete JSON strings: "..." with escaped internals
    string_re = re.compile(r'"(?:[^"\\]|\\.)*"')
    results = []
    for m in string_re.finditer(content):
        try:
            s = json.loads(m.group(0))
            if isinstance(s, str) and s.strip():
                results.append(s)
        except json.JSONDecodeError:
            pass
    return results


def _sentence_fallback(paragraphs: list[str]) -> list[str]:
    """Split paragraphs into sentences as a fallback when Groq fails."""
    sentences = []
    for para in paragraphs:
        for sent in re.split(r"(?<=[.!?])\s+", para.strip()):
            s = sent.strip()
            if s and len(s) > 20:
                sentences.append(s)
    return sentences


# ── Deduplication ─────────────────────────────────────────────────────────────

def _deduplicate(propositions: list[str], threshold: float) -> list[str]:
    """
    Remove near-duplicate propositions using BGE-M3 cosine similarity.
    Keeps the longer proposition when two are above threshold.
    """
    if len(propositions) <= 1:
        return propositions

    import numpy as np
    from src.retrieval.embedder import get_embed_model

    embed_model = get_embed_model()
    vectors = embed_model.get_text_embedding_batch(propositions)
    vecs = [np.array(v) for v in vectors]

    # Normalise
    norms = [np.linalg.norm(v) for v in vecs]
    vecs_n = [v / n if n > 0 else v for v, n in zip(vecs, norms)]

    kept = []
    dropped = set()

    for i, (prop, vec_i) in enumerate(zip(propositions, vecs_n)):
        if i in dropped:
            continue
        for j in range(i + 1, len(propositions)):
            if j in dropped:
                continue
            sim = float(np.dot(vec_i, vecs_n[j]))
            if sim >= threshold:
                # Keep the longer one
                if len(propositions[j]) > len(prop):
                    dropped.add(i)
                    break
                else:
                    dropped.add(j)
        if i not in dropped:
            kept.append(prop)

    removed = len(propositions) - len(kept)
    if removed:
        logger.info(f"[agentic_chunker] dedup removed {removed} near-duplicate propositions")
    return kept


# ── Main entry point ──────────────────────────────────────────────────────────

def chunk_to_propositions(
    chapter_map: list,
    section_map: list,
    clean_text: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    batch_size: Optional[int] = None,
    dedup_threshold: Optional[float] = None,
) -> list[TextNode]:
    """
    Transform clean document text into proposition TextNodes.

    Args:
        chapter_map:     [(char_offset, label), ...] from build_section_map_and_clean()
        section_map:     [(char_offset, title), ...] from build_section_map_and_clean()
        clean_text:      Marker-stripped document text from build_section_map_and_clean()
        api_key:         Groq API key (defaults to revamp_settings.GROQ_API_KEY)
        model:           Groq model name (defaults to revamp_settings.AGENTIC_PROPOSITION_MODEL)
        batch_size:      Paragraphs per Groq call (defaults to revamp_settings.AGENTIC_CHUNK_BATCH_SIZE)
        dedup_threshold: Cosine similarity dedup cutoff (defaults to revamp_settings.AGENTIC_DEDUP_THRESHOLD)

    Returns:
        List of TextNode objects ready for Qdrant ingestion.
    """
    api_key         = api_key         or revamp_settings.GROQ_API_KEY
    model           = model           or revamp_settings.AGENTIC_PROPOSITION_MODEL
    batch_size      = batch_size      or revamp_settings.AGENTIC_CHUNK_BATCH_SIZE
    dedup_threshold = dedup_threshold or revamp_settings.AGENTIC_DEDUP_THRESHOLD

    # ── Step 1: split into paragraphs using section_map boundaries ───────
    # PDF text uses single newlines; double-newline split only finds a handful
    # of blocks.  Instead: carve the document at section_map offsets, then
    # group lines within each section into ≤ MAX_PARA_CHARS chunks so each
    # Groq call stays well under the 6 000 TPM limit.
    _MAX_PARA_CHARS = 2000

    # Build section boundary list: [(start, end, section_title), ...]
    sec_boundaries: list[tuple[int, int, str]] = []
    if section_map:
        for i, (off, title) in enumerate(section_map):
            end = section_map[i + 1][0] if i + 1 < len(section_map) else len(clean_text)
            sec_boundaries.append((off, end, title))
        # Preamble before the first marker
        if section_map[0][0] > 0:
            sec_boundaries.insert(0, (0, section_map[0][0], "preamble"))
    else:
        sec_boundaries = [(0, len(clean_text), "document")]

    c_offsets = [o for o, _ in chapter_map]
    c_titles  = [t for _, t in chapter_map]

    para_with_offsets: list[tuple[str, int]] = []  # (text, global_offset)
    for sec_start, sec_end, _sec_title in sec_boundaries:
        sec_text = clean_text[sec_start:sec_end]
        lines    = [l.strip() for l in sec_text.split("\n") if l.strip()]

        chunk_lines: list[str] = []
        chunk_len   = 0
        chunk_start = sec_start

        for line in lines:
            if chunk_len + len(line) + 1 > _MAX_PARA_CHARS and chunk_lines:
                para_with_offsets.append(("\n".join(chunk_lines), chunk_start))
                # Advance chunk_start past the emitted block
                approx = clean_text.find(line, chunk_start)
                chunk_start = approx if approx != -1 else chunk_start
                chunk_lines = [line]
                chunk_len   = len(line)
            else:
                if not chunk_lines:
                    # Record where this chunk starts in clean_text
                    approx = clean_text.find(line, sec_start)
                    chunk_start = approx if approx != -1 else sec_start
                chunk_lines.append(line)
                chunk_len += len(line) + 1

        if chunk_lines:
            para_with_offsets.append(("\n".join(chunk_lines), chunk_start))

    raw_count = len(para_with_offsets)

    # ── Step 2: filter noise ──────────────────────────────────────────────
    filtered: list[tuple[str, int, str, str]] = []  # (text, offset, chapter, section)
    for para, offset in para_with_offsets:
        if _is_noise_paragraph(para):
            continue
        clean_para = _strip_amendment_lines(para)
        if not clean_para or len(clean_para) < 30:
            continue
        chapter = _resolve_from_map(offset, c_offsets, c_titles) or ""
        section = _resolve_from_map(offset,
                                    [o for o, _ in section_map],
                                    [t for _, t in section_map]) or ""
        filtered.append((clean_para, offset, chapter, section))

    logger.info(
        f"[agentic_chunker] {raw_count} raw paragraphs → "
        f"{len(filtered)} after noise filtering"
    )

    # ── Step 3: batch Groq calls with tqdm progress bar ───────────────────
    try:
        from tqdm import tqdm
        _tqdm = tqdm
    except ImportError:
        def _tqdm(it, **kw):  # type: ignore[misc]
            return it

    all_propositions: list[tuple[str, str, str, int, int]] = []
    # Each item: (text, chapter, section, para_idx, prop_idx_within_para)

    batches = [filtered[i : i + batch_size] for i in range(0, len(filtered), batch_size)]

    for batch_idx, batch in enumerate(_tqdm(batches, desc="Extracting propositions")):
        paragraphs = [item[0] for item in batch]
        chapters   = [item[2] for item in batch]
        sections   = [item[3] for item in batch]

        # Use the section from the first paragraph in the batch as context heading
        section_heading = sections[0] or chapters[0] or "RBI KYC Master Direction"

        raw_props = _call_groq(paragraphs, section_heading, model, api_key, batch_idx)

        for prop_idx, prop in enumerate(raw_props):
            if not prop.strip():
                continue
            # Map proposition back to the paragraph that likely produced it
            para_idx = min(prop_idx // max(len(raw_props) // len(batch), 1), len(batch) - 1)
            all_propositions.append((
                prop,
                chapters[para_idx],
                sections[para_idx],
                batch_idx * batch_size + para_idx,
                prop_idx,
            ))

    logger.info(f"[agentic_chunker] extracted {len(all_propositions)} raw propositions")

    # ── Step 4: deduplicate ───────────────────────────────────────────────
    prop_texts = [p[0] for p in all_propositions]
    deduped    = _deduplicate(prop_texts, dedup_threshold)

    # Rebuild with original metadata for the kept propositions
    prop_text_set = set(deduped)
    kept_with_meta = [item for item in all_propositions if item[0] in prop_text_set]
    # Preserve deduped order (not insertion order) — filter while preserving dedup sequence
    seen_texts: set[str] = set()
    ordered: list[tuple] = []
    for text in deduped:
        if text not in seen_texts:
            # Find the first matching item in kept_with_meta
            for item in kept_with_meta:
                if item[0] == text and text not in seen_texts:
                    ordered.append(item)
                    seen_texts.add(text)
                    break

    logger.info(f"[agentic_chunker] {len(ordered)} propositions after dedup")

    # ── Step 5: build proposition TextNodes ──────────────────────────────
    # Prepend the section heading to each proposition so BM25 can match
    # section-number queries (e.g. "section 57") against proposition text.
    nodes: list[TextNode] = []
    for prop_text, chapter, section, para_idx, prop_idx in ordered:
        has_xref = "[XREF:" in prop_text
        section_prefix = f"[{section}]\n" if section else ""
        display_text   = section_prefix + prop_text
        node = TextNode(
            node_id=str(uuid.uuid4()),
            text=display_text,
            metadata={
                "source":                "rbi_kyc_master_direction",
                "strategy":              "agentic",
                "chapter":               chapter,
                "section":               section,
                "original_paragraph_idx": para_idx,
                "proposition_idx":        prop_idx,
                "char_count":            len(display_text),
                "has_xref":              has_xref,
            },
        )
        node.excluded_embed_metadata_keys = [
            "strategy", "chapter", "original_paragraph_idx",
            "proposition_idx", "char_count", "has_xref",
        ]
        nodes.append(node)

    # ── Step 6: one summary TextNode per section ──────────────────────────
    # Proposition decomposition loses sequential structure needed for
    # procedural queries (V-CIP steps) and full-coverage queries (small
    # accounts). One summary node per section restores that context.
    from collections import defaultdict
    _MAX_SUMMARY_CHARS = 1800
    section_para_groups: dict[tuple, list[str]] = defaultdict(list)
    for para_text, _offset, chapter, section in filtered:
        key = (section or chapter or "document", chapter)
        section_para_groups[key].append(para_text)

    _SUMMARY_STRIDE = 900  # 50 % overlap between windows
    summary_nodes: list[TextNode] = []
    for (section, chapter), paras in section_para_groups.items():
        combined = "\n\n".join(paras)
        if len(combined) < 80:
            continue
        label = f"[{section}]\n" if section else ""
        # Slide a window over large sections so distant content (e.g. V-CIP
        # steps, small-account norms buried deep in section 8) gets its own node
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
            summary_nodes.append(snode)
            if start + _MAX_SUMMARY_CHARS >= len(combined):
                break
            start += _SUMMARY_STRIDE

    logger.info(
        f"[agentic_chunker] {len(nodes)} proposition nodes + "
        f"{len(summary_nodes)} section summary nodes"
    )
    return nodes + summary_nodes
