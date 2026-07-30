"""
Offline Q&A library generator — run once, commit qa_library.json to repo.

Usage:
    python scripts/generate_qa_library.py \\
        --input data/raw/rbi_kyc_master_direction.pdf \\
        --output data/qa_library.json

Splits the document into sections by heading detection, generates
8 Q&A pairs per section (configurable via QA_PAIRS_PER_SECTION in .env)
using Groq Llama-3.1-8b-instant, and writes a JSON file.

Output schema:
  {
    "generated_at": "<ISO timestamp>",
    "source_document": "rbi_kyc_master_direction",
    "model_used": "llama-3.1-8b-instant",
    "total_pairs": <int>,
    "qa_pairs": [
      {
        "question": "...",
        "answer": "...",
        "section": "...",
        "question_type": "obligation|threshold|exception|definition|process"
      }
    ]
  }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from config.revamp_settings import revamp_settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Q&A generation system prompt ─────────────────────────────────────────────

QA_GENERATION_SYSTEM_PROMPT = """You are a compliance expert specialising in RBI (Reserve Bank of India) regulations
and Indian banking law. You generate question-answer pairs for a compliance knowledge base
used by bank compliance officers and fintech developers.

WHAT YOU MUST DO:
1. Generate exactly {n_pairs} Q&A pairs per section — no more, no fewer
   The caller will set this number. Do not decide it yourself.

2. Write specific, practical questions — not generic ones
   GOOD: "What documents are acceptable for address proof for a non-resident individual?"
   BAD:  "What is KYC?" or "What does this section say?"

3. Cover all 5 question types — at least one of each per section:
   - Obligation: "What must regulated entities do when..."
   - Threshold: "What is the monetary/time limit for..."
   - Exception: "Under what conditions is X not required..."
   - Definition: "Who qualifies as a Politically Exposed Person..."
   - Process: "What is the procedure for completing V-CIP..."

4. Write complete, self-contained answers
   - No "see above", "as mentioned", "refer to Section X"
   - Include the regulatory basis: "Under Section 16(b), banks must..."
   - Include exact numbers: "INR 50,000", "60 days", "five years"
   - If the section covers an exception, the answer must state both the rule AND the exception

5. Preserve modal verbs in answers
   - SHALL / MUST → "is required to" or keep as "must"
   - MAY → "is permitted to" — never upgrade to "must"
   - SHOULD → "is expected to" — never upgrade to "must"

WHAT YOU MUST NOT DO:
- Do not generate duplicate questions (check before adding each one)
- Do not generate questions whose answers are not in the provided section text
- Do not fabricate regulatory citations — only cite sections/rules present in the input
- Do not output markdown, explanations, or preamble

OUTPUT FORMAT:
Return only valid JSON in this exact structure:
{"qa_pairs": [{"question": "...", "answer": "...", "section": "...", "question_type": "..."}]}

question_type must be one of: "obligation", "threshold", "exception", "definition", "process"
section must be the section heading passed to you in the user message.
Nothing before {. Nothing after }."""


def _build_qa_user_message(section_text: str, section_heading: str, n_pairs: int) -> str:
    return (
        f"Section heading: {section_heading}\n\n"
        f"Generate {n_pairs} Q&A pairs from the following section text.\n"
        f"Ensure coverage of obligation, threshold, exception, definition, "
        f"and process question types.\n\n"
        f"--- SECTION TEXT START ---\n"
        f"{section_text.strip()}\n"
        f"--- SECTION TEXT END ---"
    )


# ── Section splitter ──────────────────────────────────────────────────────────

_SECTION_HEADING_RE = re.compile(
    r'^(\d+\.\s+\S|CHAPTER\s+[IVX]+|PART\s+[IVX]+|SECTION\s+\d+)',
    re.IGNORECASE | re.MULTILINE,
)


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Split clean_text into (heading, body) pairs.
    Falls back to coarse chunks of ~2000 chars if no headings found.
    """
    matches = list(_SECTION_HEADING_RE.finditer(text))

    if not matches:
        # Fallback: chunk into ~2000 char blocks
        logger.warning("No section headings found — using coarse chunking")
        chunks = []
        for i in range(0, len(text), 2000):
            chunk = text[i : i + 2000].strip()
            if chunk:
                chunks.append((f"Section {i // 2000 + 1}", chunk))
        return chunks

    sections = []
    for idx, match in enumerate(matches):
        heading = match.group(0).strip()
        start   = match.end()
        end     = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body    = text[start:end].strip()
        if body and len(body) > 100:   # skip stub sections
            sections.append((heading, body))

    logger.info(f"[generate_qa_library] Found {len(sections)} sections")
    return sections


# ── Groq Q&A generation ───────────────────────────────────────────────────────

def _generate_qa_for_section(
    section_text: str,
    section_heading: str,
    model: str,
    api_key: str,
    n_pairs: int,
    section_idx: int,
    retries: int = 3,
) -> list[dict]:
    from groq import Groq

    client = Groq(api_key=api_key)

    system_prompt = QA_GENERATION_SYSTEM_PROMPT.replace("{n_pairs}", str(n_pairs))
    user_msg      = _build_qa_user_message(section_text, section_heading, n_pairs)

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_msg},
                ],
            )
            raw = resp.choices[0].message.content.strip()

            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError(f"No JSON object found: {raw[:200]!r}")

            data  = json.loads(raw[start:end])
            pairs = data.get("qa_pairs", [])
            if not isinstance(pairs, list):
                raise ValueError(f"'qa_pairs' is not a list: {type(pairs)}")

            # Attach section to any pair missing it
            for p in pairs:
                if not p.get("section"):
                    p["section"] = section_heading

            return pairs

        except Exception as exc:
            wait = 2 ** attempt
            logger.warning(
                f"[generate_qa_library] section={section_idx} attempt={attempt+1} "
                f"failed: {exc} — retrying in {wait}s"
            )
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                logger.error(
                    f"[generate_qa_library] section={section_idx} all retries exhausted — skipping"
                )
                return []

    return []


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Q&A library from RBI KYC PDF")
    parser.add_argument("--input",  default="data/raw/rbi_kyc_master_direction.pdf",
                        help="Path to source PDF")
    parser.add_argument("--output", default=revamp_settings.QA_LIBRARY_PATH,
                        help="Output JSON path")
    parser.add_argument("--n-pairs", type=int, default=revamp_settings.QA_PAIRS_PER_SECTION,
                        help="Q&A pairs per section")
    parser.add_argument("--model", default=revamp_settings.QA_GENERATION_MODEL,
                        help="Groq model name")
    args = parser.parse_args()

    api_key = revamp_settings.GROQ_API_KEY

    # ── Load PDF ──────────────────────────────────────────────────────────
    logger.info(f"Loading PDF: {args.input}")
    from src.ingestion.parser import load_pdf_with_sections
    from src.ingestion.metadata import build_section_map_and_clean

    text_marked = load_pdf_with_sections(args.input)
    _, _, clean_text = build_section_map_and_clean(text_marked)
    logger.info(f"Extracted {len(clean_text):,} chars of clean text")

    sections = _split_into_sections(clean_text)
    logger.info(f"Processing {len(sections)} sections → {len(sections) * args.n_pairs} pairs estimated")

    # ── Generate Q&A pairs ────────────────────────────────────────────────
    try:
        from tqdm import tqdm
        section_iter = tqdm(enumerate(sections), total=len(sections), desc="Generating Q&A")
    except ImportError:
        section_iter = enumerate(sections)

    all_pairs: list[dict] = []
    for idx, (heading, body) in section_iter:
        pairs = _generate_qa_for_section(
            section_text=body,
            section_heading=heading,
            model=args.model,
            api_key=api_key,
            n_pairs=args.n_pairs,
            section_idx=idx,
        )
        all_pairs.extend(pairs)
        logger.info(f"  Section '{heading[:60]}' → {len(pairs)} pairs (total so far: {len(all_pairs)})")

    # ── Write output ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    output = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "source_document": "rbi_kyc_master_direction",
        "model_used":      args.model,
        "total_pairs":     len(all_pairs),
        "qa_pairs":        all_pairs,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(f"Wrote {len(all_pairs)} Q&A pairs to {args.output}")


if __name__ == "__main__":
    main()
