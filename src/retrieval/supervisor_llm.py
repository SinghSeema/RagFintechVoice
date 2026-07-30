"""
SupervisorLLM — Stage 3 of the Hybrid RAG pipeline.

Receives two evidence streams for the same query:
  - Vector chunks  : semantic matches from Qdrant + BM25 (existing pipeline)
  - PageIndex data : structurally exact section from LLM TOC navigation

Cross-checks both streams and produces a verified, citation-rich answer.

Why a Supervisor matters:
  Vector chunks may return semantically similar but structurally adjacent text
  (e.g., "low risk" chunks appearing for a "high risk" query due to co-occurrence).
  PageIndex is structurally precise but may miss cross-section context.
  The Supervisor resolves contradictions and combines the best of both.

Public API:
    SupervisorLLM(model, max_tokens, api_key)
    supervisor.synthesize(query, vector_nodes, pageindex_result) -> str
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from llama_index.core.schema import NodeWithScore

from config.revamp_settings import revamp_settings
from src.retrieval.hybrid_base import HybridRetrievalResult

logger = logging.getLogger(__name__)

# ── Supervisor system prompt ───────────────────────────────────────────────────

_SUPERVISOR_SYSTEM = """\
You are a friendly bank compliance assistant speaking to a customer.
You receive two evidence streams for the same user query:
  VECTOR CHUNKS  — semantically retrieved passages from the document
  PAGEINDEX DATA — structurally exact section from the document's Table of Contents

Your task: synthesise ONE clear, customer-friendly answer using both streams.

STRICT RULES — follow every one:
1. Paraphrase everything in plain everyday English. NEVER copy regulatory text word-for-word.
2. Convert legal phrasing to simple language:
   "Officially Valid Documents" → "approved ID documents"
   "regulated entity" → "your bank"
   "Proof of possession of Aadhaar" → "Aadhaar card"
3. Do NOT mention source numbers, clause labels, or sub-clause codes like (a), (ab), (iii).
4. Maximum 3 short sentences. Be brief and direct.
5. If listing items, name only the 2-3 most common examples, then say "and similar documents".
6. Prefer PageIndex for exact facts (amounts, deadlines, section numbers) over Vector chunks.
7. End with one short reference if helpful, e.g. "as per RBI KYC norms" — nothing longer.
8. If the streams do not contain a clear answer, say "I don't have that detail right now, \
please check with your branch."\
"""

_SUPERVISOR_USER_TEMPLATE = """\
VECTOR CHUNKS:
{vector_text}

PAGEINDEX DATA:
Citation: {pi_citation}
Content: {pi_content}

USER QUERY: {query}

Write a plain, customer-friendly answer in 3 sentences or fewer.\
"""


# ── Supervisor class ───────────────────────────────────────────────────────────

class SupervisorLLM:
    """
    Cross-checks vector chunks vs PageIndex finding, returns verified answer.
    """

    def __init__(
        self,
        model:        Optional[str] = None,
        max_tokens:   Optional[int] = None,
        groq_api_key: Optional[str] = None,
    ) -> None:
        self.model      = model      or revamp_settings.SUPERVISOR_MODEL
        self.max_tokens = max_tokens or revamp_settings.SUPERVISOR_MAX_TOKENS
        self.api_key    = groq_api_key or revamp_settings.GROQ_API_KEY
        self._llm       = None

    def _get_llm(self):
        if self._llm is None:
            from llama_index.llms.groq import Groq
            self._llm = Groq(
                model=self.model,
                api_key=self.api_key,
                temperature=0.1,
                max_tokens=self.max_tokens,
                system_prompt=_SUPERVISOR_SYSTEM,
            )
        return self._llm

    def synthesize(
        self,
        query:           str,
        vector_nodes:    List[NodeWithScore],
        pageindex_result: HybridRetrievalResult,
    ) -> str:
        """
        Produce a verified answer from two evidence streams.

        Args:
            query:            User's original question.
            vector_nodes:     Top-K nodes from existing vector/BM25 pipeline.
            pageindex_result: Best section from PageIndex LLM navigation.

        Returns:
            Verified answer string with citations.
        """
        t0 = time.perf_counter()

        # Compact vector chunks — top-3 to stay within token budget
        vector_parts = []
        for i, nws in enumerate(vector_nodes[:3], 1):
            meta    = nws.node.metadata
            section = meta.get("section", meta.get("section_title", "Unknown section"))
            snippet = nws.node.get_content()[:600].replace("\n", " ")
            vector_parts.append(f"[Chunk {i} | {section}]\n{snippet}")
        vector_text = "\n\n".join(vector_parts) if vector_parts else "No vector chunks available."

        # PageIndex content — truncated to 1200 chars to leave room for synthesis
        pi_citation = (
            pageindex_result.citations[0]
            if pageindex_result.citations
            else "Unknown section"
        )
        pi_content = pageindex_result.content[:1200].replace("\n", " ")

        user_msg = _SUPERVISOR_USER_TEMPLATE.format(
            vector_text=vector_text,
            pi_citation=pi_citation,
            pi_content=pi_content,
            query=query,
        )

        llm      = self._get_llm()
        response = llm.complete(user_msg)
        answer   = str(response).strip()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[Supervisor] model={self.model} "
            f"vector_chunks={len(vector_nodes)} "
            f"answer_len={len(answer)} "
            f"({elapsed_ms:.0f}ms)"
        )
        return answer
