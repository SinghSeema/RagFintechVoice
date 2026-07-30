"""
Phase 3 — Voice pipeline: VAD → STT → scope-check → RAG → TTS (Pipecat + LiveKit).

Architecture
------------
LiveKitTransport (audio in, VAD)
  → DeepgramSTTService  nova-2-general streaming — transcribes speech to text
  → RAGProcessor        custom FrameProcessor — replaces the LLM step
                        1. Scope-check via Groq llama-3.1-8b-instant (fast, ~150 ms)
                           Out-of-scope → immediate polite refusal, no RAG call.
                        2. RAG query() in a ThreadPoolExecutor
                        3. Emits LLMFullResponseStartFrame → TextFrame chunks →
                           LLMFullResponseEndFrame
  → CartesiaTTSService  Sonic — streams audio before full answer is ready
    (or Deepgram Aura as fallback — swap via DEEPGRAM_API_KEY env var)
  → LiveKitTransport (audio out)

Guardrails
----------
RAGProcessor enforces three layers of guardrails before calling the RAG stack:
  1. Scope  — Groq 8b classifier decides if the question is finance/banking/
              regulatory.  Non-finance questions get a polite "out of scope"
              refusal without touching the retriever.
  2. Ethics — Questions asking for investment advice, illegal activity, or
              personal financial recommendations are declined with an
              appropriate disclaimer.
  3. PII    — No user PII is stored; audit log records only the question hash.

VAD
---
Silero VAD runs inside LiveKitParams (transport level) so speech boundaries
are detected before audio reaches STT.  stop_secs=1.5 gives enough silence
after a question before the STT batch is flushed.

TTS selection
-------------
If CARTESIA_API_KEY is set → CartesiaTTSService (primary, 90 ms TTFT).
Else if DEEPGRAM_API_KEY is set → DeepgramTTSService (fallback).
Both keys missing → RuntimeError at startup.

Debug logging
-------------
DebugFrameLogger is available for ad-hoc debugging but is NOT wired into the
default pipeline.  To enable it, insert it between any two processors:
    pipeline = Pipeline([..., DebugFrameLogger("after-stt"), stt, ...])

Usage
-----
  python -m src.voice.pipeline          # requires LiveKit + TTS env vars
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from dotenv import load_dotenv
from groq import AsyncGroq
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    Frame,
    InterruptionTaskFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport


# ── Debug frame logger ────────────────────────────────────────────────────────

# Frame types to suppress from debug logs (too noisy at audio rate)
_AUDIO_FRAME_TYPES = (AudioRawFrame, TTSAudioRawFrame)

class DebugFrameLogger(FrameProcessor):
    """
    Drop-in pipeline stage that logs every frame passing through.
    Place between any two processors to see exactly what is flowing.
    Audio frames are counted but not individually logged (too noisy).
    """

    def __init__(self, tag: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._tag = tag
        self._audio_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # MUST call super() first — it processes StartFrame which sets __started=True.
        # Without this, push_frame silently drops every frame that follows.
        await super().process_frame(frame, direction)

        dir_arrow = "↓" if direction == FrameDirection.DOWNSTREAM else "↑"

        if isinstance(frame, _AUDIO_FRAME_TYPES):
            self._audio_count += 1
            if self._audio_count % 50 == 1:           # log every 50th audio frame
                logger.debug(
                    f"[{self._tag}] {dir_arrow} {type(frame).__name__} "
                    f"(#{self._audio_count}, {len(frame.audio)} bytes, "
                    f"{frame.sample_rate}Hz)"
                )
        else:
            extra = ""
            if isinstance(frame, TranscriptionFrame):
                extra = f" text={frame.text!r}"
            elif isinstance(frame, TextFrame):
                extra = f" text={frame.text!r}"
            logger.info(f"[{self._tag}] {dir_arrow} {type(frame).__name__}{extra}")

        await self.push_frame(frame, direction)


# ── Bot persona & guardrail prompts ───────────────────────────────────────────

GREETING_TEXT = (
    "Hello! I am Finova, your RBI-compliant finance assistant. "
    "I can answer questions about KYC requirements, AML regulations, "
    "RBI master directions, banking compliance, and related financial topics. "
    "How can I help you today?"
)

# ── 3-way classifier prompt ───────────────────────────────────────────────────
#
# Single Groq 8b call (~150 ms) replaces the old YES/NO gate.
# Three outcomes:
#   FINANCE  → run full RAG pipeline
#   CHAT     → brief conversational reply via Groq 8b persona
#   BLOCKED  → polite "out of scope" refusal, no RAG call

_CLASSIFIER_PROMPT = """\
You are a strict intent classifier for Finova, a SPECIALIZED assistant whose
knowledge base covers ONLY RBI regulatory compliance topics.

Classify the user message into exactly one of three categories:

FINANCE  — Any message with substantive topical content related to Finova's domain,
  including follow-up requests and continuations of a prior finance answer:
  • KYC (Know Your Customer) requirements, documents, procedures
  • AML (Anti-Money Laundering) and CFT obligations
  • PMLA (Prevention of Money Laundering Act) provisions
  • RBI Master Directions and circulars
  • Customer due diligence: CDD, EDD (Enhanced), SDD (Simplified)
  • Account opening rules and identity verification
  • NRI banking compliance and documentation
  • Suspicious Transaction Reports (STR), Cash Transaction Reports (CTR)
  • SWIFT / correspondent banking compliance
  • Digital payment regulations (UPI, PPI) from an RBI compliance angle
  • CIBIL only in context of KYC/credit compliance, not credit scores generally
  • Follow-ups even if phrased casually: "tell me more", "elaborate", "what else?",
    "can you explain that?", "go on", "what about [any finance topic]?",
    "what does that mean?", "give an example", "is there anything else?"

CHAT     — ONLY a pure greeting, farewell, or single-word social acknowledgment
  with NO topical content whatsoever.
  Examples: "hi", "hello", "hey", "good morning", "bye", "goodbye",
  "thank you", "thanks", "okay", "ok".
  NOT CHAT: "thanks, what about NRI documents?", "okay so what are the rules?",
  "that's helpful, tell me more", "makes sense, can you elaborate?"

BLOCKED  — Everything outside Finova's compliance domain:
  • Stock market, equity trading, NSE, BSE, Nifty, Sensex, share prices
  • Mutual fund returns, portfolio management, asset allocation
  • Cryptocurrency, Bitcoin, Web3, NFT
  • IPO applications, demat accounts for investing
  • Personal finance advice (EMI calculators, savings plans, FD rates)
  • Insurance products, health coverage, LIC policies
  • Income tax filing, GST, CA advice
  • General banking products (interest rates, loan EMI, credit card offers)
  • Sports, entertainment, cooking, travel, health, politics, coding, weather
  • Self-improvement, productivity, wellness, lifestyle, daily routines, habits
  • Career advice, motivation, personal development, mindset
  • General knowledge, science, history, philosophy, relationships

IMPORTANT: When uncertain between FINANCE and BLOCKED, always choose BLOCKED.
Only classify as FINANCE if the message clearly and specifically refers to a topic
in the FINANCE list above. Novel or off-topic questions that do not match the
FINANCE list → BLOCKED.
When genuinely uncertain between FINANCE and CHAT (clear social pleasantry context
with a possible finance follow-up), choose FINANCE.

Reply with exactly one word: FINANCE, CHAT, or BLOCKED. No explanation."""

_ETHICS_KEYWORDS = (
    # Investment advice
    "buy stock", "buy share", "sell stock", "sell share",
    "invest in", "which stock", "pick stock", "best stock",
    "guaranteed return", "100% return", "sure profit", "risk free return",
    # Illegal financial activity
    "money laundering", "hide money", "avoid tax illegally",
    "evade tax", "tax evasion", "black money", "hawala",
    "benami", "shell company", "round tripping", "layering",
    # Market manipulation
    "insider trading", "insider tip", "front run", "front-run",
    "pump and dump", "pump & dump", "market manipulation",
    # Document fraud
    "fake kyc", "fake document", "forged document", "fake aadhaar",
    "fake pan", "counterfeit", "fake id",
    # Terror / crime financing
    "terror financing", "terrorist financing", "fund terrorism",
    "smuggling", "drug money", "criminal proceeds",
)

_OUT_OF_SCOPE_REPLY = (
    "That topic is outside my scope. "
    "I'm specialized in RBI regulatory compliance — KYC requirements, AML and PMLA obligations, "
    "customer due diligence, and RBI master directions. "
    "Is there anything in that space I can help you with?"
)

_ETHICS_REPLY = (
    "I'm not able to provide personal investment advice, recommendations on "
    "specific securities, or guidance on any activity that may violate financial laws. "
    "For investment decisions, please consult a SEBI-registered financial advisor."
)

# System prompt for the conversational (CHAT) reply path.
# IMPORTANT: this path is reached ONLY for pure greetings / farewells / thanks —
# the classifier has already confirmed there is no topical question here.
# The persona must NOT answer finance questions even if the user sneaks one in.
_CHAT_PERSONA_PROMPT = """\
You are Finova, a warm and professional RBI-compliant finance assistant.
The user has sent a pure social pleasantry — a greeting, farewell, or brief thanks.
Respond naturally and warmly in 1–2 short sentences.

STRICT RULES — you are in the social-pleasantry path only:
1. Do NOT answer any financial question, explain any regulation, or mention
   specific compliance terms, documents, or rules.
2. If the message contains ANY finance-related question despite reaching this path,
   reply only with: "Go ahead and ask your compliance question — I'm ready to help."
3. For greetings: welcome the user warmly and invite their question.
4. For farewells / thanks: acknowledge graciously and invite them back.
Keep it short, human, and on-topic for social exchanges only."""


async def _classify_query(question: str, groq_api_key: str) -> str:
    """
    Classify the user message into 'FINANCE', 'CHAT', or 'BLOCKED'.

    Uses Groq llama-3.1-8b-instant (~150 ms).
    Falls back to 'FINANCE' on API error (fail open — RAG handles it naturally).
    """
    try:
        client = AsyncGroq(api_key=groq_api_key)
        resp = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _CLASSIFIER_PROMPT},
                {"role": "user",   "content": question},
            ],
            max_tokens=5,
            temperature=0,
        )
        label = resp.choices[0].message.content.strip().upper().split()[0]
        if label not in ("FINANCE", "CHAT", "BLOCKED"):
            label = "FINANCE"   # unknown → let RAG handle it
        logger.info(f"Classifier | {label} | {question!r}")
        return label
    except Exception as exc:
        logger.warning(f"Classifier | error (defaulting to FINANCE): {exc}")
        return "FINANCE"


async def _conversational_reply(question: str, groq_api_key: str) -> str:
    """
    Generate a short, warm conversational response for greetings / thanks / farewells.
    Uses Groq llama-3.1-8b-instant to stay in persona as Finova.
    Falls back to a canned reply on API error.
    """
    try:
        client = AsyncGroq(api_key=groq_api_key)
        resp = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _CHAT_PERSONA_PROMPT},
                {"role": "user",   "content": question},
            ],
            max_tokens=80,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning(f"ConversationalReply | error: {exc}")
        return "You're welcome! Let me know if you have any other finance questions."


def _has_ethics_violation(question: str) -> bool:
    """Return True if the question matches a hard-coded ethics/guardrail keyword."""
    q_lower = question.lower()
    return any(kw in q_lower for kw in _ETHICS_KEYWORDS)


# ── Text helpers ───────────────────────────────────────────────────────────────

# Matches numeric citations: [1], [2], [1, 3], [1,3], [1, 2, 3]
_CITATION_NUM_RE = re.compile(r"\s*\[\d+(?:,\s*\d+)*\]")
# Matches CitationQueryEngine-style labels: [Source 2: (ab)], [Source 1: (d)], etc.
_CITATION_SOURCE_RE = re.compile(r"\s*\[Source\s+\d+[^\]]*\]", re.IGNORECASE)
# Matches leading list markers: "- ", "• ", "* ", "1. ", "1) " etc.
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-•*]|\d+[.)]) *")
# Verbose hedging prefixes the LLM sometimes emits even in voice mode
_HEDGE_RE = re.compile(
    r"^(According to (the |these )?(provided |given )?(sources?|context|documents?)[,.]?\s*)+",
    re.IGNORECASE,
)


def _strip_citations(text: str) -> str:
    """Remove citation markers and verbose hedging prefixes for spoken output."""
    text = _CITATION_NUM_RE.sub("", text)
    text = _CITATION_SOURCE_RE.sub("", text)
    text = _HEDGE_RE.sub("", text)
    return text.strip()


def _sentence_chunks(text: str) -> list[str]:
    """
    Split text into sentence-sized chunks for early TTS streaming.

    Phase 3 P0-1 fix: the old regex-only split on [.!?] left bullet-list
    answers as a single chunk (no sentence-terminal punctuation on list items),
    preventing TTS streaming — the full answer arrived as one TextFrame.

    New logic:
      1. Split on newlines first — each bullet / paragraph becomes its own unit.
      2. Strip list markers (-, •, *, 1., 1)) so TTS doesn't say "dash" aloud.
      3. Within each line, further split on sentence boundaries [.!?].
      4. Drop empty strings.

    Cartesia starts generating audio for chunk 1 while we push chunk 2 — this
    hides per-sentence TTS latency behind the pipeline, reducing perceived TTFT.
    """
    chunks: list[str] = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip leading list marker (dash, bullet, numbered)
        line = _LIST_MARKER_RE.sub("", line).strip()
        if not line:
            continue
        # Further split on sentence-terminal punctuation within the line
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            sentence = sentence.strip()
            if sentence:
                chunks.append(sentence)
    return chunks


# ── RAG frame processor ────────────────────────────────────────────────────────

class RAGProcessor(FrameProcessor):
    """
    Pipecat FrameProcessor that replaces the LLM step in the voice pipeline.

    On receiving a finalised TranscriptionFrame it:
      1. Runs the RAG query() in a thread executor (query() is synchronous).
      2. Strips citation markers from the answer.
      3. Pushes: LLMFullResponseStartFrame → TextFrame per sentence →
                 LLMFullResponseEndFrame.

    All other frames are passed through unchanged so the rest of the pipeline
    (TTS, transport) continues to work normally.
    """

    def __init__(
        self,
        index,
        nodes: list,
        storage_context,
        retriever,
        reranker,
        groq_api_key: str,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        super().__init__()
        self._index = index
        self._nodes = nodes
        self._storage_context = storage_context
        self._retriever = retriever
        self._reranker = reranker
        self._groq_api_key = groq_api_key
        self._executor = executor or ThreadPoolExecutor(max_workers=1)
        self._last_question: str = ""
        self._last_question_time: float = 0.0
        self._pending_future: Optional[asyncio.Future] = None
        self._answer_task: Optional[asyncio.Task] = None
        self._speech_active: bool = False
        # Deepgram resets its buffer after each auto-finalization, so a single
        # utterance can produce multiple TranscriptionFrames (segments) separated
        # by VAD-internal pauses.  We accumulate all segments during speech, then
        # append the Finalize response and concatenate the whole thing.
        self._transcript_segments: list[str] = []
        # True after VADUserStoppedSpeakingFrame while waiting for the Finalize
        # response.  The Finalize command is sent by DeepgramSTTService.
        self._awaiting_finalize: bool = False
        # Fallback task: if Finalize returns empty (Deepgram has no new audio),
        # fire after 800 ms and process whatever segments we already buffered.
        self._finalize_timer: Optional[asyncio.Task] = None

    def _cancel_in_flight(self) -> None:
        """Cancel any running answer task and pending RAG future."""
        if self._answer_task and not self._answer_task.done():
            self._answer_task.cancel()
        if self._pending_future and not self._pending_future.done():
            self._pending_future.cancel()

    async def _finalize_fallback(self, direction: FrameDirection) -> None:
        """Called 800 ms after VADUserStoppedSpeakingFrame if no Finalize response arrived.

        Deepgram returns an empty Finalize response when no new audio arrived after
        the last auto-finalized segment.  In that case we process the buffered
        segments directly rather than waiting indefinitely.
        """
        await asyncio.sleep(0.8)
        if not self._awaiting_finalize:
            return  # Finalize response already handled
        self._awaiting_finalize = False
        self._finalize_timer = None
        combined = " ".join(self._transcript_segments).strip()
        if not combined:
            logger.warning("RAGProcessor | Finalize timeout — no buffered segments, nothing to process")
            return
        logger.info(f"RAGProcessor | Finalize timeout — using buffered segments: {combined!r:.80}")
        from pipecat.utils.time import time_now_iso8601
        await self.process_frame(TranscriptionFrame(combined, "", time_now_iso8601()), direction)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
            self._speech_active = True
            self._transcript_segments = []
            self._awaiting_finalize = False
            if self._finalize_timer and not self._finalize_timer.done():
                self._finalize_timer.cancel()
            self._finalize_timer = None
            self._cancel_in_flight()
            logger.info(f"RAGProcessor | speech started ({type(frame).__name__}), cancelled in-flight")
            await self.push_frame(frame, direction)

        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speech_active = False
            self._awaiting_finalize = True
            await self.push_frame(frame, direction)  # triggers Finalize in DeepgramSTTService
            # Fallback: if Finalize returns empty within 800 ms, use buffered segments.
            self._finalize_timer = asyncio.create_task(self._finalize_fallback(direction))
            logger.info(
                f"RAGProcessor | speech stopped — {len(self._transcript_segments)} segment(s) buffered, "
                f"awaiting Finalize"
            )

        elif isinstance(frame, TranscriptionFrame):
            if self._speech_active:
                # Deepgram auto-finalized a segment mid-utterance.  Buffer it.
                text = frame.text.strip()
                if text:
                    self._transcript_segments.append(text)
                await self.push_frame(frame, direction)
                return

            if self._awaiting_finalize:
                # This is the Finalize response — append tail segment and process.
                self._awaiting_finalize = False
                if self._finalize_timer and not self._finalize_timer.done():
                    self._finalize_timer.cancel()
                self._finalize_timer = None
                text = frame.text.strip()
                if text:
                    self._transcript_segments.append(text)
                combined = " ".join(self._transcript_segments).strip()
                logger.info(f"RAGProcessor | Finalize received — combined: {combined!r:.80}")
                if not combined:
                    return
                from pipecat.utils.time import time_now_iso8601
                await self.process_frame(
                    TranscriptionFrame(combined, "", time_now_iso8601()), direction
                )
                return
            question = frame.text.strip()
            if not question:
                return

            # ── STT dedup: drop fragments that are substrings of the last ────
            # Groq Whisper sometimes emits the main transcription then a trailing
            # fragment (e.g. "under RBI norms.") as a second finalized frame.
            # If the new text is a substring of the last question AND arrived
            # within 5 seconds, it's a spurious fragment — skip it.
            now = time.monotonic()
            if (
                self._last_question
                and (now - self._last_question_time) < 5.0
                and question.lower() in self._last_question.lower()
            ):
                logger.warning(
                    f"RAGProcessor | dedup: dropped fragment {question!r} "
                    f"(substring of {self._last_question!r})"
                )
                return

            self._last_question = question
            self._last_question_time = now

            # Cancel any previous answer still running (e.g. user asked again
            # before the previous answer finished).
            self._cancel_in_flight()

            # Flush any TTS audio already queued in downstream processors.
            # This covers the case where _answer_task finished quickly and all
            # TextFrames were already pushed to TTS/transport before the user
            # spoke again — meaning UserStartedSpeakingFrame arrived too late to
            # cancel anything.  InterruptionTaskFrame pushed upstream is picked
            # up by the PipelineTask, which converts it into an InterruptionFrame
            # sent downstream, clearing queued audio in TTS and transport output.
            await self.push_frame(InterruptionTaskFrame(), FrameDirection.UPSTREAM)

            t_question = time.monotonic()
            logger.info(f"RAGProcessor | question: {question!r}")
            # Run _answer as a background task so process_frame returns
            # immediately and can receive UserStartedSpeakingFrame mid-answer.
            self._answer_task = asyncio.create_task(
                self._answer(question, t_question)
            )

        else:
            await self.push_frame(frame, direction)

    async def _answer(self, question: str, t_question: float) -> None:
        from src.generation.rag_chain import query

        # ── Guardrail 1: ethics hard-block (keyword, instant) ────────────────
        if _has_ethics_violation(question):
            logger.warning(f"RAGProcessor | ethics blocked: {question!r}")
            await self._push_reply(_ETHICS_REPLY, t_question)
            return

        # ── Guardrail 2: 3-way classifier (Groq 8b, ~150 ms) ─────────────────
        label = await _classify_query(question, self._groq_api_key)
        t_classified = time.monotonic()
        logger.info(
            f"RAGProcessor | classifier={label} "
            f"elapsed={1000*(t_classified - t_question):.0f}ms"
        )

        if label == "CHAT":
            # Greetings, thanks, farewells — respond naturally, skip RAG
            reply = await _conversational_reply(question, self._groq_api_key)
            await self._push_reply(reply, t_question)
            return

        if label == "BLOCKED":
            await self._push_reply(_OUT_OF_SCOPE_REPLY, t_question)
            return

        # label == "FINANCE" — fall through to RAG

        # ── RAG call ──────────────────────────────────────────────────────────
        loop = asyncio.get_running_loop()
        self._pending_future = loop.run_in_executor(
            self._executor,
            lambda: query(
                question,
                index=self._index,
                nodes=self._nodes,
                storage_context=self._storage_context,
                mode="voice",
                retriever=self._retriever,
                reranker=self._reranker,
            ),
        )
        try:
            result = await self._pending_future
        except asyncio.CancelledError:
            logger.warning(f"RAGProcessor | query cancelled (user interrupted): {question!r}")
            return
        except Exception as exc:
            logger.error(f"RAGProcessor | query FAILED: {exc}")
            await self._push_reply("Sorry, I was unable to retrieve an answer. Please try again.", t_question)
            return

        t_rag_done = time.monotonic()

        # Early exit: if the best reranker score is below the absolute floor,
        # no retrieved node has domain signal for this query — skip the LLM
        # answer and return out-of-scope rather than hallucinating a refusal.
        _RAG_FLOOR = -4.0
        if result.top_score < _RAG_FLOOR:
            logger.warning(
                f"RAGProcessor | top_score={result.top_score:.2f} < floor({_RAG_FLOOR}) "
                f"— no relevant content, returning out-of-scope reply"
            )
            await self._push_reply(_OUT_OF_SCOPE_REPLY, t_question)
            return

        answer = _strip_citations(result.answer)
        chunks = _sentence_chunks(answer)
        logger.info(
            f"RAGProcessor | sources={result.num_sources} "
            f"answer_len={len(answer)} chunks={len(chunks)} "
            f"rag_elapsed={1000*(t_rag_done - t_classified):.0f}ms "
            f"total_elapsed={1000*(t_rag_done - t_question):.0f}ms"
        )

        try:
            await self.push_frame(LLMFullResponseStartFrame())
            first_chunk = True
            for chunk in chunks:
                await self.push_frame(TextFrame(text=chunk))
                if first_chunk:
                    t_ttft = time.monotonic()
                    logger.info(
                        f"RAGProcessor | TTFT={1000*(t_ttft - t_question):.0f}ms "
                        f"first_chunk={chunk!r:.60}"
                    )
                    first_chunk = False
            await self.push_frame(LLMFullResponseEndFrame())
        except asyncio.CancelledError:
            # User interrupted mid-answer — stop pushing frames.
            # The transport-level interruption already cleared the audio queue;
            # we just need to exit cleanly without an unhandled exception.
            logger.info(f"RAGProcessor | answer push cancelled mid-stream for {question!r}")
            return

    async def _push_reply(self, text: str, t_question: float | None = None) -> None:
        """Wrap a plain text string in LLM response frames for TTS."""
        await self.push_frame(LLMFullResponseStartFrame())
        first_chunk = True
        for chunk in _sentence_chunks(text):
            await self.push_frame(TextFrame(text=chunk))
            if first_chunk and t_question is not None:
                logger.info(
                    f"RAGProcessor | TTFT={1000*(time.monotonic() - t_question):.0f}ms "
                    f"(guardrail/chat path)"
                )
                first_chunk = False
        await self.push_frame(LLMFullResponseEndFrame())


# ── TTS factory ────────────────────────────────────────────────────────────────

def _make_tts_service(
    cartesia_api_key: Optional[str],
    deepgram_api_key: Optional[str],
):
    """
    Return the appropriate TTS service.

    Priority: Cartesia (primary, 90 ms TTFT) → Deepgram (fallback).
    Both missing → RuntimeError.
    """
    if cartesia_api_key:
        from pipecat.services.cartesia.tts import CartesiaTTSService
        logger.info("TTS: Cartesia Sonic (primary)")
        return CartesiaTTSService(
            api_key=cartesia_api_key,
            voice_id="694f9389-aac1-45b6-b726-9d9369183238",  # Barbershop Man (clear, neutral)
            model="sonic-2",
            sample_rate=16000,
            encoding="pcm_s16le",
        )
    if deepgram_api_key:
        from pipecat.services.deepgram.tts import DeepgramTTSService
        logger.warning("TTS: Deepgram Aura (fallback — Cartesia key not set)")
        return DeepgramTTSService(
            api_key=deepgram_api_key,
            voice="aura-asteria-en",
            sample_rate=16000,      # must match LiveKitParams.audio_out_sample_rate
            encoding="linear16",
        )
    raise RuntimeError(
        "No TTS API key found. Set CARTESIA_API_KEY (primary) or DEEPGRAM_API_KEY (fallback)."
    )


# ── RAG stack bootstrap ────────────────────────────────────────────────────────

def _load_rag_stack(groq_api_key: str):
    """
    Load the full RAG stack from persisted storage.

    Mirrors eval_runner.load_pipeline() but returns the pre-built
    (retriever, reranker) pair so RAGProcessor never reloads the cross-encoder.
    """
    from pathlib import Path

    from src.ingestion.chunker import chunk_document
    from src.ingestion.metadata import build_section_map_and_clean
    from src.ingestion.parser import load_pdf_with_sections
    from src.retrieval.hybrid_retriever import configure_llm
    from src.retrieval.pipeline import build_pipeline
    from src.retrieval.vector_store import load_index

    # Remove stale Qdrant lock from a previous interrupted process
    qdrant_lock = Path("qdrant_storage/.lock")
    if qdrant_lock.exists():
        qdrant_lock.unlink()
        logger.warning("Removed stale qdrant_storage/.lock")

    configure_llm(groq_api_key)

    pdf_path = "data/raw/rbi_kyc_master_direction.pdf"
    logger.info(f"Parsing {pdf_path} ...")
    text_marked = load_pdf_with_sections(pdf_path)
    chapter_map, section_map, clean_text = build_section_map_and_clean(text_marked)
    leaf_nodes, storage_context = chunk_document(
        text=clean_text,
        filename=pdf_path,
        section_map=section_map,
        chapter_map=chapter_map,
        clean_text=clean_text,
    )
    logger.info(f"{len(leaf_nodes)} leaf nodes loaded")

    index = load_index(storage_context)
    logger.info("Qdrant index loaded")

    logger.info("Building retrieval pipeline (loads BGE reranker once) ...")
    retriever, reranker = build_pipeline(
        index,
        leaf_nodes,
        storage_context,
        similarity_top_k=8,   # voice: fewer candidates → fewer reranker batches (~2 vs 4)
        reranker_top_n=5,
    )
    logger.info("Retrieval pipeline ready")

    return index, leaf_nodes, storage_context, retriever, reranker


# ── Pipeline factory ───────────────────────────────────────────────────────────

async def build_voice_pipeline(
    livekit_url: str,
    livekit_token: str,
    livekit_room: str,
    groq_api_key: str,
    cartesia_api_key: Optional[str] = None,
    deepgram_api_key: Optional[str] = None,
    preloaded_rag_stack: Optional[dict] = None,
) -> tuple[PipelineTask, LiveKitTransport]:
    """
    Assemble and return the full voice pipeline as a (PipelineTask, transport) pair.

    The caller is responsible for:
      - Calling `await runner.run(task)` to start the pipeline.
      - Registering event handlers on `transport` if needed (e.g. on_participant_joined).

    Pipeline stages
    ---------------
    transport.input()   LiveKit audio in (16 kHz mono PCM, VAD enabled)
    DeepgramSTTService  nova-2-general streaming → TranscriptionFrame
    RAGProcessor        TranscriptionFrame → RAG query → TextFrame chunks
    TTS service         TextFrame → audio (Cartesia primary, Deepgram fallback)
    transport.output()  LiveKit audio out

    preloaded_rag_stack
    -------------------
    When provided (dict with keys: index, nodes, storage_context, retriever, reranker),
    _load_rag_stack() is skipped entirely — the server pre-loads this once at startup
    so the per-connection pipeline assembly is fast (~2-3 s, transport connect only).
    """
    executor = ThreadPoolExecutor(max_workers=1)

    if preloaded_rag_stack is not None:
        index           = preloaded_rag_stack["index"]
        nodes           = preloaded_rag_stack["nodes"]
        storage_context = preloaded_rag_stack["storage_context"]
        retriever       = preloaded_rag_stack["retriever"]
        reranker        = preloaded_rag_stack["reranker"]
        logger.info("build_voice_pipeline: using pre-loaded RAG stack")
    else:
        # Fallback: load synchronously in an executor (used by standalone _main()
        # and any caller that hasn't pre-warmed the stack).
        loop = asyncio.get_event_loop()
        index, nodes, storage_context, retriever, reranker = await loop.run_in_executor(
            executor, _load_rag_stack, groq_api_key
        )

    # ── Transport ──────────────────────────────────────────────────────────────
    transport = LiveKitTransport(
        url=livekit_url,
        token=livekit_token,
        room_name=livekit_room,
        params=LiveKitParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_in_channels=1,
            audio_out_enabled=True,
            audio_out_sample_rate=16000,
            audio_out_channels=1,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(
                sample_rate=16000,
                params=VADParams(
                    confidence=0.7,
                    start_secs=0.2,
                    stop_secs=1.5,   # 1.5s silence — long enough for full regulatory questions
                    min_volume=0.6,
                ),
            ),
        ),
    )

    # ── STT ────────────────────────────────────────────────────────────────────
    if not deepgram_api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is required for STT (streaming, low-latency)")
    stt = DeepgramSTTService(
        api_key=deepgram_api_key,
        sample_rate=16000,
        settings=DeepgramSTTService.Settings(
            model="nova-2-general",
            smart_format=True,
            punctuate=True,
            # endpointing=False: disable Deepgram's own mid-sentence auto-finalization.
            # Without this, Deepgram fires partial transcripts on short pauses inside a
            # sentence, e.g. "under RBI known." instead of the full question.
            # Transcripts arrive only when pipecat sends an explicit Finalize command,
            # which is triggered by Silero VAD's VADUserStoppedSpeakingFrame (stop_secs=1.5).
            endpointing=False,
            # interim_results=True (default) — required for the Finalize mechanism.
        ),
    )
    # pipecat 0.0.108 bug: language_to_service_language() returns Language enum
    # object; _build_connect_kwargs() calls str() on it → 'Language.EN' instead
    # of 'en', causing HTTP 400 on the Deepgram WebSocket handshake.
    # Fix: overwrite the stored language with the plain string after construction.
    stt._settings.language = "en"

    # ── RAG processor ──────────────────────────────────────────────────────────
    rag_proc = RAGProcessor(
        index=index,
        nodes=nodes,
        storage_context=storage_context,
        retriever=retriever,
        reranker=reranker,
        groq_api_key=groq_api_key,
        executor=executor,
    )

    # ── TTS ────────────────────────────────────────────────────────────────────
    tts = _make_tts_service(cartesia_api_key, deepgram_api_key)

    # ── Pipeline ───────────────────────────────────────────────────────────────
    pipeline = Pipeline([
        transport.input(),
        stt,
        rag_proc,
        tts,
        transport.output(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True),
    )

    return task, transport


# ── Standalone entry-point ─────────────────────────────────────────────────────

async def _main() -> None:
    load_dotenv()

    livekit_url   = os.environ["LIVEKIT_URL"]
    livekit_token = os.environ["LIVEKIT_TOKEN"]
    livekit_room  = os.environ.get("LIVEKIT_ROOM", "rag-voice")
    groq_api_key  = os.environ["GROQ_API_KEY"]
    cartesia_key  = os.environ.get("CARTESIA_API_KEY")
    deepgram_key  = os.environ.get("DEEPGRAM_API_KEY")

    task, transport = await build_voice_pipeline(
        livekit_url=livekit_url,
        livekit_token=livekit_token,
        livekit_room=livekit_room,
        groq_api_key=groq_api_key,
        cartesia_api_key=cartesia_key,
        deepgram_api_key=deepgram_key,
    )

    @transport.event_handler("on_first_participant_joined")
    async def on_joined(transport, participant):
        logger.info(f"First participant joined: {participant}")
        await task.queue_frames([
            LLMFullResponseStartFrame(),
            TextFrame(text=GREETING_TEXT),
            LLMFullResponseEndFrame()
        ])

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(_main())
