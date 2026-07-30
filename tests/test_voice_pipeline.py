"""
Unit tests for src/voice/pipeline.py

Test groups
-----------
  TestHelpers        pure functions: _strip_citations, _sentence_chunks
  TestRAGProcessor   RAGProcessor._answer() with push_frame mocked
  TestTTSFactory     _make_tts_service key-selection logic

RAGProcessor is tested by:
  1. Patching src.generation.rag_chain.query with a fake result.
  2. Mocking self.push_frame to capture emitted frames.
  3. Calling _answer() directly — this is the core logic under test.

The full LiveKit / STT / TTS stack is NOT exercised here; those need live creds.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("DEEPGRAM_API_KEY", "test-deepgram-key")

from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from src.voice.pipeline import (
    RAGProcessor,
    _make_tts_service,
    _sentence_chunks,
    _strip_citations,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_processor() -> RAGProcessor:
    return RAGProcessor(
        index=None, nodes=[], storage_context=None,
        retriever=None, reranker=None,
    )


def _fake_result(answer: str) -> MagicMock:
    r = MagicMock()
    r.answer = answer
    r.num_sources = 1
    r.sources = []
    return r


async def _collect_answer(processor: RAGProcessor, question: str) -> list:
    """
    Call processor._answer(question) with push_frame replaced by an AsyncMock.
    Returns the list of frames pushed in order.
    """
    pushed: list = []

    async def _capture(frame, direction=FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    processor.push_frame = _capture
    await processor._answer(question)
    return pushed


# ── TestHelpers ────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_strip_citations_removes_inline_markers(self):
        assert _strip_citations("Answer [1] and [2] text.") == "Answer and text."

    def test_strip_citations_leading_marker(self):
        assert _strip_citations("[1] First sentence.") == "First sentence."

    def test_strip_citations_no_markers(self):
        assert _strip_citations("Clean text.") == "Clean text."

    def test_strip_citations_consecutive(self):
        assert _strip_citations("Fact[1][2] confirmed.") == "Fact confirmed."

    def test_sentence_chunks_splits_on_period(self):
        assert _sentence_chunks("One. Two. Three.") == ["One.", "Two.", "Three."]

    def test_sentence_chunks_splits_on_exclamation_question(self):
        chunks = _sentence_chunks("Hi! Really? Yes.")
        assert chunks == ["Hi!", "Really?", "Yes."]

    def test_sentence_chunks_single_sentence(self):
        assert _sentence_chunks("Only one.") == ["Only one."]

    def test_sentence_chunks_empty_string(self):
        assert _sentence_chunks("") == []

    def test_sentence_chunks_whitespace_only(self):
        assert _sentence_chunks("   ") == []

    def test_sentence_chunks_strips_each_chunk(self):
        chunks = _sentence_chunks("  Hello.  World.  ")
        assert all(c == c.strip() for c in chunks)


# ── TestRAGProcessor ───────────────────────────────────────────────────────────

class TestRAGProcessor:

    @pytest.mark.asyncio
    async def test_answer_emits_start_text_end_frames(self):
        answer = "KYC requires Aadhaar. PAN is also needed."
        with patch("src.generation.rag_chain.query", return_value=_fake_result(answer)):
            frames = await _collect_answer(_make_processor(), "What is KYC?")

        types = [type(f).__name__ for f in frames]
        assert types[0]  == "LLMFullResponseStartFrame"
        assert types[-1] == "LLMFullResponseEndFrame"
        assert "TextFrame" in types

    @pytest.mark.asyncio
    async def test_answer_splits_into_sentence_chunks(self):
        answer = "First sentence. Second sentence. Third sentence."
        with patch("src.generation.rag_chain.query", return_value=_fake_result(answer)):
            frames = await _collect_answer(_make_processor(), "Question?")

        texts = [f.text for f in frames if isinstance(f, TextFrame)]
        assert texts == ["First sentence.", "Second sentence.", "Third sentence."]

    @pytest.mark.asyncio
    async def test_citations_stripped_before_tts(self):
        answer = "Aadhaar required [1]. PAN also needed [2]."
        with patch("src.generation.rag_chain.query", return_value=_fake_result(answer)):
            frames = await _collect_answer(_make_processor(), "KYC docs?")

        for f in frames:
            if isinstance(f, TextFrame):
                assert "[" not in f.text, f"citation leaked: {f.text!r}"

    @pytest.mark.asyncio
    async def test_rag_error_emits_fallback_text(self):
        with patch("src.generation.rag_chain.query", side_effect=RuntimeError("Groq 500")):
            frames = await _collect_answer(_make_processor(), "What is KYC?")

        types = [type(f).__name__ for f in frames]
        assert "LLMFullResponseStartFrame" in types
        assert "LLMFullResponseEndFrame"   in types
        texts = [f.text for f in frames if isinstance(f, TextFrame)]
        assert len(texts) == 1
        assert "Sorry" in texts[0] or "unable" in texts[0].lower()

    @pytest.mark.asyncio
    async def test_frame_order_start_before_texts_before_end(self):
        answer = "Answer sentence one. Answer sentence two."
        with patch("src.generation.rag_chain.query", return_value=_fake_result(answer)):
            frames = await _collect_answer(_make_processor(), "Q?")

        types = [type(f).__name__ for f in frames]
        start_i = types.index("LLMFullResponseStartFrame")
        end_i   = types.index("LLMFullResponseEndFrame")
        text_is = [i for i, t in enumerate(types) if t == "TextFrame"]
        assert start_i < min(text_is)
        assert end_i   > max(text_is)

    @pytest.mark.asyncio
    async def test_process_frame_ignores_non_finalized_transcription(self):
        proc = _make_processor()
        pushed = []

        async def _capture(frame, direction=FrameDirection.DOWNSTREAM):
            pushed.append(frame)

        proc.push_frame = _capture

        # patch process_frame's call to super() to avoid TaskManager requirement
        with patch.object(proc.__class__.__bases__[0], "process_frame", new=AsyncMock()):
            tf = TranscriptionFrame(text="partial", user_id="u1", timestamp="", finalized=False)
            await proc.process_frame(tf, FrameDirection.DOWNSTREAM)

        assert not any(isinstance(f, LLMFullResponseStartFrame) for f in pushed)

    @pytest.mark.asyncio
    async def test_process_frame_ignores_empty_question(self):
        proc = _make_processor()
        pushed = []

        async def _capture(frame, direction=FrameDirection.DOWNSTREAM):
            pushed.append(frame)

        proc.push_frame = _capture

        with patch.object(proc.__class__.__bases__[0], "process_frame", new=AsyncMock()):
            tf = TranscriptionFrame(text="   ", user_id="u1", timestamp="", finalized=True)
            await proc.process_frame(tf, FrameDirection.DOWNSTREAM)

        assert not any(isinstance(f, LLMFullResponseStartFrame) for f in pushed)

    @pytest.mark.asyncio
    async def test_single_sentence_answer_one_text_frame(self):
        answer = "KYC is mandatory for all accounts."
        with patch("src.generation.rag_chain.query", return_value=_fake_result(answer)):
            frames = await _collect_answer(_make_processor(), "What is KYC?")

        texts = [f.text for f in frames if isinstance(f, TextFrame)]
        assert len(texts) == 1
        assert texts[0] == "KYC is mandatory for all accounts."


# ── TestTTSFactory ─────────────────────────────────────────────────────────────

class TestTTSFactory:

    def test_cartesia_key_selects_cartesia(self):
        with patch("pipecat.services.cartesia.tts.CartesiaTTSService.__init__", return_value=None):
            svc = _make_tts_service(cartesia_api_key="ckey", deepgram_api_key=None)
        from pipecat.services.cartesia.tts import CartesiaTTSService
        assert isinstance(svc, CartesiaTTSService)

    def test_deepgram_key_selects_deepgram_when_no_cartesia(self):
        with patch("pipecat.services.deepgram.tts.DeepgramTTSService.__init__", return_value=None):
            svc = _make_tts_service(cartesia_api_key=None, deepgram_api_key="dkey")
        from pipecat.services.deepgram.tts import DeepgramTTSService
        assert isinstance(svc, DeepgramTTSService)

    def test_cartesia_takes_priority_over_deepgram(self):
        with patch("pipecat.services.cartesia.tts.CartesiaTTSService.__init__", return_value=None):
            svc = _make_tts_service(cartesia_api_key="ckey", deepgram_api_key="dkey")
        from pipecat.services.cartesia.tts import CartesiaTTSService
        assert isinstance(svc, CartesiaTTSService)

    def test_no_keys_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="No TTS API key"):
            _make_tts_service(cartesia_api_key=None, deepgram_api_key=None)

    def test_empty_string_cartesia_key_falls_back_to_deepgram(self):
        with patch("pipecat.services.deepgram.tts.DeepgramTTSService.__init__", return_value=None):
            svc = _make_tts_service(cartesia_api_key="", deepgram_api_key="dkey")
        from pipecat.services.deepgram.tts import DeepgramTTSService
        assert isinstance(svc, DeepgramTTSService)
