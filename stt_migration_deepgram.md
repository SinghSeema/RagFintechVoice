# STT Migration: GroqSTTService → DeepgramSTTService

**Date:** April 2026  
**File changed:** `src/voice/pipeline.py`  
**pipecat version:** 0.0.108

---

## Why we migrated

| | GroqSTTService (Whisper) | DeepgramSTTService (nova-2) |
|---|---|---|
| Mode | Batch — buffers full audio, then transcribes | Streaming — transcribes incrementally |
| Latency | High (~1–2 s after speech ends) | Low (~200–400 ms after speech ends) |
| VAD integration | Relies on pipecat's internal VAD stop to flush buffer | Native Finalize mechanism: sends `Finalize` WS message on VAD stop |
| Mid-sentence fragments | Rare | Possible (auto-endpointing on short pauses) |

Groq/Whisper was causing noticeable delay between the user finishing speaking and the bot starting to answer. Deepgram streaming eliminates most of that gap.

---

## Migration change

**Before:**
```python
from pipecat.services.groq import GroqSTTService

stt = GroqSTTService(
    api_key=groq_api_key,
    model="whisper-large-v3-turbo",
    language="en",
)
```

**After:**
```python
from pipecat.services.deepgram.stt import DeepgramSTTService

stt = DeepgramSTTService(
    api_key=deepgram_api_key,
    sample_rate=16000,
    settings=DeepgramSTTService.Settings(
        model="nova-2-general",
        smart_format=True,
        punctuate=True,
        endpointing=False,
        # interim_results=True (default) — required for Finalize mechanism
    ),
)
stt._settings.language = "en"   # bug workaround — see Bug 1 below
```

---

## Bugs encountered and fixes

### Bug 1 — HTTP 400 on Deepgram WebSocket handshake

**Symptom:**
```
Unexpected error when initializing websocket connection
aiohttp.client_exceptions.ClientResponseError: 400, message='Bad Request'
```

**Root cause (pipecat 0.0.108):**  
`DeepgramSTTService._build_connect_kwargs()` calls `language_to_service_language()` which returns a `Language` enum object (e.g. `Language.EN`). The code then does `str(language)` → `'Language.EN'` instead of `'en'`. Deepgram rejects the malformed language code.

**What we tried first (wrong):**  
Passing `live_options=LiveOptions(language="en-IN")` — `en-IN` is not a valid Deepgram language code, still 400.

**Fix:**  
Bypass the broken helper entirely by using the `Settings` dataclass and overwriting the language field as a plain string after construction:
```python
stt._settings.language = "en"
```

---

### Bug 2 — WebSocket connected but no transcription at all

**Symptom:**  
Deepgram WebSocket connected successfully, VAD frames appeared in logs, but no `TranscriptionFrame` ever arrived.

**Root cause:**  
We had set `interim_results=False` in settings. pipecat's Deepgram integration uses interim results to drive its VAD→Finalize mechanism. With them disabled, the `Finalize` command is never sent and no final transcript is returned.

**Fix:**  
Remove `interim_results=False` — leave it at the default `True`.

---

### Bug 3 — `TranscriptionFrame.finalized` always False

**Symptom:**  
Code checking `if frame.finalized:` to gate processing never triggered — even on frames that were clearly final transcriptions.

**Root cause (pipecat 0.0.108):**  
pipecat never sets `finalized=True` when pushing `TranscriptionFrame` from Deepgram's `is_final=True` WebSocket response. The field defaults to `False` and is never updated.

**Fix:**  
Stop relying on `.finalized`. Instead use frame type:
- `TranscriptionFrame` → final segment (use for processing)
- `InterimTranscriptionFrame` → partial/interim (ignore)

---

### Bug 4 — `VADUserStartedSpeakingFrame` not triggering barge-in / interruption

**Symptom:**  
User speaking while bot was answering did not cancel the in-flight answer. The `UserStartedSpeakingFrame` handler in `RAGProcessor` was never reached.

**Root cause:**  
`VADUserStartedSpeakingFrame` (emitted by Silero VAD at the transport level) is a completely separate class — it does **not** inherit from `UserStartedSpeakingFrame`. The `isinstance` check only matched `UserStartedSpeakingFrame`.

**Fix:**
```python
# Before
if isinstance(frame, UserStartedSpeakingFrame):

# After
if isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
```

---

### Bug 5 — Partial query sent to RAG mid-utterance

**Symptom:**  
For a long question like *"What are the officially valid documents accepted for individual KYC under RBI?"*, RAG received only `"under RBI known."` — the tail of an internal segment, not the full question.

**Root cause:**  
Deepgram's auto-endpointing fires on short pauses inside a long sentence, sending the text up to that point as a `TranscriptionFrame`. With `endpointing` enabled (default), these mid-utterance partial transcripts reached the RAG processor immediately.

**Fix — two parts:**

1. Set `endpointing=False` in Deepgram settings to suppress Deepgram's own auto-segmentation.
2. Add a `_speech_active` gate in `RAGProcessor`: while the user is still speaking (between `VADUserStartedSpeakingFrame` and `VADUserStoppedSpeakingFrame`), buffer all `TranscriptionFrame`s — do not process them yet.

```python
if self._speech_active:
    if text:
        self._transcript_segments.append(text)
    await self.push_frame(frame, direction)
    return
```

---

### Bug 6 — Two separate RAG queries for one utterance

**Symptom:**  
Logs showed two `RAGProcessor | question:` entries for a single utterance:
```
RAGProcessor | question: '...officially valid document'
RAGProcessor | question: 'accepted for individual KYC under RBI?'
```

**Root cause:**  
Even with `endpointing=False`, Deepgram still resets its internal buffer after each auto-finalized segment. So one utterance produced:
- **Segment 1** (`TranscriptionFrame` mid-speech): `"...officially valid document"` — buffered correctly by the `_speech_active` gate
- **Segment 2** (Finalize response after VAD stop): `"accepted for individual KYC under RBI?"` — only the tail

Because the Finalize response contained only the tail (Deepgram had already cleared segment 1 from its buffer), the combined text was still incomplete.

Previously with `endpointing=True`, both segments reached the RAG processor as independent queries.

**Fix — segment accumulation + Finalize assembly:**

`RAGProcessor` now accumulates all segments in a list during the speech window, then concatenates them when the Finalize response arrives:

```
VADUserStartedSpeakingFrame  →  reset _transcript_segments = []
TranscriptionFrame (mid-speech)  →  append to _transcript_segments, do NOT process
VADUserStoppedSpeakingFrame  →  set _awaiting_finalize=True, push frame downstream
                                 (pipecat sends Finalize to Deepgram)
                                 start 800ms fallback timer
TranscriptionFrame (Finalize response)  →  append tail, join all segments, process combined
```

**800ms fallback timer:**  
If the Finalize response is empty (Deepgram has no new audio because the last auto-segment already consumed all speech), the fallback fires and processes the buffered segments directly:

```python
async def _finalize_fallback(self, direction):
    await asyncio.sleep(0.8)
    if not self._awaiting_finalize:
        return   # Finalize already handled
    combined = " ".join(self._transcript_segments).strip()
    if combined:
        await self.process_frame(TranscriptionFrame(combined, "", time_now_iso8601()), direction)
```

---

## Final state — `RAGProcessor` speech/transcript state machine

```
_speech_active      True between VADUserStarted and VADUserStopped
_transcript_segments  list of str; all TranscriptionFrame texts buffered during speech
_awaiting_finalize  True after VADUserStopped while waiting for Finalize response
_finalize_timer     asyncio.Task for 800ms fallback
```

**Frame flow (happy path — one or more segments):**

```
VADUserStartedSpeakingFrame
  → _speech_active=True, segments=[], cancel in-flight answer

TranscriptionFrame (mid-utterance, Deepgram auto-segment)
  → buffered in _transcript_segments, not processed

VADUserStoppedSpeakingFrame
  → _speech_active=False, _awaiting_finalize=True
  → pushed downstream (triggers pipecat Finalize to Deepgram)
  → 800ms fallback timer started

TranscriptionFrame (Finalize response — may be empty or tail segment)
  → _awaiting_finalize=False, timer cancelled
  → append tail to _transcript_segments
  → combined = join(_transcript_segments)
  → process_frame(TranscriptionFrame(combined)) → RAG pipeline
```

---

## Key logs to watch

```
RAGProcessor | speech started (VADUserStartedSpeakingFrame), cancelled in-flight
RAGProcessor | speech stopped — 2 segment(s) buffered, awaiting Finalize
RAGProcessor | Finalize received — combined: 'What are the officially valid documents...'
RAGProcessor | question: 'What are the officially valid documents...'
Classifier | FINANCE | 'What are the officially valid documents...'
RAGProcessor | classifier=FINANCE elapsed=148ms
RAGProcessor | sources=5 answer_len=312 chunks=4 rag_elapsed=1240ms total_elapsed=1390ms
RAGProcessor | TTFT=1392ms first_chunk='According to RBI Master Direction...'
```

Fallback path (empty Finalize):
```
RAGProcessor | Finalize timeout — using buffered segments: 'What are the officially...'
```

---

## Notes

- **Model used:** `nova-2-general` (free tier). `nova-2-finance` may offer better accuracy on banking/regulatory terms but requires a paid plan.
- **`DebugFrameLogger`** is still present in `pipeline.py` as a class but not wired into the pipeline. Insert it between any two stages for ad-hoc debugging:
  ```python
  pipeline = Pipeline([..., DebugFrameLogger("after-stt"), stt, ...])
  ```
- The `stt._settings.language = "en"` workaround is a patch for pipecat 0.0.108. Check if it's still needed when upgrading pipecat.
