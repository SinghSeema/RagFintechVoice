# Task: Two-Gate Endpointing — Silence Threshold + Semantic Completeness
**Project:** RagFintechVoice (Voice Layer)  
**Stack:** Python · Pipecat · Deepgram Nova · Groq · LlamaIndex RAG chain  
**Goal:** Fix the problem where user pauses mid-query trigger the RAG pipeline prematurely. Implement a two-gate system: silence threshold gate (Deepgram config) + semantic completeness gate (lightweight LLM check).

---

## Context You Must Understand First

### The actual failure mode
The current pipeline fires the RAG chain the moment Deepgram reports end-of-speech. This causes two distinct failure types:

**Type 1 — Silence trigger (Deepgram fires too early)**  
User says: "Mera account ka..." *(pauses 400ms to think)* "...KYC kab complete hoga?"  
Current behaviour: Pipeline fires on "Mera account ka..." — incomplete query reaches the LLM.

**Type 2 — Semantic trigger (silence threshold is fine, but utterance is still incomplete)**  
User says: "If I have a savings account and also a fixed deposit..." *(silence 900ms)*  
Even with a generous silence threshold, this is an incomplete conditional — the user hasn't stated what they want to know. Current behaviour: fires anyway.

Both types need to be caught. They are solved at different layers.

### Current voice pipeline architecture
```
Microphone → Deepgram STT (streaming) → on_transcript() callback
                                              ↓
                                         RAG chain (fires immediately)  ← problem is here
                                              ↓
                                         Cartesia TTS → Speaker
```

### Target architecture after this task
```
Microphone → Deepgram STT (streaming)
                    ↓
         Pipecat internal _on_utterance_end()
                    ↓ broadcasts
         UserStoppedSpeakingFrame into pipeline
                    ↓
         EndpointingGate (FrameProcessor)   ← your code lives here
         [GATE 1] utterance_end_ms ≥ 1000ms (Deepgram config — already enforced before frame arrives)
         [GATE 2] SemanticGate — is this a complete query?
                    ↓ only if both gates pass — pushes TranscriptionFrame downstream
         LLM / RAG chain
                    ↓
         Cartesia TTS → Speaker

Note: VADUserStoppedSpeakingFrame (from VAD layer) bypasses EndpointingGate entirely — 
it is passed through untouched via the else branch.
```

### Existing project structure (do not break these)
```
ragfintech/
├── voice/
│   ├── pipeline.py          # Pipecat pipeline setup — THIS is where your work goes
│   ├── deepgram_handler.py  # Deepgram STT config and callbacks — extend this
│   └── tts_handler.py       # Cartesia TTS — do NOT modify
├── rag_chain.py             # Do NOT modify
├── config/
│   └── settings.py          # Extend with new .env keys
└── .env                     # Extend with new keys
```

---

## What To Build

---

## Gate 1: Deepgram Silence Threshold

This is a configuration change, not a code architecture change. Do it first — it's the foundation.

### Changes to `deepgram_handler.py`

Update the Deepgram connection options:

```python
# deepgram_handler.py — update DeepgramSTTOptions
options = DeepgramSTTOptions(
    model="nova-2",
    language="en-IN",          # Keep as-is — critical for Indian accent handling
    smart_format=True,
    interim_results=True,       # Must remain True — needed for Gate 2 to see partials
    utterance_end_ms=str(settings.DEEPGRAM_UTTERANCE_END_MS),   # was hardcoded or default
    endpointing=settings.DEEPGRAM_ENDPOINTING_MS,               # was hardcoded or default
    vad_events=True,            # Enable VAD events — needed to detect speech start/end
)
```

**What these params do:**
- `endpointing`: milliseconds of silence before Deepgram considers speech ended. Default is ~10ms — far too aggressive. Set to 500ms minimum.
- `utterance_end_ms`: milliseconds after the last word before Deepgram emits a final `UtteranceEnd` event. Set to 800ms minimum. This is the primary gate.
- `vad_events=True`: enables `SpeechStarted` and `UtteranceEnd` events separately from transcript events — required for the two-gate logic.

**Add to `.env`:**
```env
DEEPGRAM_UTTERANCE_END_MS=1000
DEEPGRAM_ENDPOINTING_MS=500
```

These are tunable. Start with 1000ms / 500ms. If users still get cut off mid-sentence, increase `UTTERANCE_END_MS` to 1200ms. If responses feel sluggish, decrease to 800ms.

### How Pipecat delivers UtteranceEnd — important

Do NOT register an `on_utterance_end` callback. Do NOT call `_on_utterance_end` directly. Pipecat handles this internally:

```python
# Inside Pipecat's Deepgram integration (do not touch this):
async def _on_utterance_end(self, message):
    await self._call_event_handler("on_utterance_end", message)
    await self.broadcast_frame(UserStoppedSpeakingFrame)   # ← this is what you catch
```

Pipecat converts Deepgram's `UtteranceEnd` event into `UserStoppedSpeakingFrame` and broadcasts it into the pipeline automatically. Your only job is to catch that frame in a `FrameProcessor`.

### Frame types — know the difference

```
UserStoppedSpeakingFrame      ← from Deepgram UtteranceEnd (silence timer based)
VADUserStoppedSpeakingFrame   ← from VAD layer (energy based)
```

`EndpointingGate` listens ONLY to `UserStoppedSpeakingFrame`. `VADUserStoppedSpeakingFrame` must pass through untouched via the `else` branch. Do not import `VADUserStoppedSpeakingFrame` into `EndpointingGate` — if you catch it, you reintroduce the double-trigger problem.

### Create `voice/endpointing_gate.py` — the frame processor

```python
"""
endpointing_gate.py

Pipecat FrameProcessor that implements two-gate endpointing.
Sits between DeepgramSTTService and the LLM in the pipeline.

Gate 1: Only acts on UserStoppedSpeakingFrame (Deepgram utterance_end_ms already enforced upstream).
Gate 2: SemanticGate — checks if accumulated transcript is a complete, answerable query.

VADUserStoppedSpeakingFrame bypasses this processor entirely (passed through via else branch).
"""

from pipecat.frames.frames import TranscriptionFrame, UserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameProcessor
from voice.semantic_gate import SemanticGate
import logging

logger = logging.getLogger(__name__)

class EndpointingGate(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._buffer: list[str] = []       # accumulates TranscriptionFrame finals
        self._pending_prefix: str = ""     # holds incomplete utterance across UtteranceEnd events
        self._gate = SemanticGate()

    async def process_frame(self, frame, direction):

        if isinstance(frame, TranscriptionFrame) and frame.final:
            # Accumulate final transcript segments — do not fire yet
            if frame.text.strip():
                self._buffer.append(frame.text.strip())
            await self.push_frame(frame, direction)   # pass through for logging

        elif isinstance(frame, UserStoppedSpeakingFrame):
            # Gate 1 has already passed — Deepgram only emits this after utterance_end_ms silence
            new_speech = " ".join(self._buffer).strip()
            self._buffer.clear()

            if not new_speech:
                # Nothing transcribed — pass frame through, don't invoke gate
                await self.push_frame(frame, direction)
                return

            # Prepend any held incomplete fragment from previous UtteranceEnd
            utterance = f"{self._pending_prefix} {new_speech}".strip()

            result = await self._gate.check(utterance)

            logger.info(
                f"EndpointingGate | complete={result.complete} | "
                f"confidence={result.confidence:.2f} | "
                f"latency={result.latency_ms:.0f}ms | "
                f"reason='{result.reason}' | "
                f"utterance='{utterance[:60]}'"
            )

            if self._gate.should_fire(result):
                self._pending_prefix = ""
                # Push a clean TranscriptionFrame downstream — this triggers the LLM/RAG chain
                await self.push_frame(
                    TranscriptionFrame(
                        text=utterance,
                        user_id=frame.user_id,
                        timestamp=frame.timestamp
                    ),
                    direction
                )
            else:
                # Hold — carry incomplete fragment forward
                self._pending_prefix = utterance
                logger.info(f"EndpointingGate: HELD — pending_prefix='{self._pending_prefix[:60]}'")

                # Safety valve — if prefix grows beyond 200 chars, user changed topic
                if len(self._pending_prefix) > 200:
                    logger.warning(
                        f"EndpointingGate: pending_prefix overflow, clearing. "
                        f"Was: '{self._pending_prefix[:80]}'"
                    )
                    self._pending_prefix = ""
                # Swallow the frame — do not push_frame — RAG does not fire

        else:
            # Everything else (including VADUserStoppedSpeakingFrame) passes through untouched
            await self.push_frame(frame, direction)
```

### Wire `EndpointingGate` into `pipeline.py`

```python
from voice.endpointing_gate import EndpointingGate

pipeline = Pipeline([
    transport.input(),
    stt,                    # DeepgramSTTService
    EndpointingGate(),      # ← between STT and LLM
    llm,
    tts,
    transport.output()
])
```

Nothing else in `pipeline.py` needs to change. Do not add `pending_prefix` as a module-level variable in `pipeline.py` — it lives inside `EndpointingGate` as `self._pending_prefix`.

---

## Gate 2: Semantic Completeness Check

This is the main engineering task. After Gate 1 passes (Deepgram fires `UtteranceEnd`), the accumulated transcript goes through a lightweight completeness classifier before the RAG chain is triggered.

### Create `voice/semantic_gate.py`

```python
"""
semantic_gate.py

Lightweight semantic completeness gate for voice query endpointing.
Sits between Deepgram STT output and the RAG chain.
Determines whether a user's utterance is a complete, answerable query
before firing the expensive RAG pipeline.
"""
```

### Core classification logic

```python
COMPLETENESS_SYSTEM_PROMPT = """You are a voice query classifier for a fintech banking assistant.

Your ONLY job is to determine if a user's spoken utterance is a COMPLETE, ANSWERABLE query.

A query is COMPLETE if:
- It asks a full question ("What documents do I need for KYC?")
- It makes a full statement or request ("Tell me about the KYC process")
- It is a short, unambiguous intent ("KYC status", "account balance")
- It contains a greeting that implies an implicit request ("Hello", "Hi there")

A query is INCOMPLETE if:
- It is a sentence fragment ("If I have a savings account and")
- It starts a conditional without completing it ("When the RBI says that")
- It is a partial question ("What is the re—")
- It trails off mid-thought ("So basically what I want to know is")
- It ends with a conjunction or preposition ("KYC documents for")

Respond with ONLY a JSON object. No explanation, no preamble.
{"complete": true, "confidence": 0.95, "reason": "full question with clear intent"}
OR
{"complete": false, "confidence": 0.88, "reason": "incomplete conditional, missing the question"}"""
```

### SemanticGate class

```python
import json
import logging
import asyncio
from dataclasses import dataclass
from groq import AsyncGroq
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class CompletenessResult:
    complete: bool
    confidence: float
    reason: str
    utterance: str
    latency_ms: float

class SemanticGate:
    def __init__(self):
        self._client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self._model = settings.SEMANTIC_GATE_MODEL
        self._threshold = settings.SEMANTIC_GATE_CONFIDENCE_THRESHOLD
        self._enabled = settings.SEMANTIC_GATE_ENABLED
        self._fallback_policy = settings.SEMANTIC_GATE_FALLBACK_POLICY
    
    async def check(self, utterance: str) -> CompletenessResult:
        """
        Returns CompletenessResult. Caller decides whether to fire RAG.
        Does NOT call the RAG chain itself — single responsibility.
        """
        if not self._enabled:
            # Gate disabled via config — pass everything through
            return CompletenessResult(
                complete=True, confidence=1.0,
                reason="gate disabled", utterance=utterance, latency_ms=0
            )
        
        if len(utterance.strip()) < 3:
            return CompletenessResult(
                complete=False, confidence=0.99,
                reason="utterance too short", utterance=utterance, latency_ms=0
            )
        
        import time
        start = time.monotonic()
        
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": COMPLETENESS_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Utterance: {utterance}"}
                ],
                max_tokens=60,       # tiny response — just the JSON
                temperature=0.0,     # deterministic classification
                timeout=2.0,         # hard 2s timeout — must not block voice pipeline
            )
            
            latency_ms = (time.monotonic() - start) * 1000
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            
            return CompletenessResult(
                complete=parsed["complete"],
                confidence=parsed.get("confidence", 0.0),
                reason=parsed.get("reason", ""),
                utterance=utterance,
                latency_ms=latency_ms
            )
        
        except json.JSONDecodeError as e:
            latency_ms = (time.monotonic() - start) * 1000
            logger.warning(f"SemanticGate: JSON parse failed for '{utterance[:50]}': {e}")
            return self._fallback_result(utterance, latency_ms)
        
        except asyncio.TimeoutError:
            latency_ms = (time.monotonic() - start) * 1000
            logger.warning(f"SemanticGate: timeout after {latency_ms:.0f}ms for '{utterance[:50]}'")
            return self._fallback_result(utterance, latency_ms)
        
        except Exception as e:
            latency_ms = (time.monotonic() - start) * 1000
            logger.error(f"SemanticGate: unexpected error: {e}")
            return self._fallback_result(utterance, latency_ms)
    
    def _fallback_result(self, utterance: str, latency_ms: float) -> CompletenessResult:
        """
        On any failure, apply the configured fallback policy.
        PASS = fire the RAG pipeline anyway (prioritise responsiveness)
        HOLD = drop the utterance (prioritise accuracy)
        Default should be PASS — better to fire on a partial than to silently drop a complete query.
        """
        fallback_complete = self._fallback_policy == "PASS"
        return CompletenessResult(
            complete=fallback_complete,
            confidence=0.0,
            reason=f"gate_error:fallback_{self._fallback_policy}",
            utterance=utterance,
            latency_ms=latency_ms
        )
    
    def should_fire(self, result: CompletenessResult) -> bool:
        """
        Decision function. Separates the classification from the firing decision.
        Fires if: complete=True AND confidence >= threshold.
        """
        return result.complete and result.confidence >= self._threshold
```

### Wire it into `pipeline.py`

```python
from voice.endpointing_gate import EndpointingGate

pipeline = Pipeline([
    transport.input(),
    stt,                    # DeepgramSTTService — Gate 1 enforced here via utterance_end_ms
    EndpointingGate(),      # Gate 2 — semantic completeness check + pending_prefix accumulation
    llm,
    tts,
    transport.output()
])
```

---

## Partial Transcript Accumulation — How pending_prefix Works

`pending_prefix` is an instance variable on `EndpointingGate` (`self._pending_prefix`), not a module-level variable. It lives entirely inside the frame processor.

**The problem it solves:**

When Gate 2 holds an incomplete utterance, that speech is gone from `self._buffer`. Deepgram will not repeat it. The next `UserStoppedSpeakingFrame` only contains what the user said *after* the hold.

Without `pending_prefix`:
```
UserStoppedSpeakingFrame → new_speech = "If I have a savings account and"
                         → gate says INCOMPLETE → swallowed
UserStoppedSpeakingFrame → new_speech = "what is the minimum balance?"
                         → gate says COMPLETE → RAG gets "what is the minimum balance?"
                         → decontextualised — bad answer
```

With `pending_prefix`:
```
UserStoppedSpeakingFrame → new_speech  = "If I have a savings account and"
                         → utterance   = "" + "If I have a savings account and"
                         → gate says INCOMPLETE
                         → self._pending_prefix = "If I have a savings account and"

UserStoppedSpeakingFrame → new_speech  = "what is the minimum balance?"
                         → utterance   = "If I have a savings account and what is the minimum balance?"
                         → gate says COMPLETE
                         → self._pending_prefix = ""   ← cleared
                         → RAG gets full reconstructed query ✓
```

**The 200-char safety valve:**

If `pending_prefix` grows beyond 200 chars, the user has almost certainly abandoned their original thought and started a new topic. Clear it and start fresh rather than prepending a stale fragment to an unrelated query.

The full implementation is in `EndpointingGate.process_frame` above — no additional code needed here.

---

## Configuration via .env

```env
# --- Gate 1: Deepgram Silence Threshold ---
DEEPGRAM_UTTERANCE_END_MS=1000
DEEPGRAM_ENDPOINTING_MS=500

# --- Gate 2: Semantic Completeness Gate ---
SEMANTIC_GATE_ENABLED=true
SEMANTIC_GATE_MODEL=groq/llama-3.1-8b-instant
SEMANTIC_GATE_CONFIDENCE_THRESHOLD=0.75
# PASS = fire RAG on gate failure (prioritise responsiveness)
# HOLD = drop utterance on gate failure (prioritise accuracy)
SEMANTIC_GATE_FALLBACK_POLICY=PASS
```

**Tuning guide (add as comments in `.env`):**
- `CONFIDENCE_THRESHOLD=0.75` — good default. Raise to 0.85 if too many partials still get through. Lower to 0.65 if complete queries are being held.
- `FALLBACK_POLICY=PASS` — keep this as PASS in production. A held complete query is more frustrating to a user than a slightly early trigger.
- `SEMANTIC_GATE_ENABLED=false` — use this to A/B test: disable gate, observe false triggers in logs, re-enable.

---

## Observability: Logging and Metrics

Add structured logging throughout so you can actually see what the gate is doing in production.

### Log every gate decision
The log line in `on_utterance_end` above is required. Format it so it's greppable:

```
SemanticGate | complete=False | confidence=0.91 | latency=43ms | reason='incomplete conditional' | utterance='If I have a savings account and'
SemanticGate | complete=True  | confidence=0.97 | latency=38ms | reason='full question'          | utterance='What documents do I need for KYC?'
```

### Add a `GateMetrics` counter class in `voice/semantic_gate.py`

```python
from dataclasses import dataclass, field

@dataclass
class GateMetrics:
    total_checks: int = 0
    passed: int = 0
    held: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    
    @property
    def pass_rate(self) -> float:
        return self.passed / self.total_checks if self.total_checks else 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_checks if self.total_checks else 0.0
    
    def summary(self) -> str:
        return (
            f"Gate metrics: {self.total_checks} checks | "
            f"{self.pass_rate:.1%} pass rate | "
            f"{self.avg_latency_ms:.0f}ms avg latency | "
            f"{self.errors} errors"
        )
```

Attach one `GateMetrics` instance to `SemanticGate`. Log `metrics.summary()` every 20 checks.

---

## Test Script

Create `scripts/test_semantic_gate.py` — runs the gate against known complete and incomplete utterances without needing a live microphone.

```python
# scripts/test_semantic_gate.py
# Usage: python scripts/test_semantic_gate.py
# Tests the semantic gate against a fixed set of known complete/incomplete utterances.

COMPLETE_UTTERANCES = [
    "What documents do I need for KYC?",
    "Mera KYC complete hua kya?",                  # Hindi — should still pass
    "Tell me about video KYC",
    "What is the minimum balance for a savings account?",
    "How long does KYC take?",
    "Hi",                                           # greeting — pass through
    "Account balance",                              # short but complete intent
    "Can I do KYC online?",
]

INCOMPLETE_UTTERANCES = [
    "If I have a savings account and",
    "So basically what I wanted to ask is",
    "The RBI says that in case of",
    "KYC documents for",                            # trailing preposition
    "What is the re",                               # cut off mid-word
    "Mera account ka",                              # incomplete possessive
    "When the bank",
]
```

For each utterance, print: expected label, gate decision, confidence, reason, latency. Flag any misclassifications as `[WRONG]`.

Expected output format:
```
COMPLETE utterances:
  ✓  "What documents do I need for KYC?"        → PASS  (conf=0.97, 41ms)
  ✓  "Mera KYC complete hua kya?"               → PASS  (conf=0.89, 44ms)
  ✗  "Account balance"                           → HOLD  (conf=0.71, 38ms)  [WRONG — below threshold]

INCOMPLETE utterances:
  ✓  "If I have a savings account and"           → HOLD  (conf=0.94, 39ms)
  ✓  "KYC documents for"                         → HOLD  (conf=0.91, 37ms)
```

If you see `[WRONG]` results, adjust `SEMANTIC_GATE_CONFIDENCE_THRESHOLD` in `.env` accordingly.

---

## Files To Create / Modify

| File | Action | Notes |
|------|--------|-------|
| `voice/endpointing_gate.py` | Create new | `EndpointingGate` FrameProcessor — all gate logic + `pending_prefix` lives here |
| `voice/semantic_gate.py` | Create new | `SemanticGate`, `CompletenessResult`, `GateMetrics` |
| `voice/deepgram_handler.py` | Modify | Update `DeepgramSTTOptions` with `utterance_end_ms` + `endpointing` config only — no callback changes needed |
| `voice/pipeline.py` | Modify | Add `EndpointingGate()` between `stt` and `llm` — one line change |
| `config/settings.py` | Extend | Add all new `.env` keys |
| `.env` | Extend | Add Gate 1 + Gate 2 config keys |
| `scripts/test_semantic_gate.py` | Create new | Offline test harness — tests `SemanticGate` directly without live mic |

**Do NOT modify:** `rag_chain.py`, `tts_handler.py`, `ingestion/`, `vector_store/`

---

## Definition of Done

- [ ] `SEMANTIC_GATE_ENABLED=false` + `DEEPGRAM_UTTERANCE_END_MS=1000` → pipeline fires on `UserStoppedSpeakingFrame` only, no semantic check
- [ ] `SEMANTIC_GATE_ENABLED=true` → pipeline fires only when both gates pass
- [ ] `VADUserStoppedSpeakingFrame` passes through `EndpointingGate` untouched — never triggers RAG
- [ ] `self._pending_prefix` correctly stitches incomplete fragments across consecutive `UserStoppedSpeakingFrame` events
- [ ] `scripts/test_semantic_gate.py` passes ≥ 13/15 known utterances correctly at default threshold
- [ ] Every gate decision logged in greppable format: `EndpointingGate | complete=... | confidence=... | latency=...ms`
- [ ] Gate latency stays under 200ms p95 (2s hard timeout on Groq call ensures this)
- [ ] `SEMANTIC_GATE_FALLBACK_POLICY=PASS` means gate errors never silently drop user queries
- [ ] No existing RAG chain behaviour changed

---

## Known Limitations — Document These in Code

1. **Hindi/Hinglish classification** — the gate prompt is English-only. Groq will usually classify Hinglish correctly by intent, but confidence scores will be lower for mixed-language utterances. If `CONFIDENCE_THRESHOLD=0.75` causes too many held Hindi queries, lower to `0.65` for those users. A proper fix requires a bilingual completeness prompt — out of scope for this task but note it as a `# TODO` comment.

2. **First-word latency** — Gate 2 adds ~40–80ms to the start-of-response latency. This is acceptable (total pipeline latency target is 1–1.5s). Do not attempt to optimise this away.

3. **Streaming STT partials** — Gate 2 only sees the utterance after UtteranceEnd. It does not see interim partials. This is intentional — classifying partials in real-time would add noise.
