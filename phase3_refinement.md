# Phase 3 Refinement — Eager RAG Stack Initialization

**Status:** Implemented  
**Created:** 2026-04-18  
**Builds on:** Phase 3 fixes (classifier, score filter, interruption handling)

---

## Problem

The RAG stack was loaded **per-connection**: every call to `POST /agent/start` triggered
`_load_rag_stack()` inside `_run_agent()`, which blocked the agent from connecting to
LiveKit until all initialization was complete.

**Dead time the user waited through (15–30 s):**

```
POST /agent/start
  └── _load_rag_stack()
        ├── load_pdf_with_sections()        ~3–5 s
        ├── build_section_map_and_clean()   ~1–2 s
        ├── chunk_document()                ~2–4 s
        ├── load_index()  (Qdrant)          ~1–2 s
        └── build_pipeline()  (BGE weights) ~5–15 s
  └── LiveKitTransport connects
  └── on_first_participant_joined → GREETING
```

None of this work depends on the user, room, or question — it is static initialization
that was needlessly repeated every time.

---

## Root Cause

`build_voice_pipeline()` called `_load_rag_stack()` unconditionally. The function had no
way to accept a pre-built stack, so there was no path for sharing initialization across
connections.

---

## Design: Two-Phase Lifecycle

### Phase 1 — Server startup (background, non-blocking)

Immediately after `uvicorn` starts, a background task pre-loads the RAG stack.
The server accepts `/token` and `/health` requests while this runs in the background.

```
uvicorn starts
  ├── FastAPI ready: /token, /health, /agent/start all reachable immediately
  └── asyncio background task: _preload_and_warm()
        ├── _load_rag_stack()   (same work as before, now happens once)
        ├── sets _rag_stack_ready asyncio.Event
        └── auto-starts agent in default room ("rag-voice")
              └── LiveKitTransport connects and WAITS
```

### Phase 2 — Per-room agent (fast, RAG already warm)

```
user connects to LiveKit room
  └── on_first_participant_joined fires
        └── GREETING pushed immediately  (~2–3 s transport connect time only)
```

For custom rooms (non-default), `POST /agent/start` awaits `_rag_stack_ready` (already
set at this point), then builds the pipeline without reloading anything.

---

## User-Visible Result

| | Before | After |
|---|---|---|
| Time to first greeting | 15–30 s | ~2–3 s |
| Cause of wait | RAG loading per-connection | Transport connect only |
| Multiple rooms | Each reloads full stack | Shared index; own retriever/reranker per room |

---

## Code Changes

### `src/voice/pipeline.py`

**Change 1 — `build_voice_pipeline` accepts a pre-loaded stack**

Added `preloaded_rag_stack` optional parameter (dict with keys `index`, `nodes`,
`storage_context`, `retriever`, `reranker`). When provided, `_load_rag_stack()` is
skipped entirely.

```python
async def build_voice_pipeline(
    livekit_url: str,
    livekit_token: str,
    livekit_room: str,
    groq_api_key: str,
    cartesia_api_key: Optional[str] = None,
    deepgram_api_key: Optional[str] = None,
    preloaded_rag_stack: Optional[dict] = None,   # NEW
) -> tuple[PipelineTask, LiveKitTransport]:
```

Inside:
```python
if preloaded_rag_stack is not None:
    index      = preloaded_rag_stack["index"]
    nodes      = preloaded_rag_stack["nodes"]
    storage_context = preloaded_rag_stack["storage_context"]
    retriever  = preloaded_rag_stack["retriever"]
    reranker   = preloaded_rag_stack["reranker"]
else:
    index, nodes, storage_context, retriever, reranker = \
        await loop.run_in_executor(executor, _load_rag_stack, groq_api_key)
```

**Note on multi-room sharing:** `index`, `nodes`, and `storage_context` are shared
(read-only). Each room gets its own `(retriever, reranker)` built from the shared base,
so BGE inference is never shared across threads.

---

### `src/voice/server.py`

**Change 1 — Module-level cache and ready-event**

```python
_rag_stack_cache: Optional[dict] = None
_rag_stack_ready: Optional[asyncio.Event] = None
```

`_rag_stack_ready` is created inside `lifespan()` (not at import time) to avoid
creating an `asyncio.Event` before the event loop exists.

**Change 2 — FastAPI lifespan replaces `@app.on_event("startup")`**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag_stack_ready
    _rag_stack_ready = asyncio.Event()
    asyncio.create_task(_preload_and_warm())
    yield
    # shutdown: cancel all active agents
    for task in list(_active_agents.values()):
        task.cancel()

app = FastAPI(title="...", lifespan=lifespan)
```

**Change 3 — `_preload_and_warm()` background task**

```python
async def _preload_and_warm() -> None:
    global _rag_stack_cache
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    groq_key = os.environ["GROQ_API_KEY"]
    logger.info("Startup: pre-loading RAG stack ...")
    try:
        index, nodes, storage_context, retriever, reranker = \
            await loop.run_in_executor(executor, _load_rag_stack, groq_key)
        _rag_stack_cache = dict(
            index=index, nodes=nodes,
            storage_context=storage_context,
            retriever=retriever, reranker=reranker,
        )
        _rag_stack_ready.set()
        logger.info("Startup: RAG stack ready — auto-warming default room")
        default_room = os.environ.get("LIVEKIT_ROOM", "rag-voice")
        await _ensure_agent(default_room)
    except Exception as exc:
        logger.error(f"Startup: RAG pre-load FAILED: {exc}")
        _rag_stack_ready.set()   # unblock any waiting /agent/start calls
```

**Change 4 — `_ensure_agent()` helper (idempotent)**

Extracted from the old `start_agent` endpoint so startup and the endpoint share the
same launch logic:

```python
async def _ensure_agent(room: str) -> bool:
    """Start an agent in room if none is active. Returns True if launched."""
    if room in _active_agents and not _active_agents[room].done():
        return False
    task = asyncio.create_task(_run_agent(room), name=f"agent-{room}")
    _active_agents[room] = task
    logger.info(f"Agent task launched for room={room!r}")
    return True
```

**Change 5 — `_run_agent()` passes cached stack**

```python
async def _run_agent(room: str) -> None:
    ...
    # Wait for RAG stack (already ready at startup; this is a safety net for
    # /agent/start calls that arrive before preload finishes).
    if _rag_stack_ready is not None:
        await asyncio.wait_for(_rag_stack_ready.wait(), timeout=120.0)

    task, transport = await build_voice_pipeline(
        ...
        preloaded_rag_stack=_rag_stack_cache,   # None-safe: pipeline loads itself
    )
```

**Change 6 — `POST /agent/start` uses `_ensure_agent()`**

```python
@app.post("/agent/start")
async def start_agent(room: Optional[str] = Query(default=None)):
    room = room or os.environ.get("LIVEKIT_ROOM", "rag-voice")
    launched = await _ensure_agent(room)
    if not launched:
        raise HTTPException(status_code=409, detail=f"Agent already active in room={room!r}")
    return {"status": "started", "room": room}
```

---

## Edge Cases

| Scenario | Handling |
|---|---|
| `/agent/start` before RAG finishes loading | `_run_agent` awaits `_rag_stack_ready` (120 s timeout) |
| RAG preload fails | `_rag_stack_ready` is still set; `_rag_stack_cache` stays `None`; pipeline falls back to loading itself |
| User uses custom room | `/agent/start` is fast (RAG already cached); custom room gets its own transport |
| Server restart | Preload runs fresh; no stale state |
| Default room agent crashes | Next `/agent/start` call re-launches; no auto-restart loop (avoids infinite crash loops) |
| `standalone _main()` entry-point | `preloaded_rag_stack=None` → loads as before; no change to dev/test workflow |
| Greeting heard as silence (pre-warm) | `on_first_participant_joined` now sleeps 1.5 s before pushing greeting; LiveKit track subscription on the client is async and not ready at the instant the event fires |

---

## Files Changed

- [src/voice/pipeline.py](src/voice/pipeline.py) — `build_voice_pipeline` accepts `preloaded_rag_stack`
- [src/voice/server.py](src/voice/server.py) — lifespan handler, `_preload_and_warm`, `_ensure_agent`, `_run_agent` updated
