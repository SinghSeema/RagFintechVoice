"""
Phase 3 — LiveKit voice agent server (eager RAG initialization).

Responsibilities
----------------
1. Token endpoint  GET /token
   Generates a short-lived LiveKit JWT for a mobile/web client to join a room.
   The client passes ?room=<name>&identity=<user-id> as query params.

2. Agent launcher  POST /agent/start
   Starts the Pipecat voice pipeline as the RAG agent in the specified room.
   Fast (~2-3 s) because the RAG stack is pre-loaded at server startup.

3. Health          GET /health
   Returns {"status": "ok"} — used by LiveKit Cloud to verify the agent is up.

Architecture notes
------------------
- The FastAPI app runs in the main asyncio event loop.
- On startup: the RAG stack (PDF parse, Qdrant index, BGE reranker) is loaded
  ONCE in a background task (_preload_and_warm). This takes 15–30 s but happens
  before any user connects, so the user never waits for it.
- After the RAG stack is ready, the default room ("rag-voice") is pre-warmed:
  an agent connects to LiveKit and waits. When the user joins, the greeting fires
  immediately (no pipeline assembly delay).
- _rag_stack_cache holds the loaded stack; _rag_stack_ready (asyncio.Event) is
  set when loading completes. _run_agent awaits it before building the pipeline.
- Only one active pipeline per room is allowed; a second /agent/start for the
  same room returns 409.
- Token TTL: client tokens expire in 1 hour; agent tokens expire in 6 hours.

Startup sequence
----------------
  uvicorn starts
    ├── FastAPI ready: /token, /health immediately reachable
    └── background: _preload_and_warm()
          ├── _load_rag_stack()   (PDF → Qdrant → BGE, ~15–30 s)
          ├── _rag_stack_ready.set()
          └── _ensure_agent("rag-voice")
                └── agent waits in LiveKit room

  user joins LiveKit room
    └── on_first_participant_joined → GREETING  (~2–3 s from joining)

Usage
-----
  uvicorn src.voice.server:app --host 0.0.0.0 --port 8001 --reload

  # Generate a token for the React Native client:
  curl "http://localhost:8001/token?room=rag-voice&identity=user-001"

  # Start the RAG agent in a room (auto-started for default room at startup):
  curl -X POST "http://localhost:8001/agent/start?room=rag-voice"

Environment variables (all required unless noted)
--------------------------------------------------
  LIVEKIT_URL        wss://…livekit.cloud
  LIVEKIT_API_KEY    APImEY…
  LIVEKIT_API_SECRET TDO2PW…
  GROQ_API_KEY       gsk_…
  DEEPGRAM_API_KEY   2eb7ed… (TTS — primary until Cartesia is re-enabled)
  CARTESIA_API_KEY   (optional — if set, takes over as TTS primary)
  LIVEKIT_ROOM       (optional — default: rag-voice)
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from livekit.api import AccessToken, VideoGrants
from loguru import logger

load_dotenv()

# ── RAG stack cache (populated once at startup) ────────────────────────────────
# Holds: index, nodes, storage_context, retriever, reranker
_rag_stack_cache: Optional[dict] = None
# Set when loading completes (or fails). _run_agent awaits this before
# building the pipeline so /agent/start is never blocked on a cold stack.
_rag_stack_ready: Optional[asyncio.Event] = None

# Active pipeline tasks keyed by room name; prevents duplicate agents per room.
_active_agents: dict[str, asyncio.Task] = {}


# ── Startup / shutdown lifecycle ───────────────────────────────────────────────

async def _preload_and_warm() -> None:
    """
    Background task: load the RAG stack once, then pre-warm the default room.

    Runs concurrently with the server — /token and /health are reachable while
    this is in progress.  _run_agent awaits _rag_stack_ready before connecting
    to LiveKit, so no agent starts with a None stack.
    """
    global _rag_stack_cache
    from src.voice.pipeline import _load_rag_stack

    groq_key = os.environ["GROQ_API_KEY"]
    executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()

    logger.info("Startup: pre-loading RAG stack (PDF → Qdrant → BGE reranker) ...")
    try:
        index, nodes, storage_context, retriever, reranker = \
            await loop.run_in_executor(executor, _load_rag_stack, groq_key)
        _rag_stack_cache = dict(
            index=index,
            nodes=nodes,
            storage_context=storage_context,
            retriever=retriever,
            reranker=reranker,
        )
        logger.info("Startup: RAG stack ready")
    except Exception as exc:
        logger.error(f"Startup: RAG pre-load FAILED: {exc}")
        # Still set the event so /agent/start callers aren't blocked forever.
        # _rag_stack_cache stays None; build_voice_pipeline falls back to
        # loading the stack itself.
    finally:
        _rag_stack_ready.set()

    # Auto-warm the default room so the agent is waiting before any user joins.
    default_room = os.environ.get("LIVEKIT_ROOM", "rag-voice")
    logger.info(f"Startup: auto-warming default room {default_room!r}")
    await _ensure_agent(default_room)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag_stack_ready
    _rag_stack_ready = asyncio.Event()
    asyncio.create_task(_preload_and_warm())
    yield
    # Shutdown: cancel all active agent tasks cleanly.
    for task in list(_active_agents.values()):
        if not task.done():
            task.cancel()


app = FastAPI(title="RagFintechVoice — LiveKit agent server", lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dev only — tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Token helpers ──────────────────────────────────────────────────────────────

def _make_token(
    room: str,
    identity: str,
    ttl: timedelta,
    *,
    can_publish: bool = True,
    can_subscribe: bool = True,
    hidden: bool = False,
    agent: bool = False,
) -> str:
    api_key    = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]

    grants = VideoGrants(
        room_join=True,
        room=room,
        can_publish=can_publish,
        can_subscribe=can_subscribe,
        hidden=hidden,
        agent=agent,
    )
    token = (
        AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_ttl(ttl)
        .with_grants(grants)
        .to_jwt()
    )
    return token


# ── Agent runner ───────────────────────────────────────────────────────────────

async def _ensure_agent(room: str) -> bool:
    """
    Start an agent in room if none is currently active.

    Returns True if a new task was launched, False if one was already running.
    Idempotent — safe to call from both startup auto-warm and /agent/start.
    """
    if room in _active_agents and not _active_agents[room].done():
        return False
    task = asyncio.create_task(_run_agent(room), name=f"agent-{room}")
    _active_agents[room] = task
    logger.info(f"Agent task launched for room={room!r}")
    return True


async def _run_agent(room: str) -> None:
    """Build and run the Pipecat pipeline for one room session."""
    from src.voice.pipeline import build_voice_pipeline
    from pipecat.pipeline.runner import PipelineRunner

    livekit_url  = os.environ["LIVEKIT_URL"]
    groq_key     = os.environ["GROQ_API_KEY"]
    cartesia_key = os.environ.get("CARTESIA_API_KEY")
    deepgram_key = os.environ.get("DEEPGRAM_API_KEY")

    # Wait for the RAG stack to finish loading before building the pipeline.
    # At startup auto-warm time this will already be set; for /agent/start
    # calls that race the preload it acts as a safety net (120 s hard timeout).
    if _rag_stack_ready is not None:
        try:
            await asyncio.wait_for(_rag_stack_ready.wait(), timeout=120.0)
        except asyncio.TimeoutError:
            logger.error(f"Agent for room={room!r}: timed out waiting for RAG stack")
            _active_agents.pop(room, None)
            return

    # Agent token — NOT hidden so the browser client sees the track
    agent_token = _make_token(
        room=room,
        identity=f"rag-agent-{room}",
        ttl=timedelta(hours=6),
        can_publish=True,
        can_subscribe=True,
        hidden=False,   # hidden=True can suppress TrackSubscribed on the client
        agent=False,
    )

    logger.info(f"Agent starting in room={room!r}")
    try:
        task, transport = await build_voice_pipeline(
            livekit_url=livekit_url,
            livekit_token=agent_token,
            livekit_room=room,
            groq_api_key=groq_key,
            cartesia_api_key=cartesia_key,
            deepgram_api_key=deepgram_key,
            preloaded_rag_stack=_rag_stack_cache,   # None-safe: pipeline loads itself
        )

        from pipecat.frames.frames import LLMFullResponseStartFrame, LLMFullResponseEndFrame, TextFrame
        from src.voice.pipeline import GREETING_TEXT

        @transport.event_handler("on_first_participant_joined")
        async def on_first_joined(transport, participant):
            logger.info(f"First participant joined room={room!r}: {participant}")
            # In the pre-warm flow the agent is already in the room when the user
            # joins.  LiveKit track subscription is async on the client side (~1–2 s
            # after joining), so if we push the greeting immediately the audio is
            # sent before anyone is subscribed and the user hears silence.
            # A short sleep gives the client time to complete track setup.
            await asyncio.sleep(1.5)
            await task.queue_frames([
                LLMFullResponseStartFrame(),
                TextFrame(text=GREETING_TEXT),
                LLMFullResponseEndFrame(),
            ])

        @transport.event_handler("on_participant_connected")
        async def on_connected_participant(transport, participant):
            logger.info(f"Participant connected room={room!r}: {participant}")

        @transport.event_handler("on_participant_left")
        async def on_left(transport, participant):
            logger.info(f"Participant left room={room!r}: {participant}")

        runner = PipelineRunner()
        await runner.run(task)
    except asyncio.CancelledError:
        logger.info(f"Agent for room={room!r} cancelled")
    except Exception as exc:
        logger.error(f"Agent for room={room!r} crashed: {exc}")
    finally:
        _active_agents.pop(room, None)
        logger.info(f"Agent for room={room!r} stopped")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the frontend page — eliminates the need for a separate static server."""
    html = Path("index.html")
    if not html.exists():
        raise HTTPException(status_code=404, detail="index.html not found in working directory")
    return FileResponse(html, media_type="text/html")


@app.get("/livekit-client.umd.js")
async def livekit_sdk():
    """Serve the vendored LiveKit JS SDK so the browser can load it."""
    sdk = Path("livekit-client.umd.js")
    if not sdk.exists():
        raise HTTPException(status_code=404, detail="livekit-client.umd.js not found")
    return FileResponse(sdk, media_type="application/javascript")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/token")
async def get_token(
    room: str = Query(default=None, description="LiveKit room name"),
    identity: str = Query(..., description="Unique participant identity (e.g. user-001)"),
):
    """
    Generate a LiveKit JWT for a client (mobile app / browser) to join a room.

    The client passes this token directly to the LiveKit SDK — no server-side
    session state is kept.
    """
    room = room or os.environ.get("LIVEKIT_ROOM", "rag-voice")
    token = _make_token(
        room=room,
        identity=identity,
        ttl=timedelta(hours=1),
    )
    return {
        "token": token,
        "room": room,
        "livekit_url": os.environ["LIVEKIT_URL"],
    }


@app.post("/agent/start")
async def start_agent(
    room: Optional[str] = Query(default=None, description="LiveKit room name"),
):
    """
    Launch the RAG voice agent in the specified room.

    For the default room this is usually a no-op — the agent is auto-started at
    server startup.  Use this endpoint to start agents in custom rooms, or to
    restart after /agent/stop.

    Returns 409 if an agent is already active in that room.
    """
    room = room or os.environ.get("LIVEKIT_ROOM", "rag-voice")
    launched = await _ensure_agent(room)
    if not launched:
        raise HTTPException(
            status_code=409,
            detail=f"Agent already active in room={room!r}. Call /agent/stop first.",
        )
    return {"status": "started", "room": room}


@app.post("/agent/stop")
async def stop_agent(
    room: Optional[str] = Query(default=None, description="LiveKit room name"),
):
    """Cancel the active agent for a room (graceful shutdown)."""
    room = room or os.environ.get("LIVEKIT_ROOM", "rag-voice")
    task = _active_agents.get(room)
    if not task or task.done():
        raise HTTPException(status_code=404, detail=f"No active agent in room={room!r}")
    task.cancel()
    return {"status": "stopping", "room": room}


@app.get("/agent/status")
async def agent_status():
    """List all rooms and whether their agent task is active."""
    return {
        room: "active" if not task.done() else "stopped"
        for room, task in _active_agents.items()
    }
