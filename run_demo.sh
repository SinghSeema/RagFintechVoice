#!/usr/bin/env bash
# run_demo.sh — starts both servers; Ctrl-C kills both cleanly
set -e

# Resolve project root regardless of where the script is called from
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Activate venv if present and not already active
if [ -z "$VIRTUAL_ENV" ] && [ -f "$ROOT/.venv/bin/activate" ]; then
  source "$ROOT/.venv/bin/activate"
fi

# Load .env
if [ -f "$ROOT/.env" ]; then
  set -a; source "$ROOT/.env"; set +a
fi

# ── Qdrant ──────────────────────────────────────────────────────────────────
QDRANT_CONTAINER="rag-qdrant"
QDRANT_STARTED_BY_US=0

if ! docker ps --format '{{.Names}}' | grep -q "^${QDRANT_CONTAINER}$"; then
  echo "Starting Qdrant   → http://localhost:6333"
  docker run -d \
    --name "$QDRANT_CONTAINER" \
    -p 6333:6333 \
    -v "$ROOT/qdrant_storage:/qdrant/storage" \
    qdrant/qdrant:latest >/dev/null
  QDRANT_STARTED_BY_US=1

  # Wait until Qdrant is ready (up to 15 s)
  for i in $(seq 1 15); do
    if curl -sf http://localhost:6333/readyz >/dev/null 2>&1; then break; fi
    sleep 1
  done
else
  echo "Qdrant already running ($QDRANT_CONTAINER)"
fi

# Kill both servers when this script exits (Ctrl-C or error)
cleanup() {
  echo ""
  echo "Stopping servers..."
  kill "$TEXT_PID" "$VOICE_PID" 2>/dev/null
  wait "$TEXT_PID" "$VOICE_PID" 2>/dev/null
  if [ "$QDRANT_STARTED_BY_US" -eq 1 ]; then
    echo "Stopping Qdrant..."
    docker stop "$QDRANT_CONTAINER" >/dev/null
    docker rm   "$QDRANT_CONTAINER" >/dev/null
  fi
  echo "Done."
}
trap cleanup EXIT INT TERM

echo "Starting Text API  → http://localhost:8000"
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 2>&1 | sed 's/^/[text]  /' &
TEXT_PID=$!

echo "Starting Voice API → http://localhost:8001"
uvicorn src.voice.server:app --host 0.0.0.0 --port 8001 2>&1 | sed 's/^/[voice] /' &
VOICE_PID=$!

echo ""
echo "  Chat tab  → http://localhost:8001"
echo "  Voice tab → http://localhost:8001"
echo ""
echo "Press Ctrl-C to stop both servers."
echo ""

wait "$TEXT_PID" "$VOICE_PID"
