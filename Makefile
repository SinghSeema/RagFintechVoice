.PHONY: demo build rebuild stop install

# ── Demo (both servers, single Ctrl-C to stop) ──────────────────────────────
demo:
	@chmod +x run_demo.sh && ./run_demo.sh

# ── Install Python deps ──────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Build RAG index (skip if storage already exists) ────────────────────────
build:
	@if [ -d storage ] && [ -d qdrant_storage ]; then \
	  echo "Index already built — skipping (use 'make rebuild' to force)."; \
	else \
	  python -m src.ingestion.chunker && python -m src.retrieval.vector_store; \
	fi

# ── Force-rebuild index from scratch ────────────────────────────────────────
rebuild:
	rm -rf storage/ qdrant_storage/
	python -m src.ingestion.chunker
	python -m src.retrieval.vector_store

# ── Kill any stray server processes ─────────────────────────────────────────
stop:
	-lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	-lsof -ti:8001 | xargs kill -9 2>/dev/null || true
	@echo "Ports 8000 and 8001 cleared."
