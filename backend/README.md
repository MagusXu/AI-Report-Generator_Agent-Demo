# Backend

FastAPI backend for the local AI Report Generator Agent Demo.

## Current Scope

This backend provides the local RAG and report workflow APIs:

- health and runtime configuration checks
- report template and section state
- uploaded reference document metadata
- file parsing for supported document formats
- text chunking
- DashScope `text-embedding-v4` embedding calls
- local ChromaDB indexing and retrieval
- DashScope `qwen3.6-35b-a3b` generation calls
- section version storage in SQLite
- simulated export records

## Local Commands

Use Python 3.12 through `uv`.

```bash
uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Connectivity check:

```bash
uv run python scripts/check_dashscope.py
```

RAG pipeline check:

```bash
uv run python scripts/check_rag_pipeline.py
```

The connectivity and RAG checks call live DashScope APIs. The RAG check also writes a sample chunk to the local Chroma store.

## Local Data

The backend keeps local runtime files under ignored paths:

- `config.env`
- `data/`
- `uploads/`
- `exports/`
- `.venv/`
