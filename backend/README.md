# Backend

FastAPI backend for the local AI Report Generator Agent Demo.

中文总览见仓库根目录 [README.zh-CN.md](../README.zh-CN.md)。

## Scope

- Health / runtime configuration
- Report template and section state (SQLite)
- Document upload, parse, chunk, embed, Chroma index/retrieve
- Section generation (sync + SSE stream) with DashScope `qwen3.6-35b-a3b`
- Table pipeline (`table_service.py`): AI synthesize + verbatim cite
- Citation normalization and section version persistence
- Simulated export records

## Local Commands

Python 3.12 via `uv`:

```bash
uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

```bash
curl http://127.0.0.1:8000/health
uv run python scripts/check_dashscope.py
uv run python scripts/check_rag_pipeline.py
```

Connectivity and RAG checks call live DashScope APIs. The RAG check also writes a sample chunk to the local Chroma store.

## Local Data (gitignored)

- `config.env`
- `data/`
- `uploads/`
- `exports/`
- `.venv/`
