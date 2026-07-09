# AI Report Generator Agent Demo

Local full-stack demo for an AI-assisted internal report generation product with retrieval-augmented generation.

The product scenario is an investment-banking internal industry risk research report. Users manage a report project, select reference documents, generate and edit sections independently, keep section versions, preview the full report, and simulate exporting the final deliverable.

## Current Stage

Current stage: local FastAPI + React + SQLite + ChromaDB + DashScope RAG.

The app supports:

- file upload
- text extraction from uploaded reference documents
- chunking
- `text-embedding-v4` embeddings
- local ChromaDB persistent vector store
- section-level retrieval
- `qwen3.6-35b-a3b` generation
- chunk-level citation markers and hover previews

## Working Features

- Report project basics: name, industry, year, language
- Reference documents: upload, type labels, selection checkboxes, indexing status
- Report template: standard data center industry risk report structure
- Section switching through a dropdown
- Per-section prompt editing
- AI-assisted prompt drafting modal with six guiding fields
- RAG section generation that creates a new version every time
- Citation markers in generated content with hover source preview
- Manual edit saved as a separate version
- Current version confirmation and historical version switching
- Full report preview assembled from current section versions
- Pre-export checks for missing sections and unconfirmed versions
- Simulated Word/PDF export records

## Architecture

- Frontend: React + TypeScript + Vite
- Backend: Python FastAPI
- Metadata store: SQLite
- Vector store: ChromaDB persistent client
- LLM API: Alibaba Cloud Model Studio mainland China `qwen3.6-35b-a3b`
- Embedding API: Alibaba Cloud Model Studio mainland China `text-embedding-v4`

## Directory Layout

```text
.
├── frontend/
├── backend/
├── docs/
├── .env.example
├── .gitignore
├── AGENTS.md
└── README.md
```

## Backend

Use Python 3.12 through `uv`.

```bash
cd backend
/opt/homebrew/bin/uv sync
/opt/homebrew/bin/uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

RAG check:

```bash
cd backend
/opt/homebrew/bin/uv run python scripts/check_rag_pipeline.py
```

The RAG check calls the live model and embedding APIs and writes to the local Chroma store.

## Frontend

Use Node 22.

```bash
cd frontend
/opt/homebrew/opt/node@22/bin/npm install
/opt/homebrew/opt/node@22/bin/npm run dev
```

Local URL:

```text
http://127.0.0.1:5173
```

## Model Configuration

Real credentials are stored locally in:

```text
backend/config.env
```

The file is ignored by Git.

Required values:

```text
DASHSCOPE_API_KEY=
WORKSPACE_ID=
```

Do not commit `.env`, `config.env`, uploaded files, local Chroma data, SQLite data, generated exports, dependency folders, or Python cache files.
