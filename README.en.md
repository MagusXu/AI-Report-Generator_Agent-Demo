# AI Report Generator Agent Demo

Local full-stack demo for an AI-assisted internal report generation product with retrieval-augmented generation (RAG).

**中文版请见 [README.md](./README.md)。**

The product scenario is an investment-banking internal industry risk research report. Users manage a report project, select reference documents, generate and edit sections independently, keep section versions, insert tables, preview the full report, and simulate exporting the final deliverable.

## Current Stage

Local FastAPI + React + SQLite + ChromaDB + DashScope RAG. **Docker is not required for v1.**

### Working Features

- Report project basics: name, industry, year, language
- Reference documents: upload, type labels, selection, parse/index status, preview, reindex, delete
- Fixed report template (industry risk structure) with section switching
- Per-section prompt editing and AI-assisted prompt drafting
- RAG section generation with **SSE streaming** and a simple on-screen execution trace
- Tables per section (up to 5):
  - **AI synthesize**: user title + column specs → model fills a structured table
  - **Verbatim cite**: user **title** (display) + **description** (search only) → retrieve candidates → confirm chunk → insert original table
- Inline citations `[ref:chunk_id]` rendered as hoverable source chips (fullwidth `【ref:…】` normalized)
- Markdown-ish body rendering: headings, bold, tables with caption under the table in small gray text
- Manual edit / version history / confirm current version
- Full report preview and pre-export checks
- Simulated Word/PDF export records (no real binary generation in v1)

## Architecture

| Layer | Choice |
|--------|--------|
| Frontend | React + TypeScript + Vite |
| Backend | Python FastAPI |
| Metadata | SQLite |
| Vectors | ChromaDB persistent client |
| LLM | Alibaba Cloud Model Studio (CN) `qwen3.6-35b-a3b` |
| Embedding | Alibaba Cloud Model Studio (CN) `text-embedding-v4` |

```text
Upload docs → parse → chunk → embed → Chroma
                         ↓
Section generate (stream):
  prepare → retrieve → stream narrative deltas
         → (optional) insert/synthesize tables
         → normalize citations → persist version
```

## Directory Layout

```text
.
├── frontend/                 # Vite React app (App.tsx is the main UI)
├── backend/
│   ├── app/
│   │   ├── main.py           # HTTP APIs + SSE orchestration
│   │   ├── config.py
│   │   ├── database.py
│   │   └── services/         # parser, chunker, embed, LLM, vector, tables
│   ├── scripts/              # connectivity / RAG smoke checks
│   └── README.md
├── docs/                     # repo-local docs only
├── .env.example
├── README.md                 # Chinese (default)
└── README.en.md              # English (this file)
```

## Generation Flow (Streaming)

`POST /api/sections/{id}/generate` with `stream: true` returns `text/event-stream`.

| Event | Meaning |
|--------|---------|
| `status` | Pipeline phase tip (`prepare` → `retrieve` → `generating` → optional `tables` / `tables_done` → `persist`) |
| `delta` | Narrative token chunk `{ "content": "..." }` |
| `done` | Final `{ "workspace": ... }` |
| `error` | `{ "detail": "..." }` |

The UI shows an **execution process** list from `status` messages while streaming body text into the preview.

## Tables

- Narrative generation must emit placeholders `<<TABLE:1>>`, `<<TABLE:2>>`, … instead of inventing table markdown.
- **Synthesize**: title + columns (+ notes) → second LLM call builds JSON cells; unverified cells marked `[[?…?]]`.
- **Verbatim**: `description` is only the retrieval query; `title` is the display caption shown **under** the table in gray small text. Candidate search: `POST /api/sections/{id}/table-candidates`.
- Stored table captions use an internal marker `[[表题：…]]` (legacy `## title` before a table is still recognized).

## Citations

- Model / pipeline markers: `[ref:chunk_id]`
- Frontend maps ids through `workspace.citations` to document name, type, locator, and snippet
- Invalid refs stripped; trailing “资料来源：…” footers are removed (citations stay inline only)

## Key APIs

| Method | Path |
|--------|------|
| GET | `/health`, `/api/runtime`, `/api/workspace`, `/api/report-preview`, `/api/report-template` |
| PATCH | `/api/project` |
| POST | `/api/project/new` |
| POST | `/api/documents/upload-file` |
| PATCH / DELETE / GET / POST | `/api/documents/{id}/selection`, `…/{id}`, `…/{id}/preview`, `…/{id}/reindex` |
| PATCH | `/api/sections/{id}/prompt` |
| POST | `/api/sections/{id}/generate`, `…/table-candidates`, `…/manual-edit`, `…/select-version`, `…/confirm`, `…/enhance-prompt`, `…/clear` |
| POST | `/api/exports` |

## Backend Setup

Python 3.12 via `uv`:

```bash
cd backend
uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Health: `http://127.0.0.1:8000/health`

Checks (call live DashScope APIs):

```bash
uv run python scripts/check_dashscope.py
uv run python scripts/check_rag_pipeline.py
```

## Frontend Setup

Node 22:

```bash
cd frontend
npm install
npm run dev
```

Local URL: `http://127.0.0.1:5173`

## Model Configuration

Copy `.env.example` into ignored `backend/config.env` (or equivalent) and set:

```text
DASHSCOPE_API_KEY=
WORKSPACE_ID=
LLM_BASE_URL=https://${WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3.6-35b-a3b
EMBEDDING_BASE_URL=https://${WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
DATABASE_URL=sqlite:///./data/app.db
CHROMA_PATH=./data/chroma
UPLOAD_DIR=./uploads
EXPORT_DIR=./exports
CORS_ORIGINS=http://127.0.0.1:5173,http://127.0.0.1:5174
```

**Do not commit** real API keys, `config.env`, uploaded source files, Chroma/SQLite data, exports, `node_modules`, or virtualenvs.

## Out of Scope (v1)

- Docker / cloud deployment packaging
- Real Word/PDF binary generation
- Auth, multi-user, permissions
- Cloud-hosted vector DB
