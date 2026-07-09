# Frontend

React + TypeScript + Vite frontend for the local AI Report Generator Agent Demo.

## Current Scope

This frontend provides the product workspace for the local RAG demo:

- report project setup and summary
- reference document upload and indexing status
- section selector for the report template
- per-section prompt editing
- AI-assisted prompt drafting modal
- RAG generation controls connected to the FastAPI backend
- generated section versions, manual edits, and current-version confirmation
- citation markers with source previews
- full report preview and simulated export checks

## Local Commands

Use Node 22.

```bash
npm install
npm run dev
```

Local URL:

```text
http://127.0.0.1:5173
```

Build check:

```bash
npm run build
```
