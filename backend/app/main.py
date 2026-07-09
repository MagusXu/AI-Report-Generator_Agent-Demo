from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from platform import python_version
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.database import PROJECT_ID, get_connection, init_database, json_loads, row_to_dict, utc_now
from app.services.chunker import (
    TextChunk,
    chunk_document,
    evaluate_chunk_quality_gate,
    format_chunk_index_summary,
    summarize_chunk_quality,
)
from app.services.document_parser import DocumentParserError, parse_document
from app.services.embedding_client import EmbeddingClient, EmbeddingClientError
from app.services.llm_client import LLMClient, LLMClientError, LLMResult
from app.services.vector_store import VectorStore


settings = get_settings()
init_database()

DEFAULT_SYSTEM_PROMPT = """你是一名投行内部行业风险研究员。你必须只基于给定资料写作，不要编造具体数据。

写作要求：
1. 使用投行内部行业风险研究报告口吻。
2. 结构为：核心判断、关键依据、风险提示、银行业务含义。
3. 每个主要判断后尽量给出引用标注，格式必须是 [ref:chunk_id]。
4. 如果资料不足，请明确写“现有资料不足以支持更细判断”，不要编造。"""

app = FastAPI(
    title="AI Report Generator API",
    version="0.3.0",
    description="Local backend for the AI-assisted report generation demo.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProjectUpdate(BaseModel):
    name: str
    industry: str
    year: str
    language: str


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1)
    industry: str = Field(min_length=1)
    year: str = Field(min_length=1)
    language: str = Field(min_length=1)


class DocumentCreate(BaseModel):
    name: str = Field(min_length=1)
    type: str = Field(min_length=1)


class DocumentSelection(BaseModel):
    selected: bool


class PromptUpdate(BaseModel):
    prompt: str = Field(min_length=1)


class GenerateRequest(BaseModel):
    prompt: str | None = None
    system_prompt: str | None = None
    reference_ids: list[str] = Field(default_factory=list)
    style: str = Field(default="strict")
    length: str = Field(default="medium")
    stream: bool = False
    retrieval_top_k: int = Field(default=20, ge=1, le=40)
    per_document_limit: int = Field(default=3, ge=1, le=10)
    use_parent_context: bool = True
    temperature: float = Field(default=0.2, ge=0, lt=2)
    max_tokens: int = Field(default=2200, ge=300, le=6000)


class ManualEditRequest(BaseModel):
    content: str = Field(min_length=1)


class SelectVersionRequest(BaseModel):
    version_id: str


class PromptAssistRequest(BaseModel):
    objective: str = ""
    key_questions: str = ""
    geography: str = ""
    risk_dimensions: str = ""
    tone_length: str = ""
    exclusions: str = ""


class ExportRequest(BaseModel):
    format: str = Field(pattern="^(Word|PDF)$")


def fetch_documents(conn) -> list[dict]:
    rows = conn.execute("SELECT * FROM documents ORDER BY created_at, id").fetchall()
    return [{**dict(row), "selected": bool(row["selected"])} for row in rows]


def fetch_versions(conn, section_id: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT * FROM section_versions
        WHERE section_id = ?
        ORDER BY version_number DESC
        """,
        (section_id,),
    ).fetchall()
    versions = []
    for row in rows:
        item = dict(row)
        item["reference_ids"] = json_loads(item["reference_ids"], [])
        versions.append(item)
    return versions


def fetch_sections(conn) -> list[dict]:
    rows = conn.execute("SELECT * FROM sections ORDER BY order_index").fetchall()
    sections = []
    for row in rows:
        section = dict(row)
        section["confirmed"] = bool(section["confirmed"])
        section["versions"] = fetch_versions(conn, section["id"])
        section["current_version"] = next(
            (item for item in section["versions"] if item["id"] == section["current_version_id"]),
            None,
        )
        sections.append(section)
    return sections


def fetch_project(conn) -> dict:
    project = row_to_dict(conn.execute("SELECT * FROM projects WHERE id = ?", (PROJECT_ID,)).fetchone())
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def fetch_export_records(conn) -> list[dict]:
    rows = conn.execute("SELECT * FROM export_records ORDER BY created_at DESC").fetchall()
    records = []
    for row in rows:
        item = dict(row)
        item["issues"] = json_loads(item["issues"], [])
        records.append(item)
    return records


def fetch_citations(conn) -> dict[str, dict]:
    citations: dict[str, dict] = {}
    doc_rows = conn.execute("SELECT * FROM documents").fetchall()
    for row in doc_rows:
        doc = dict(row)
        citations[doc["id"]] = {
            "id": doc["id"],
            "kind": "document",
            "document_id": doc["id"],
            "document_name": doc["name"],
            "document_type": doc["type"],
            "source_locator": "文档摘要",
            "text": doc["summary"],
        }

    chunk_rows = conn.execute(
        """
        SELECT c.*, d.name AS document_name, d.type AS document_type
        FROM document_chunks c
        JOIN documents d ON d.id = c.document_id
        """
    ).fetchall()
    for row in chunk_rows:
        chunk = dict(row)
        citations[chunk["id"]] = {
            "id": chunk["id"],
            "kind": "chunk",
            "document_id": chunk["document_id"],
            "document_name": chunk["document_name"],
            "document_type": chunk["document_type"],
            "source_locator": chunk["source_locator"],
            "text": chunk["text"][:420],
        }
    return citations


def fetch_ai_call_logs(conn) -> list[dict]:
    ensure_ai_call_logs_table(conn)
    rows = conn.execute(
        """
        SELECT * FROM ai_call_logs
        ORDER BY created_at DESC
        LIMIT 20
        """
    ).fetchall()
    logs = []
    for row in rows:
        item = dict(row)
        item["metadata"] = json_loads(item.get("metadata"), {})
        logs.append(item)
    return logs


def workspace_payload(conn) -> dict:
    return {
        "project": fetch_project(conn),
        "documents": fetch_documents(conn),
        "sections": fetch_sections(conn),
        "citations": fetch_citations(conn),
        "ai_call_logs": fetch_ai_call_logs(conn),
        "export_records": fetch_export_records(conn),
    }


def get_section_or_404(conn, section_id: str) -> dict:
    section = row_to_dict(conn.execute("SELECT * FROM sections WHERE id = ?", (section_id,)).fetchone())
    if section is None:
        raise HTTPException(status_code=404, detail="Section not found")
    return section


def get_document_or_404(conn, document_id: str) -> dict:
    document = row_to_dict(conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone())
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    document["selected"] = bool(document["selected"])
    return document


def get_documents_by_ids(conn, reference_ids: list[str]) -> list[dict]:
    if not reference_ids:
        rows = conn.execute("SELECT * FROM documents WHERE selected = 1 ORDER BY created_at, id").fetchall()
        return [dict(row) for row in rows]

    placeholders = ",".join("?" for _ in reference_ids)
    rows = conn.execute(
        f"SELECT * FROM documents WHERE id IN ({placeholders}) ORDER BY created_at, id",
        reference_ids,
    ).fetchall()
    return [dict(row) for row in rows]


def next_version_number(conn, section_id: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(version_number), 0) + 1 AS next_number FROM section_versions WHERE section_id = ?",
        (section_id,),
    ).fetchone()
    return int(row["next_number"])


def create_version(
    conn,
    section_id: str,
    source: str,
    summary: str,
    content: str,
    reference_ids: list[str],
) -> dict:
    version_id = f"ver_{uuid4().hex[:12]}"
    version_number = next_version_number(conn, section_id)
    conn.execute(
        """
        INSERT INTO section_versions (
            id, section_id, version_number, source, summary, content, reference_ids, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            version_id,
            section_id,
            version_number,
            source,
            summary,
            content,
            json.dumps(reference_ids, ensure_ascii=False),
            utc_now(),
        ),
    )
    conn.execute(
        "UPDATE sections SET current_version_id = ?, confirmed = 0 WHERE id = ?",
        (version_id, section_id),
    )
    return fetch_versions(conn, section_id)[0]


def ensure_ai_call_logs_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_call_logs (
            id TEXT PRIMARY KEY,
            operation TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            metadata TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )


def record_ai_call(
    conn,
    operation: str,
    model: str,
    usage: dict[str, int | None] | None,
    metadata: dict | None = None,
) -> None:
    ensure_ai_call_logs_table(conn)
    usage = usage or {}
    conn.execute(
        """
        INSERT INTO ai_call_logs (
            id, operation, model, prompt_tokens, completion_tokens, total_tokens, metadata, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"call_{uuid4().hex[:12]}",
            operation,
            model,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
            json.dumps(metadata or {}, ensure_ascii=False),
            utc_now(),
        ),
    )


def clear_report_versions(conn) -> None:
    ensure_ai_call_logs_table(conn)
    conn.execute("DELETE FROM section_versions")
    conn.execute("DELETE FROM export_records")
    conn.execute("DELETE FROM ai_call_logs")
    conn.execute("UPDATE sections SET current_version_id = NULL, confirmed = 0")


def export_check(sections: list[dict], documents: list[dict]) -> list[str]:
    issues = []
    missing = [section["title"] for section in sections if section["level"] == 1 and not section["current_version"]]
    unconfirmed = [section["title"] for section in sections if section["current_version"] and not section["confirmed"]]
    selected_docs = [doc for doc in documents if doc["selected"]]
    indexed_docs = [doc for doc in selected_docs if doc.get("parse_status") == "indexed" and doc.get("chunk_count", 0) > 0]

    if missing:
        issues.append(f"以下一级章节尚未生成当前版本：{', '.join(missing)}")
    if unconfirmed:
        issues.append(f"以下章节已有版本但尚未确认：{', '.join(unconfirmed[:6])}")
    if len(selected_docs) < 2:
        issues.append("当前勾选参考文档少于 2 份，建议补充资料后再导出。")
    if not indexed_docs:
        issues.append("当前没有已入库的真实参考文档，建议上传并完成索引后再导出。")
    return issues


def build_rag_query(project: dict, section: dict, prompt: str) -> str:
    return f"{project['industry']} {project['year']} {section['title']} {prompt}"


def build_rag_messages(
    project: dict,
    section: dict,
    prompt: str,
    chunks: list[dict],
    system_prompt: str,
    use_parent_context: bool = True,
) -> list[dict[str, str]]:
    context_lines = []
    seen_parent_ids: set[str] = set()
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        parent_id = str(metadata.get("parent_id") or chunk["id"])
        if use_parent_context and parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)
        context_text = str(metadata.get("parent_text") or chunk["text"]) if use_parent_context else str(chunk["text"])
        context_lines.append(
            f"资料 {index}\n"
            f"chunk_id: {chunk['id']}\n"
            f"文档: {metadata.get('document_name')} / {metadata.get('document_type')}\n"
            f"标题路径: {metadata.get('heading_path') or '未识别'}\n"
            f"位置: {metadata.get('source_locator')}\n"
            f"内容: {context_text}"
        )

    user = f"""请为报告《{project['name']}》生成章节「{section['title']}」。

报告基础信息：
- 行业：{project['industry']}
- 年份：{project['year']}
- 语言：{project['language']}

章节 prompt：
{prompt}

可引用资料：
{chr(10).join(context_lines)}
"""
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


def ensure_citations(content: str, chunk_ids: list[str]) -> str:
    known_ids = set(chunk_ids)
    used_ids = set(re.findall(r"\[ref:([^\]]+)\]", content))
    invalid_ids = used_ids - known_ids
    for invalid_id in invalid_ids:
        content = content.replace(f"[ref:{invalid_id}]", "")

    if not re.search(r"\[ref:[^\]]+\]", content) and chunk_ids:
        content = f"{content.rstrip()}\n\n资料来源：{', '.join(f'[ref:{chunk_id}]' for chunk_id in chunk_ids[:3])}"
    return content.strip()


def generation_settings_snapshot(payload: GenerateRequest) -> dict:
    return {
        "style": payload.style,
        "length": payload.length,
        "stream": payload.stream,
        "retrieval_top_k": payload.retrieval_top_k,
        "per_document_limit": payload.per_document_limit,
        "use_parent_context": payload.use_parent_context,
        "temperature": payload.temperature,
        "max_tokens": payload.max_tokens,
    }


def apply_per_document_limit(chunks: list[dict], per_document_limit: int) -> list[dict]:
    counts: dict[str, int] = {}
    selected: list[dict] = []
    for chunk in chunks:
        document_id = str(chunk.get("metadata", {}).get("document_id") or "")
        count = counts.get(document_id, 0)
        if count >= per_document_limit:
            continue
        counts[document_id] = count + 1
        selected.append(chunk)
    return selected


def loggable_chunk(chunk: dict) -> dict:
    metadata = chunk.get("metadata", {})
    return {
        "id": chunk.get("id"),
        "document_id": metadata.get("document_id"),
        "document_name": metadata.get("document_name"),
        "document_type": metadata.get("document_type"),
        "heading_path": metadata.get("heading_path"),
        "source_locator": metadata.get("source_locator"),
        "distance": chunk.get("distance"),
        "text": chunk.get("text"),
        "parent_id": metadata.get("parent_id"),
        "parent_text": metadata.get("parent_text"),
    }


def citation_map_from_chunks(chunks: list[dict]) -> dict:
    return {str(chunk.get("id")): loggable_chunk(chunk) for chunk in chunks}


def format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def prepare_section_generation(conn, section_id: str, payload: GenerateRequest) -> dict:
    section = get_section_or_404(conn, section_id)
    if payload.prompt:
        conn.execute("UPDATE sections SET prompt = ? WHERE id = ?", (payload.prompt, section_id))
        section["prompt"] = payload.prompt

    project = fetch_project(conn)
    docs = get_documents_by_ids(conn, payload.reference_ids)
    indexed_docs = [doc for doc in docs if doc.get("parse_status") == "indexed" and doc.get("chunk_count", 0) > 0]
    if not indexed_docs:
        raise HTTPException(status_code=400, detail="请先上传并勾选至少一份已完成入库的真实参考文档。")

    query = build_rag_query(project, section, section["prompt"])
    query_embedding_client = EmbeddingClient()
    try:
        query_embedding = query_embedding_client.embed_query(query)
    except EmbeddingClientError as exc:
        raise HTTPException(status_code=502, detail=f"Embedding 调用失败：{exc}") from exc

    retrieved_chunks = VectorStore().query(
        query_embedding=query_embedding,
        document_ids=[doc["id"] for doc in indexed_docs],
        top_k=payload.retrieval_top_k,
    )
    retrieved_chunks = apply_per_document_limit(retrieved_chunks, payload.per_document_limit)
    if not retrieved_chunks:
        raise HTTPException(status_code=400, detail="未检索到可用于生成的资料片段。")

    system_prompt = (payload.system_prompt or DEFAULT_SYSTEM_PROMPT).strip() or DEFAULT_SYSTEM_PROMPT
    messages = build_rag_messages(
        project,
        section,
        section["prompt"],
        retrieved_chunks,
        system_prompt=system_prompt,
        use_parent_context=payload.use_parent_context,
    )
    return {
        "section": section,
        "project": project,
        "indexed_docs": indexed_docs,
        "query": query,
        "query_embedding_client": query_embedding_client,
        "retrieved_chunks": retrieved_chunks,
        "system_prompt": system_prompt,
        "messages": messages,
    }


def persist_section_generation(
    conn,
    *,
    section_id: str,
    section: dict,
    project: dict,
    indexed_docs: list[dict],
    retrieved_chunks: list[dict],
    query: str,
    query_embedding_client: EmbeddingClient,
    payload: GenerateRequest,
    system_prompt: str,
    messages: list[dict[str, str]],
    llm_result: LLMResult,
    content: str,
    reference_ids: list[str],
) -> dict:
    create_version(
        conn,
        section_id=section_id,
        source="RAG + Qwen",
        summary=f"基于 {len(indexed_docs)} 份资料、{len(retrieved_chunks)} 个片段生成真实 RAG 版本。",
        content=content,
        reference_ids=reference_ids,
    )
    record_ai_call(
        conn,
        operation="章节检索 embedding",
        model=settings.embedding_model,
        usage=query_embedding_client.total_usage,
        metadata={
            "section_id": section_id,
            "query": query,
            "generation_settings": generation_settings_snapshot(payload),
            "retrieved_chunks": len(retrieved_chunks),
            "reference_ids": reference_ids,
        },
    )
    record_ai_call(
        conn,
        operation="章节生成 LLM",
        model=settings.llm_model,
        usage=llm_result.usage,
        metadata={
            "section_id": section_id,
            "section_title": section["title"],
            "project": project,
            "generation_settings": generation_settings_snapshot(payload),
            "system_prompt": system_prompt,
            "llm_api_request": llm_result.request_payload,
            "llm_api_response": llm_result.response_payload,
            "prompt_structure": messages,
            "output_content": content,
            "raw_output_content": llm_result.content,
            "retrieved_chunks": [loggable_chunk(chunk) for chunk in retrieved_chunks],
            "reference_ids": reference_ids,
            "citation_map": citation_map_from_chunks(retrieved_chunks),
            "citation_rendering_rule": "正文中的 [ref:chunk_id] 会在前端通过 workspace.citations[chunk_id] 映射到 document_chunks 与 documents 表，从而显示文档名、类型、source_locator 和片段文本。",
        },
    )
    return workspace_payload(conn)


def stream_section_generation(section_id: str, payload: GenerateRequest) -> Iterator[str]:
    with get_connection() as conn:
        try:
            context = prepare_section_generation(conn, section_id, payload)
            section = context["section"]
            project = context["project"]
            indexed_docs = context["indexed_docs"]
            retrieved_chunks = context["retrieved_chunks"]
            messages = context["messages"]
            system_prompt = context["system_prompt"]
            query = context["query"]
            query_embedding_client = context["query_embedding_client"]

            yield format_sse(
                "status",
                {
                    "phase": "generating",
                    "message": f"已检索 {len(retrieved_chunks)} 个资料片段，开始生成…",
                },
            )

            llm_client = LLMClient()
            content_parts: list[str] = []
            try:
                for delta in llm_client.iter_generate(
                    messages=messages,
                    max_tokens=payload.max_tokens,
                    temperature=payload.temperature,
                ):
                    content_parts.append(delta)
                    yield format_sse("delta", {"content": delta})
            except LLMClientError as exc:
                yield format_sse("error", {"detail": f"LLM 调用失败：{exc}"})
                return

            raw_content = "".join(content_parts).strip()
            if not raw_content:
                yield format_sse("error", {"detail": "LLM 未返回任何内容"})
                return

            reference_ids = [chunk["id"] for chunk in retrieved_chunks]
            content = ensure_citations(raw_content, reference_ids)
            llm_result = LLMResult(
                content=raw_content,
                usage=llm_client.last_usage,
                request_payload=_redacted_stream_request(messages, payload),
                response_payload={
                    "stream": True,
                    "events_sample": llm_client.last_stream_events,
                    "assembled_content": raw_content,
                    "usage": llm_client.last_usage,
                },
            )
            workspace = persist_section_generation(
                conn,
                section_id=section_id,
                section=section,
                project=project,
                indexed_docs=indexed_docs,
                retrieved_chunks=retrieved_chunks,
                query=query,
                query_embedding_client=query_embedding_client,
                payload=payload,
                system_prompt=system_prompt,
                messages=messages,
                llm_result=llm_result,
                content=content,
                reference_ids=reference_ids,
            )
            yield format_sse("done", {"workspace": workspace})
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail, ensure_ascii=False)
            yield format_sse("error", {"detail": detail})
        except Exception as exc:
            yield format_sse("error", {"detail": str(exc)})


def _redacted_stream_request(messages: list[dict[str, str]], payload: GenerateRequest) -> dict:
    return {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": payload.temperature,
        "max_tokens": payload.max_tokens,
        "stream": True,
    }


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name.strip() or "uploaded-document"
    return re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", name)


def document_type_from_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return "PDF"
    if suffix == ".docx":
        return "Word"
    if suffix == ".xlsx":
        return "Excel"
    if suffix == ".md":
        return "Markdown"
    if suffix == ".txt":
        return "文本"
    return "资料"


def scalar_metadata(value: object) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


def build_vector_chunks(doc_id: str, document_name: str, document_type: str, chunks: list[TextChunk]) -> list[dict]:
    vector_chunks = []
    for chunk in chunks:
        metadata = {
            **chunk.metadata,
            "document_id": doc_id,
            "document_name": document_name,
            "document_type": document_type,
            "chunk_index": chunk.chunk_index,
            "source_locator": chunk.source_locator,
        }
        vector_chunks.append(
            {
                "id": chunk.id,
                "text": chunk.text,
                "metadata": {key: scalar_metadata(value) for key, value in metadata.items()},
            }
        )
    return vector_chunks


def remove_document_chunks_from_indexes(conn, document_id: str) -> None:
    rows = conn.execute("SELECT id FROM document_chunks WHERE document_id = ?", (document_id,)).fetchall()
    chunk_ids = [row["id"] for row in rows]
    if chunk_ids:
        try:
            VectorStore().delete_chunks(chunk_ids)
        except Exception:
            pass
    conn.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))


def index_document_file(doc_id: str, document_name: str, document_type: str, upload_path: Path) -> tuple[list[TextChunk], list[dict], dict, dict]:
    parsed = parse_document(upload_path)
    chunks = chunk_document(parsed, document_id=doc_id, document_type=document_type)
    if not chunks:
        raise DocumentParserError("No text chunks were generated from this document.")

    quality = summarize_chunk_quality(chunks)
    gate_error = evaluate_chunk_quality_gate(quality)
    if gate_error:
        raise DocumentParserError(gate_error)

    embedding_client = EmbeddingClient()
    embeddings = embedding_client.embed_texts([chunk.text for chunk in chunks])
    vector_chunks = build_vector_chunks(doc_id, document_name, document_type, chunks)
    VectorStore().upsert_chunks(vector_chunks, embeddings)
    return chunks, vector_chunks, quality, embedding_client.total_usage


def save_indexed_chunks(
    conn,
    doc_id: str,
    chunks: list[TextChunk],
    vector_chunks: list[dict],
    quality: dict,
) -> None:
    remove_document_chunks_from_indexes(conn, doc_id)
    conn.executemany(
        """
        INSERT INTO document_chunks (
            id, document_id, chunk_index, text, source_locator, metadata, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                chunk.id,
                doc_id,
                chunk.chunk_index,
                chunk.text,
                chunk.source_locator,
                json.dumps(
                    {
                        **vector_chunks[index]["metadata"],
                        "quality_status": quality["status"],
                        "quality_score": quality["score"],
                    },
                    ensure_ascii=False,
                ),
                utc_now(),
            )
            for index, chunk in enumerate(chunks)
        ],
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "ai-report-generator-backend",
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/runtime")
def runtime() -> dict[str, object]:
    return {
        "python": python_version(),
        "fastapi": version("fastapi"),
        "llm": {
            "model": settings.llm_model,
            "base_url": settings.llm_base_url,
            "api_key_configured": bool(settings.dashscope_api_key),
            "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
        },
        "embedding": {
            "model": settings.embedding_model,
            "base_url": settings.embedding_base_url,
        },
        "storage": {
            "database_url": settings.database_url,
            "chroma_path": settings.chroma_path,
            "upload_dir": settings.upload_dir,
            "export_dir": settings.export_dir,
        },
    }


@app.get("/api/workspace")
def get_workspace() -> dict:
    with get_connection() as conn:
        return workspace_payload(conn)


@app.patch("/api/project")
def update_project(payload: ProjectUpdate) -> dict:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE projects
            SET name = ?, industry = ?, year = ?, language = ?, updated_at = ?
            WHERE id = ?
            """,
            (payload.name, payload.industry, payload.year, payload.language, utc_now(), PROJECT_ID),
        )
        return workspace_payload(conn)


@app.post("/api/project/new")
def create_new_report(payload: ProjectCreate) -> dict:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE projects
            SET name = ?, industry = ?, year = ?, language = ?, updated_at = ?
            WHERE id = ?
            """,
            (payload.name, payload.industry, payload.year, payload.language, utc_now(), PROJECT_ID),
        )
        clear_report_versions(conn)
        return workspace_payload(conn)


@app.post("/api/documents/upload")
def upload_document(payload: DocumentCreate) -> dict:
    with get_connection() as conn:
        doc_id = f"doc_{uuid4().hex[:10]}"
        conn.execute(
            """
            INSERT INTO documents (
                id, name, type, selected, summary, created_at, upload_status, parse_status, chunk_count
            )
            VALUES (?, ?, ?, 1, ?, ?, 'manual', 'not_indexed', 0)
            """,
            (
                doc_id,
                payload.name,
                payload.type,
                f"本地手动新增资料《{payload.name}》。该资料尚未解析入库，只用于流程占位。",
                utc_now(),
            ),
        )
        return workspace_payload(conn)


@app.post("/api/documents/upload-file")
async def upload_file(file: UploadFile = File(...), type: str = Form("")) -> dict:
    original_name = sanitize_filename(file.filename or "uploaded-document")
    document_type = type or document_type_from_suffix(original_name)
    doc_id = f"doc_{uuid4().hex[:10]}"
    upload_path = Path(settings.upload_dir) / f"{doc_id}_{original_name}"
    upload_path.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO documents (
                id, name, type, selected, summary, created_at, file_path,
                upload_status, parse_status, chunk_count, error_message
            )
            VALUES (?, ?, ?, 1, ?, ?, ?, 'uploaded', 'parsing', 0, NULL)
            """,
            (
                doc_id,
                original_name,
                document_type,
                "文件已上传，正在解析和写入向量库。",
                utc_now(),
                str(upload_path),
            ),
        )

    try:
        upload_path.write_bytes(await file.read())
        chunks, vector_chunks, quality, embedding_usage = index_document_file(
            doc_id=doc_id,
            document_name=original_name,
            document_type=document_type,
            upload_path=upload_path,
        )

        with get_connection() as conn:
            save_indexed_chunks(conn, doc_id, chunks, vector_chunks, quality)
            conn.execute(
                """
                UPDATE documents
                SET summary = ?, parse_status = 'indexed', chunk_count = ?, indexed_at = ?,
                    error_message = NULL
                WHERE id = ?
                """,
                (
                    format_chunk_index_summary(len(chunks), quality),
                    len(chunks),
                    utc_now(),
                    doc_id,
                ),
            )
            record_ai_call(
                conn,
                operation="文档入库 embedding",
                model=settings.embedding_model,
                usage=embedding_usage,
                metadata={"document_id": doc_id, "chunk_count": len(chunks), "quality_score": quality["score"]},
            )
            return workspace_payload(conn)
    except (DocumentParserError, EmbeddingClientError, ValueError) as exc:
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE documents
                SET parse_status = 'failed', error_message = ?, summary = ?
                WHERE id = ?
                """,
                (str(exc), f"解析或入库失败：{exc}", doc_id),
            )
            return workspace_payload(conn)


@app.patch("/api/documents/{document_id}/selection")
def update_document_selection(document_id: str, payload: DocumentSelection) -> dict:
    with get_connection() as conn:
        result = conn.execute(
            "UPDATE documents SET selected = ? WHERE id = ?",
            (1 if payload.selected else 0, document_id),
        )
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        return workspace_payload(conn)


@app.get("/api/documents/{document_id}/preview")
def preview_document(document_id: str) -> dict:
    with get_connection() as conn:
        document = get_document_or_404(conn, document_id)
        rows = conn.execute(
            """
            SELECT id, chunk_index, text, source_locator, metadata, created_at
            FROM document_chunks
            WHERE document_id = ?
            ORDER BY chunk_index
            """,
            (document_id,),
        ).fetchall()
        chunk_items = []
        text_chunks = []
        for row in rows:
            item = dict(row)
            metadata = json_loads(item.pop("metadata"), {})
            item["metadata"] = metadata
            chunk_items.append(item)
            text_chunks.append(
                TextChunk(
                    id=item["id"],
                    text=item["text"],
                    chunk_index=item["chunk_index"],
                    source_locator=item["source_locator"],
                    metadata=metadata,
                )
            )
        document["quality_metrics"] = summarize_chunk_quality(text_chunks)
        document["chunks"] = chunk_items[:50]
        return document


@app.post("/api/documents/{document_id}/reindex")
def reindex_document(document_id: str) -> dict:
    with get_connection() as conn:
        document = get_document_or_404(conn, document_id)
        file_path = document.get("file_path")
        if not file_path:
            raise HTTPException(status_code=400, detail="这份资料没有本地上传文件，无法重新解析入库。")
        upload_path = Path(file_path)
        if not upload_path.exists():
            raise HTTPException(status_code=404, detail="本地上传文件不存在，无法重新解析入库。")
        conn.execute(
            """
            UPDATE documents
            SET parse_status = 'parsing', error_message = NULL, summary = ?
            WHERE id = ?
            """,
            ("正在用新版 chunking 策略重新解析和写入向量库。", document_id),
        )

    try:
        chunks, vector_chunks, quality, embedding_usage = index_document_file(
            doc_id=document_id,
            document_name=document["name"],
            document_type=document["type"],
            upload_path=Path(document["file_path"]),
        )
        with get_connection() as conn:
            save_indexed_chunks(conn, document_id, chunks, vector_chunks, quality)
            conn.execute(
                """
                UPDATE documents
                SET summary = ?, parse_status = 'indexed', chunk_count = ?, indexed_at = ?,
                    error_message = NULL
                WHERE id = ?
                """,
                (
                    format_chunk_index_summary(len(chunks), quality),
                    len(chunks),
                    utc_now(),
                    document_id,
                ),
            )
            record_ai_call(
                conn,
                operation="文档重新入库 embedding",
                model=settings.embedding_model,
                usage=embedding_usage,
                metadata={"document_id": document_id, "chunk_count": len(chunks), "quality_score": quality["score"]},
            )
            return workspace_payload(conn)
    except (DocumentParserError, EmbeddingClientError, ValueError) as exc:
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE documents
                SET parse_status = 'failed', error_message = ?, summary = ?
                WHERE id = ?
                """,
                (str(exc), f"重新解析或入库失败：{exc}", document_id),
            )
            return workspace_payload(conn)


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str) -> dict:
    with get_connection() as conn:
        document = get_document_or_404(conn, document_id)
        rows = conn.execute("SELECT id FROM document_chunks WHERE document_id = ?", (document_id,)).fetchall()
        chunk_ids = [row["id"] for row in rows]

        if chunk_ids:
            try:
                VectorStore().delete_chunks(chunk_ids)
            except Exception:
                # Chroma cleanup should not block removing the document from the app workspace.
                pass

        conn.execute("DELETE FROM document_chunks WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))

        file_path = document.get("file_path")
        if file_path:
            try:
                Path(file_path).unlink(missing_ok=True)
            except OSError:
                pass

        return workspace_payload(conn)


@app.patch("/api/sections/{section_id}/prompt")
def update_prompt(section_id: str, payload: PromptUpdate) -> dict:
    with get_connection() as conn:
        get_section_or_404(conn, section_id)
        conn.execute("UPDATE sections SET prompt = ? WHERE id = ?", (payload.prompt, section_id))
        return workspace_payload(conn)


@app.post("/api/sections/{section_id}/generate")
def generate_section(section_id: str, payload: GenerateRequest):
    if payload.stream:
        return StreamingResponse(
            stream_section_generation(section_id, payload),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    with get_connection() as conn:
        context = prepare_section_generation(conn, section_id, payload)
        section = context["section"]
        project = context["project"]
        indexed_docs = context["indexed_docs"]
        retrieved_chunks = context["retrieved_chunks"]
        messages = context["messages"]
        system_prompt = context["system_prompt"]
        query = context["query"]
        query_embedding_client = context["query_embedding_client"]

        try:
            llm_result = LLMClient().generate(
                messages=messages,
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                stream=False,
            )
        except LLMClientError as exc:
            raise HTTPException(status_code=504, detail=f"LLM 调用失败：{exc}") from exc

        reference_ids = [chunk["id"] for chunk in retrieved_chunks]
        content = ensure_citations(llm_result.content, reference_ids)
        return persist_section_generation(
            conn,
            section_id=section_id,
            section=section,
            project=project,
            indexed_docs=indexed_docs,
            retrieved_chunks=retrieved_chunks,
            query=query,
            query_embedding_client=query_embedding_client,
            payload=payload,
            system_prompt=system_prompt,
            messages=messages,
            llm_result=llm_result,
            content=content,
            reference_ids=reference_ids,
        )


@app.post("/api/sections/{section_id}/clear")
def clear_section(section_id: str) -> dict:
    with get_connection() as conn:
        get_section_or_404(conn, section_id)
        conn.execute("DELETE FROM section_versions WHERE section_id = ?", (section_id,))
        conn.execute(
            "UPDATE sections SET current_version_id = NULL, confirmed = 0 WHERE id = ?",
            (section_id,),
        )
        return workspace_payload(conn)


@app.post("/api/sections/{section_id}/manual-edit")
def manual_edit_section(section_id: str, payload: ManualEditRequest) -> dict:
    with get_connection() as conn:
        section = get_section_or_404(conn, section_id)
        current_version_id = section["current_version_id"]
        reference_ids: list[str] = []
        if current_version_id:
            row = conn.execute("SELECT reference_ids FROM section_versions WHERE id = ?", (current_version_id,)).fetchone()
            reference_ids = json_loads(row["reference_ids"], []) if row else []

        create_version(
            conn,
            section_id=section_id,
            source="人工编辑",
            summary="基于当前章节正文保存了人工编辑版本。",
            content=payload.content,
            reference_ids=reference_ids,
        )
        return workspace_payload(conn)


@app.post("/api/sections/{section_id}/select-version")
def select_version(section_id: str, payload: SelectVersionRequest) -> dict:
    with get_connection() as conn:
        get_section_or_404(conn, section_id)
        row = conn.execute(
            "SELECT id FROM section_versions WHERE id = ? AND section_id = ?",
            (payload.version_id, section_id),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Version not found")
        conn.execute(
            "UPDATE sections SET current_version_id = ?, confirmed = 1 WHERE id = ?",
            (payload.version_id, section_id),
        )
        return workspace_payload(conn)


@app.post("/api/sections/{section_id}/confirm")
def confirm_section(section_id: str) -> dict:
    with get_connection() as conn:
        section = get_section_or_404(conn, section_id)
        if not section["current_version_id"]:
            raise HTTPException(status_code=400, detail="Section has no current version")
        conn.execute("UPDATE sections SET confirmed = 1 WHERE id = ?", (section_id,))
        return workspace_payload(conn)


@app.post("/api/sections/{section_id}/enhance-prompt")
def enhance_prompt(section_id: str, payload: PromptAssistRequest) -> dict[str, str]:
    with get_connection() as conn:
        section = get_section_or_404(conn, section_id)
        project = fetch_project(conn)

    prompt = f"""你是一名投行内部行业研究员。请为《{project['name']}》撰写章节「{section['title']}」。

报告基础信息：
- 行业：{project['industry']}
- 年份：{project['year']}
- 语言：{project['language']}

写作目标：{payload.objective or '形成可用于行业风险判断和客户准入讨论的分析段落。'}
重点问题：{payload.key_questions or '市场事实、核心判断、风险含义、银行业务启示。'}
地域范围：{payload.geography or '全球、中国、香港、东南亚，按章节需要取舍。'}
风险维度：{payload.risk_dimensions or '政策合规、供需、电力、资本开支、客户集中度、竞争格局。'}
语气和篇幅：{payload.tone_length or '投行内部研究报告口吻，克制、结构化，约 600-900 字。'}
排除内容：{payload.exclusions or '避免营销化表达，不编造具体数据，不输出无来源的绝对判断。'}

请按「结论先行 - 关键依据 - 风险提示 - 业务含义」组织内容，并在引用资料观点时保留来源标注。"""
    return {"prompt": prompt}


@app.get("/api/report-preview")
def report_preview() -> dict:
    with get_connection() as conn:
        project = fetch_project(conn)
        sections = fetch_sections(conn)
        documents = fetch_documents(conn)
        issues = export_check(sections, documents)
        return {
            "project": project,
            "sections": sections,
            "citations": fetch_citations(conn),
            "issues": issues,
        }


@app.post("/api/exports")
def create_export(payload: ExportRequest) -> dict:
    with get_connection() as conn:
        sections = fetch_sections(conn)
        documents = fetch_documents(conn)
        issues = export_check(sections, documents)
        export_id = f"export_{uuid4().hex[:10]}"
        status = "已生成模拟导出记录" if not issues else "已生成但需复核"
        conn.execute(
            """
            INSERT INTO export_records (id, format, status, issues, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (export_id, payload.format, status, json.dumps(issues, ensure_ascii=False), utc_now()),
        )
        return workspace_payload(conn)


@app.get("/api/report-template")
def report_template() -> dict[str, object]:
    with get_connection() as conn:
        sections = fetch_sections(conn)
    return {"name": "数据中心行业风险研究报告", "sections": sections}
