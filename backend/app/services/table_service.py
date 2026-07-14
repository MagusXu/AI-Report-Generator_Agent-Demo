from __future__ import annotations

import json
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from app.services.embedding_client import EmbeddingClient, EmbeddingClientError
from app.services.llm_client import LLMClient, LLMClientError, LLMResult
from app.services.vector_store import VectorStore
from app.database import json_loads


MAX_TABLES = 5
TABLE_CANDIDATES_TOP_K = 5
TABLE_SYNTHESIZE_TEMPERATURE = 0.1
TABLE_SYNTHESIZE_MAX_TOKENS = 1800
UNVERIFIED_CELL_MARKER = "[[?"
UNVERIFIED_CELL_END = "?]]"
PLACEHOLDER_PATTERN = re.compile(r"<<TABLE:(\d+)>>")
CANDIDATE_TABLE_MAX_ROWS = 10


def _split_table_row(line: str) -> list[str]:
    line = line.strip()
    if "|" not in line:
        return [cell.strip() for cell in re.split(r"\s{2,}", line) if cell.strip()]
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    meaningful = [cell.strip() for cell in cells if cell.strip()]
    if not meaningful:
        return True
    return all(re.fullmatch(r":?-{3,}:?", cell) for cell in meaningful)


def _trim_empty_edge_columns(matrix: list[list[str]]) -> list[list[str]]:
    if not matrix:
        return matrix
    column_count = max(len(row) for row in matrix)
    normalized = [row + [""] * (column_count - len(row)) for row in matrix]

    leading = 0
    while leading < column_count and all(not row[leading].strip() for row in normalized):
        leading += 1
    trailing = column_count
    while trailing > leading and all(not row[trailing - 1].strip() for row in normalized):
        trailing -= 1
    return [row[leading:trailing] for row in normalized]


def _normalize_table_matrix(rows: list[list[str]]) -> list[list[str]] | None:
    filtered: list[list[str]] = []
    for row in rows:
        if _is_separator_row(row):
            continue
        if len([cell for cell in row if cell.strip()]) >= 2:
            filtered.append(row)
    if len(filtered) < 2:
        return None
    matrix = _trim_empty_edge_columns(filtered)
    if len(matrix) < 2 or len(matrix[0]) < 2:
        return None
    column_count = len(matrix[0])
    return [row + [""] * (column_count - len(row)) for row in matrix]


def _longest_table_line_run(lines: list[str]) -> list[str]:
    best: list[str] = []
    current: list[str] = []
    for line in lines:
        cells = _split_table_row(line)
        if len(cells) >= 2:
            current.append(line)
            continue
        if len(current) > len(best):
            best = current
        current = []
    if len(current) > len(best):
        best = current
    return best


def extract_table_preview(text: str, *, max_rows: int = CANDIDATE_TABLE_MAX_ROWS) -> dict[str, Any] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    pipe_lines = [line for line in lines if "|" in line]
    candidate_lines = pipe_lines if len(pipe_lines) >= 2 else _longest_table_line_run(lines)
    matrix = _normalize_table_matrix([_split_table_row(line) for line in candidate_lines])
    if matrix is None:
        return None

    header = matrix[0]
    body_rows = matrix[1:]
    visible_rows = body_rows[:max_rows]
    return {
        "parsed": True,
        "columns": header,
        "rows": visible_rows,
        "truncated": len(body_rows) > max_rows,
        "total_rows": len(body_rows),
    }


def build_candidate_preview(text: str) -> dict[str, Any]:
    source = text.strip()
    table_preview = extract_table_preview(source)
    if table_preview:
        return {
            "preview_text": "",
            "table_preview": table_preview,
        }

    preview_lines = [line for line in source.splitlines() if line.strip()][:4]
    return {
        "preview_text": "\n".join(preview_lines)[:320],
        "table_preview": None,
    }

TABLE_SYNTHESIZE_SYSTEM = """你是投行内部行业研究报告的表格生成助手。
你必须只基于给定资料填表，不要编造具体数据。

输出要求：
1. 只输出 JSON，不要 Markdown 代码块，不要解释。
2. JSON 格式：
{
  "rows": [
    {
      "cells": [
        {"column": "列名", "value": "单元格值", "refs": ["chunk_id"]}
      ]
    }
  ]
}
3. 每个单元格必须列出 refs（chunk_id 数组）；若资料不足，value 填「资料不足」且 refs 为空数组。
4. 不要输出正文，不要输出引用标注 [ref:...]，引用只放在 refs 字段。"""


class TableColumnSpec(BaseModel):
    name: str = Field(min_length=1)
    description: str = ""


class SynthesizeTableConfig(BaseModel):
    mode: Literal["synthesize"] = "synthesize"
    title: str = Field(min_length=1)
    columns: list[TableColumnSpec] = Field(min_length=1)
    notes: str = ""


class VerbatimTableConfig(BaseModel):
    mode: Literal["verbatim"] = "verbatim"
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    confirmed_chunk_id: str = Field(min_length=1)


TableConfig = Annotated[SynthesizeTableConfig | VerbatimTableConfig, Field(discriminator="mode")]


class TableCandidatesRequest(BaseModel):
    description: str = Field(min_length=1)
    reference_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(default=TABLE_CANDIDATES_TOP_K, ge=1, le=10)


class TableProcessingResult(BaseModel):
    content: str
    extra_reference_ids: list[str] = Field(default_factory=list)
    table_warnings: list[str] = Field(default_factory=list)
    table_metadata: list[dict[str, Any]] = Field(default_factory=list)


class TableValidationError(ValueError):
    pass


def validate_table_configs(tables: list[SynthesizeTableConfig | VerbatimTableConfig]) -> None:
    if len(tables) > MAX_TABLES:
        raise TableValidationError(f"每个章节最多支持 {MAX_TABLES} 张表格")
    for index, table in enumerate(tables, start=1):
        if isinstance(table, VerbatimTableConfig):
            if not table.title.strip():
                raise TableValidationError(f"表格 {index} 缺少标题")
            if not table.description.strip():
                raise TableValidationError(f"表格 {index} 缺少原表检索描述")
            if not table.confirmed_chunk_id.strip():
                raise TableValidationError(f"表格 {index} 尚未确认引用的原表 chunk")
        elif isinstance(table, SynthesizeTableConfig):
            if not table.title.strip():
                raise TableValidationError(f"表格 {index} 缺少标题")
            if not table.columns:
                raise TableValidationError(f"表格 {index} 至少需要 1 列定义")


def build_table_query_suffix(tables: list[SynthesizeTableConfig | VerbatimTableConfig]) -> str:
    if not tables:
        return ""
    parts: list[str] = []
    for index, table in enumerate(tables, start=1):
        if isinstance(table, SynthesizeTableConfig):
            columns = ", ".join(column.name for column in table.columns)
            parts.append(f"表格{index}:{table.title} 列:{columns}")
        else:
            parts.append(f"表格{index}:{table.title} 检索:{table.description}")
    return " ".join(parts)


def format_tables_for_narrative_prompt(tables: list[SynthesizeTableConfig | VerbatimTableConfig]) -> str:
    lines: list[str] = []
    for index, table in enumerate(tables, start=1):
        if isinstance(table, SynthesizeTableConfig):
            column_lines = "\n".join(
                f"    - {column.name}：{column.description or '按资料填写'}" for column in table.columns
            )
            notes = f"\n    备注：{table.notes}" if table.notes.strip() else ""
            lines.append(
                f"- 表格 {index}（AI 汇总）：{table.title}\n"
                f"  列定义：\n{column_lines}{notes}"
            )
        else:
            lines.append(
                f"- 表格 {index}（引用原表）：{table.title}\n"
                f"  检索说明：{table.description}\n"
                f"  将在正文相关段落后插入原表内容。"
            )
    return "\n".join(lines)


def narrative_table_instructions(tables: list[SynthesizeTableConfig | VerbatimTableConfig]) -> str:
    placeholders = ", ".join(f"<<TABLE:{index}>>" for index in range(1, len(tables) + 1))
    return f"""【本章节需插入表格】
{format_tables_for_narrative_prompt(tables)}

表格占位符规则：
1. 正文按常规结构撰写，不要自行编写任何表格内容。
2. 在最适合承接对应表格的段落后输出占位符；通常建议在「关键依据」相关论述之后。
3. 每张表恰好输出一个占位符，按顺序使用：{placeholders}
4. 找不到合适位置时，可将占位符放在「关键依据」最后一段之后。"""


def search_table_candidates(
    *,
    description: str,
    document_ids: list[str],
    top_k: int = TABLE_CANDIDATES_TOP_K,
) -> list[dict[str, Any]]:
    embedding_client = EmbeddingClient()
    try:
        query_embedding = embedding_client.embed_query(description)
    except EmbeddingClientError as exc:
        raise RuntimeError(f"Embedding 调用失败：{exc}") from exc

    vector_store = VectorStore()
    raw_results = vector_store.query(
        query_embedding=query_embedding,
        document_ids=document_ids,
        top_k=max(top_k * 4, 20),
    )

    table_rows: list[dict[str, Any]] = []
    other_rows: list[dict[str, Any]] = []
    for row in raw_results:
        metadata = row.get("metadata") or {}
        if str(metadata.get("block_type") or "") == "table":
            table_rows.append(row)
        else:
            other_rows.append(row)

    ranked = table_rows + other_rows
    seen_ids: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for row in ranked:
        chunk_id = str(row.get("id") or "")
        if not chunk_id or chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        metadata = row.get("metadata") or {}
        preview_source = str(metadata.get("parent_text") or row.get("text") or "")
        preview_payload = build_candidate_preview(preview_source)
        candidates.append(
            {
                "chunk_id": chunk_id,
                "document_id": str(metadata.get("document_id") or ""),
                "document_name": str(metadata.get("document_name") or ""),
                "source_locator": str(metadata.get("source_locator") or ""),
                "preview_text": preview_payload["preview_text"],
                "table_preview": preview_payload["table_preview"],
                "block_type": str(metadata.get("block_type") or "text"),
                "score": row.get("distance"),
            }
        )
        if len(candidates) >= top_k:
            break
    return candidates


def fetch_chunk_record(conn, chunk_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT c.*, d.name AS document_name, d.type AS document_type
        FROM document_chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.id = ?
        """,
        (chunk_id,),
    ).fetchone()
    if row is None:
        return None
    item = dict(row)
    item["metadata"] = json_loads(item.get("metadata"), {})
    return item


def format_table_with_caption(body: str, title: str, trailing: str = "") -> str:
    caption = title.strip()
    parts = [body.strip()]
    if caption:
        parts.append(f"[[表题：{caption}]]")
    if trailing.strip():
        parts.append(trailing.strip())
    return "\n\n".join(part for part in parts if part)


def build_verbatim_table_body(chunk: dict[str, Any], title: str) -> str:
    metadata = chunk.get("metadata") or {}
    chunk_id = str(chunk["id"])
    fallback = f"{metadata.get('document_name')} / {chunk.get('source_locator')}"
    caption = title.strip() or str(fallback)
    body = str(metadata.get("parent_text") or chunk.get("text") or "").strip()
    return format_table_with_caption(body, caption, f"[ref:{chunk_id}]")


def replace_or_append_tables(content: str, table_bodies: list[str]) -> str:
    if not table_bodies:
        return content.strip()

    result = content
    appended: list[str] = []
    for index, body in enumerate(table_bodies, start=1):
        placeholder = f"<<TABLE:{index}>>"
        if placeholder in result:
            result = result.replace(placeholder, body.strip(), 1)
        else:
            appended.append(body.strip())

    if appended:
        suffix = "\n\n".join(appended)
        result = f"{result.rstrip()}\n\n{suffix}"
    return result.strip()


def _format_context_lines(chunks: list[dict[str, Any]], use_parent_context: bool = True) -> list[str]:
    lines: list[str] = []
    seen_parent_ids: set[str] = set()
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        parent_id = str(metadata.get("parent_id") or chunk.get("id"))
        if use_parent_context and parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)
        context_text = str(metadata.get("parent_text") or chunk.get("text") or "") if use_parent_context else str(chunk.get("text") or "")
        lines.append(
            f"资料 {index}\n"
            f"chunk_id: {chunk.get('id')}\n"
            f"文档: {metadata.get('document_name')} / {metadata.get('document_type')}\n"
            f"位置: {metadata.get('source_locator')}\n"
            f"内容: {context_text}"
        )
    return lines


def build_synthesize_table_messages(
    *,
    project: dict[str, Any],
    section: dict[str, Any],
    table: SynthesizeTableConfig,
    chunks: list[dict[str, Any]],
    use_parent_context: bool = True,
) -> list[dict[str, str]]:
    column_lines = "\n".join(
        f"- {column.name}：{column.description or '按资料填写'}" for column in table.columns
    )
    notes = f"\n备注：{table.notes}" if table.notes.strip() else ""
    user = f"""请为报告《{project['name']}》章节「{section['title']}」生成汇总表格。

表格标题：{table.title}
列定义：
{column_lines}{notes}

可引用资料：
{chr(10).join(_format_context_lines(chunks, use_parent_context=use_parent_context))}

请输出 JSON。"""
    return [{"role": "system", "content": TABLE_SYNTHESIZE_SYSTEM}, {"role": "user", "content": user}]


def extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM 未返回有效 JSON 表格")
    return json.loads(text[start : end + 1])


def _normalize_lookup(value: str) -> str:
    return re.sub(r"\s+", "", value.lower())


def _value_supported(value: str, refs: list[str], chunks_by_id: dict[str, dict[str, Any]]) -> bool:
    if not value or value == "资料不足":
        return True
    if not refs:
        return False
    normalized_value = _normalize_lookup(value)
    digits = re.findall(r"\d+(?:\.\d+)?", value)
    for ref in refs:
        chunk = chunks_by_id.get(ref)
        if chunk is None:
            continue
        haystack = _normalize_lookup(
            f"{chunk.get('text') or ''} {((chunk.get('metadata') or {}).get('parent_text') or '')}"
        )
        if normalized_value and normalized_value in haystack:
            return True
        if digits and any(digit in haystack for digit in digits):
            return True
        if value in haystack:
            return True
    return False


def wrap_unverified_cell(value: str) -> str:
    return f"{UNVERIFIED_CELL_MARKER}{value}{UNVERIFIED_CELL_END}"


def render_table_json(
    payload: dict[str, Any],
    columns: list[TableColumnSpec],
    chunks_by_id: dict[str, dict[str, Any]],
) -> tuple[str, list[str], list[str]]:
    column_names = [column.name for column in columns]
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("表格 JSON 缺少 rows 数组")

    rendered_rows: list[str] = [
        "| " + " | ".join(column_names) + " |",
        "| " + " | ".join("---" for _ in column_names) + " |",
    ]
    extra_reference_ids: list[str] = []
    warnings: list[str] = []

    for row_index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        cells = row.get("cells")
        if not isinstance(cells, list):
            continue
        values_by_column: dict[str, dict[str, Any]] = {}
        for cell in cells:
            if not isinstance(cell, dict):
                continue
            column = str(cell.get("column") or "").strip()
            if not column:
                continue
            values_by_column[column] = cell

        row_values: list[str] = []
        for column_name in column_names:
            cell = values_by_column.get(column_name, {})
            value = str(cell.get("value") or "资料不足").strip() or "资料不足"
            refs = [str(item) for item in (cell.get("refs") or []) if str(item).strip()]
            for ref in refs:
                if ref not in extra_reference_ids:
                    extra_reference_ids.append(ref)
            if value != "资料不足" and not _value_supported(value, refs, chunks_by_id):
                value = wrap_unverified_cell(value)
                warnings.append(f"表格第 {row_index} 行「{column_name}」未能与引用资料对齐")
            row_values.append(value)
        rendered_rows.append("| " + " | ".join(row_values) + " |")

    return "\n".join(rendered_rows), extra_reference_ids, warnings


def build_synthesized_table_body(
    *,
    table: SynthesizeTableConfig,
    llm_result: LLMResult,
    chunks_by_id: dict[str, dict[str, Any]],
) -> tuple[str, list[str], list[str]]:
    payload = extract_json_object(llm_result.content)
    rendered, extra_reference_ids, warnings = render_table_json(payload, table.columns, chunks_by_id)
    body = format_table_with_caption(rendered, table.title)
    return body, extra_reference_ids, warnings


def process_tables_for_section(
    conn,
    *,
    project: dict[str, Any],
    section: dict[str, Any],
    narrative_content: str,
    tables: list[SynthesizeTableConfig | VerbatimTableConfig],
    retrieved_chunks: list[dict[str, Any]],
    use_parent_context: bool = True,
) -> TableProcessingResult:
    if not tables:
        return TableProcessingResult(content=narrative_content.strip())

    chunks_by_id = {str(chunk.get("id")): chunk for chunk in retrieved_chunks}
    table_bodies: list[str] = []
    extra_reference_ids: list[str] = []
    table_warnings: list[str] = []
    table_metadata: list[dict[str, Any]] = []

    for index, table in enumerate(tables, start=1):
        if isinstance(table, VerbatimTableConfig):
            chunk = fetch_chunk_record(conn, table.confirmed_chunk_id)
            if chunk is None:
                raise TableValidationError(f"表格 {index} 引用的 chunk 不存在：{table.confirmed_chunk_id}")
            body = build_verbatim_table_body(chunk, table.title)
            table_bodies.append(body)
            extra_reference_ids.append(table.confirmed_chunk_id)
            table_metadata.append(
                {
                    "index": index,
                    "mode": "verbatim",
                    "title": table.title,
                    "chunk_id": table.confirmed_chunk_id,
                    "description": table.description,
                }
            )
            continue

        messages = build_synthesize_table_messages(
            project=project,
            section=section,
            table=table,
            chunks=retrieved_chunks,
            use_parent_context=use_parent_context,
        )
        llm_client = LLMClient()
        try:
            llm_result = llm_client.generate(
                messages=messages,
                max_tokens=TABLE_SYNTHESIZE_MAX_TOKENS,
                temperature=TABLE_SYNTHESIZE_TEMPERATURE,
                stream=False,
            )
        except LLMClientError as exc:
            raise RuntimeError(f"表格 {index} 生成失败：{exc}") from exc

        body, table_refs, warnings = build_synthesized_table_body(
            table=table,
            llm_result=llm_result,
            chunks_by_id=chunks_by_id,
        )
        table_bodies.append(body)
        extra_reference_ids.extend(ref for ref in table_refs if ref not in extra_reference_ids)
        table_warnings.extend(warnings)
        table_metadata.append(
            {
                "index": index,
                "mode": "synthesize",
                "title": table.title,
                "warnings": warnings,
                "llm_usage": llm_result.usage,
                "llm_api_request": llm_result.request_payload,
                "llm_api_response": llm_result.response_payload,
                "prompt_structure": messages,
                "output_content": llm_result.content,
            }
        )

    final_content = replace_or_append_tables(narrative_content, table_bodies)
    return TableProcessingResult(
        content=final_content,
        extra_reference_ids=extra_reference_ids,
        table_warnings=table_warnings,
        table_metadata=table_metadata,
    )


def content_has_unverified_table_cells(content: str | None) -> bool:
    if not content:
        return False
    return UNVERIFIED_CELL_MARKER in content


def scan_unverified_table_sections(sections: list[dict[str, Any]]) -> list[str]:
    flagged: list[str] = []
    for section in sections:
        current = section.get("current_version") or {}
        content = current.get("content") if isinstance(current, dict) else None
        if content_has_unverified_table_cells(content):
            flagged.append(section["title"])
    return flagged
