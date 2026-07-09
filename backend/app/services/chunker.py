from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import mean
from uuid import uuid4

from app.services.document_parser import ParsedBlock, ParsedDocument


TEXT_CHILD_TARGET = 820
TEXT_PARENT_TARGET = 2200
TABLE_ROWS_PER_CHILD = 18
TABLE_ROWS_PER_PARENT = 45


@dataclass(frozen=True)
class TextChunk:
    id: str
    text: str
    chunk_index: int
    source_locator: str
    metadata: dict = field(default_factory=dict)


def chunk_document(document: ParsedDocument, document_id: str) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    parent_index = 0
    for group in _semantic_groups(document.blocks):
        if group["block_type"] == "table":
            new_chunks, parent_index = _chunk_table_group(group, document_id, len(chunks), parent_index)
        else:
            new_chunks, parent_index = _chunk_text_group(group, document_id, len(chunks), parent_index)
        chunks.extend(new_chunks)
    return chunks


def summarize_chunk_quality(chunks: list[TextChunk]) -> dict:
    lengths = [len(chunk.text) for chunk in chunks]
    if not lengths:
        return {
            "status": "poor",
            "score": 0,
            "chunk_count": 0,
            "avg_chars": 0,
            "min_chars": 0,
            "max_chars": 0,
            "short_chunks": 0,
            "long_chunks": 0,
            "heading_coverage": 0,
            "table_chunks": 0,
            "warnings": ["没有生成可检索片段。"],
        }

    short_chunks = sum(1 for length in lengths if length < 180)
    long_chunks = sum(1 for length in lengths if length > 1700)
    heading_chunks = sum(1 for chunk in chunks if chunk.metadata.get("heading_path"))
    table_chunks = sum(1 for chunk in chunks if chunk.metadata.get("block_type") == "table")
    parent_chunks = sum(1 for chunk in chunks if chunk.metadata.get("parent_text"))
    heading_coverage = round(heading_chunks / len(chunks), 2)
    parent_coverage = round(parent_chunks / len(chunks), 2)

    warnings: list[str] = []
    if short_chunks / len(chunks) > 0.25:
        warnings.append("短片段比例偏高，可能存在过碎切分。")
    if long_chunks:
        warnings.append("存在超长片段，可能降低检索精度。")
    if heading_coverage < 0.35:
        warnings.append("标题路径覆盖偏低，建议检查原文是否缺少清晰章节标题。")
    if parent_coverage < 0.9:
        warnings.append("部分片段缺少 parent 上下文。")

    score = 100
    score -= min(30, short_chunks * 4)
    score -= min(25, long_chunks * 8)
    if heading_coverage < 0.35:
        score -= 12
    if parent_coverage < 0.9:
        score -= 8
    score = max(0, score)
    status = "good" if score >= 82 else "warn" if score >= 60 else "poor"

    return {
        "status": status,
        "score": score,
        "chunk_count": len(chunks),
        "avg_chars": round(mean(lengths)),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "short_chunks": short_chunks,
        "long_chunks": long_chunks,
        "heading_coverage": heading_coverage,
        "parent_coverage": parent_coverage,
        "table_chunks": table_chunks,
        "warnings": warnings,
    }


def _semantic_groups(blocks: list[ParsedBlock]) -> list[dict]:
    groups: list[dict] = []
    current: dict | None = None

    for block in blocks:
        text = _normalize(block.text)
        if not text:
            continue
        if block.block_type == "heading":
            continue
        heading_path = " > ".join(block.heading_path)
        group_key = (block.block_type, heading_path, block.source_locator if block.block_type == "table" else "")

        if current is None or current["key"] != group_key or _group_text_length(current) > TEXT_PARENT_TARGET * 1.6:
            current = {
                "key": group_key,
                "block_type": block.block_type,
                "heading_path": heading_path,
                "source_locators": [],
                "page_start": block.page_start,
                "page_end": block.page_end,
                "texts": [],
            }
            groups.append(current)

        current["texts"].append(text)
        current["source_locators"].append(block.source_locator)
        current["page_start"] = _min_page(current["page_start"], block.page_start)
        current["page_end"] = _max_page(current["page_end"], block.page_end)

    return groups


def _chunk_text_group(group: dict, document_id: str, chunk_offset: int, parent_index: int) -> tuple[list[TextChunk], int]:
    text = "\n\n".join(group["texts"])
    paragraphs = _paragraph_units(text)
    child_ranges = _child_ranges(paragraphs, target_size=TEXT_CHILD_TARGET)
    chunks: list[TextChunk] = []
    local_total = len(child_ranges)
    for local_index, (start, end) in enumerate(child_ranges):
        child_text = "\n\n".join(paragraphs[start:end]).strip()
        parent_text, parent_start, parent_end = _parent_window(paragraphs, start, end, target_size=TEXT_PARENT_TARGET)
        parent_id = f"parent_{document_id}_{parent_index}"
        parent_index += 1
        source_locator = _source_locator(group, local_index, local_total)
        metadata = _base_metadata(group, parent_id, parent_text, source_locator)
        metadata.update(
            {
                "paragraph_start": start + 1,
                "paragraph_end": end,
                "parent_paragraph_start": parent_start + 1,
                "parent_paragraph_end": parent_end,
                "parent_differs_from_child": parent_text.strip() != child_text.strip(),
            }
        )
        chunks.append(
            TextChunk(
                id=f"chunk_{document_id}_{uuid4().hex[:10]}",
                text=child_text,
                chunk_index=chunk_offset + len(chunks),
                source_locator=source_locator,
                metadata=metadata,
            )
        )
    return chunks, parent_index


def _chunk_table_group(group: dict, document_id: str, chunk_offset: int, parent_index: int) -> tuple[list[TextChunk], int]:
    lines = [line for text in group["texts"] for line in text.splitlines() if line.strip()]
    if not lines:
        return [], parent_index
    header = lines[0]
    data_lines = lines[1:] or lines
    parent_batches = _batched_rows(data_lines, TABLE_ROWS_PER_PARENT)
    chunks: list[TextChunk] = []

    for parent_batch in parent_batches:
        parent_id = f"parent_{document_id}_{parent_index}"
        parent_index += 1
        parent_text = "\n".join([header, *parent_batch]) if parent_batch and parent_batch[0] != header else "\n".join(parent_batch)
        child_batches = _batched_rows(parent_batch, TABLE_ROWS_PER_CHILD)
        for child_batch in child_batches:
            child_text = "\n".join([header, *child_batch]) if child_batch and child_batch[0] != header else "\n".join(child_batch)
            source_locator = _source_locator(group, len(chunks), len(child_batches))
            metadata = _base_metadata(group, parent_id, parent_text, source_locator)
            metadata["block_type"] = "table"
            chunks.append(
                TextChunk(
                    id=f"chunk_{document_id}_{uuid4().hex[:10]}",
                    text=child_text,
                    chunk_index=chunk_offset + len(chunks),
                    source_locator=source_locator,
                    metadata=metadata,
                )
            )
    return chunks, parent_index


def _paragraph_units(text: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= TEXT_CHILD_TARGET:
            units.append(paragraph)
            continue
        pieces = re.split(r"(?<=[。！？；;.!?])\s*", paragraph)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if len(piece) <= TEXT_CHILD_TARGET:
                units.append(piece)
            else:
                units.extend(piece[index : index + TEXT_CHILD_TARGET] for index in range(0, len(piece), TEXT_CHILD_TARGET))
    return units


def _child_ranges(paragraphs: list[str], target_size: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    current_len = 0
    for index, paragraph in enumerate(paragraphs):
        paragraph_len = len(paragraph)
        if current_len and current_len + paragraph_len + 2 > target_size:
            ranges.append((start, index))
            start = index
            current_len = paragraph_len
        else:
            current_len += paragraph_len + (2 if current_len else 0)
    if start < len(paragraphs):
        ranges.append((start, len(paragraphs)))
    return ranges


def _parent_window(paragraphs: list[str], child_start: int, child_end: int, target_size: int) -> tuple[str, int, int]:
    start = child_start
    end = child_end

    def window_len(left: int, right: int) -> int:
        return sum(len(paragraph) for paragraph in paragraphs[left:right]) + max(0, right - left - 1) * 2

    while start > 0 and window_len(start - 1, end) <= target_size:
        start -= 1
    while end < len(paragraphs) and window_len(start, end + 1) <= target_size:
        end += 1
    return "\n\n".join(paragraphs[start:end]).strip(), start, end


def _base_metadata(group: dict, parent_id: str, parent_text: str, source_locator: str) -> dict:
    return {
        "block_type": group["block_type"],
        "heading_path": group["heading_path"],
        "page_start": group["page_start"] or "",
        "page_end": group["page_end"] or "",
        "source_locator": source_locator,
        "parent_id": parent_id,
        "parent_text": parent_text[:2600],
    }


def _source_locator(group: dict, local_index: int, local_total: int) -> str:
    source = _compact_locators(group["source_locators"])
    heading = group["heading_path"]
    page = ""
    if group["page_start"] and group["page_end"]:
        page = f"第 {group['page_start']}-{group['page_end']} 页" if group["page_start"] != group["page_end"] else f"第 {group['page_start']} 页"
    parts = [part for part in (page or source, heading) if part]
    suffix = f"片段 {local_index + 1}/{local_total}" if local_total > 1 else "片段 1"
    return " / ".join([*parts, suffix])


def _compact_locators(locators: list[str]) -> str:
    unique = []
    for locator in locators:
        if locator not in unique:
            unique.append(locator)
    if len(unique) <= 2:
        return "、".join(unique)
    return f"{unique[0]} 至 {unique[-1]}"


def _batched_rows(rows: list[str], batch_size: int) -> list[list[str]]:
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def _group_text_length(group: dict) -> int:
    return sum(len(text) for text in group["texts"])


def _min_page(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _max_page(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _normalize(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)
