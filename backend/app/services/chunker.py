from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import mean, median
from uuid import uuid4

from app.config import get_settings
from app.services.document_parser import ParsedBlock, ParsedDocument


@dataclass(frozen=True)
class ChunkSettings:
    child_target_chars: int
    child_min_chars: int
    child_max_chars: int
    child_overlap_chars: int
    parent_target_chars: int
    group_max_chars: int
    table_rows_per_child: int
    table_rows_per_parent: int
    heading_prefix_enabled: bool
    fallback_pages_per_group: int
    heading_chunks_enabled: bool


_DOCUMENT_TYPE_PROFILES: dict[str, dict[str, int | bool]] = {
    "PDF": {
        "child_target_chars": 900,
        "child_min_chars": 350,
        "fallback_pages_per_group": 2,
    },
    "Excel": {
        "child_min_chars": 180,
        "table_rows_per_child": 24,
    },
    "Word": {
        "child_target_chars": 780,
        "child_min_chars": 280,
    },
    "Markdown": {
        "child_target_chars": 760,
        "child_min_chars": 250,
    },
    "文本": {
        "child_min_chars": 250,
    },
}


def get_chunk_settings(document_type: str | None = None) -> ChunkSettings:
    settings = get_settings()
    profile = _DOCUMENT_TYPE_PROFILES.get((document_type or "").strip(), {})

    def pick(name: str, default: int | bool) -> int | bool:
        return profile.get(name, default)

    return ChunkSettings(
        child_target_chars=int(pick("child_target_chars", settings.chunk_child_target_chars)),
        child_min_chars=int(pick("child_min_chars", settings.chunk_child_min_chars)),
        child_max_chars=int(pick("child_max_chars", settings.chunk_child_max_chars)),
        child_overlap_chars=int(pick("child_overlap_chars", settings.chunk_child_overlap_chars)),
        parent_target_chars=int(pick("parent_target_chars", settings.chunk_parent_target_chars)),
        group_max_chars=int(pick("group_max_chars", settings.chunk_group_max_chars)),
        table_rows_per_child=int(pick("table_rows_per_child", settings.table_rows_per_child)),
        table_rows_per_parent=int(pick("table_rows_per_parent", settings.table_rows_per_parent)),
        heading_prefix_enabled=bool(pick("heading_prefix_enabled", settings.chunk_heading_prefix_enabled)),
        fallback_pages_per_group=int(pick("fallback_pages_per_group", settings.chunk_fallback_pages_per_group)),
        heading_chunks_enabled=bool(pick("heading_chunks_enabled", settings.chunk_heading_chunks_enabled)),
    )


@dataclass(frozen=True)
class TextChunk:
    id: str
    text: str
    chunk_index: int
    source_locator: str
    metadata: dict = field(default_factory=dict)


def chunk_document(
    document: ParsedDocument,
    document_id: str,
    document_type: str | None = None,
) -> list[TextChunk]:
    settings = get_chunk_settings(document_type)
    chunks: list[TextChunk] = []
    parent_index = 0
    for group in _coalesce_short_text_groups(_semantic_groups(document.blocks, settings), settings):
        if group["block_type"] == "table":
            new_chunks, parent_index = _chunk_table_group(group, document_id, len(chunks), parent_index, settings)
        else:
            new_chunks, parent_index = _chunk_text_group(group, document_id, len(chunks), parent_index, settings)
        chunks.extend(new_chunks)
    chunks.extend(
        _build_heading_chunks(
            document.blocks,
            document_id,
            settings,
            len(chunks),
            _collect_section_bodies(document.blocks),
        )
    )
    return chunks


def _collect_section_bodies(blocks: list[ParsedBlock]) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    for block in blocks:
        if block.block_type not in {"text", "table"}:
            continue
        heading_path = " > ".join(block.heading_path).strip()
        if not heading_path:
            continue
        text = block.text.strip()
        if text:
            sections.setdefault(heading_path, []).append(text)
    return {path: "\n\n".join(parts) for path, parts in sections.items()}


def _section_parent_text(heading_path: str, section_bodies: dict[str, str], max_chars: int) -> str:
    body = section_bodies.get(heading_path, "").strip()
    if not body:
        return heading_path
    combined = f"{heading_path}\n\n{body}".strip()
    return combined[:max_chars]


def evaluate_chunk_quality_gate(quality: dict) -> str | None:
    settings = get_settings()
    if settings.chunk_quality_gate_mode != "block":
        return None
    if quality.get("score", 0) >= settings.chunk_quality_min_score:
        return None
    warnings = quality.get("warnings") or ["chunk 质量评分过低。"]
    return f"Chunk 质量评分 {quality.get('score', 0)}/100 低于阈值 {settings.chunk_quality_min_score}：{'；'.join(warnings)}"


def format_chunk_index_summary(chunk_count: int, quality: dict) -> str:
    summary = (
        f"已解析并入库，生成 {chunk_count} 个 chunks（含可选章节标题块）；"
        f"chunk 质量评分 {quality['score']}/100（{quality['status']}）。"
    )
    warnings = quality.get("warnings") or []
    if warnings:
        summary += " 提示：" + "；".join(warnings)
    return summary


def summarize_chunk_quality(chunks: list[TextChunk]) -> dict:
    settings = get_settings()
    content_chunks = [chunk for chunk in chunks if chunk.metadata.get("chunk_role") != "heading"]
    heading_chunks = len(chunks) - len(content_chunks)
    lengths = [len(chunk.text) for chunk in content_chunks]
    if not lengths:
        return {
            "status": "poor",
            "score": 0,
            "chunk_count": 0,
            "avg_chars": 0,
            "median_chars": 0,
            "min_chars": 0,
            "max_chars": 0,
            "short_chunks": 0,
            "long_chunks": 0,
            "heading_coverage": 0,
            "parent_coverage": 0,
            "parent_gain_ratio": 0,
            "table_chunks": 0,
            "heading_chunks": 0,
            "warnings": ["没有生成可检索片段。"],
        }

    short_threshold = min(180, max(120, settings.chunk_child_min_chars // 2))
    short_chunks = sum(1 for length in lengths if length < short_threshold)
    long_chunks = sum(1 for length in lengths if length > settings.chunk_child_max_chars)
    heading_path_chunks = sum(1 for chunk in content_chunks if chunk.metadata.get("heading_path"))
    table_chunks = sum(1 for chunk in content_chunks if chunk.metadata.get("block_type") == "table")
    parent_chunks = sum(1 for chunk in content_chunks if chunk.metadata.get("parent_text"))
    heading_coverage = round((heading_path_chunks + heading_chunks) / max(1, len(chunks)), 2)
    parent_coverage = round(parent_chunks / len(content_chunks), 2) if content_chunks else 0
    median_chars = round(median(lengths))
    parent_gains = []
    for chunk in content_chunks:
        child_len = _effective_child_chars(chunk)
        parent_len = len(str(chunk.metadata.get("parent_text") or ""))
        if child_len > 0:
            parent_gains.append((parent_len - child_len) / child_len)
    parent_gain_ratio = round(mean(parent_gains), 2) if parent_gains else 0.0

    warnings: list[str] = []
    if content_chunks and short_chunks / len(content_chunks) > 0.2:
        warnings.append("短片段比例偏高，可能存在过碎切分。")
    if median_chars < 350:
        warnings.append("片段中位长度偏低，检索粒度可能过碎。")
    if long_chunks:
        warnings.append("存在超长片段，可能降低检索精度。")
    if heading_coverage < 0.35:
        warnings.append("标题路径覆盖偏低，建议检查原文是否缺少清晰章节标题。")
    if parent_coverage < 0.9:
        warnings.append("部分片段缺少 parent 上下文。")
    if parent_gain_ratio < 0.35:
        warnings.append("parent 上下文扩展幅度偏小，生成阶段收益有限。")

    score = 100
    score -= min(30, short_chunks * 4)
    if median_chars < 350:
        score -= 15
    score -= min(25, long_chunks * 8)
    if heading_coverage < 0.35:
        score -= 12
    if parent_coverage < 0.9:
        score -= 8
    if parent_gain_ratio < 0.35:
        score -= 10
    score = max(0, score)
    status = "good" if score >= 82 else "warn" if score >= 60 else "poor"

    return {
        "status": status,
        "score": score,
        "chunk_count": len(chunks),
        "content_chunk_count": len(content_chunks),
        "avg_chars": round(mean(lengths)) if lengths else 0,
        "median_chars": median_chars,
        "min_chars": min(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
        "short_chunks": short_chunks,
        "long_chunks": long_chunks,
        "heading_coverage": heading_coverage,
        "parent_coverage": parent_coverage,
        "parent_gain_ratio": parent_gain_ratio,
        "table_chunks": table_chunks,
        "heading_chunks": heading_chunks,
        "warnings": warnings,
    }


def _semantic_groups(blocks: list[ParsedBlock], settings: ChunkSettings) -> list[dict]:
    groups: list[dict] = []
    current: dict | None = None

    for block in blocks:
        text = _normalize(block.text)
        if not text:
            continue
        if block.block_type == "heading":
            continue
        heading_path = " > ".join(block.heading_path)
        group_key = _group_key(block, heading_path, settings)

        if current is None or current["key"] != group_key or _group_text_length(current) > settings.group_max_chars:
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


def _group_key(block: ParsedBlock, heading_path: str, settings: ChunkSettings) -> tuple[str, str, str]:
    if block.block_type == "table":
        return (block.block_type, heading_path, block.source_locator)
    if heading_path:
        return (block.block_type, heading_path, "")
    page_bucket = _page_bucket(block.page_start, settings.fallback_pages_per_group)
    return (block.block_type, "", f"pages:{page_bucket}")


def _page_bucket(page: int | None, pages_per_group: int) -> int:
    if page is None or pages_per_group <= 0:
        return 0
    return (page - 1) // pages_per_group


def _coalesce_short_text_groups(groups: list[dict], settings: ChunkSettings) -> list[dict]:
    if not groups:
        return groups

    merged: list[dict] = []
    buffer: dict | None = None

    def flush() -> None:
        nonlocal buffer
        if buffer is not None:
            merged.append(buffer)
            buffer = None

    for group in groups:
        if group["block_type"] != "text":
            flush()
            merged.append(group)
            continue

        if buffer is None:
            buffer = _clone_group(group)
            continue

        combined_len = _group_text_length(buffer) + _group_text_length(group)
        same_root = _shared_root_heading(buffer["heading_path"], group["heading_path"])
        should_merge = same_root and (
            _group_text_length(buffer) < settings.child_min_chars
            or combined_len <= settings.child_target_chars
        )
        if should_merge:
            buffer["texts"].extend(group["texts"])
            buffer["source_locators"].extend(group["source_locators"])
            buffer["page_start"] = _min_page(buffer["page_start"], group["page_start"])
            buffer["page_end"] = _max_page(buffer["page_end"], group["page_end"])
            buffer["heading_path"] = group["heading_path"] or buffer["heading_path"]
            continue

        flush()
        buffer = _clone_group(group)

    flush()
    return merged


def _clone_group(group: dict) -> dict:
    return {
        "key": group["key"],
        "block_type": group["block_type"],
        "heading_path": group["heading_path"],
        "source_locators": list(group["source_locators"]),
        "page_start": group["page_start"],
        "page_end": group["page_end"],
        "texts": list(group["texts"]),
    }


def _shared_root_heading(left: str, right: str) -> bool:
    if not left and not right:
        return True
    if not left or not right:
        return False
    return left.split(" > ", 1)[0] == right.split(" > ", 1)[0]


def _chunk_text_group(
    group: dict,
    document_id: str,
    chunk_offset: int,
    parent_index: int,
    settings: ChunkSettings,
) -> tuple[list[TextChunk], int]:
    text = "\n\n".join(group["texts"])
    paragraphs = _paragraph_units(text, settings.child_target_chars, settings.child_max_chars)
    child_ranges = _child_ranges(paragraphs, target_size=settings.child_target_chars)
    child_ranges = _merge_short_ranges(paragraphs, child_ranges, settings.child_min_chars)
    chunks: list[TextChunk] = []
    local_total = len(child_ranges)
    for local_index, (start, end) in enumerate(child_ranges):
        core_text = "\n\n".join(paragraphs[start:end]).strip()
        child_text = _child_text_with_overlap(
            paragraphs,
            start,
            core_text,
            overlap_chars=settings.child_overlap_chars if local_index > 0 else 0,
            max_chars=settings.child_max_chars,
        )
        child_text = _apply_heading_prefix(child_text, group["heading_path"], settings)
        parent_text, parent_start, parent_end = _parent_window(
            paragraphs,
            start,
            end,
            target_size=settings.parent_target_chars,
        )
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
                "has_overlap_prefix": child_text.strip() != core_text.strip(),
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


def _chunk_table_group(
    group: dict,
    document_id: str,
    chunk_offset: int,
    parent_index: int,
    settings: ChunkSettings,
) -> tuple[list[TextChunk], int]:
    lines = [line for text in group["texts"] for line in text.splitlines() if line.strip()]
    if not lines:
        return [], parent_index
    header = lines[0]
    data_lines = lines[1:] or []
    if not data_lines:
        data_lines = [header]
        header = ""

    if len(data_lines) <= settings.table_rows_per_child:
        parent_text = "\n".join([line for line in [header, *data_lines] if line])
        child_text = _apply_heading_prefix(parent_text, group["heading_path"], settings)
        parent_id = f"parent_{document_id}_{parent_index}"
        parent_index += 1
        metadata = _base_metadata(group, parent_id, parent_text, _source_locator(group, 0, 1))
        metadata["block_type"] = "table"
        return [
            TextChunk(
                id=f"chunk_{document_id}_{uuid4().hex[:10]}",
                text=child_text,
                chunk_index=chunk_offset,
                source_locator=_source_locator(group, 0, 1),
                metadata=metadata,
            )
        ], parent_index

    parent_batches = _batched_rows(data_lines, settings.table_rows_per_parent)
    child_batches_per_parent = [_batched_rows(parent_batch, settings.table_rows_per_child) for parent_batch in parent_batches]
    total_children = sum(len(child_batches) for child_batches in child_batches_per_parent)
    chunks: list[TextChunk] = []
    global_child_index = 0

    for parent_batch, child_batches in zip(parent_batches, child_batches_per_parent, strict=True):
        parent_id = f"parent_{document_id}_{parent_index}"
        parent_index += 1
        parent_text = "\n".join([line for line in [header, *parent_batch] if line])
        for child_batch in child_batches:
            child_text = "\n".join([line for line in [header, *child_batch] if line])
            child_text = _apply_heading_prefix(child_text, group["heading_path"], settings)
            source_locator = _source_locator(group, global_child_index, total_children)
            global_child_index += 1
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


def _build_heading_chunks(
    blocks: list[ParsedBlock],
    document_id: str,
    settings: ChunkSettings,
    chunk_offset: int,
    section_bodies: dict[str, str],
) -> list[TextChunk]:
    if not settings.heading_chunks_enabled:
        return []

    seen_paths: set[str] = set()
    heading_chunks: list[TextChunk] = []
    for block in blocks:
        if block.block_type != "heading":
            continue
        heading_path = " > ".join(block.heading_path) or block.text.strip()
        if not heading_path or heading_path in seen_paths:
            continue
        seen_paths.add(heading_path)
        child_text = f"[章节标题] {heading_path}"
        parent_text = _section_parent_text(heading_path, section_bodies, settings.parent_target_chars)
        source_locator = " / ".join(
            part
            for part in (
                f"第 {block.page_start} 页" if block.page_start else block.source_locator,
                heading_path,
                "章节标题",
            )
            if part
        )
        metadata = {
            "block_type": "heading",
            "heading_path": heading_path,
            "page_start": block.page_start or "",
            "page_end": block.page_end or "",
            "source_locator": source_locator,
            "chunk_role": "heading",
            "parent_id": f"heading_{document_id}_{len(seen_paths)}",
            "parent_text": parent_text[:2600],
            "parent_differs_from_child": parent_text.strip() != child_text.strip(),
            "section_body_chars": len(section_bodies.get(heading_path, "")),
        }
        heading_chunks.append(
            TextChunk(
                id=f"chunk_{document_id}_{uuid4().hex[:10]}",
                text=child_text,
                chunk_index=chunk_offset + len(heading_chunks),
                source_locator=source_locator,
                metadata=metadata,
            )
        )
    return heading_chunks


def _apply_heading_prefix(text: str, heading_path: str, settings: ChunkSettings) -> str:
    if not settings.heading_prefix_enabled or not heading_path:
        return text
    prefix = f"[章节] {heading_path}"
    if text.startswith(prefix):
        return text
    return f"{prefix}\n\n{text}"


def _effective_child_chars(chunk: TextChunk) -> int:
    text = chunk.text
    for prefix in ("[章节] ", "[章节标题] "):
        if text.startswith(prefix):
            separator = text.find("\n\n")
            if separator == -1:
                return len(text)
            core = text[separator + 2 :].strip()
            return len(core) if core else len(text)
    return len(text)


def _paragraph_units(text: str, target_size: int, max_size: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= target_size:
            units.append(paragraph)
            continue
        pieces = re.split(r"(?<=[。！？；;.!?])\s*", paragraph)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            if len(piece) <= target_size:
                units.append(piece)
            else:
                units.extend(_hard_split_text(piece, target_size, max_size))
    return units


def _hard_split_text(text: str, target_size: int, max_size: int) -> list[str]:
    if len(text) <= max_size:
        return [text]
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + target_size, len(text))
        if end < len(text):
            for separator in ("，", ",", "、", "；", ";", " "):
                pivot = text.rfind(separator, start + target_size // 2, end)
                if pivot > start:
                    end = pivot + 1
                    break
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        start = end
    return pieces


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


def _merge_short_ranges(
    paragraphs: list[str],
    ranges: list[tuple[int, int]],
    min_chars: int,
) -> list[tuple[int, int]]:
    if not ranges or min_chars <= 0:
        return ranges

    merged: list[tuple[int, int]] = []
    index = 0
    while index < len(ranges):
        start, end = ranges[index]
        while index + 1 < len(ranges) and _range_len(paragraphs, start, end) < min_chars:
            index += 1
            end = ranges[index][1]
        merged.append((start, end))
        index += 1

    if len(merged) >= 2 and _range_len(paragraphs, *merged[-1]) < min_chars:
        last_start, last_end = merged.pop()
        prev_start, _ = merged.pop()
        merged.append((prev_start, last_end))

    return merged


def _range_len(paragraphs: list[str], start: int, end: int) -> int:
    return sum(len(paragraph) for paragraph in paragraphs[start:end]) + max(0, end - start - 1) * 2


def _child_text_with_overlap(paragraphs: list[str], start: int, core_text: str, overlap_chars: int, max_chars: int) -> str:
    if start == 0 or overlap_chars <= 0:
        return core_text
    prefix = _overlap_prefix(paragraphs, start, overlap_chars)
    if not prefix:
        return core_text
    combined = f"{prefix}\n\n{core_text}".strip()
    if len(combined) > max_chars:
        return core_text
    return combined


def _overlap_prefix(paragraphs: list[str], start: int, overlap_chars: int) -> str:
    parts: list[str] = []
    total = 0
    for index in range(start - 1, -1, -1):
        paragraph = paragraphs[index]
        projected = total + len(paragraph) + (2 if parts else 0)
        if parts and projected > overlap_chars:
            break
        parts.insert(0, paragraph)
        total = projected
    return "\n\n".join(parts).strip()


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
