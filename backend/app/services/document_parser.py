from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document
from openpyxl import load_workbook
from pypdf import PdfReader


@dataclass(frozen=True)
class ParsedBlock:
    text: str
    source_locator: str
    block_type: str = "text"
    page_start: int | None = None
    page_end: int | None = None
    heading_path: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ParsedDocument:
    blocks: list[ParsedBlock]

    @property
    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks if block.text.strip())


class DocumentParserError(RuntimeError):
    pass


def parse_document(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return _parse_text(path)
    if suffix == ".pdf":
        return _parse_pdf(path)
    if suffix == ".docx":
        return _parse_docx(path)
    if suffix == ".xlsx":
        return _parse_xlsx(path)
    raise DocumentParserError(f"Unsupported file type: {suffix or 'unknown'}")


def _parse_text(path: Path) -> ParsedDocument:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks: list[ParsedBlock] = []
    current_heading: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            blocks.append(
                ParsedBlock(
                    text="\n".join(buffer).strip(),
                    source_locator="全文",
                    block_type="text",
                    heading_path=tuple(current_heading),
                )
            )
            buffer.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        heading = _markdown_heading(line) or _plain_heading(line)
        if heading:
            flush()
            current_heading = _update_heading_path(current_heading, heading[0], heading[1])
            blocks.append(
                ParsedBlock(
                    text=heading[1],
                    source_locator="全文",
                    block_type="heading",
                    heading_path=tuple(current_heading),
                )
            )
        else:
            buffer.append(line)
    flush()

    if not blocks:
        raise DocumentParserError("Text file contains no readable text.")
    return ParsedDocument(blocks=blocks)


def _parse_pdf(path: Path) -> ParsedDocument:
    reader = PdfReader(str(path))
    page_lines: list[list[str]] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        page_lines.append(_clean_lines(text))

    repeated = _repeated_margin_lines(page_lines)
    blocks: list[ParsedBlock] = []
    current_heading: list[str] = []
    for index, lines in enumerate(page_lines, start=1):
        clean_lines = [line for line in lines if line not in repeated]
        if not clean_lines:
            continue
        page_blocks, current_heading = _blocks_from_lines(
            clean_lines,
            source_locator=f"第 {index} 页",
            page=index,
            heading_path=current_heading,
        )
        blocks.extend(page_blocks)

    if not blocks:
        raise DocumentParserError("PDF contains no extractable text. OCR is not supported in this demo stage.")
    return ParsedDocument(blocks=blocks)


def _parse_docx(path: Path) -> ParsedDocument:
    document = Document(str(path))
    blocks: list[ParsedBlock] = []
    current_heading: list[str] = []
    paragraph_index = 0
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        paragraph_index += 1
        style_name = (paragraph.style.name or "").lower() if paragraph.style else ""
        heading = _docx_heading(style_name, text) or _plain_heading(text)
        if heading:
            current_heading = _update_heading_path(current_heading, heading[0], heading[1])
            blocks.append(
                ParsedBlock(
                    text=heading[1],
                    source_locator=f"段落 {paragraph_index}",
                    block_type="heading",
                    heading_path=tuple(current_heading),
                )
            )
        else:
            blocks.append(
                ParsedBlock(
                    text=text,
                    source_locator=f"段落 {paragraph_index}",
                    block_type="text",
                    heading_path=tuple(current_heading),
                )
            )

    table_index = 0
    for table in document.tables:
        table_index += 1
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            blocks.append(
                ParsedBlock(
                    text="\n".join(rows),
                    source_locator=f"表格 {table_index}",
                    block_type="table",
                    heading_path=tuple(current_heading),
                )
            )

    if not blocks:
        raise DocumentParserError("DOCX contains no readable text.")
    return ParsedDocument(blocks=blocks)


def _parse_xlsx(path: Path) -> ParsedDocument:
    workbook = load_workbook(str(path), read_only=True, data_only=True)
    blocks: list[ParsedBlock] = []
    for sheet in workbook.worksheets:
        rows = []
        for row_index, row in enumerate(sheet.iter_rows(values_only=True), start=1):
            values = [str(value).strip() for value in row if value is not None and str(value).strip()]
            if values:
                rows.append(f"第 {row_index} 行：" + " | ".join(values))
        if rows:
            blocks.append(
                ParsedBlock(
                    text="\n".join(rows),
                    source_locator=f"工作表 {sheet.title}",
                    block_type="table",
                    heading_path=(sheet.title,),
                )
            )
    if not blocks:
        raise DocumentParserError("XLSX contains no readable cells.")
    return ParsedDocument(blocks=blocks)


def _blocks_from_lines(
    lines: list[str],
    source_locator: str,
    page: int,
    heading_path: list[str],
) -> tuple[list[ParsedBlock], list[str]]:
    blocks: list[ParsedBlock] = []
    buffer: list[str] = []
    current_heading = list(heading_path)

    def flush() -> None:
        if buffer:
            blocks.append(
                ParsedBlock(
                    text=_repair_wrapped_lines(buffer),
                    source_locator=source_locator,
                    block_type="text",
                    page_start=page,
                    page_end=page,
                    heading_path=tuple(current_heading),
                )
            )
            buffer.clear()

    for line in lines:
        heading = _plain_heading(line)
        if heading:
            flush()
            current_heading = _update_heading_path(current_heading, heading[0], heading[1])
            blocks.append(
                ParsedBlock(
                    text=heading[1],
                    source_locator=source_locator,
                    block_type="heading",
                    page_start=page,
                    page_end=page,
                    heading_path=tuple(current_heading),
                )
            )
        else:
            buffer.append(line)
    flush()
    return blocks, current_heading


def _clean_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return lines


def _clean_text(text: str) -> str:
    return "\n".join(_clean_lines(text))


def _repeated_margin_lines(page_lines: list[list[str]]) -> set[str]:
    candidates: list[str] = []
    for lines in page_lines:
        candidates.extend(lines[:3])
        candidates.extend(lines[-3:])
    counts = Counter(line for line in candidates if 3 <= len(line) <= 80)
    threshold = max(3, int(len(page_lines) * 0.35))
    repeated = {line for line, count in counts.items() if count >= threshold}
    return {line for line in repeated if not _plain_heading(line)}


def _repair_wrapped_lines(lines: list[str]) -> str:
    paragraphs: list[str] = []
    current = ""
    for line in lines:
        if _looks_like_table_row(line):
            if current:
                paragraphs.append(current)
                current = ""
            paragraphs.append(line)
            continue
        if not current:
            current = line
            continue
        if _should_join(current, line):
            current = f"{current}{line}"
        else:
            paragraphs.append(current)
            current = line
    if current:
        paragraphs.append(current)
    return "\n\n".join(paragraphs)


def _should_join(previous: str, current: str) -> bool:
    if previous.endswith(("。", "；", "：", ":", ".", "?", "？", "!", "！", "）", ")")):
        return False
    if re.match(r"^[-•·●▪]|^\d+[.)、]", current):
        return False
    if len(previous) < 18:
        return False
    return True


def _looks_like_table_row(line: str) -> bool:
    return "|" in line or bool(re.search(r"\s{2,}", line))


def _markdown_heading(line: str) -> tuple[int, str] | None:
    match = re.match(r"^(#{1,6})\s+(.+)$", line)
    if not match:
        return None
    return len(match.group(1)), match.group(2).strip()


def _docx_heading(style_name: str, text: str) -> tuple[int, str] | None:
    match = re.match(r"heading\s+(\d+)", style_name)
    if match:
        return min(6, int(match.group(1))), text.strip()
    return None


def _plain_heading(line: str) -> tuple[int, str] | None:
    text = line.strip()
    if len(text) > 80:
        return None
    if _looks_like_table_row(text):
        return None
    patterns = [
        (r"^[一二三四五六七八九十]+[、.．]\s*(.+)$", 1),
        (r"^第[一二三四五六七八九十\d]+[章节部分]\s*(.+)$", 1),
        (r"^\d+[、.．]\s+(.+)$", 2),
        (r"^[（(]\d+[）)]\s*(.+)$", 4),
    ]
    for pattern, level in patterns:
        if re.match(pattern, text):
            return level, text
    return None


def _update_heading_path(current: list[str], level: int, title: str) -> list[str]:
    level = max(1, min(level, 6))
    next_path = current[: level - 1]
    next_path.append(title.strip())
    return next_path
