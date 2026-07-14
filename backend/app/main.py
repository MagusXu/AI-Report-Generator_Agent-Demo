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
from app.services.table_service import (
    SynthesizeTableConfig,
    TableCandidatesRequest,
    TableConfig,
    TableValidationError,
    VerbatimTableConfig,
    build_table_query_suffix,
    narrative_table_instructions,
    process_tables_for_section,
    scan_unverified_table_sections,
    search_table_candidates,
    validate_table_configs,
)
from app.services.vector_store import VectorStore


settings = get_settings()
init_database()

DEFAULT_SYSTEM_PROMPT = """你是一个面向银行内部使用的行业研究报告生成助手，目标是生成一份关于“消费电子行业风险分析”的中文书面报告，用于商业银行的授信审批、行业限额管理、内部评级和风险政策支持等场景。报告必须满足以下全局约束，并在所有章节中保持风格与逻辑的一致性。

### 一、整体定位与语气

- 报告定位：面向商业银行的风险管理、授信审批、行业研究等专业读者，服务于信用风险识别、额度与结构设计和内部政策制定。
- 语气要求：使用正式、专业、审慎的书面中文，避免口语化、情绪化或营销式表达；保持冷静、客观、中性，不使用夸张和煽动性措辞。
- 立场与视角：全文站在银行风险管理与授信决策视角，而不是企业自我宣传或纯投资研究视角；任何结论和判断，都需要回到“对信用风险和授信安全性的影响”这一主线。

### 二、结构与逻辑一致性

- 逻辑主线：所有章节内容都应围绕以下链条进行隐含或显式的叙述：“宏观与政策环境 → 行业景气与结构 → 企业商业模式与运营财务特征 → 信用风险与银行资产质量/资本占用”。
- 章节之间的衔接：即使不显式写出章节标题，段落内容也应在逻辑上相互支撑，避免前后结论冲突；后面的分析（财务、授信建议等）要以此前的行业和风险特征为基础，而不是孤立罗列。
- 论证方式：优先采用“现象描述 → 原因分析 → 风险影响 → 银行关注点/应对方向”的结构展开论述，每个核心段落尽量保持这一模式。

### 三、内容边界与信息使用规范

- 行业边界：报告默认涵盖消费电子整机、零部件与模组、代工制造、品牌渠道与电商等相关业务，但不深入到与消费电子关联度较低的其他电子或硬件行业。
- 数据与事实使用：以趋势、区间和定性描述为主，避免编造具体年份、精确数据或引用真实公司未公开信息；可以使用“近年来”“过去几个年度”“整体呈现增长/放缓趋势”等模糊时间与趋势表达。
- 案例与公司名称：不出现具体公司全称或任何可能指向单一企业的敏感信息；如需案例说明，一律使用高度抽象的“某大型品牌商”“某头部代工企业”“某跨境电商卖家”等泛化表达。

### 四、风险视角与评估框架

- 风险类型覆盖：全文需要系统覆盖宏观环境风险、行业结构性风险、商业模式与盈利模式风险、财务与信用风险、运营与供应链风险、技术与产品生命周期风险、法律合规与 ESG 风险等维度。
- 关注重点：在每一类风险的分析中，都要明确指出其对企业现金流、偿债能力、授信回收风险、不良率和资本占用的影响路径，而不仅仅停留在行业现象层面。
- 风险程度与方向：采用“风险较高/中等/偏低”“波动较大/相对稳定”“集中度较高/分散”等方向性判断，不要求也不允许给出精确量化评级结果，但要避免模糊到无法指导授信实践。

### 五、写作风格与表达规范

- 语言规范：使用规范金融与风险管理术语，如“信用风险、违约概率、资产质量、资本占用、授信政策、额度管理、担保与抵质押、营运资本、现金流压力”等；避免网络用语和口语化表达。
- 段落结构：段落内部保持信息密度适中，不使用长句堆砌导致理解困难；对于复杂内容可适度使用分句和分段，但保持整体连贯性与严谨性。
- 价值判断：所有判断应基于行业通行认知或合理推理，避免出现绝对化表达（如“必然”“肯定”“毫无风险”），改用“可能”“倾向于”“在一定条件下”等审慎措辞。

### 六、风格一致性与可读性

- 全文一致性：在不同章节生成内容时，保持语气、用词、风险视角和逻辑主线一致，不出现明显风格断裂或立场变化。
- 信息密度控制：保持适度高的信息密度和分析深度，但避免堆砌概念和重复表述；对同一风险点不在不同章节无意义重复，而是通过不同角度加深理解。
- 可读性：内容面向具备金融与风控背景的读者，不需要对基础概念进行过度科普，但应确保段落结构清晰，方便授信与风险管理人员快速获取结论与重点。

在生成具体章节内容时，始终遵守上述全局约束，并默认读者是银行内部的风险管理与授信专业人员，以消费电子行业的信用风险识别和授信决策支持为最核心目标。"""

PROMPT_ASSIST_SYSTEM = """你是商业银行内部使用的「章节生成 Prompt」编写助手。
你的任务：为《消费电子行业风险分析》报告的指定章节，编写一段可直接交给后续 RAG 写作模型使用的「章节 Prompt」；你不撰写报告正文。

报告定位：
- 面向银行风险管理、授信审批、行业研究等专业读者。
- 服务于信用风险识别、额度与结构设计、内部政策制定。
- 全文站在银行授信与风险管理视角，而非企业宣传或纯投资研究视角。
- 结论与分析最终应落到「对信用风险、授信安全性和资产质量的影响」。

你输出的章节 Prompt 必须同时满足：
1. 只输出一段可直接粘贴使用的中文章节 Prompt，不要解释、不要前后缀、不要 Markdown 代码块。
2. 不要输出报告正文，不要输出 [ref:chunk_id]，不要编造具体数值、年份或真实公司全称。
3. 优先沿用并强化「当前默认 Prompt」中的写作目标；结合「用户补充要求」做修订，而不是推倒重来。
4. 与整份报告大纲保持衔接：明确本章在「宏观/政策 → 行业景气与结构 → 商业模式与运营财务 → 信用风险与银行影响」链条中的位置，避免与前后章节重复堆砌同一风险点。
5. 输出结构尽量固定为以下四段（可用【】标题）：
   【写作目标】本章要回答什么问题、覆盖哪些分析维度、如何落到银行授信关注点。
   【写作风格】正式审慎、结构清晰；可说明条块/分类对比/因果链条等展开方式。
   【数据引用要求】以定性、趋势、区间表述为主；禁止要求编造精确数据；案例仅用泛化称谓（如「某大型品牌商」）。
   【篇幅建议】给出段落/条块数量与大致字数区间，与本章功能匹配。
6. 篇幅控制在约 350–650 字，表述克制、可执行；不要写成空泛模板口号。
7. 除非用户明确要求，不要在 Prompt 里规定插入表格或占位符；表格由系统在生成阶段另行配置。"""

PROMPT_ASSIST_MAX_TOKENS = 1400
PROMPT_ASSIST_TEMPERATURE = 0.3

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
    tables: list[TableConfig] = Field(default_factory=list)
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
    content: str = Field(min_length=1)


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
        raise HTTPException(status_code=404, detail="未找到报告项目")
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
        LIMIT 50
        """
    ).fetchall()
    logs = []
    for row in rows:
        item = dict(row)
        item["metadata"] = json_loads(item.get("metadata"), {})
        logs.append(item)
    return logs


def workspace_payload(conn) -> dict:
    sections = fetch_sections(conn)
    documents = fetch_documents(conn)
    return {
        "project": fetch_project(conn),
        "documents": documents,
        "sections": sections,
        "citations": fetch_citations(conn),
        "ai_call_logs": fetch_ai_call_logs(conn),
        "export_records": fetch_export_records(conn),
        "export_issues": export_check(sections, documents),
    }


def get_section_or_404(conn, section_id: str) -> dict:
    section = row_to_dict(conn.execute("SELECT * FROM sections WHERE id = ?", (section_id,)).fetchone())
    if section is None:
        raise HTTPException(status_code=404, detail="未找到章节")
    return section


def get_document_or_404(conn, document_id: str) -> dict:
    document = row_to_dict(conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone())
    if document is None:
        raise HTTPException(status_code=404, detail="未找到参考文档")
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
        issues.append(f"以下一级章节尚未生成：{'、'.join(missing)}")
    if unconfirmed:
        issues.append(f"以下章节尚未确认：{'、'.join(unconfirmed[:6])}")
    if not indexed_docs:
        issues.append("当前没有已入库的参考文档")
    unverified_sections = scan_unverified_table_sections(sections)
    if unverified_sections:
        issues.append(f"以下章节含待核实表格数据：{'、'.join(unverified_sections[:6])}")
    return issues


def build_rag_query(project: dict, section: dict, prompt: str, tables: list | None = None) -> str:
    base = f"{project['industry']} {project['year']} {section['title']} {prompt}"
    suffix = build_table_query_suffix(tables or [])
    return f"{base} {suffix}".strip()


def build_rag_messages(
    project: dict,
    section: dict,
    prompt: str,
    chunks: list[dict],
    system_prompt: str,
    use_parent_context: bool = True,
    tables: list | None = None,
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
{_table_prompt_block(tables)}

可引用资料：
{chr(10).join(context_lines)}

写作与引用要求：
1. 必须只基于上述资料写作，不要编造具体数据。
2. 每个主要判断后尽量给出引用标注，格式必须是 [ref:chunk_id]。
3. 如果资料不足，请明确写“现有资料不足以支持更细判断”，不要编造。
"""
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


def _table_prompt_block(tables: list | None) -> str:
    if not tables:
        return ""
    return f"\n\n{narrative_table_instructions(tables)}\n"


def ensure_citations(content: str, chunk_ids: list[str]) -> str:
    # 兼容模型偶发输出的全角括号引用，统一成前端可渲染的 [ref:...]
    content = re.sub(r"【ref:([^】]+)】", r"[ref:\1]", content)
    # 引用只保留正文内联标注，不在文末单独罗列资料来源
    content = re.sub(r"(?:\n|^)[ \t]*资料来源：[^\n]*\s*$", "", content)

    known_ids = set(chunk_ids)
    used_ids = set(re.findall(r"\[ref:([^\]]+)\]", content))
    invalid_ids = used_ids - known_ids
    for invalid_id in invalid_ids:
        content = content.replace(f"[ref:{invalid_id}]", "")

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
        "tables": [table.model_dump() for table in payload.tables],
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


def prepare_section_base(conn, section_id: str, payload: GenerateRequest) -> dict:
    section = get_section_or_404(conn, section_id)
    if payload.prompt:
        conn.execute("UPDATE sections SET prompt = ? WHERE id = ?", (payload.prompt, section_id))
        section["prompt"] = payload.prompt

    if payload.tables:
        try:
            validate_table_configs(payload.tables)
        except TableValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    project = fetch_project(conn)
    docs = get_documents_by_ids(conn, payload.reference_ids)
    indexed_docs = [doc for doc in docs if doc.get("parse_status") == "indexed" and doc.get("chunk_count", 0) > 0]
    if not indexed_docs:
        raise HTTPException(status_code=400, detail="请先上传并勾选至少一份已完成入库的真实参考文档。")

    query = build_rag_query(project, section, section["prompt"], payload.tables)
    return {
        "section": section,
        "project": project,
        "indexed_docs": indexed_docs,
        "query": query,
        "tables": payload.tables,
    }


def retrieve_section_chunks(base: dict, payload: GenerateRequest) -> dict:
    query_embedding_client = EmbeddingClient()
    try:
        query_embedding = query_embedding_client.embed_query(base["query"])
    except EmbeddingClientError as exc:
        raise HTTPException(status_code=502, detail=f"Embedding 调用失败：{exc}") from exc

    retrieved_chunks = VectorStore().query(
        query_embedding=query_embedding,
        document_ids=[doc["id"] for doc in base["indexed_docs"]],
        top_k=payload.retrieval_top_k,
    )
    retrieved_chunks = apply_per_document_limit(retrieved_chunks, payload.per_document_limit)
    if not retrieved_chunks:
        raise HTTPException(status_code=400, detail="未检索到可用于生成的资料片段。")

    return {
        "query_embedding_client": query_embedding_client,
        "retrieved_chunks": retrieved_chunks,
    }


def assemble_section_context(base: dict, retrieval: dict, payload: GenerateRequest) -> dict:
    system_prompt = (payload.system_prompt or DEFAULT_SYSTEM_PROMPT).strip() or DEFAULT_SYSTEM_PROMPT
    messages = build_rag_messages(
        base["project"],
        base["section"],
        base["section"]["prompt"],
        retrieval["retrieved_chunks"],
        system_prompt=system_prompt,
        use_parent_context=payload.use_parent_context,
        tables=payload.tables,
    )
    return {
        **base,
        **retrieval,
        "system_prompt": system_prompt,
        "messages": messages,
    }


def prepare_section_generation(conn, section_id: str, payload: GenerateRequest) -> dict:
    base = prepare_section_base(conn, section_id, payload)
    retrieval = retrieve_section_chunks(base, payload)
    return assemble_section_context(base, retrieval, payload)


def finalize_generated_content(
    conn,
    *,
    narrative_content: str,
    context: dict,
    payload: GenerateRequest,
) -> tuple[str, list[str], dict]:
    retrieved_chunks = context["retrieved_chunks"]
    reference_ids = [chunk["id"] for chunk in retrieved_chunks]
    table_result = process_tables_for_section(
        conn,
        project=context["project"],
        section=context["section"],
        narrative_content=narrative_content,
        tables=payload.tables,
        retrieved_chunks=retrieved_chunks,
        use_parent_context=payload.use_parent_context,
    )
    merged_reference_ids = list(reference_ids)
    for chunk_id in table_result.extra_reference_ids:
        if chunk_id not in merged_reference_ids:
            merged_reference_ids.append(chunk_id)
    content = ensure_citations(table_result.content, merged_reference_ids)
    table_metadata = {
        "table_warnings": table_result.table_warnings,
        "tables": table_result.table_metadata,
    }
    return content, merged_reference_ids, table_metadata


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
    table_metadata: dict | None = None,
) -> dict:
    table_count = len(payload.tables)
    summary = f"基于 {len(indexed_docs)} 份资料、{len(retrieved_chunks)} 个片段生成真实 RAG 版本。"
    if table_count:
        summary = f"{summary} 含 {table_count} 张表格。"
    create_version(
        conn,
        section_id=section_id,
        source="RAG + Qwen",
        summary=summary,
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
            "table_metadata": table_metadata or {},
        },
    )
    for table_item in (table_metadata or {}).get("tables", []):
        if table_item.get("mode") != "synthesize":
            continue
        record_ai_call(
            conn,
            operation=f"表格汇总 LLM · 表{table_item.get('index')}",
            model=settings.llm_model,
            usage=table_item.get("llm_usage") or {},
            metadata={
                "section_id": section_id,
                "section_title": section["title"],
                "table_index": table_item.get("index"),
                "table_title": table_item.get("title"),
                "warnings": table_item.get("warnings") or [],
                "llm_api_request": table_item.get("llm_api_request"),
                "llm_api_response": table_item.get("llm_api_response"),
                "prompt_structure": table_item.get("prompt_structure"),
                "output_content": table_item.get("output_content"),
            },
        )
    return workspace_payload(conn)


def stream_section_generation(section_id: str, payload: GenerateRequest) -> Iterator[str]:
    with get_connection() as conn:
        try:
            yield format_sse(
                "status",
                {"phase": "prepare", "message": "校验章节与参考资料…"},
            )
            base = prepare_section_base(conn, section_id, payload)

            yield format_sse(
                "status",
                {
                    "phase": "retrieve",
                    "message": f"正在检索 {len(base['indexed_docs'])} 份已选资料…",
                },
            )
            retrieval = retrieve_section_chunks(base, payload)
            context = assemble_section_context(base, retrieval, payload)

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
                    "message": f"已命中 {len(retrieved_chunks)} 个资料片段，开始生成正文…",
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

            if payload.tables:
                yield format_sse(
                    "status",
                    {"phase": "tables", "message": f"正文已完成，正在处理 {len(payload.tables)} 张表格…"},
                )

            try:
                content, reference_ids, table_metadata = finalize_generated_content(
                    conn,
                    narrative_content=raw_content,
                    context=context,
                    payload=payload,
                )
            except (TableValidationError, RuntimeError) as exc:
                yield format_sse("error", {"detail": str(exc)})
                return

            if payload.tables:
                yield format_sse(
                    "status",
                    {
                        "phase": "tables_done",
                        "message": "表格已插入",
                        "content": content,
                        "table_warnings": table_metadata.get("table_warnings", []),
                    },
                )

            yield format_sse(
                "status",
                {"phase": "persist", "message": "正在保存章节版本…"},
            )

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
                table_metadata=table_metadata,
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
            raise HTTPException(status_code=404, detail="未找到参考文档")
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

        try:
            content, reference_ids, table_metadata = finalize_generated_content(
                conn,
                narrative_content=llm_result.content,
                context=context,
                payload=payload,
            )
        except TableValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

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
            table_metadata=table_metadata,
        )


@app.post("/api/sections/{section_id}/table-candidates")
def table_candidates(section_id: str, payload: TableCandidatesRequest) -> dict:
    with get_connection() as conn:
        get_section_or_404(conn, section_id)
        docs = get_documents_by_ids(conn, payload.reference_ids)
        indexed_docs = [doc for doc in docs if doc.get("parse_status") == "indexed" and doc.get("chunk_count", 0) > 0]
        if not indexed_docs:
            raise HTTPException(status_code=400, detail="请先上传并勾选至少一份已完成入库的真实参考文档。")

    try:
        candidates = search_table_candidates(
            description=payload.description,
            document_ids=[doc["id"] for doc in indexed_docs],
            top_k=payload.top_k,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"candidates": candidates}


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
            raise HTTPException(status_code=404, detail="未找到版本")
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
            raise HTTPException(status_code=400, detail="当前章节尚无版本")
        conn.execute("UPDATE sections SET confirmed = 1 WHERE id = ?", (section_id,))
        return workspace_payload(conn)


def format_report_outline(sections: list[dict]) -> str:
    lines: list[str] = []
    for section in sections:
        if section["level"] == 1:
            lines.append(f"- {section['number']}、{section['title']}")
        else:
            lines.append(f"  - {section['number']}. {section['title']}")
    return "\n".join(lines)


def section_parent_title(sections: list[dict], section: dict) -> str:
    parent_id = section.get("parent_id")
    if not parent_id:
        return "无"
    parent = next((item for item in sections if item["id"] == parent_id), None)
    if parent is None:
        return "无"
    return f"{parent['number']}、{parent['title']}"


def format_selected_reference_documents(documents: list[dict]) -> str:
    selected = [document for document in documents if document.get("selected")]
    if not selected:
        return "当前未勾选参考文档"
    return "\n".join(f"- {document['name']}（{document['type']}）" for document in selected)


def build_prompt_assist_messages(
    project: dict,
    section: dict,
    sections: list[dict],
    documents: list[dict],
    user_content: str,
) -> list[dict[str, str]]:
    section_label = (
        f"{section['number']}、{section['title']}"
        if section["level"] == 1
        else f"{section['number']}. {section['title']}"
    )
    user = f"""请为以下报告章节编写一段「章节生成 Prompt」（给后续 RAG 写作模型使用，不是写正文）。

【报告】{project['name']} | 行业 {project['industry']} | {project['year']} | {project['language']}
【章节】{section_label}（level {section['level']}，所属：{section_parent_title(sections, section)}）
【报告大纲】
{format_report_outline(sections)}
【已勾选参考文档】
{format_selected_reference_documents(documents)}
【当前默认 Prompt】
{section['prompt']}
【用户补充要求】
{user_content}

请直接输出优化后的章节 Prompt，不要解释。"""
    return [{"role": "system", "content": PROMPT_ASSIST_SYSTEM}, {"role": "user", "content": user}]


@app.post("/api/sections/{section_id}/enhance-prompt")
def enhance_prompt(section_id: str, payload: PromptAssistRequest) -> dict[str, object]:
    user_content = payload.content.strip()
    if not user_content:
        raise HTTPException(status_code=400, detail="请填写内容要求")

    with get_connection() as conn:
        section = get_section_or_404(conn, section_id)
        project = fetch_project(conn)
        sections = fetch_sections(conn)
        documents = fetch_documents(conn)

    messages = build_prompt_assist_messages(project, section, sections, documents, user_content)

    try:
        llm_result = LLMClient().generate(
            messages=messages,
            max_tokens=PROMPT_ASSIST_MAX_TOKENS,
            temperature=PROMPT_ASSIST_TEMPERATURE,
            stream=False,
        )
    except LLMClientError as exc:
        raise HTTPException(status_code=502, detail=f"LLM 调用失败：{exc}") from exc

    prompt = llm_result.content.strip()
    if not prompt:
        raise HTTPException(status_code=502, detail="LLM 未返回任何内容")

    with get_connection() as conn:
        record_ai_call(
            conn,
            operation="Prompt 辅助生成 LLM",
            model=settings.llm_model,
            usage=llm_result.usage,
            metadata={
                "section_id": section_id,
                "section_title": section["title"],
                "project": project,
                "user_content": user_content,
                "selected_documents": format_selected_reference_documents(documents),
                "llm_api_request": llm_result.request_payload,
                "llm_api_response": llm_result.response_payload,
                "prompt_structure": messages,
                "output_content": prompt,
            },
        )
        workspace = workspace_payload(conn)

    return {"prompt": prompt, "source": "llm", "workspace": workspace}


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
        status = "导出记录已创建" if not issues else "导出前需复核"
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
        project = fetch_project(conn)
        sections = fetch_sections(conn)
    return {"name": project["name"], "sections": sections}
