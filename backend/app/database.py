from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.config import get_settings


PROJECT_ID = "project_dc_2026"

SEED_PROJECT = {
    "id": PROJECT_ID,
    "name": "数据中心行业风险研究报告",
    "industry": "数据中心",
    "year": "2026",
    "language": "中文",
}

SEED_DOCUMENTS = [
    {
        "id": "doc_industry_report",
        "name": "2026 数据中心行业研究摘录",
        "type": "研报",
        "selected": 1,
        "summary": "覆盖全球和中国数据中心市场规模、算力需求、资本开支和运营商竞争格局。",
    },
    {
        "id": "doc_hk_policy",
        "name": "香港数据中心政策文件",
        "type": "政策",
        "selected": 1,
        "summary": "整理香港土地、电力、绿色建筑和跨境数据合规相关政策约束。",
    },
    {
        "id": "doc_sea_market",
        "name": "东南亚云基础设施市场数据",
        "type": "数据",
        "selected": 1,
        "summary": "包含新加坡、马来西亚、印尼、泰国等市场的云服务需求和供给指标。",
    },
    {
        "id": "doc_power_cost",
        "name": "亚太电力价格与供给风险表",
        "type": "数据",
        "selected": 0,
        "summary": "对比亚太主要区域电价、可再生能源比例、电网稳定性和新增容量约束。",
    },
]

SEED_SECTIONS = [
    {
        "id": "industry_intro",
        "number": "一",
        "title": "行业介绍",
        "level": 1,
        "parent_id": None,
        "order_index": 1,
        "prompt": "请以投行内部行业研究报告口吻，概述数据中心行业定义、核心需求、主要参与方和研究边界，强调后续风险分析所需的行业背景。",
    },
    {
        "id": "data_center_classification",
        "number": "1",
        "title": "数据中心分类",
        "level": 2,
        "parent_id": "industry_intro",
        "order_index": 2,
        "prompt": "请分类说明数据中心的主要类型，包括企业自建、IDC、云服务商自建、边缘数据中心等，并比较其客户、资产特征和风险暴露。",
    },
    {
        "id": "value_chain",
        "number": "2",
        "title": "行业产业链",
        "level": 2,
        "parent_id": "industry_intro",
        "order_index": 3,
        "prompt": "请梳理数据中心产业链，覆盖上游设备与能源、中游建设运营、下游云计算及企业客户，并说明关键议价因素。",
    },
    {
        "id": "operating_models",
        "number": "3",
        "title": "运营模式",
        "level": 2,
        "parent_id": "industry_intro",
        "order_index": 4,
        "prompt": "请比较批发型、零售型、托管型和自营型数据中心运营模式，分析收入确认、客户结构、资本开支和现金流特点。",
    },
    {
        "id": "market_overview",
        "number": "二",
        "title": "产业市场情况",
        "level": 1,
        "parent_id": None,
        "order_index": 5,
        "prompt": "请概述全球与亚太数据中心市场趋势，说明需求驱动因素、供给瓶颈和区域差异，为后续分市场分析铺垫。",
    },
    {
        "id": "global_market",
        "number": "1",
        "title": "全球整体市场情况介绍",
        "level": 2,
        "parent_id": "market_overview",
        "order_index": 6,
        "prompt": "请介绍全球数据中心市场规模、增长驱动、主要区域、客户需求变化和 AI 算力对新增容量的影响。",
    },
    {
        "id": "china_market",
        "number": "2",
        "title": "中国市场",
        "level": 2,
        "parent_id": "market_overview",
        "order_index": 7,
        "prompt": "请分析中国数据中心市场的供需结构、政策约束、东数西算影响、价格压力和主要运营商竞争情况。",
    },
    {
        "id": "hongkong_market",
        "number": "3",
        "title": "香港市场",
        "level": 2,
        "parent_id": "market_overview",
        "order_index": 8,
        "prompt": "请分析香港数据中心市场的定位、跨境连接优势、土地和电力瓶颈、合规要求及金融客户需求。",
    },
    {
        "id": "sea_market",
        "number": "4",
        "title": "东南亚市场",
        "level": 2,
        "parent_id": "market_overview",
        "order_index": 9,
        "prompt": "请比较东南亚主要数据中心市场，重点分析新加坡外溢、马来西亚承接、印尼和泰国需求增长及基础设施风险。",
    },
    {
        "id": "risk_analysis",
        "number": "三",
        "title": "行业风险分析",
        "level": 1,
        "parent_id": None,
        "order_index": 10,
        "prompt": "请从政策合规、市场供需、能源与碳排、资本开支、客户集中度、技术替代和区域竞争角度，提炼数据中心行业主要风险。",
    },
    {
        "id": "business_expansion",
        "number": "四",
        "title": "业务拓展策略",
        "level": 1,
        "parent_id": None,
        "order_index": 11,
        "prompt": "请结合银行投行业务视角，提出数据中心行业业务拓展策略，包括目标客户筛选、区域优先级、融资产品和风险准入建议。",
    },
    {
        "id": "appendix",
        "number": "五",
        "title": "附件",
        "level": 1,
        "parent_id": None,
        "order_index": 12,
        "prompt": "请列出本报告建议附录，包括术语表、市场数据表、资料清单、假设说明和后续尽调问题。",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _database_path() -> Path:
    settings = get_settings()
    if settings.database_url.startswith("sqlite:///"):
        raw_path = settings.database_url.replace("sqlite:///", "", 1)
        return Path(raw_path)
    return Path("./data/app.db")


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    db_path = _database_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def json_loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    return json.loads(value)


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row["name"] == column for row in rows)


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    if not _column_exists(conn, table, column):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_database() -> None:
    settings = get_settings()
    for path in (settings.upload_path, settings.export_path, settings.chroma_data_path):
        Path(path).mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                industry TEXT NOT NULL,
                year TEXT NOT NULL,
                language TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                selected INTEGER NOT NULL DEFAULT 1,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sections (
                id TEXT PRIMARY KEY,
                number TEXT NOT NULL,
                title TEXT NOT NULL,
                level INTEGER NOT NULL,
                parent_id TEXT,
                order_index INTEGER NOT NULL,
                prompt TEXT NOT NULL,
                current_version_id TEXT,
                confirmed INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS section_versions (
                id TEXT PRIMARY KEY,
                section_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                source TEXT NOT NULL,
                summary TEXT NOT NULL,
                content TEXT NOT NULL,
                reference_ids TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(section_id) REFERENCES sections(id)
            );

            CREATE TABLE IF NOT EXISTS export_records (
                id TEXT PRIMARY KEY,
                format TEXT NOT NULL,
                status TEXT NOT NULL,
                issues TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                source_locator TEXT NOT NULL,
                metadata TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            """
        )

        _add_column_if_missing(conn, "documents", "file_path", "TEXT")
        _add_column_if_missing(conn, "documents", "upload_status", "TEXT NOT NULL DEFAULT 'sample'")
        _add_column_if_missing(conn, "documents", "parse_status", "TEXT NOT NULL DEFAULT 'not_indexed'")
        _add_column_if_missing(conn, "documents", "chunk_count", "INTEGER NOT NULL DEFAULT 0")
        _add_column_if_missing(conn, "documents", "indexed_at", "TEXT")
        _add_column_if_missing(conn, "documents", "error_message", "TEXT")

        project_count = conn.execute("SELECT COUNT(*) AS count FROM projects").fetchone()["count"]
        if project_count:
            return

        now = utc_now()
        conn.execute(
            """
            INSERT INTO projects (id, name, industry, year, language, updated_at)
            VALUES (:id, :name, :industry, :year, :language, :updated_at)
            """,
            {**SEED_PROJECT, "updated_at": now},
        )
        conn.executemany(
            """
            INSERT INTO documents (
                id, name, type, selected, summary, created_at, upload_status, parse_status, chunk_count
            )
            VALUES (
                :id, :name, :type, :selected, :summary, :created_at, 'sample', 'not_indexed', 0
            )
            """,
            [{**doc, "created_at": now} for doc in SEED_DOCUMENTS],
        )
        conn.executemany(
            """
            INSERT INTO sections (
                id, number, title, level, parent_id, order_index, prompt, current_version_id, confirmed
            )
            VALUES (
                :id, :number, :title, :level, :parent_id, :order_index, :prompt, NULL, 0
            )
            """,
            SEED_SECTIONS,
        )
