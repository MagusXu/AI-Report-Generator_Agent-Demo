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
    "name": "消费电子行业风险研究报告",
    "industry": "消费电子",
    "year": "2026",
    "language": "中文",
}

SEED_DOCUMENTS = [
    {
        "id": "doc_industry_report",
        "name": "2026 消费电子行业研究摘录",
        "type": "研报",
        "selected": 1,
        "summary": "覆盖消费电子产业链、需求周期、竞争格局、财务特征及商业银行授信关注点。",
    },
]

SEED_SECTIONS = [
    {
        "id": "report_purpose_scope",
        "number": "一",
        "title": "报告目的与适用范围",
        "level": 1,
        "parent_id": None,
        "order_index": 1,
        "prompt": """【写作目标】从商业银行授信与信用风险管理视角，说明本报告撰写目的、在银行内部的具体使用场景（如授信审批、行业限额管理、内部评级模型支持等），以及报告覆盖的消费电子行业边界和客户类型。
【写作风格】采用正式、严谨的银行内部行业研究报告风格，使用标准金融与风险管理术语，结构清晰、逻辑自洽，避免口语化和主观情绪表达。
【数据引用要求】如需引用数据，仅在必要时给出宏观或行业层面的定性描述，例如“近年来行业竞争加剧、信用风险上升”，不必给出具体数值和年份，重点突出银行视角下的研究目的、使用场景与适用范围。
【篇幅建议】控制在 2–3 个自然段，总字数约 300–500 字，重点回答“为什么写这份报告、谁来用、适用于哪些客户与业务”。""",
    },
    {
        "id": "industry_overview_value_chain",
        "number": "二",
        "title": "行业概况与产业链结构",
        "level": 1,
        "parent_id": None,
        "order_index": 2,
        "prompt": """【写作目标】系统梳理消费电子行业的定义和范围，描述上游零部件与材料、中游整机制造与代工、下游品牌商与渠道、电商平台等主要环节的产业链结构及价值分布。
【写作风格】采用概括性、框架化的行业研究风格，先给出行业定义，再按产业链上下游分层描述，行文保持客观、简洁，适度使用分点或分段结构。
【数据引用要求】用定性方式描述行业规模与发展阶段，可以使用“全球市场规模持续增长/增速放缓”等表述，不必给出具体统计口径；通过文字说明上游、中游、下游各环节的主要参与者类型和价值创造方式。
【篇幅建议】控制在 3–4 个自然段，总字数约 500–700 字，可辅以少量条列式描述产业链环节，保证银行授信人员能够快速理解行业整体结构。""",
    },
    {
        "id": "macro_external_risks",
        "number": "三",
        "title": "宏观与外部环境风险",
        "level": 1,
        "parent_id": None,
        "order_index": 3,
        "prompt": """【写作目标】从宏观经济、居民消费、利率与汇率、政策与监管、全球贸易环境等维度，分析影响消费电子行业的外部环境风险，强调其对行业景气度、企业收入与盈利波动以及银行资产质量的潜在影响。
【写作风格】采用风险条块式写法，以宏观经济、消费环境、利率与汇率、政策与监管、全球贸易为小标题或段落划分，每一块先描述现象，再指出对应的风险点。
【数据引用要求】以定性分析为主，可引用“经济增速放缓、消费信心波动、汇率双向波动加剧、监管趋严”等常见表述；如需要示例数据，使用区间或趋势描述，而非具体点值，避免过度依赖单一数据源。
【篇幅建议】控制在 4–6 个小条块，总字数约 600–800 字，确保每类外部环境因素与消费电子行业及银行信用风险之间的关联关系有清晰说明。""",
    },
    {
        "id": "structural_industry_risks",
        "number": "四",
        "title": "行业结构性风险",
        "level": 1,
        "parent_id": None,
        "order_index": 4,
        "prompt": """【写作目标】围绕竞争格局、市场集中度、产业链上下游议价能力、供需匹配与库存周期、区域和渠道结构等方面，分析消费电子行业的结构性风险特征，突出对企业盈利稳定性和违约概率的长期影响。
【写作风格】采用分析型、逻辑分解式风格，从竞争格局、集中度、供需周期、库存风险、区域与渠道结构等维度逐层展开；每个维度先描述现状，再分析对盈利和信用风险的中长期影响。
【数据引用要求】尽量用“高度集中、竞争激烈、价格战普遍、库存周期明显”等概括性词汇说明行业结构特征；在不依赖具体数据的情况下指明风险方向和相对强度，必要时可引用“部分细分领域呈寡头格局”等定性判断。
【篇幅建议】控制在 4–5 个自然段或条块，总字数约 600–900 字，保证结构性风险的主线清晰，便于后续授信政策与限额设置。""",
    },
    {
        "id": "business_model_risks",
        "number": "五",
        "title": "商业模式与盈利模式风险",
        "level": 1,
        "parent_id": None,
        "order_index": 5,
        "prompt": """【写作目标】分类描述消费电子行业主要商业模式（品牌商、ODM/EMS代工、零部件供应商、电商/渠道商等）及其盈利模式，并分析各类模式在收入稳定性、成本结构、毛利率波动和现金流特征上的风险点，从商业银行授信视角指出关注重点。
【写作风格】采用分类对比式写作，将品牌商、ODM/EMS、零部件供应商、电商/渠道商等若干类型分开描述；对每类模式使用统一模板：“商业模式 → 盈利模式 → 风险特征”。
【数据引用要求】主要使用定性描述，例如“毛利率较高但营销费用占比大”、“订单集中度高、毛利率较低但周转速度快”；不必给出具体毛利率或费用率数值，重点在于帮助授信人员识别不同模式的风险画像。
【篇幅建议】按客户类型分 3–5 个条块，总字数约 700–900 字，每类模式用 1–2 个自然段说明，避免冗长细节，突出差异化风险。""",
    },
    {
        "id": "financial_credit_risks",
        "number": "六",
        "title": "财务与信用风险分析",
        "level": 1,
        "parent_id": None,
        "order_index": 6,
        "prompt": """【写作目标】从财务报表和关键指标角度分析消费电子企业的信用风险，包括收入和利润波动、毛利率和费用率变化、营运资本结构（应收、存货、应付）、现金流质量、资本结构与偿债能力等，形成适用于授信评审的财务风险分析段落。
【写作风格】采用偏“信用评级报告”风格，以财务报表结构和关键比率为主线，从盈利能力、营运资本、现金流、资本结构和偿债能力几个维度进行条理化分析。
【数据引用要求】可以引用典型财务指标的方向性描述，如“收入与利润波动较大”、“应收和存货占总资产比重偏高”、“经营性现金流波动明显”；避免虚构具体数值，只强调指标在消费电子行业的普遍特征及对违约风险的含义。
【篇幅建议】控制在 4–6 个条块，总字数约 700–1000 字，形成一段可直接嵌入授信审查材料的财务风险分析模板。""",
    },
    {
        "id": "operations_supply_chain_risks",
        "number": "七",
        "title": "运营与供应链风险",
        "level": 1,
        "parent_id": None,
        "order_index": 7,
        "prompt": """【写作目标】围绕产能布局、生产管理、供应商集中度、采购与库存管理、物流与交付能力等运营环节，分析消费电子企业在供应链中面临的运营风险，并说明这些风险如何传导为交货风险、订单流失和现金流压力，对银行信贷安全性的影响。
【写作风格】采用运营流程视角，从产能布局、生产管理、供应商集中度、采购与库存管理、物流与交付等环节依次展开，强调运营事件如何转化为现金流与信用风险。
【数据引用要求】使用定性或示例型描述，如“部分企业产能高度集中于单一地区”、“关键零部件供应商数量有限”、“库存结构中成品占比偏高”等，不需要给出具体数量或地名，重点说明风险机制。
【篇幅建议】控制在 3–5 个自然段，总字数约 600–800 字，每个段落聚焦一个运营环节及其风险传导路径。""",
    },
    {
        "id": "tech_product_lifecycle_risks",
        "number": "八",
        "title": "技术迭代与产品生命周期风险",
        "level": 1,
        "parent_id": None,
        "order_index": 8,
        "prompt": """【写作目标】从技术迭代速度、产品生命周期短、创新成功率、研发投入强度、关键技术路线选择等方面，分析消费电子行业的技术与产品风险；说明技术失误或创新不足如何导致产品滞销、库存减值和盈利能力恶化，并与信用风险联系起来。
【写作风格】采用“产品周期 + 技术演进”双主线写法，先描述消费电子产品生命周期短、更新频率高的行业特性，再分别讨论技术路线选择和创新成功率对企业经营和信用风险的影响。
【数据引用要求】用趋势和案例型描述，如“技术迭代周期缩短”、“若新品未被市场认可，可能导致库存积压和减值”；不需要举具体机型或年份，重点在于抽象出通用风险模式。
【篇幅建议】控制在 3–4 个自然段，总字数约 500–700 字，形成可复用的“技术与产品风险”分析段落。""",
    },
    {
        "id": "legal_compliance_esg_risks",
        "number": "九",
        "title": "法律、合规与ESG风险",
        "level": 1,
        "parent_id": None,
        "order_index": 9,
        "prompt": """【写作目标】梳理消费电子企业在法律与合规方面的主要风险，包括知识产权纠纷、产品质量与安全责任、数据与隐私合规、环保与劳工规范等，同时从ESG视角说明这些因素对企业声誉、经营持续性及银行信用敞口的影响。
【写作风格】采用风险清单式写法，按法律合规（知识产权、质量责任、数据隐私）、监管要求和 ESG（环境、劳工、供应链管理、公司治理）分条说明，每条先列出风险类型，再说明可能影响。
【数据引用要求】使用概括性表达，如“涉及专利侵权诉讼风险”、“可能面临产品召回和质量索赔”、“供应链环境与劳工合规要求提升”；无需引用具体法规条款号或案例细节。
【篇幅建议】控制在 4–6 个条目，总字数约 600–800 字，结构清晰，便于银行内部合规与风险团队快速对照。""",
    },
    {
        "id": "risk_transmission_scenario",
        "number": "十",
        "title": "风险传导机制与情景分析框架",
        "level": 1,
        "parent_id": None,
        "order_index": 10,
        "prompt": """【写作目标】构建消费电子行业风险从宏观环境、行业层面传导到企业经营与财务指标、再传导到银行不良资产和资本占用的链条描述，设计基准/悲观/乐观情景下销量、价格、汇率、融资环境等关键变量变动，对企业现金流和违约风险进行定性情景分析。
【写作风格】采用因果链条和情景分析结合的写法，先画出“宏观环境 → 行业景气 → 企业经营与财务指标 → 银行资产质量和资本占用”的文字化传导路径，再设计基准/悲观/乐观三个情景并进行定性描述。
【数据引用要求】情景部分使用“销量下降幅度较大/略有上升”、“价格竞争加剧”、“融资环境收紧”等方向性表达，不设置具体百分比或数值，重点在于风险识别和逻辑关系，而非量化模型。
【篇幅建议】控制在 3–5 个自然段，总字数约 700–900 字，其中传导机制部分约 2 段，情景框架部分约 1–3 段。""",
    },
    {
        "id": "credit_risk_metrics_policy",
        "number": "十一",
        "title": "授信风险评估指标体系与政策建议",
        "level": 1,
        "parent_id": None,
        "order_index": 11,
        "prompt": """【写作目标】从银行内部风险管理视角，设计针对消费电子行业的授信评估指标体系，包括行业层面景气和周期指标、企业层面财务与运营指标、质性评估维度；在此基础上提出授信政策建议，如额度管理、期限结构、担保/抵质押安排、约束性条款设置等。
【写作风格】采用“指标体系 + 政策建议”结构写法，先分行业层面、企业层面、质性维度列出授信评估重点，再提出额度管理、期限结构、担保安排、约束性条款等政策建议，行文保持规范、可操作。
【数据引用要求】指标体系部分用定性描述指标含义，如“应收账款占总资产比重”，不需要填具体数值；政策建议部分避免涉及具体内部等级或限额数字，只给出通用原则和方向。
【篇幅建议】控制在 4–6 个条块，总字数约 800–1000 字，使该章节可直接作为银行内部授信政策附件的框架描述。""",
    },
    {
        "id": "conclusion_bank_implications",
        "number": "十二",
        "title": "结论与对银行业务的启示",
        "level": 1,
        "parent_id": None,
        "order_index": 12,
        "prompt": """【写作目标】在总结消费电子行业整体风险画像的基础上，概括对商业银行对公授信、供应链金融、投行及资产组合管理的主要启示，强调该行业在风险、收益与资本占用之间的权衡，并给出对未来一段时间内行业风险趋势的简要判断。
【写作风格】采用高度概括、偏决策支持风格，总结消费电子行业的整体风险特征与趋势，并从对公授信、供应链金融、投行、资产组合管理等角度提炼出 3–5 条关键启示，避免重复前文细节。
【数据引用要求】使用定性判断和趋势性表述，如“整体风险水平处于中高位”、“需强化额度管理与结构化授信安排”；不需要任何数值型数据或模型输出，保持决策摘要的简洁性。
【篇幅建议】控制在 2–3 个自然段，总字数约 400–600 字，可直接用于对管理层或授信委员会的口径总结。""",
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


def _seed_section_signature() -> list[tuple[str, str, str, int, str | None, int, str]]:
    return [
        (
            section["id"],
            section["number"],
            section["title"],
            section["level"],
            section["parent_id"],
            section["order_index"],
            section["prompt"],
        )
        for section in SEED_SECTIONS
    ]


def _existing_section_signature(conn: sqlite3.Connection) -> list[tuple[str, str, str, int, str | None, int, str]]:
    rows = conn.execute(
        """
        SELECT id, number, title, level, parent_id, order_index, prompt
        FROM sections
        ORDER BY order_index, id
        """
    ).fetchall()
    return [
        (
            row["id"],
            row["number"],
            row["title"],
            row["level"],
            row["parent_id"],
            row["order_index"],
            row["prompt"],
        )
        for row in rows
    ]


def sync_seed_sections(conn: sqlite3.Connection) -> bool:
    """Replace section outline when seed structure/prompts diverge. Clears section versions."""
    if _existing_section_signature(conn) == _seed_section_signature():
        return False

    conn.execute("DELETE FROM section_versions")
    conn.execute("DELETE FROM sections")
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
    return True


def sync_seed_project(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        UPDATE projects
        SET name = ?, industry = ?, year = ?, language = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            SEED_PROJECT["name"],
            SEED_PROJECT["industry"],
            SEED_PROJECT["year"],
            SEED_PROJECT["language"],
            utc_now(),
            PROJECT_ID,
        ),
    )


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
        if not project_count:
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
            return

        # Existing DB: keep uploaded docs; replace section outline only when seed diverges.
        if sync_seed_sections(conn):
            sync_seed_project(conn)
            conn.execute("DELETE FROM export_records")
            table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'ai_call_logs'"
            ).fetchone()
            if table:
                conn.execute("DELETE FROM ai_call_logs")
