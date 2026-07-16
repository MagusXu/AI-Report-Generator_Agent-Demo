# AI 报告生成 Agent Demo

本地全栈 Demo：面向投行内部「行业风险研究」类报告的 **RAG 辅助写作**。

用户可管理报告项目与参考资料，按模板分章节检索生成、编辑与版本确认，插入表格，全文预览，并模拟导出交付物。

## 当前阶段

本地 **FastAPI + React + SQLite + ChromaDB + 通义（DashScope）**。**v1 不要求 Docker。**

### 已具备能力

- 报告基础信息：名称、行业、年份、语言
- 参考资料：上传、类型、勾选、解析入库状态、预览、重新入库、删除
- PDF 解析除正文外用 pdfplumber 抽取表格块（识别「图表 N」类表题、跨页续表继承表题，并去重与表格重复的正文）
- 固定报告模板（行业风险结构）与章节切换
- 章节 Prompt 编辑，以及 AI 辅助起草 Prompt
- 章节 RAG 生成，支持 **SSE 流式输出**，前端展示简要「执行过程」阶段提示
- 章节表格（最多 5 张）：
  - **AI 汇总**：用户填写标题与列定义 → 二次调用模型填充结构化表格
  - **引用原表**：单独「表格标题」（展示用）+「原表描述」（仅检索）→ 检索候选 → 确认 chunk → 插入原表内容
- 正文引用 `[ref:chunk_id]` 渲染为可悬停的来源标签（兼容全角 `【ref:…】`）
- 正文展示支持标题、加粗等格式；表格标题在表格下方以灰色小字展示（不与正文标题同等处理）
- 手工编辑 / 历史版本 / 确认当前版本
- 整份预览与导出前检查
- 模拟 Word/PDF 导出记录（v1 不真正生成文件二进制）

## 技术架构

| 层级 | 选型 |
|------|------|
| 前端 | React + TypeScript + Vite |
| 后端 | Python FastAPI |
| 元数据 | SQLite |
| 向量库 | ChromaDB 本地持久化 |
| 大模型 | 阿里云百炼（国内）`qwen3.6-35b-a3b` |
| 向量模型 | 阿里云百炼（国内）`text-embedding-v4` |

```text
上传资料 → 解析 → 分块 → 向量化 → 写入 Chroma
                         ↓
章节生成（流式）：
  校验准备 → 检索 → 流式输出正文
         →（可选）插入 / 汇总表格
         → 规范化引用 → 落库版本
```

## 目录结构

```text
.
├── frontend/                 # Vite React（主界面在 App.tsx）
├── backend/
│   ├── app/
│   │   ├── main.py           # HTTP API 与 SSE 编排
│   │   ├── config.py
│   │   ├── database.py
│   │   └── services/         # 解析、分块、向量、LLM、表格等
│   ├── scripts/              # 连通性 / RAG 冒烟检查
│   └── README.md
├── docs/                     # 仅放必须随仓库的文档
├── .env.example
└── README.md
```

## 流式生成逻辑

`POST /api/sections/{id}/generate` 在 `stream: true` 时返回 `text/event-stream`。

| 事件 | 含义 |
|------|------|
| `status` | 流水线阶段提示（`prepare` → `retrieve` → `generating` → 可选 `truncated` / `tables` / `tables_done` → `persist`；`truncated` 表示正文达到 `max_tokens` 上限，可能被截断） |
| `delta` | 正文增量 `{ "content": "..." }` |
| `done` | 最终 `{ "workspace": ... }` |
| `error` | `{ "detail": "..." }` |

前端根据 `status` 展示「执行过程」列表，同时将 `delta` 写入正文预览。

## 表格

- 正文生成时应输出占位符 `<<TABLE:1>>`、`<<TABLE:2>>`…，不要自行编造表格 Markdown；正文缺失占位符时表格会兜底追加到章节末尾。选择表格时 `max_tokens` 会自动抬高到下限 1800，降低正文截断概率。
- **AI 汇总**：标题 + 列定义（+ 备注）→ 按表格描述单独检索资料（优先表格类 chunk）→ 二次 LLM 生成 JSON 单元格；无法对齐资料的格子标记为 `[[?…?]]`。对模型偶发的畸形 JSON（同一 cell 合并多列、列标签错位）会按值的输出顺序整行重排修复。
- **引用原表**：`description` 只用于检索；`title` 才是展示用表题，渲染在表格下方灰色小字。候选检索：`POST /api/sections/{id}/table-candidates`。
- 落库表题使用内部标记 `[[表题：…]]`（仍兼容旧内容里表格前的 `## 标题`）。

## 引用

- 模型 / 流水线标记：`[ref:chunk_id]`
- 前端通过 `workspace.citations` 映射到文档名、类型、定位与片段
- 无效引用会被剥离；文末「资料来源：…」不再保留（引用只保留正文内联）

## 主要 API

| 方法 | 路径 |
|------|------|
| GET | `/health`、`/api/runtime`、`/api/workspace`、`/api/report-preview`、`/api/report-template` |
| PATCH | `/api/project` |
| POST | `/api/project/new` |
| POST | `/api/documents/upload-file`、`/api/documents/upload`（流程占位，手动登记不解析） |
| PATCH / DELETE / GET / POST | `/api/documents/{id}/selection`、`…/{id}`、`…/{id}/preview`、`…/{id}/reindex` |
| PATCH | `/api/sections/{id}/prompt` |
| POST | `/api/sections/{id}/generate`、`…/table-candidates`、`…/manual-edit`、`…/select-version`、`…/confirm`、`…/enhance-prompt`、`…/clear` |
| POST | `/api/exports` |

## 后端启动

Python 3.12，使用 `uv`：

```bash
cd backend
uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

健康检查：`http://127.0.0.1:8000/health`

连通性与 RAG 检查（会调用真实 DashScope 接口）：

```bash
uv run python scripts/check_dashscope.py
uv run python scripts/check_rag_pipeline.py
```

## 前端启动

Node 22：

```bash
cd frontend
npm install
npm run dev
```

本地地址：`http://127.0.0.1:5173`

## 模型与环境变量

将 `.env.example` 复制为本地忽略文件 `backend/config.env`（或等价配置），并填写：

```text
DASHSCOPE_API_KEY=
WORKSPACE_ID=
LLM_BASE_URL=https://${WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3.6-35b-a3b
ENABLE_THINKING=false
EMBEDDING_BASE_URL=https://${WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
DATABASE_URL=sqlite:///./data/app.db
CHROMA_PATH=./data/chroma
UPLOAD_DIR=./uploads
EXPORT_DIR=./exports
CORS_ORIGINS=http://127.0.0.1:5173
```

**切勿提交**真实 API Key、`config.env`、上传原文、Chroma/SQLite 数据、导出物、`node_modules` 或虚拟环境。

