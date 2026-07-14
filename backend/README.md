# 后端

本地 AI 报告生成 Agent Demo 的 FastAPI 后端。

项目总览见仓库根目录 [README.md](../README.md)。

## 职责范围

- 健康检查与运行时配置
- 报告模板与章节状态（SQLite）
- 参考资料上传、解析、分块、向量化与 Chroma 检索
- 章节生成（同步 / SSE 流式），模型为通义 `qwen3.6-35b-a3b`
- 表格流水线（`table_service.py`）：AI 汇总与引用原表
- 引用规范化与章节版本落库
- 模拟导出记录

## 本地命令

Python 3.12，使用 `uv`：

```bash
uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

```bash
curl http://127.0.0.1:8000/health
uv run python scripts/check_dashscope.py
uv run python scripts/check_rag_pipeline.py
```

连通性与 RAG 检查会调用真实 DashScope 接口；RAG 检查还会向本地 Chroma 写入样例片段。

## 本地数据（已 gitignore）

- `config.env`
- `data/`
- `uploads/`
- `exports/`
- `.venv/`
