from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.services.embedding_client import EmbeddingClient  # noqa: E402
from app.services.llm_client import LLMClient  # noqa: E402
from app.services.vector_store import VectorStore  # noqa: E402


def main() -> int:
    settings = get_settings()
    if not settings.dashscope_api_key:
        print("RAG check failed: DASHSCOPE_API_KEY is not configured")
        return 1

    sample_chunk = {
        "id": "check_chunk_data_center_power",
        "text": "数据中心项目的核心风险包括电力供给、PUE约束、客户上架率、资本开支和长期租约质量。香港市场还需要关注土地供应和跨境数据合规。",
        "metadata": {
            "document_id": "check_doc",
            "document_name": "RAG 自检样本文档",
            "document_type": "测试",
            "chunk_index": 0,
            "source_locator": "自检样本",
        },
    }

    embedding_client = EmbeddingClient()
    chunk_embedding = embedding_client.embed_texts([sample_chunk["text"]])[0]
    VectorStore().upsert_chunks([sample_chunk], [chunk_embedding])

    query_embedding = embedding_client.embed_query("数据中心行业风险有哪些")
    results = VectorStore().query(query_embedding, document_ids=["check_doc"], top_k=1)
    if not results:
        print("RAG check failed: ChromaDB returned no results")
        return 1

    result = LLMClient().generate(
        messages=[
            {"role": "system", "content": "你是投行行业风险研究助手。"},
            {
                "role": "user",
                "content": (
                    "请基于以下资料，用一句话总结数据中心项目风险，并保留引用 [ref:check_chunk_data_center_power]。\n"
                    f"资料：{results[0]['text']}"
                ),
            },
        ],
        max_tokens=120,
    )
    print(f"RAG OK: retrieved {results[0]['id']}")
    print(f"LLM answer: {result.content}")
    print(f"LLM usage: {result.usage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
