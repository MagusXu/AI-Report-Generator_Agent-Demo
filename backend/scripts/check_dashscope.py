from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402


def post_json(url: str, api_key: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def require_config() -> None:
    settings = get_settings()
    missing = []
    if not settings.dashscope_api_key:
        missing.append("DASHSCOPE_API_KEY")
    if not settings.workspace_id and "${WORKSPACE_ID}" in settings.llm_base_url:
        missing.append("WORKSPACE_ID")
    if missing:
        raise RuntimeError(f"Missing required .env values: {', '.join(missing)}")


def check_chat() -> None:
    settings = get_settings()
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": "你是一个连接性测试助手。"},
            {"role": "user", "content": "请只回复：模型连通正常"},
        ],
        "max_tokens": 32,
        "temperature": 0,
    }
    data = post_json(url, settings.dashscope_api_key, payload)
    text = data["choices"][0]["message"]["content"]
    print(f"LLM OK: {settings.llm_model} -> {text}")


def check_embedding() -> None:
    settings = get_settings()
    url = f"{settings.embedding_base_url.rstrip('/')}/embeddings"
    payload = {
        "model": settings.embedding_model,
        "input": ["数据中心行业风险研究报告连接性测试"],
    }
    data = post_json(url, settings.dashscope_api_key, payload)
    vector = data["data"][0]["embedding"]
    print(f"Embedding OK: {settings.embedding_model} -> dimension {len(vector)}")


def main() -> int:
    try:
        require_config()
        check_chat()
        check_embedding()
    except Exception as exc:
        print(f"Connectivity check failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
