from __future__ import annotations

import httpx

from app.config import get_settings


class EmbeddingClientError(RuntimeError):
    pass


class EmbeddingClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.last_usage: dict[str, int | None] = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        self.total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def embed_texts(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        if not texts:
            return []
        if not self.settings.dashscope_api_key:
            raise EmbeddingClientError("DASHSCOPE_API_KEY is not configured")

        vectors: list[list[float]] = []
        self.total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_texts([text], batch_size=1)
        if not vectors:
            raise EmbeddingClientError("Embedding response was empty")
        return vectors[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.settings.embedding_base_url.rstrip('/')}/embeddings"
        payload = {
            "model": self.settings.embedding_model,
            "input": texts,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.dashscope_api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=120) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise EmbeddingClientError(
                f"Embedding request failed: HTTP {exc.response.status_code} {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingClientError(f"Embedding request failed: {exc}") from exc

        data = response.json()
        self.last_usage = _normalize_usage(data.get("usage"))
        _add_usage(self.total_usage, self.last_usage)
        try:
            rows = sorted(data["data"], key=lambda item: item.get("index", 0))
            return [row["embedding"] for row in rows]
        except (KeyError, TypeError) as exc:
            raise EmbeddingClientError("Embedding response shape is not compatible with OpenAI embeddings") from exc


def _normalize_usage(raw_usage: object) -> dict[str, int | None]:
    if not isinstance(raw_usage, dict):
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    return {
        "prompt_tokens": _int_or_none(raw_usage.get("prompt_tokens")),
        "completion_tokens": _int_or_none(raw_usage.get("completion_tokens")),
        "total_tokens": _int_or_none(raw_usage.get("total_tokens")),
    }


def _add_usage(total: dict[str, int], usage: dict[str, int | None]) -> None:
    for key in total:
        value = usage.get(key)
        if isinstance(value, int):
            total[key] += value


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
