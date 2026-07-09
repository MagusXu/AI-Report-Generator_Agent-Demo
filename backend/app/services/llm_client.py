from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

from app.config import get_settings


class LLMClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResult:
    content: str
    usage: dict[str, int | None]
    request_payload: dict
    response_payload: dict


class LLMClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.last_usage: dict[str, int | None] = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
        self.last_stream_events: list[dict] = []

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1400,
        temperature: float = 0.2,
        stream: bool = False,
    ) -> LLMResult:
        if not self.settings.dashscope_api_key:
            raise LLMClientError("DASHSCOPE_API_KEY is not configured")

        url = f"{self.settings.llm_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        headers = {
            "Authorization": f"Bearer {self.settings.dashscope_api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=180) as client:
                if stream:
                    return self._generate_stream(client, url, payload, headers)
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMClientError(f"LLM request failed: HTTP {exc.response.status_code} {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise LLMClientError(f"LLM request failed: {exc}") from exc

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMClientError("LLM response shape is not compatible with OpenAI chat completions") from exc

        return LLMResult(
            content=content,
            usage=_normalize_usage(data.get("usage")),
            request_payload=_redacted_request_payload(payload),
            response_payload=data,
        )

    def iter_generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1400,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        if not self.settings.dashscope_api_key:
            raise LLMClientError("DASHSCOPE_API_KEY is not configured")

        url = f"{self.settings.llm_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        headers = {
            "Authorization": f"Bearer {self.settings.dashscope_api_key}",
            "Content-Type": "application/json",
        }

        usage: dict[str, int | None] = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        stream_events: list[dict] = []

        try:
            with httpx.Client(timeout=180) as client:
                with client.stream("POST", url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        delta, usage, stream_events = self._consume_stream_line(line, usage, stream_events)
                        if delta:
                            yield delta
        except httpx.HTTPStatusError as exc:
            raise LLMClientError(f"LLM request failed: HTTP {exc.response.status_code} {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise LLMClientError(f"LLM request failed: {exc}") from exc

        self.last_usage = usage
        self.last_stream_events = stream_events

    def _consume_stream_line(
        self,
        line: str,
        usage: dict[str, int | None],
        stream_events: list[dict],
    ) -> tuple[str | None, dict[str, int | None], list[dict]]:
        if not line or not line.startswith("data:"):
            return None, usage, stream_events
        raw = line.removeprefix("data:").strip()
        if raw == "[DONE]":
            return None, usage, stream_events
        try:
            data = json.loads(raw)
        except ValueError:
            return None, usage, stream_events
        if len(stream_events) < 40:
            stream_events.append(data)
        usage = _merge_usage(usage, _normalize_usage(data.get("usage")))
        try:
            delta = data["choices"][0]["delta"].get("content")
        except (KeyError, IndexError, TypeError, AttributeError):
            delta = None
        return delta, usage, stream_events

    def _generate_stream(
        self,
        client: httpx.Client,
        url: str,
        payload: dict,
        headers: dict[str, str],
    ) -> LLMResult:
        content_parts: list[str] = []
        usage: dict[str, int | None] = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
        stream_events: list[dict] = []
        with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                delta, usage, stream_events = self._consume_stream_line(line, usage, stream_events)
                if delta:
                    content_parts.append(delta)

        content = "".join(content_parts).strip()
        if not content:
            raise LLMClientError("LLM stream response did not contain content")
        self.last_usage = usage
        self.last_stream_events = stream_events
        return LLMResult(
            content=content,
            usage=usage,
            request_payload=_redacted_request_payload(payload),
            response_payload={
                "stream": True,
                "events_sample": stream_events,
                "assembled_content": content,
                "usage": usage,
            },
        )


def _normalize_usage(raw_usage: object) -> dict[str, int | None]:
    if not isinstance(raw_usage, dict):
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    return {
        "prompt_tokens": _int_or_none(raw_usage.get("prompt_tokens")),
        "completion_tokens": _int_or_none(raw_usage.get("completion_tokens")),
        "total_tokens": _int_or_none(raw_usage.get("total_tokens")),
    }


def _merge_usage(left: dict[str, int | None], right: dict[str, int | None]) -> dict[str, int | None]:
    return {key: right.get(key) if right.get(key) is not None else left.get(key) for key in left}


def _redacted_request_payload(payload: dict) -> dict:
    return json.loads(json.dumps(payload, ensure_ascii=False))


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
