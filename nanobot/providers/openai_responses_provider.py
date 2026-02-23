"""OpenAI Responses API provider (API key based)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import json_repair

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.responses_compat import (
    convert_messages as _convert_messages,
    convert_tools as _convert_tools,
)

DEFAULT_RESPONSES_URL = "https://api.openai.com/v1/responses"
_FINISH_REASON_MAP = {"completed": "stop", "incomplete": "length", "failed": "error", "cancelled": "error"}


class OpenAIResponsesProvider(LLMProvider):
    """Call OpenAI's Responses API directly with an API key."""

    def __init__(
        self,
        api_key: str,
        api_base: str | None = None,
        default_model: str = "openai-responses/gpt-5",
    ):
        super().__init__(api_key=api_key, api_base=api_base)
        self.default_model = default_model
        self.url = _responses_url(api_base)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        request_options: dict[str, Any] | None = None,
    ) -> LLMResponse:
        selected_model = _strip_responses_model_prefix(model or self.default_model)
        instructions, input_items = _convert_messages(messages)

        body: dict[str, Any] = {
            "model": selected_model,
            "store": False,
            "input": input_items,
            "max_output_tokens": max(1, max_tokens),
            "parallel_tool_calls": True,
            "temperature": temperature,
        }
        if instructions:
            body["instructions"] = instructions
        reasoning_effort = _reasoning_effort(request_options)
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
        if tools:
            body["tools"] = _convert_tools(tools)
            body["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            payload = await _post_json_with_fallbacks(self.url, headers, body)
            return _parse_response(payload)
        except Exception as e:
            return LLMResponse(
                content=f"Error calling OpenAI Responses: {str(e)}",
                finish_reason="error",
            )

    def get_default_model(self) -> str:
        return self.default_model


def _responses_url(api_base: str | None) -> str:
    base = (api_base or DEFAULT_RESPONSES_URL).rstrip("/")
    if base.endswith("/responses"):
        return base
    if base.endswith("/v1"):
        return f"{base}/responses"
    return f"{base}/v1/responses"


def _strip_responses_model_prefix(model: str) -> str:
    model_lower = model.lower()
    if model_lower.startswith("openai-responses/") or model_lower.startswith("openai_responses/"):
        return model.split("/", 1)[1]
    return model


async def _post_json_with_fallbacks(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
) -> dict[str, Any]:
    """Retry request after removing unsupported optional fields."""
    working = dict(body)
    for _ in range(3):
        try:
            return await _post_json(url, headers, working)
        except RuntimeError as e:
            message = str(e)
            if "temperature" in working and _is_temperature_error(message):
                working.pop("temperature", None)
                continue
            if "reasoning" in working and _is_reasoning_effort_error(message):
                working.pop("reasoning", None)
                continue
            raise
    raise RuntimeError("OpenAI Responses request failed after fallbacks.")


async def _post_json(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=body)
    if response.status_code != 200:
        raise RuntimeError(_friendly_error(response.status_code, response.text))
    try:
        payload = response.json()
    except Exception as e:
        raise RuntimeError(f"Invalid JSON response: {str(e)}") from e
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid Responses API payload")
    return payload


def _parse_response(payload: dict[str, Any]) -> LLMResponse:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[ToolCallRequest] = []

    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                blocks = item.get("content")
                if not isinstance(blocks, list):
                    continue
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") in {"output_text", "text"}:
                        text = block.get("text")
                        if isinstance(text, str) and text:
                            content_parts.append(text)
            elif item_type == "reasoning":
                summary = item.get("summary")
                if isinstance(summary, list):
                    for chunk in summary:
                        if not isinstance(chunk, dict):
                            continue
                        text = chunk.get("text")
                        if isinstance(text, str) and text:
                            reasoning_parts.append(text)
            elif item_type == "function_call":
                call_id = item.get("call_id") or "call_0"
                item_id = item.get("id") or "fc_0"
                args_raw = item.get("arguments") or "{}"
                args = _loads_json(args_raw)
                tool_calls.append(
                    ToolCallRequest(
                        id=f"{call_id}|{item_id}",
                        name=item.get("name"),
                        arguments=args if isinstance(args, dict) else {"raw": str(args)},
                    )
                )

    top_level_text = payload.get("output_text")
    content = "".join(content_parts) or (top_level_text if isinstance(top_level_text, str) else None)
    reasoning_content = "\n".join(reasoning_parts) or None

    usage_payload = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    prompt_tokens = _first_int(usage_payload.get("input_tokens"), usage_payload.get("prompt_tokens"))
    completion_tokens = _first_int(
        usage_payload.get("output_tokens"),
        usage_payload.get("completion_tokens"),
    )
    total_tokens = _first_int(usage_payload.get("total_tokens"))

    usage: dict[str, int] = {}
    if prompt_tokens is not None:
        usage["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        usage["completion_tokens"] = completion_tokens
    if total_tokens is not None:
        usage["total_tokens"] = total_tokens

    status = payload.get("status")
    finish_reason = _FINISH_REASON_MAP.get(status if isinstance(status, str) else "completed", "stop")

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
        reasoning_content=reasoning_content,
    )


def _first_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, int):
            return value
    return None


def _loads_json(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json_repair.loads(raw)
        except Exception:
            return {"raw": raw}
    return raw


def _reasoning_effort(request_options: dict[str, Any] | None) -> str | None:
    if not isinstance(request_options, dict):
        return None
    value = request_options.get("reasoning_effort")
    if isinstance(value, str):
        value = value.lower().strip()
        if value in {"low", "medium", "high"}:
            return value
    return None


def _is_temperature_error(message: str) -> bool:
    text = message.lower()
    if "temperature" not in text:
        return False
    return any(part in text for part in ("unsupported", "unknown", "invalid", "not allowed"))


def _is_reasoning_effort_error(message: str) -> bool:
    text = message.lower()
    if "reasoning" not in text:
        return False
    return any(part in text for part in ("unsupported", "unknown", "invalid", "not allowed"))


def _friendly_error(status_code: int, raw: str) -> str:
    message = ""
    try:
        data = json.loads(raw)
        error = data.get("error") if isinstance(data, dict) else None
        if isinstance(error, dict):
            msg = error.get("message")
            if isinstance(msg, str):
                message = msg
    except Exception:
        message = ""

    if status_code == 401:
        return "Invalid OpenAI API key."
    if status_code == 429:
        return "OpenAI rate limit exceeded. Please retry later."
    if message:
        return f"HTTP {status_code}: {message}"
    return f"HTTP {status_code}: {raw}"
