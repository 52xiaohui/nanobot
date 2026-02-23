"""Shared compatibility helpers for Responses-style providers."""

from __future__ import annotations

import json
from typing import Any


def convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI chat function schema to Responses function schema."""
    converted: list[dict[str, Any]] = []
    for tool in tools:
        fn = (tool.get("function") or {}) if tool.get("type") == "function" else tool
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") or {}
        converted.append(
            {
                "type": "function",
                "name": name,
                "description": fn.get("description") or "",
                "parameters": params if isinstance(params, dict) else {},
            }
        )
    return converted


def convert_messages(
    messages: list[dict[str, Any]],
    *,
    join_system_messages: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """Convert internal chat-completion-style history to Responses input items."""
    system_parts: list[str] = []
    system_prompt = ""
    input_items: list[dict[str, Any]] = []

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            if isinstance(content, str) and content:
                if join_system_messages:
                    system_parts.append(content)
                else:
                    system_prompt = content
            continue

        if role == "user":
            input_items.append(convert_user_message(content))
            continue

        if role == "assistant":
            if isinstance(content, str) and content:
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}],
                    }
                )
            for tool_call in msg.get("tool_calls", []) or []:
                fn = tool_call.get("function") or {}
                call_id, item_id = split_tool_call_id(tool_call.get("id"))
                call_id = call_id or f"call_{idx}"
                item_id = item_id or f"fc_{idx}"
                input_items.append(
                    {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": call_id,
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments") or "{}",
                    }
                )
            continue

        if role == "tool":
            call_id, _ = split_tool_call_id(msg.get("tool_call_id"))
            output = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }
            )

    instructions = "\n\n".join(system_parts) if join_system_messages else system_prompt
    return instructions, input_items


def convert_user_message(content: Any) -> dict[str, Any]:
    """Convert user content (text or multimodal list) to Responses content blocks."""
    if isinstance(content, str):
        return {"role": "user", "content": [{"type": "input_text", "text": content}]}

    if isinstance(content, list):
        converted: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                converted.append({"type": "input_text", "text": item.get("text", "")})
                continue
            if item.get("type") == "image_url":
                image = item.get("image_url") or {}
                url = image.get("url")
                if url:
                    converted.append({"type": "input_image", "image_url": url, "detail": "auto"})
        if converted:
            return {"role": "user", "content": converted}

    return {"role": "user", "content": [{"type": "input_text", "text": ""}]}


def split_tool_call_id(tool_call_id: Any) -> tuple[str, str | None]:
    """Split composite `<call_id>|<item_id>` tool id used by Responses providers."""
    if isinstance(tool_call_id, str) and tool_call_id:
        if "|" in tool_call_id:
            call_id, item_id = tool_call_id.split("|", 1)
            return call_id, item_id or None
        return tool_call_id, None
    return "call_0", None
