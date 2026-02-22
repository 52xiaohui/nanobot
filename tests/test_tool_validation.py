import json
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool, register_web_tools
from nanobot.config.schema import WebToolsConfig


class SampleTool(Tool):
    @property
    def name(self) -> str:
        return "sample"

    @property
    def description(self) -> str:
        return "sample tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2},
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
                "mode": {"type": "string", "enum": ["fast", "full"]},
                "meta": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "flags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["tag"],
                },
            },
            "required": ["query", "count"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


def test_validate_params_missing_required() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi"})
    assert "missing required count" in "; ".join(errors)


def test_validate_params_type_and_range() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 0})
    assert any("count must be >= 1" in e for e in errors)

    errors = tool.validate_params({"query": "hi", "count": "2"})
    assert any("count should be integer" in e for e in errors)


def test_validate_params_enum_and_min_length() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "h", "count": 2, "mode": "slow"})
    assert any("query must be at least 2 chars" in e for e in errors)
    assert any("mode must be one of" in e for e in errors)


def test_validate_params_nested_object_and_array() -> None:
    tool = SampleTool()
    errors = tool.validate_params(
        {
            "query": "hi",
            "count": 2,
            "meta": {"flags": [1, "ok"]},
        }
    )
    assert any("missing required meta.tag" in e for e in errors)
    assert any("meta.flags[0] should be string" in e for e in errors)


def test_validate_params_ignores_unknown_fields() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 2, "extra": "x"})
    assert errors == []


async def test_registry_returns_validation_error() -> None:
    reg = ToolRegistry()
    reg.register(SampleTool())
    result = await reg.execute("sample", {"query": "hi"})
    assert "Invalid parameters" in result


async def test_web_search_mcp_maps_params_and_formats_results() -> None:
    calls: list[tuple[str, dict]] = []

    async def _mcp_execute(name: str, params: dict) -> str:
        calls.append((name, params))
        return json.dumps(
            {
                "results": [
                    {
                        "title": "Result A",
                        "url": "https://example.com/a",
                        "content": "Snippet A",
                    }
                ]
            }
        )

    tool = WebSearchTool(
        backend="mcp",
        mcp_server="tavily",
        mcp_tool="tavily_search",
        mcp_executor=_mcp_execute,
    )
    result = await tool.execute(query="nanobot", count=99)

    assert calls == [
        (
            "mcp_tavily_tavily_search",
            {"query": "nanobot", "max_results": 10},
        )
    ]
    assert "Results for: nanobot" in result
    assert "https://example.com/a" in result


async def test_web_fetch_mcp_maps_params_and_applies_truncation() -> None:
    calls: list[tuple[str, dict]] = []

    async def _mcp_execute(name: str, params: dict) -> str:
        calls.append((name, params))
        return json.dumps(
            {
                "results": [
                    {"url": "https://example.com/a", "content": "1234567890"},
                ]
            }
        )

    tool = WebFetchTool(
        backend="mcp",
        max_chars=6,
        mcp_server="tavily",
        mcp_tool="tavily_extract",
        mcp_executor=_mcp_execute,
    )
    raw = await tool.execute(url="https://example.com/a", extractMode="text")
    data = json.loads(raw)

    assert calls == [
        (
            "mcp_tavily_tavily_extract",
            {"urls": ["https://example.com/a"], "format": "text"},
        )
    ]
    assert data["extractor"] == "mcp:mcp_tavily_tavily_extract"
    assert data["truncated"] is True
    assert data["text"] == "## htt"


def test_register_web_tools_respects_disabled_backends() -> None:
    cfg = WebToolsConfig()
    cfg.search.backend = "disabled"
    cfg.fetch.backend = "disabled"

    reg = ToolRegistry()
    register_web_tools(reg, cfg)

    assert not reg.has("web_search")
    assert not reg.has("web_fetch")
