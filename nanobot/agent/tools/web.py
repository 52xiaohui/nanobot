"""Web tools: web_search and web_fetch with pluggable backends."""

from __future__ import annotations

import html
import json
import os
import re
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.config.schema import WebToolsConfig

ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[str]]

# Shared constants
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"
MAX_REDIRECTS = 5  # Limit redirects to prevent DoS attacks


def _strip_tags(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def _normalize(text: str) -> str:
    """Normalize whitespace."""
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _validate_url(url: str) -> tuple[bool, str]:
    """Validate URL: must be http(s) with valid domain."""
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
        if not p.netloc:
            return False, "Missing domain"
        return True, ""
    except Exception as e:
        return False, str(e)


def _safe_json_loads(text: str) -> Any | None:
    """Best-effort JSON parsing."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _mcp_wrapped_tool_name(server: str, tool: str) -> str:
    """Build wrapper name for MCP tools registered by connect_mcp_servers()."""
    server = (server or "").strip()
    tool = (tool or "").strip()
    if not server or not tool:
        return ""
    return f"mcp_{server}_{tool}"


def _is_tool_not_found_error(result: str, tool_name: str) -> bool:
    return (
        "Error: Tool '" in result
        and tool_name in result
        and "not found" in result.lower()
    )


def _extract_search_rows(payload: Any) -> list[dict[str, str]]:
    """Extract search rows from common API shapes."""
    rows: list[dict[str, str]] = []

    def _push(item: Any) -> None:
        if not isinstance(item, dict):
            return
        title = item.get("title") or item.get("name") or ""
        url = item.get("url") or item.get("link") or ""
        snippet = (
            item.get("description")
            or item.get("snippet")
            or item.get("content")
            or item.get("raw_content")
            or ""
        )
        if title or url or snippet:
            rows.append({"title": str(title), "url": str(url), "snippet": str(snippet)})

    if isinstance(payload, dict):
        web = payload.get("web")
        if isinstance(web, dict):
            for item in web.get("results", []):
                _push(item)
        for key in ("results", "data", "items"):
            val = payload.get(key)
            if isinstance(val, list):
                for item in val:
                    _push(item)
    elif isinstance(payload, list):
        for item in payload:
            _push(item)

    return rows


def _extract_fetch_text(payload: Any, fallback_url: str) -> str:
    """Extract readable text content from common fetch/extract API shapes."""
    if isinstance(payload, dict):
        # Tavily-like shape: {"results": [{"url": "...", "content": "..."}]}
        if isinstance(payload.get("results"), list):
            blocks: list[str] = []
            for item in payload["results"]:
                if not isinstance(item, dict):
                    continue
                item_url = str(item.get("url") or fallback_url)
                content = item.get("content") or item.get("raw_content") or item.get("text") or ""
                if content:
                    blocks.append(f"## {item_url}\n\n{content}")
            if blocks:
                return "\n\n".join(blocks)

        for key in ("content", "raw_content", "text"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                return val

    elif isinstance(payload, list):
        blocks: list[str] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            item_url = str(item.get("url") or fallback_url)
            content = item.get("content") or item.get("raw_content") or item.get("text") or ""
            if content:
                blocks.append(f"## {item_url}\n\n{content}")
        if blocks:
            return "\n\n".join(blocks)

    return json.dumps(payload, ensure_ascii=False, indent=2)


class WebSearchTool(Tool):
    """Search the web using a configured backend."""

    name = "web_search"
    description = "Search the web. Returns titles, URLs, and snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {
                "type": "integer",
                "description": "Results (1-10)",
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 5,
        backend: str = "brave",
        mcp_server: str = "tavily",
        mcp_tool: str = "tavily_search",
        mcp_executor: ToolExecutor | None = None,
    ):
        self.api_key = api_key
        self.max_results = max_results
        self.backend = (backend or "brave").strip().lower()
        self.mcp_server = mcp_server
        self.mcp_tool = mcp_tool
        self._mcp_execute = mcp_executor

    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        n = min(max(count or self.max_results, 1), 10)

        if self.backend == "disabled":
            return "Error: web_search is disabled by configuration"
        if self.backend == "mcp":
            return await self._execute_mcp(query=query, count=n)
        return await self._execute_brave(query=query, count=n)

    async def _execute_brave(self, query: str, count: int) -> str:
        api_key = self.api_key or os.environ.get("BRAVE_API_KEY", "")
        if not api_key:
            return (
                "Error: Brave Search API key not configured. "
                "Set it in ~/.nanobot/config.json under tools.web.search.apiKey "
                "(or export BRAVE_API_KEY), then restart the gateway."
            )

        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": count},
                    headers={
                        "Accept": "application/json",
                        "X-Subscription-Token": api_key,
                    },
                    timeout=10.0,
                )
                r.raise_for_status()

            results = r.json().get("web", {}).get("results", [])
            if not results:
                return f"No results for: {query}"

            lines = [f"Results for: {query}\n"]
            for i, item in enumerate(results[:count], 1):
                lines.append(f"{i}. {item.get('title', '')}\n   {item.get('url', '')}")
                if desc := item.get("description"):
                    lines.append(f"   {desc}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    async def _execute_mcp(self, query: str, count: int) -> str:
        if not self._mcp_execute:
            return "Error: tools.web.search.backend='mcp' requires an MCP executor"

        wrapped_name = _mcp_wrapped_tool_name(self.mcp_server, self.mcp_tool)
        if not wrapped_name:
            return "Error: tools.web.search.mcpServer and mcpTool must be configured for MCP backend"
        if wrapped_name == self.name:
            return "Error: web_search MCP backend points to itself"

        raw = await self._mcp_execute(
            wrapped_name,
            {"query": query, "max_results": count},
        )
        if _is_tool_not_found_error(raw, wrapped_name):
            return (
                f"Error: MCP tool '{wrapped_name}' not found. "
                "Check tools.web.search.mcpServer and tools.web.search.mcpTool."
            )

        payload = _safe_json_loads(raw)
        if payload is None:
            return raw

        rows = _extract_search_rows(payload)
        if not rows:
            return raw

        lines = [f"Results for: {query}\n"]
        for i, row in enumerate(rows[:count], 1):
            lines.append(f"{i}. {row['title']}\n   {row['url']}")
            if row["snippet"]:
                lines.append(f"   {row['snippet']}")
        return "\n".join(lines)


class WebFetchTool(Tool):
    """Fetch and extract content from a URL using a configured backend."""

    name = "web_fetch"
    description = "Fetch URL and extract readable content (HTML -> markdown/text)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "extractMode": {"type": "string", "enum": ["markdown", "text"], "default": "markdown"},
            "maxChars": {"type": "integer", "minimum": 100},
        },
        "required": ["url"],
    }

    def __init__(
        self,
        max_chars: int = 50000,
        backend: str = "builtin",
        mcp_server: str = "tavily",
        mcp_tool: str = "tavily_extract",
        mcp_executor: ToolExecutor | None = None,
    ):
        self.max_chars = max_chars
        self.backend = (backend or "builtin").strip().lower()
        self.mcp_server = mcp_server
        self.mcp_tool = mcp_tool
        self._mcp_execute = mcp_executor

    async def execute(
        self,
        url: str,
        extractMode: str = "markdown",
        maxChars: int | None = None,
        **kwargs: Any,
    ) -> str:
        max_chars = maxChars or self.max_chars

        # Validate URL before fetching
        is_valid, error_msg = _validate_url(url)
        if not is_valid:
            return json.dumps({"error": f"URL validation failed: {error_msg}", "url": url}, ensure_ascii=False)

        if self.backend == "disabled":
            return json.dumps({"error": "web_fetch is disabled by configuration", "url": url}, ensure_ascii=False)
        if self.backend == "mcp":
            return await self._execute_mcp(url=url, extract_mode=extractMode, max_chars=max_chars)
        return await self._execute_builtin(url=url, extract_mode=extractMode, max_chars=max_chars)

    async def _execute_builtin(self, url: str, extract_mode: str, max_chars: int) -> str:
        from readability import Document

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=MAX_REDIRECTS,
                timeout=30.0,
            ) as client:
                r = await client.get(url, headers={"User-Agent": USER_AGENT})
                r.raise_for_status()

            ctype = r.headers.get("content-type", "")

            # JSON
            if "application/json" in ctype:
                text, extractor = json.dumps(r.json(), indent=2, ensure_ascii=False), "json"
            # HTML
            elif "text/html" in ctype or r.text[:256].lower().startswith(("<!doctype", "<html")):
                doc = Document(r.text)
                content = (
                    self._to_markdown(doc.summary())
                    if extract_mode == "markdown"
                    else _strip_tags(doc.summary())
                )
                text = f"# {doc.title()}\n\n{content}" if doc.title() else content
                extractor = "readability"
            else:
                text, extractor = r.text, "raw"

            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]

            return json.dumps(
                {
                    "url": url,
                    "finalUrl": str(r.url),
                    "status": r.status_code,
                    "extractor": extractor,
                    "truncated": truncated,
                    "length": len(text),
                    "text": text,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps({"error": str(e), "url": url}, ensure_ascii=False)

    async def _execute_mcp(self, url: str, extract_mode: str, max_chars: int) -> str:
        if not self._mcp_execute:
            return json.dumps(
                {"error": "tools.web.fetch.backend='mcp' requires an MCP executor", "url": url},
                ensure_ascii=False,
            )

        wrapped_name = _mcp_wrapped_tool_name(self.mcp_server, self.mcp_tool)
        if not wrapped_name:
            return json.dumps(
                {
                    "error": "tools.web.fetch.mcpServer and mcpTool must be configured for MCP backend",
                    "url": url,
                },
                ensure_ascii=False,
            )
        if wrapped_name == self.name:
            return json.dumps(
                {"error": "web_fetch MCP backend points to itself", "url": url},
                ensure_ascii=False,
            )

        try:
            raw = await self._mcp_execute(
                wrapped_name,
                {"urls": [url], "format": extract_mode},
            )
            if _is_tool_not_found_error(raw, wrapped_name):
                return json.dumps(
                    {
                        "error": (
                            f"MCP tool '{wrapped_name}' not found. Check "
                            "tools.web.fetch.mcpServer and tools.web.fetch.mcpTool."
                        ),
                        "url": url,
                    },
                    ensure_ascii=False,
                )

            payload = _safe_json_loads(raw)
            text = _extract_fetch_text(payload, fallback_url=url) if payload is not None else raw
            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]

            return json.dumps(
                {
                    "url": url,
                    "finalUrl": url,
                    "status": 200,
                    "extractor": f"mcp:{wrapped_name}",
                    "truncated": truncated,
                    "length": len(text),
                    "text": text,
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps({"error": str(e), "url": url}, ensure_ascii=False)

    def _to_markdown(self, html_text: str) -> str:
        """Convert HTML to markdown."""
        # Convert links, headings, lists before stripping tags
        text = re.sub(
            r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>([\s\S]*?)</a>',
            lambda m: f"[{_strip_tags(m[2])}]({m[1]})",
            html_text,
            flags=re.I,
        )
        text = re.sub(
            r"<h([1-6])[^>]*>([\s\S]*?)</h\1>",
            lambda m: f'\n{"#" * int(m[1])} {_strip_tags(m[2])}\n',
            text,
            flags=re.I,
        )
        text = re.sub(r"<li[^>]*>([\s\S]*?)</li>", lambda m: f"\n- {_strip_tags(m[1])}", text, flags=re.I)
        text = re.sub(r"</(p|div|section|article)>", "\n\n", text, flags=re.I)
        text = re.sub(r"<(br|hr)\s*/?>", "\n", text, flags=re.I)
        return _normalize(_strip_tags(text))


def register_web_tools(
    registry: ToolRegistry,
    web_config: WebToolsConfig,
    mcp_executor: ToolExecutor | None = None,
) -> None:
    """Register web_search and web_fetch based on config backends."""
    search_backend = (web_config.search.backend or "brave").lower()
    if search_backend != "disabled":
        registry.register(
            WebSearchTool(
                api_key=web_config.search.api_key or None,
                max_results=web_config.search.max_results,
                backend=search_backend,
                mcp_server=web_config.search.mcp_server,
                mcp_tool=web_config.search.mcp_tool,
                mcp_executor=mcp_executor,
            )
        )

    fetch_backend = (web_config.fetch.backend or "builtin").lower()
    if fetch_backend != "disabled":
        registry.register(
            WebFetchTool(
                max_chars=web_config.fetch.max_chars,
                backend=fetch_backend,
                mcp_server=web_config.fetch.mcp_server,
                mcp_tool=web_config.fetch.mcp_tool,
                mcp_executor=mcp_executor,
            )
        )
