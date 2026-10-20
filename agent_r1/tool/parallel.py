"""Built-in web tools backed by Parallel's Streamable HTTP Search MCP server."""

import asyncio
import math
import os
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from .base import BaseTool
from .schema import ToolResponse

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "Agent-R1 (+https://github.com/AgentR1/Agent-R1)"


async def _call_mcp(name: str, arguments: dict[str, Any], api_key: str, timeout: float) -> str:
    # Keep MCP optional for existing recipes that only use local tools.
    import httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    headers = {"User-Agent": USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
        async with streamable_http_client(MCP_URL, http_client=client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                if name not in {tool.name for tool in tools.tools}:
                    return f"Error: Parallel Search MCP does not expose {name}."
                result = await session.call_tool(name, arguments)
                text = "\n".join(block.text for block in result.content if block.type == "text")
                if result.isError:
                    return f"Error: Parallel Search MCP: {text}"
                return text or "Parallel Search MCP returned no text results."


class _ParallelTool(BaseTool):
    mcp_name: str

    def mcp_arguments(self, args: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    async def execute(self, args: dict[str, Any], **kwargs) -> tuple[ToolResponse, None, dict]:
        options = kwargs.get("tools_kwargs", {}).get("parallel", {})
        if options.get("enabled", True) is False:
            return ToolResponse(text="Error: Parallel web access is disabled."), None, {}
        try:
            arguments = self.mcp_arguments(args)
            timeout = float(options.get("timeout", 30))
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("Parallel timeout must be a positive finite number.")
            api_key = options.get("api_key", os.environ.get("PARALLEL_API_KEY", ""))
            if not isinstance(api_key, str):
                raise ValueError("Parallel api_key must be a string.")
        except (ValueError, TypeError) as exc:
            return ToolResponse(text=f"Error: {exc}"), None, {}

        # The environment owns this ID. A direct standalone tool run is one new
        # research run; callers making related direct calls can supply an ID.
        arguments["session_id"] = kwargs.get("conversation_id") or uuid4().hex
        try:
            text = await asyncio.wait_for(_call_mcp(self.mcp_name, arguments, api_key, timeout), timeout=timeout)
        except ImportError:
            text = "Error: Install web search dependencies with pip install -r requirements-search.txt."
        except asyncio.TimeoutError:
            text = "Error: Parallel Search MCP request timed out."
        except Exception:
            # Never expose headers/credentials or silently select another provider.
            text = "Error: Parallel Search MCP request failed. Check service availability and configured credentials."
        return ToolResponse(text=text), None, {}


@BaseTool.register("parallel_search")
class ParallelSearchTool(_ParallelTool):
    name = "parallel_search"
    mcp_name = "web_search"
    description = "Search the web for current information. Returns source URLs and relevant excerpts."
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "A focused web search query."}},
        "required": ["query"],
    }

    def mcp_arguments(self, args: dict[str, Any]) -> dict[str, Any]:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonempty string.")
        return {"objective": query, "search_queries": [query]}


@BaseTool.register("parallel_fetch")
class ParallelFetchTool(_ParallelTool):
    name = "parallel_fetch"
    mcp_name = "web_fetch"
    description = "Read relevant content from a web page URL, optionally focused on an objective."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The HTTP or HTTPS page URL to read."},
            "objective": {"type": "string", "description": "What to find on the page (up to 200 characters)."},
        },
        "required": ["url"],
    }

    def mcp_arguments(self, args: dict[str, Any]) -> dict[str, Any]:
        url = args.get("url")
        if not isinstance(url, str) or any(char.isspace() for char in url):
            raise ValueError("url must be an HTTP or HTTPS URL.")
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("url must be an HTTP or HTTPS URL without credentials.")
        arguments: dict[str, Any] = {"urls": [url]}
        if "objective" in args:
            objective = args["objective"]
            if not isinstance(objective, str) or len(objective) > 200:
                raise ValueError("objective must be a string of at most 200 characters.")
            arguments["objective"] = objective
        return arguments
