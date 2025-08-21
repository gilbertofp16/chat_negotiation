# mcp_client.py
import asyncio
import os
import sys
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def mcp_search_async(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Spawn the MCP server via stdio and call its web_search tool.
    """
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_servers.web_search_server"],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Call the tool; returns a ToolResponse
            tool_result = await session.call_tool(
                "web_search",
                arguments={"query": query, "max_results": max_results},
            )

            items: List[Dict[str, Any]] = []
            # Parse structured results if present
            for part in getattr(tool_result, "content", []) or []:
                # json parts are common for structured tool returns
                if getattr(part, "type", "") == "json":
                    data = part.data
                    if isinstance(data, list):
                        items.extend(data)
                    elif isinstance(data, dict):
                        items.append(data)
                elif getattr(part, "type", "") == "text":
                    items.append({"title": "", "url": "", "snippet": part.text})
            return items


def mcp_search_sync(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Synchronous convenience wrapper.
    """
    return asyncio.run(mcp_search_async(query, max_results=max_results))


def web_context_from_results(items: List[Dict[str, Any]]) -> str:
    """
    Turn tool results into a compact string for prompts.
    """
    lines = []
    for i, it in enumerate(items, 1):
        title = it.get("title", "").strip()
        url = it.get("url", "").strip()
        snippet = (it.get("snippet", "") or "").replace("\n", " ")
        lines.append(f"- {i}. {title} | {url}\n  {snippet}")
    return "\n".join(lines)
