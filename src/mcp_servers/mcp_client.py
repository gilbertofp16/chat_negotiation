# mcp_client.py
import asyncio
import os
import sys
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def mcp_browse_async(topic: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Spawns the MCP server via stdio and calls its browse_bsw tool.
    """
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_servers.web_search_server"],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # Call the browse_bsw tool
            tool_result = await session.call_tool(
                "browse_bsw",
                arguments={"topic": topic, "max_results": max_results},
            )

            # The tool response can have multiple parts (e.g., stdout, json).
            # We need to find the JSON part that contains our actual data.
            for part in getattr(tool_result, "content", []) or []:
                if getattr(part, "type", "") == "json":
                    data = part.data
                    if isinstance(data, dict):
                        # This is the data we want.
                        return data

            # If we get here, the tool call succeeded but no JSON was returned.
            # This can happen if the server-side tool has an error before it
            # can return a JSON payload.
            return {"summary": "Error: The tool executed but did not return a valid JSON response.", "sources": []}


def mcp_browse_sync(topic: str, max_results: int = 3) -> Dict[str, Any]:
    """
    Synchronous convenience wrapper for mcp_browse_async.
    """
    return asyncio.run(mcp_browse_async(topic, max_results=max_results))


def bsw_context_from_results(result: Dict[str, Any]) -> str:
    """
    Turns the browse_bsw tool's dictionary output into a compact string for prompts.
    """
    summary = result.get("summary", "No summary provided.").strip()
    sources = result.get("sources", [])

    if not summary and not sources:
        return "[No web context found]"

    lines = [f"Web Search Summary:\n{summary}"]

    if sources:
        lines.append("\nSources:")
        for i, source in enumerate(sources, 1):
            title = source.get("title", "No Title").strip()
            url = source.get("url", "").strip()
            lines.append(f"- {i}. {title} ({url})")

    return "\n".join(lines)
