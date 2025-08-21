# mcp_servers/web_search_server.py
import os
from typing import Dict, List

from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient

mcp = FastMCP("WebSearchMCP")


def _tavily() -> TavilyClient:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("Set TAVILY_API_KEY in your environment")
    return TavilyClient(api_key=key)


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search the web and return a compact list of {title, url, snippet}.
    """
    client = _tavily()
    res = client.search(query=query, max_results=max_results, include_domains=None)
    out: List[Dict] = []
    for r in res.get("results", []):
        out.append({"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")[:500]})
    return out


if __name__ == "__main__":
    # stdio is the simplest transport for local clients
    mcp.run(transport="stdio")
