from __future__ import annotations

import asyncio
import logging
import os
import sys
from functools import lru_cache

from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
from dotenv import load_dotenv

logger = logging.getLogger("mcp-bsw-py")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.handlers[:] = [handler]
logger.propagate = False


load_dotenv()

mcp = FastMCP("mcp-bsw-py")


@lru_cache(maxsize=1)
def _tavily() -> TavilyClient:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        # don’t cache error, clear cache
        _tavily.cache_clear()
        raise RuntimeError("Set TAVILY_API_KEY")
    return TavilyClient(api_key=key)


@mcp.tool()
async def browse_bsw(topic: str, max_results: int = 3):
    """
    Searches the web for a topic related to Black Swan negotiation techniques.
    Returns a summary and a list of sources.
    """
    logger.warning(
        f"⚡️ browse_bsw CALLED with topic={topic!r}, max_results={max_results}"
    )

    query = f"Black Swan negotiation technique: {topic}"

    try:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(
            None, lambda: _tavily().search(query=query, max_results=max_results)
        )
        results = res.get("results", [])
        if not results:
            return {
                "summary": f"No results found for the topic '{topic}'.",
                "sources": [],
            }

        summary_parts = []
        sources = []
        for r in results:
            snippet = (r.get("content", "") or "")[:400]
            summary_parts.append(snippet)
            sources.append(
                {
                    "title": r.get("title", "No Title"),
                    "url": r.get("url", ""),
                }
            )

        return {
            "summary": "\n\n".join(summary_parts).strip(),
            "sources": sources,
        }

    except Exception as e:
        logger.exception(f"Error during search for topic {topic!r}: {e}")
        return {
            "summary": f"An internal error occurred during the web search: {e}",
            "sources": [],
        }


if __name__ == "__main__":
    # Runs over stdio; no HTTP server, no FastAPI.
    mcp.run(transport="stdio")
