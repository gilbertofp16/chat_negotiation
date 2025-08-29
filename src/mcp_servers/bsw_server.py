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
async def browse_bsw(question: str, max_results: int = 5):
    """
    Performs an advanced web search on a negotiation-related topic.

    Returns a list of search result objects, each containing a title, url,
    and a detailed content snippet. This provides rich, structured information
    for answering user questions.
    """
    logger.warning(
        f"⚡️ browse_bsw CALLED with question={question!r}, max_results={max_results}"
    )

    query = f"negotiation techniques: {question}"

    try:
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(
            None,
            lambda: _tavily().search(
                query=query, max_results=max_results, search_depth="advanced"
            ),
        )
        api_results = res.get("results", [])
        if not api_results:
            return {"results": []}

        # Create a structured list of results for the agent
        structured_results = []
        for r in api_results:
            structured_results.append(
                {
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "content": (r.get("content", "") or "")[:4000],
                }
            )

        return {"results": structured_results}

    except Exception as e:
        logger.exception(f"Error during search for question {question!r}: {e}")
        return {
            "results": [],
            "error": f"An internal error occurred during the web search: {e}",
        }


if __name__ == "__main__":
    # Runs over stdio; no HTTP server, no FastAPI.
    mcp.run(transport="stdio")
