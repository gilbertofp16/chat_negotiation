import logging
import os
from typing import Dict, List

from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient

# Configure logging for the server process
logging.basicConfig(level=logging.INFO, format="%(asctime)s - MCP_SERVER - %(levelname)s - %(message)s")

mcp = FastMCP("WebSearchMCP")


def _tavily() -> TavilyClient:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("Set TAVILY_API_KEY")
    return TavilyClient(api_key=key)


@mcp.tool()
def browse_bsw(topic: str, max_results: int = 3) -> Dict:
    """
    Searches the web for a topic related to Black Swan negotiation techniques.
    Returns a summary and a list of sources.
    """
    logging.info(f"Received request to browse for topic: {topic}")

    # Construct a focused query to keep the search within the desired domain.
    query = f"Black Swan negotiation technique: {topic}"

    try:
        logging.info(f"Performing Tavily search with query: '{query}'")
        res = _tavily().search(query=query, max_results=max_results)
        logging.info("Tavily search successful.")

        # Process and format the results into a summary and sources.
        results = res.get("results", [])
        if not results:
            logging.warning(f"No results found for the topic '{topic}'.")
            return {"summary": f"No results found for the topic '{topic}'.", "sources": []}

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

        summary = "\n\n".join(summary_parts).strip()
        logging.info(f"Returning summary and {len(sources)} sources.")
        return {"summary": summary, "sources": sources}

    except Exception as e:
        # Log the full exception details on the server side
        logging.error(f"An error occurred during the search for topic '{topic}': {e}", exc_info=True)
        # Return a clear error message in the JSON payload
        return {"summary": f"An internal error occurred during the web search: {e}", "sources": []}


if __name__ == "__main__":
    mcp.run(transport="stdio")
