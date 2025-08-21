from langchain_core.tools import Tool

from src.mcp_servers.mcp_client import bsw_context_from_results, mcp_browse_sync


def get_bsw_tool() -> Tool:
    """
    Creates a LangChain Tool for browsing Black Swan negotiation techniques.

    This tool wraps the MCP client, allowing a LangChain agent to decide when
    to call the external `browse_bsw` microservice.
    """
    return Tool(
        name="browse_bsw",
        description="Use this tool to search the web for information on Black Swan negotiation techniques to sanity-check or enrich your answer. Input should be a specific negotiation topic (e.g., 'mirroring', 'calibrated questions').",
        func=lambda topic: bsw_context_from_results(mcp_browse_sync(topic)),
    )
