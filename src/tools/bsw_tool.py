import os
from langchain_core.tools import Tool
from langchain_mcp_adapters.client import MultiServerMCPClient

_client: MultiServerMCPClient | None = None
_bsw_tool: Tool | None = None


async def get_bsw_tool() -> Tool:
    """
    Returns the 'browse_bsw' MCP tool as a LangChain Tool.
    Spawns the MCP server (bsw_server.py) via stdio.
    """
    global _client, _bsw_tool
    if _bsw_tool:
        return _bsw_tool

    # Absolute path to your server
    server_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "mcp_servers", "bsw_server.py")
    )

    _client = MultiServerMCPClient(
        {
            "bsw": {
                "command": "python",
                "args": [server_path],
                "transport": "stdio",
            }
        }
    )

    tools = await _client.get_tools()
    for t in tools:
        if t.name == "browse_bsw":
            _bsw_tool = t
            return t

    raise RuntimeError("browse_bsw tool not found on MCP server")
