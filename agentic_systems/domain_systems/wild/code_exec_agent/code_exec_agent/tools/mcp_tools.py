import os
from crewai.mcp import MCPServerStdio









mcp_local = MCPServerStdio(
    command="python",
    args=["mcps/local_mcp.py"],
    cache_tools_list=True,
)
