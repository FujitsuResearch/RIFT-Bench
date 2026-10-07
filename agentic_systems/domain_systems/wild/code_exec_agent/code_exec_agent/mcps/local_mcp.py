from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain_community.utilities import GoogleSerperAPIWrapper
import asyncio

load_dotenv(override=True)


search_mcp = FastMCP("search_mcp_tools")


@search_mcp.tool()
def web_search(query: str, num_results: int = 5) -> str:
    """Perform a web search and return the results."""
    search = GoogleSerperAPIWrapper()
    results = search.results(query, num_results=num_results)
    return results


# Create the main MCP server
main_mcp = FastMCP("main_MCP_server")


async def setup():
    # Unncessary for this case but here incase we will want to add more servers later.
    await main_mcp.import_server(search_mcp, prefix="search")


if __name__ == "__main__":
    # Run the server
    asyncio.run(setup())
    main_mcp.run(transport="stdio")
