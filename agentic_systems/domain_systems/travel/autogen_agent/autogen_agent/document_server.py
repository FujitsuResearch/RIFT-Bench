from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("document_mcp")


@mcp.tool()
def write_travel_plan_document(title: str, markdown_content: str) -> str:
    """Save a local markdown travel plan document."""
    safe_title = "".join(ch.lower() if ch.isalnum() else "_" for ch in title).strip("_")
    if not safe_title:
        safe_title = "travel_plan"
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_path = out_dir / f"{safe_title}_{timestamp}.md"
    content = f"# {title}\n\n{markdown_content.strip()}\n"
    file_path.write_text(content, encoding="utf-8")
    return f"Created travel plan document: {file_path.name}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
