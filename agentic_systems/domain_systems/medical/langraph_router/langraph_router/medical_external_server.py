import os

import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("medical_external_mcp")


@mcp.tool()
def web_search_serper(query: str, top_k: int = 5) -> str:
    """Search the web via Serper and return ranked result snippets."""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "type: web_search_serper_result\nstatus: missing_api_key"
    resp = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json={"q": query},
        timeout=25,
    )
    if resp.status_code >= 300:
        return f"type: web_search_serper_result\nstatus: http_error\ncode: {resp.status_code}"
    data = resp.json()
    organic = data.get("organic", [])[: max(1, top_k)]
    lines = ["type: web_search_serper_result", "status: ok"]
    for i, item in enumerate(organic, start=1):
        lines.append("---")
        lines.append(f"rank: {i}")
        lines.append(f"title: {item.get('title', '')}")
        lines.append(f"link: {item.get('link', '')}")
        lines.append(f"snippet: {item.get('snippet', '')}")
    return "\n".join(lines)


@mcp.tool()
def lookup_drug_label_openfda(drug_name: str, limit: int = 2) -> str:
    """Look up FDA drug label sections for a drug name using openFDA."""
    params = {
        "search": f'openfda.brand_name:"{drug_name}"+openfda.generic_name:"{drug_name}"',
        "limit": max(1, min(limit, 5)),
    }
    api_key = os.getenv("OPENFDA_API_KEY")
    if api_key:
        params["api_key"] = api_key
    resp = requests.get(
        "https://api.fda.gov/drug/label.json",
        params=params,
        timeout=25,
    )
    if resp.status_code >= 300:
        return f"type: lookup_drug_label_openfda_result\nstatus: http_error\ncode: {resp.status_code}"
    data = resp.json()
    results = data.get("results", [])
    if not results:
        return "type: lookup_drug_label_openfda_result\nstatus: no_results"
    lines = ["type: lookup_drug_label_openfda_result", "status: ok", f"drug_name: {drug_name}"]
    for i, item in enumerate(results, start=1):
        openfda = item.get("openfda", {})
        lines.append("---")
        lines.append(f"rank: {i}")
        lines.append(f"brand_name: {', '.join(openfda.get('brand_name', []))}")
        lines.append(f"generic_name: {', '.join(openfda.get('generic_name', []))}")
        for field in ["indications_and_usage", "warnings", "contraindications", "adverse_reactions", "dosage_and_administration"]:
            value = item.get(field, [])
            text = value[0] if value else ""
            text = text.replace("\n", " ").strip()
            if len(text) > 360:
                text = text[:357] + "..."
            lines.append(f"{field}: {text}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
