from mcp.server.fastmcp import FastMCP

from medical_rag import retrieve

mcp = FastMCP("medical_rag_mcp")


def _render_hits(name: str, hits: list[dict]) -> str:
    lines = [f"type: {name}", f"count: {len(hits)}"]
    for h in hits:
        lines.append("---")
        lines.append(f"rank: {h['rank']}")
        lines.append(f"id: {h['id']}")
        lines.append(f"source: {h['source']}")
        lines.append(f"score: {h['score']:.4f}")
        lines.append(f"question: {h['question']}")
        lines.append(f"answer: {h['answer']}")
        md = h.get("metadata", {})
        lines.append(f"topic: {md.get('topic', '')}")
        lines.append(f"question_type: {md.get('question_type', '')}")
    return "\n".join(lines)


@mcp.tool()
def search_medical_knowledge(
    query: str,
    top_k: int = 5,
    topic: str = "",
    question_type: str = "",
) -> str:
    """Retrieve relevant medical knowledge entries for a query with optional filters."""
    hits = retrieve(query=query, top_k=top_k, topic=topic, question_type=question_type)
    return _render_hits("search_medical_knowledge_result", hits)


@mcp.tool()
def answer_medical_question_from_rag(question: str, top_k: int = 5) -> str:
    """Answer a medical question using top RAG hits and include citation IDs."""
    hits = retrieve(query=question, top_k=top_k)
    if not hits:
        return "type: answer_medical_question_from_rag_result\nstatus: no_results"
    answer = hits[0]["answer"]
    lines = [
        "type: answer_medical_question_from_rag_result",
        "status: ok",
        f"question: {question}",
        f"answer: {answer}",
        "citations:",
    ]
    for h in hits:
        lines.append(f"- {h['id']} (score={h['score']:.4f})")
    return "\n".join(lines)


@mcp.tool()
def summarize_medical_topic(topic: str, focus: str = "", top_k: int = 5) -> str:
    """Summarize a medical topic from retrieved passages with brief bullet points."""
    query = topic if not focus else f"{topic} {focus}"
    hits = retrieve(query=query, top_k=top_k, topic=topic)
    if not hits:
        hits = retrieve(query=query, top_k=top_k)
    if not hits:
        return "type: summarize_medical_topic_result\nstatus: no_results"

    bullets = []
    for h in hits[:3]:
        text = h["answer"].replace("\n", " ").strip()
        if len(text) > 240:
            text = text[:237] + "..."
        bullets.append(f"- {text}")

    lines = [
        "type: summarize_medical_topic_result",
        "status: ok",
        f"topic: {topic}",
        f"focus: {focus}",
        "summary:",
        *bullets,
        "citations:",
    ]
    for h in hits:
        lines.append(f"- {h['id']} (score={h['score']:.4f})")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")
