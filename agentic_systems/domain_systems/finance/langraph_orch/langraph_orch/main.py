import argparse
import asyncio
import hashlib
import os
import sys
from pathlib import Path
from typing import Literal, TypedDict

import mlflow
try:
    import mlflow.langchain as mlflow_langchain
except Exception:
    mlflow_langchain = None
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from mlflow.entities import SpanType

from llm import LLM
from portfolio_state import initialize_portfolio_state

PYTHON_EXE = sys.executable

RouteName = Literal["market", "retrieval", "actions", "calculations", "final"]

ORCH_PROMPT = """You are a finance orchestrator node.
You receive the user query first, inspect worker outputs, and decide the next graph node.
Return only JSON with this schema:
{"next_worker":"market|retrieval|actions|calculations|final","worker_query":"...","final_answer":"..."}

Use retrieval for cash, holdings, and position details.
Use actions for buy and sell requests.
Use calculations for position value and portfolio value calculations.
Use market for external market data (price/fundamentals/news-like market info).
If more work is needed, set final_answer to an empty string.
If the request is complete, set next_worker to "final" and provide final_answer as plain text only.
Do not put JSON, markdown, or code fences inside final_answer.
Do not include intermediate reasoning, worker transcripts, or tool logs in final_answer.
Never ask follow-up questions.
"""

WORKER_PROMPT_TEMPLATE = """You are {name}_worker.
Handle only {name} tasks.
Use only your tools when needed.
Never invent facts. Every claim must come from tool results.
If no relevant tool result is available, return status: incomplete_tool_execution.
"""


class FinanceState(TypedDict, total=False):
    query: str
    next_worker: RouteName
    worker_query: str
    worker_outputs: list[str]
    final_answer: str
    iterations: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run finance LangGraph orchestrator.")
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--exp_name", type=str, default="finance_langraph_orchestrator")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def parse_orchestrator_decision(content: str) -> dict:
    import json

    try:
        start = content.index("{")
        end = content.rindex("}") + 1
        d = json.loads(content[start:end])
    except Exception:
        return {"next_worker": "final", "worker_query": "", "final_answer": content}
    nw = str(d.get("next_worker", "final"))
    if nw not in {"market", "retrieval", "actions", "calculations", "final"}:
        nw = "final"
    return {"next_worker": nw, "worker_query": str(d.get("worker_query", "")), "final_answer": str(d.get("final_answer", ""))}


def _last_message_text(result: dict) -> str:
    msgs = result.get("messages") or []
    if not msgs:
        return ""
    content = msgs[-1].content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                txt = item.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(parts).strip()
    return str(content)


@mlflow.trace(name="finance_query", span_type=SpanType.AGENT)
async def run_mas(query: str, verbose: bool = False) -> str:
    os.environ["FINANCE_PORTFOLIO_RUN_KEY"] = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
    initialize_portfolio_state()
    key = os.getenv("FINANCIAL_DATASET_API_KEY")
    if not key:
        raise ValueError("Missing required API key env var: FINANCIAL_DATASET_API_KEY")
    base_dir = Path(__file__).resolve().parent
    llm = LLM().client

    market_tools = await MultiServerMCPClient({"market": {"transport": "streamable_http", "url": "https://mcp.financialdatasets.ai/api", "headers": {"X-API-KEY": key}}}).get_tools()
    retrieval_tools = await MultiServerMCPClient({"retrieval": {"transport": "stdio", "command": PYTHON_EXE, "args": ["portfolio_retrieval_server.py"], "cwd": str(base_dir)}}).get_tools()
    actions_tools = await MultiServerMCPClient({"actions": {"transport": "stdio", "command": PYTHON_EXE, "args": ["portfolio_actions_server.py"], "cwd": str(base_dir)}}).get_tools()
    calc_tools = await MultiServerMCPClient({"calculations": {"transport": "stdio", "command": PYTHON_EXE, "args": ["portfolio_calculations_server.py"], "cwd": str(base_dir)}}).get_tools()

    async def orchestrator_node(state: FinanceState) -> dict:
        outputs = "\n\n".join(state.get("worker_outputs", [])) or "No worker outputs yet."
        iterations = state.get("iterations", 0)
        if iterations >= 10:
            return {"next_worker": "final", "final_answer": outputs, "iterations": iterations}
        agent = create_agent(model=llm, tools=[], system_prompt=ORCH_PROMPT)
        r = await agent.ainvoke({"messages": [{"role": "user", "content": f"Query:\n{state['query']}\n\nWorker outputs:\n{outputs}"}]})
        content = _last_message_text(r)
        d = parse_orchestrator_decision(content)
        if d["next_worker"] == "final" and not d.get("final_answer", "").strip():
            d["final_answer"] = outputs if outputs != "No worker outputs yet." else "No response generated."
        d["iterations"] = iterations + 1
        return d

    async def _run_worker(name: str, tools, state: FinanceState) -> dict:
        task = state.get("worker_query") or state["query"]
        agent = create_agent(model=llm, tools=tools, system_prompt=WORKER_PROMPT_TEMPLATE.format(name=name))
        r = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})
        out = _last_message_text(r) or "No response."
        return {
            "worker_outputs": [*(state.get("worker_outputs", [])), f"{name}_output:\n{out}"],
            "worker_query": "",
            "next_worker": "orchestrator",
        }

    async def market_node(state: FinanceState) -> dict:
        return await _run_worker("market", market_tools, state)

    async def retrieval_node(state: FinanceState) -> dict:
        return await _run_worker("retrieval", retrieval_tools, state)

    async def actions_node(state: FinanceState) -> dict:
        return await _run_worker("actions", actions_tools, state)

    async def calculations_node(state: FinanceState) -> dict:
        return await _run_worker("calculations", calc_tools, state)

    graph = StateGraph(FinanceState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("market", market_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("actions", actions_node)
    graph.add_node("calculations", calculations_node)
    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        lambda s: s.get("next_worker", "final"),
        {
            "market": "market",
            "retrieval": "retrieval",
            "actions": "actions",
            "calculations": "calculations",
            "final": END,
            "orchestrator": "orchestrator",
        },
    )
    for n in ["market", "retrieval", "actions", "calculations"]:
        graph.add_edge(n, "orchestrator")
    app = graph.compile()
    result = await app.ainvoke({"query": query, "worker_outputs": [], "iterations": 0})
    return result.get("final_answer") or "\n\n".join(result.get("worker_outputs", [])) or "No response generated."


async def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    if mlflow_langchain is not None:
        mlflow_langchain.autolog(log_traces=True, silent=True)
    with mlflow.start_run():
        answer = await run_mas(args.query, verbose=args.verbose)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
