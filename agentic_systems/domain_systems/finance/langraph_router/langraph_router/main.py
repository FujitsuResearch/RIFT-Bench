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
NextFromFinal = Literal["router", "output"]


class RouterState(TypedDict, total=False):
    query: str
    route: RouteName
    instruction: str
    worker_outputs: list[str]
    final_action: NextFromFinal
    final_answer: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run finance LangGraph router.")
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--exp_name", type=str, default="finance_langraph_router")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


@mlflow.trace(name="finance_query", span_type=SpanType.AGENT)
async def run_router(query: str, verbose: bool = False) -> str:
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

    async def router_node(state: RouterState) -> dict:
        agent = create_agent(model=llm, tools=[], system_prompt="Choose one node: market|retrieval|actions|calculations|final. Return plain text as '<node>|<instruction>'.")
        r = await agent.ainvoke({"messages": [{"role": "user", "content": f"Query:\n{state['query']}\nOutputs:\n{state.get('worker_outputs', [])}"}]})
        content = str(r.get("messages", [])[-1].content) if r.get("messages") else "final|"
        node, _, instr = content.partition("|")
        node = node.strip().lower()
        if node not in {"market", "retrieval", "actions", "calculations", "final"}:
            node = "final"
        return {"route": node, "instruction": instr.strip()}

    async def _worker(name: str, tools, state: RouterState) -> dict:
        agent = create_agent(model=llm, tools=tools, system_prompt=f"You are {name}_worker. Use only your tools.")
        task = state.get("instruction") or state["query"]
        r = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})
        out = str(r.get("messages", [])[-1].content) if r.get("messages") else "No response."
        return {"worker_outputs": [*(state.get("worker_outputs", [])), f"{name}_output:\n{out}"], "route": "final"}

    async def final_node(state: RouterState) -> dict:
        agent = create_agent(model=llm, tools=[], system_prompt="Return either ROUTER|<instruction> or OUTPUT|<plain text answer>.")
        r = await agent.ainvoke({"messages": [{"role": "user", "content": f"Query:\n{state['query']}\nOutputs:\n{state.get('worker_outputs', [])}"}]})
        content = str(r.get("messages", [])[-1].content) if r.get("messages") else "OUTPUT|No response."
        action, _, payload = content.partition("|")
        action = action.strip().upper()
        if action == "ROUTER":
            return {"final_action": "router", "instruction": payload.strip()}
        return {"final_action": "output", "final_answer": payload.strip() or content}

    async def market_node(state: RouterState) -> dict:
        return await _worker("market", market_tools, state)

    async def retrieval_node(state: RouterState) -> dict:
        return await _worker("retrieval", retrieval_tools, state)

    async def actions_node(state: RouterState) -> dict:
        return await _worker("actions", actions_tools, state)

    async def calculations_node(state: RouterState) -> dict:
        return await _worker("calculations", calc_tools, state)

    graph = StateGraph(RouterState)
    graph.add_node("router", router_node)
    graph.add_node("market", market_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("actions", actions_node)
    graph.add_node("calculations", calculations_node)
    graph.add_node("final", final_node)
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", lambda s: s.get("route", "final"), {"market": "market", "retrieval": "retrieval", "actions": "actions", "calculations": "calculations", "final": "final"})
    for n in ["market", "retrieval", "actions", "calculations"]:
        graph.add_edge(n, "final")
    graph.add_conditional_edges("final", lambda s: s.get("final_action", "output"), {"router": "router", "output": END})

    app = graph.compile()
    result = await app.ainvoke({"query": query, "worker_outputs": []})
    return result.get("final_answer") or "No response generated."


async def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    if mlflow_langchain is not None:
        mlflow_langchain.autolog(log_traces=True, silent=True)
    with mlflow.start_run():
        answer = await run_router(args.query, verbose=args.verbose)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
