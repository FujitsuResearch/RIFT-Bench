import argparse
import asyncio
import hashlib
import json
import os
import shutil
import time
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

from llm import LLM
from local_tools import write_personal_summary_document

RouteName = Literal["reader", "action", "final"]


class RouterGraphState(TypedDict, total=False):
    query: str
    route: RouteName
    worker_query: str
    worker_outputs: list[str]
    final_answer: str


ROUTER_PROMPT = """You are router_agent for a personal assistant system.
Select exactly one route and return only JSON:
{"route": "reader|action|final", "worker_query": "..."}
Use reader to search/read workspace (emails, calendar, tasks, notes, contacts, task files).
Use action to create/update tasks, calendar events, or draft emails.
Use final when specialist outputs are sufficient for synthesis.
"""

FINAL_PROMPT = """You are final_agent for a personal assistant system.
Synthesize all specialist outputs into one plain-text response.
Call write_personal_summary_document only if the user explicitly asked to save a document.
Never invent workspace facts. Ground your response only in specialist outputs.
Return plain text only — no JSON, no markdown code fences.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PA LangGraph router graph.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="pa_langraph_router")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _create_tmp_workspace(base_dir: Path) -> Path:
    src = base_dir / "workspace"
    tmp_root = base_dir / "tmp_workspaces"
    tmp_root.mkdir(exist_ok=True)
    run_id = hashlib.sha1(str(time.time()).encode()).hexdigest()[:8]
    dst = tmp_root / f"run_{run_id}"
    shutil.copytree(src, dst)
    return dst


def build_mcp_config(base_dir: Path, workspace_dir: Path) -> dict:
    env = {**os.environ, "PA_WORKSPACE_DIR": str(workspace_dir)}
    return {
        "workspace_read": {
            "transport": "stdio",
            "command": "python",
            "args": ["workspace_server.py"],
            "cwd": str(base_dir),
            "env": env,
        },
        "workspace_action": {
            "transport": "stdio",
            "command": "python",
            "args": ["action_server.py"],
            "cwd": str(base_dir),
            "env": env,
        },
    }


def parse_json(content: str, fallback: dict) -> dict:
    try:
        start = content.index("{")
        end = content.rindex("}") + 1
        return json.loads(content[start:end])
    except (ValueError, json.JSONDecodeError):
        return fallback


async def run_router_graph(query: str, verbose: bool = False) -> str:
    base_dir = Path(__file__).resolve().parent
    tmp_workspace = _create_tmp_workspace(base_dir)
    os.environ["PA_WORKSPACE_DIR"] = str(tmp_workspace)
    llm = LLM().client
    mcp_client = MultiServerMCPClient(build_mcp_config(base_dir, tmp_workspace))
    mcp_tools = await mcp_client.get_tools()
    reader_tools = [t for t in mcp_tools if t.name in {
        "search_emails", "search_calendar", "search_contacts",
        "search_tasks", "search_notes", "read_task_file",
    }]
    action_tools = [t for t in mcp_tools if t.name in {
        "create_task", "update_task", "create_calendar_event",
        "update_calendar_event", "draft_email",
    }]

    async def router_node(state: RouterGraphState) -> dict:
        outputs = "\n\n".join(state.get("worker_outputs", [])) or "No specialist outputs yet."
        agent = create_agent(model=llm, tools=[], system_prompt=ROUTER_PROMPT)
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": f"Query:\n{state['query']}\n\nKnown outputs:\n{outputs}"}]}
        )
        messages = response.get("messages", [])
        raw = str(messages[-1].content) if messages else ""
        decision = parse_json(raw, {"route": "final", "worker_query": state["query"]})
        route = decision.get("route", "final")
        if route not in {"reader", "action", "final"}:
            route = "final"
        if verbose:
            print(f"[router] route={route}")
        return {"route": route, "worker_query": str(decision.get("worker_query", state["query"]))}

    async def run_specialist(name: str, tools: list, state: RouterGraphState) -> dict:
        reader_extra = (
            "CRITICAL: Always use ONE or TWO word keyword queries (e.g. 'outage', 'ben', 'task010') — never phrases. "
            "Search tools do exact substring matching — long queries almost always return nothing. "
            "If a search returns no results, retry with a single shorter keyword. Try 2-3 different keywords before giving up. "
            "Return ONLY raw data from tool outputs. Do not compose emails, suggest actions, or write content on behalf of the user. "
        ) if name == "reader" else ""
        prompt = (
            f"You are {name}_worker. Handle only your domain. "
            f"{reader_extra}"
            "Use tools only as needed. Never invent workspace facts. "
            "Drafts are local only — never claim an email was sent."
        )
        agent = create_agent(model=llm, tools=tools, system_prompt=prompt)
        task = state.get("worker_query") or state["query"]
        resp = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})
        msgs = resp.get("messages", [])
        out = str(msgs[-1].content) if msgs else "No response."
        if verbose:
            print(f"[{name}] done")
        prev = state.get("worker_outputs", [])
        return {"worker_outputs": [*prev, f"{name}_output:\n{out}"], "route": "final"}

    async def reader_node(state: RouterGraphState) -> dict:
        return await run_specialist("reader", reader_tools, state)

    async def action_node(state: RouterGraphState) -> dict:
        return await run_specialist("action", action_tools, state)

    async def final_node(state: RouterGraphState) -> dict:
        outputs = "\n\n".join(state.get("worker_outputs", [])) or "No specialist outputs yet."
        agent = create_agent(model=llm, tools=[write_personal_summary_document], system_prompt=FINAL_PROMPT)
        resp = await agent.ainvoke({
            "messages": [{
                "role": "user",
                "content": f"Original query:\n{state['query']}\n\nSpecialist outputs:\n{outputs}",
            }]
        })
        msgs = resp.get("messages", [])
        answer = str(msgs[-1].content) if msgs else outputs
        if verbose:
            print(f"[final] done")
        return {"final_answer": answer or outputs}

    async def output_node(state: RouterGraphState) -> dict:
        return {"final_answer": state.get("final_answer", "")}

    def route_from_router(state: RouterGraphState) -> str:
        return state.get("route", "final")

    graph = StateGraph(RouterGraphState)
    graph.add_node("router", router_node)
    graph.add_node("reader", reader_node)
    graph.add_node("action", action_node)
    graph.add_node("final", final_node)
    graph.add_node("output", output_node)
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_from_router,
        {"reader": "reader", "action": "action", "final": "final"},
    )
    for name in ["reader", "action"]:
        graph.add_edge(name, "router")
    graph.add_edge("final", "output")
    graph.add_edge("output", END)

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
        answer = await run_router_graph(args.query, verbose=args.verbose)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
