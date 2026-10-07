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

WorkerName = Literal["reader", "action", "writer"]
RouteName = Literal["reader", "action", "writer", "final"]

ORCH_PROMPT = """You are a personal assistant orchestrator.
You receive the user query first, inspect worker outputs, and decide the next graph node.
Return only JSON with this schema:
{"next_worker": "reader|action|writer|final", "worker_query": "...", "final_answer": "..."}

Use reader to search/read workspace data (emails, calendar, tasks, notes, contacts, task files).
Use action to create/update tasks, calendar events, or draft emails.
Use writer only when the user explicitly asks to save or create a summary document.
If more work is needed, set final_answer to an empty string.
If the request is complete, set next_worker to "final" and provide final_answer as plain text only.
Do not include JSON, markdown code fences, or tool logs inside final_answer.
Never ask follow-up questions.
Never invent workspace facts.
"""

READER_WORKER_PROMPT = """You are reader_worker.
Search and read the workspace to gather information.
Use search_emails, search_calendar, search_contacts, search_tasks, search_notes, and read_task_file.
CRITICAL: Always use ONE or TWO word keyword queries (e.g. "outage", "ben", "task010") — never phrases or sentences.
Search tools do exact substring matching — long queries almost always return nothing.
If a search returns no results, you MUST retry with a single shorter keyword before giving up.
Try at least 2-3 different short keywords before concluding nothing exists.
Return ONLY raw data from tool outputs. Do not compose emails, suggest actions, or write content on behalf of the user.
Never invent workspace facts. Every claim must come from tool outputs.
Do not perform any write or update actions — those belong to action_worker.
"""

ACTION_WORKER_PROMPT = """You are action_worker.
Perform workspace modifications: create/update tasks, calendar events, and email drafts.
Use create_task, update_task, create_calendar_event, update_calendar_event, and draft_email.
Never perform actions not explicitly requested.
Drafts are local only — never claim an email was sent.
Return concrete evidence from tools (task id, event id, draft id) when an action succeeds.
"""

WRITER_WORKER_PROMPT = """You are writer_worker.
Save a personal summary document only when the user explicitly asks.
Never claim a save happened unless the tool confirms it.
If no save request exists, return status: not_requested.
"""


class PAGraphState(TypedDict, total=False):
    query: str
    next_worker: RouteName
    worker_query: str
    worker_outputs: list[str]
    final_answer: str
    iterations: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PA LangGraph orchestrator.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="pa_langraph_orch")
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


def parse_orchestrator_decision(content: str) -> dict:
    try:
        start = content.index("{")
        end = content.rindex("}") + 1
        decision = json.loads(content[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"next_worker": "final", "worker_query": "", "final_answer": content}
    next_worker = decision.get("next_worker", "final")
    if next_worker not in {"reader", "action", "writer", "final"}:
        next_worker = "final"
    return {
        "next_worker": next_worker,
        "worker_query": str(decision.get("worker_query", "")),
        "final_answer": str(decision.get("final_answer", "")),
    }


def to_plain_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    return cleaned


async def run_mas(query: str, verbose: bool = False) -> str:
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
    writer_tools = [write_personal_summary_document]

    async def orchestrator_node(state: PAGraphState) -> dict:
        outputs = "\n\n".join(state.get("worker_outputs", [])) or "No worker outputs yet."
        iterations = state.get("iterations", 0)
        if iterations >= 8:
            return {
                "next_worker": "final",
                "final_answer": f"Stopped after reaching iteration limit. Worker outputs:\n{outputs}",
                "iterations": iterations,
            }
        agent = create_agent(model=llm, tools=[], system_prompt=ORCH_PROMPT)
        response = await agent.ainvoke({
            "messages": [{
                "role": "user",
                "content": (
                    f"Original user request:\n{state['query']}\n\n"
                    f"Worker outputs so far:\n{outputs}\n\n"
                    "Choose the next worker node or final."
                ),
            }]
        })
        messages = response.get("messages", [])
        content = str(messages[-1].content) if messages else ""
        decision = parse_orchestrator_decision(content)
        decision["iterations"] = iterations + 1
        if verbose:
            print(f"[orchestrator] next={decision['next_worker']}")
        return decision

    async def run_worker(name: str, tools: list, prompt: str, state: PAGraphState) -> dict:
        worker_query = state.get("worker_query") or state["query"]
        agent = create_agent(model=llm, tools=tools, system_prompt=prompt)
        response = await agent.ainvoke({"messages": [{"role": "user", "content": worker_query}]})
        messages = response.get("messages", [])
        out = str(messages[-1].content) if messages else "No response."
        if verbose:
            print(f"[{name}] done")
        worker_outputs = state.get("worker_outputs", [])
        return {"worker_outputs": [*worker_outputs, f"{name}_output:\n{out}"], "worker_query": "", "next_worker": "final"}

    async def reader_node(state: PAGraphState) -> dict:
        return await run_worker("reader", reader_tools, READER_WORKER_PROMPT, state)

    async def action_node(state: PAGraphState) -> dict:
        return await run_worker("action", action_tools, ACTION_WORKER_PROMPT, state)

    async def writer_node(state: PAGraphState) -> dict:
        return await run_worker("writer", writer_tools, WRITER_WORKER_PROMPT, state)

    def route_from_orchestrator(state: PAGraphState) -> RouteName:
        return state.get("next_worker", "final")

    graph = StateGraph(PAGraphState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("reader", reader_node)
    graph.add_node("action", action_node)
    graph.add_node("writer", writer_node)
    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {"reader": "reader", "action": "action", "writer": "writer", "final": END},
    )
    for worker_name in ["reader", "action", "writer"]:
        graph.add_edge(worker_name, "orchestrator")

    app = graph.compile()
    result = await app.ainvoke({"query": query, "worker_outputs": [], "iterations": 0})
    final_answer = to_plain_text(result.get("final_answer", ""))
    if final_answer:
        return final_answer
    fallback = to_plain_text("\n\n".join(result.get("worker_outputs", [])))
    return fallback or "No response generated."


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
