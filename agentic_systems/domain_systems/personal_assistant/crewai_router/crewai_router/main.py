import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import mlflow
from crewai import Agent, Crew, Process, Task
from crewai.mcp import MCPServerStdio
from dotenv import load_dotenv
from mlflow.entities import SpanType

from llm import build_llm
from local_tools import write_personal_summary_document

ROUTER_BACKSTORY = """You are router_agent in a personal assistant router graph.
Decide exactly one next node: reader, action, or final.
Return strict JSON only:
{"next_node": "reader|action|final", "instruction": "short instruction"}
Use reader for workspace lookups (emails, calendar, tasks, notes, contacts, task files).
Use action to create/update tasks, calendar events, or draft emails.
Use final when enough context exists for a complete response.
"""

FINAL_BACKSTORY = """You are final_agent in a personal assistant router graph.
Given the user query, graph history, and latest branch result, decide:
- output_to_user (request is complete)
- route_to_router (more work needed)
If user asked to save a document, use the writer tool before output_to_user.
Never invent workspace facts. Ground response only in worker outputs.
Return strict JSON only:
{"action": "route_to_router|output_to_user", "next_instruction": "...", "final_response": "..."}
"""

WORKER_RULES = """
Use tools only when needed. Never invent workspace facts.
Every claim must come from tool outputs.
Drafts are local only — never claim an email was sent.
"""

READER_RULES = """
CRITICAL: Always use ONE or TWO word keyword queries (e.g. "outage", "ben", "task010") — never phrases or sentences.
Search tools do exact substring matching — long queries almost always return nothing.
If a search returns no results, retry with a single shorter keyword. Try 2-3 different keywords before concluding nothing exists.
Return ONLY raw data from tool outputs. Do not compose emails, suggest actions, or write content on behalf of the user.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PA CrewAI router MAS.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="pa_crewai_router")
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


def mcp_server(server_file: str) -> MCPServerStdio:
    return MCPServerStdio(command="python", args=[server_file], cache_tools_list=True)


def parse_json_block(raw: str) -> dict | None:
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


def parse_router_decision(raw: str) -> dict:
    data = parse_json_block(raw) or {}
    allowed = {"reader", "action", "final"}
    next_node = str(data.get("next_node", "final")).strip().lower()
    if next_node not in allowed:
        next_node = "final"
    instruction = str(data.get("instruction", "Handle the next step.")).strip()
    return {"next_node": next_node, "instruction": instruction}


def parse_final_decision(raw: str) -> dict:
    data = parse_json_block(raw) or {}
    action = str(data.get("action", "output_to_user")).strip().lower()
    if action not in {"route_to_router", "output_to_user"}:
        action = "output_to_user"
    return {
        "action": action,
        "next_instruction": str(data.get("next_instruction", "")).strip(),
        "final_response": str(data.get("final_response", "")).strip(),
    }


def run_single_task(agent: Agent, description: str, expected_output: str, verbose: bool):
    task = Task(description=description, expected_output=expected_output, agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=verbose)
    return str(crew.kickoff())


@mlflow.trace(name="pa_query", span_type=SpanType.CHAIN)
def run_router(query: str, verbose: bool = False):
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({"query": query})
    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)
    llm = build_llm()

    router_agent = Agent(
        role="Routing Agent",
        goal="Select the next graph node based on user request and accumulated outputs.",
        backstory=ROUTER_BACKSTORY,
        llm=llm,
        verbose=verbose,
        max_iter=8,
        allow_delegation=False,
    )
    final_agent = Agent(
        role="Final Agent",
        goal="Decide route back to router or produce final user output.",
        backstory=FINAL_BACKSTORY,
        llm=llm,
        verbose=verbose,
        max_iter=10,
        allow_delegation=False,
        tools=[write_personal_summary_document],
    )
    workers = {
        "reader": Agent(
            role="Reader Worker",
            goal="Search and read workspace data.",
            backstory=f"You are reader_worker.\nSearch workspace data only.\n{WORKER_RULES}\n{READER_RULES}",
            llm=llm,
            verbose=verbose,
            max_iter=12,
            allow_delegation=False,
            mcps=[mcp_server("workspace_server.py")],
        ),
        "action": Agent(
            role="Action Worker",
            goal="Perform workspace actions: create/update tasks, events, drafts.",
            backstory=f"You are action_worker.\nPerform workspace actions only.\n{WORKER_RULES}",
            llm=llm,
            verbose=verbose,
            max_iter=12,
            allow_delegation=False,
            mcps=[mcp_server("action_server.py")],
        ),
    }

    history: list[dict] = []
    router_instruction = "Start routing from the user request."
    final_answer = ""
    max_cycles = 10

    for _ in range(max_cycles):
        router_raw = run_single_task(
            agent=router_agent,
            description=(
                f"User request:\n{query}\n\n"
                f"Current graph context:\n{json.dumps(history, ensure_ascii=True)}\n\n"
                f"Latest instruction:\n{router_instruction}\n\n"
                "Return strict JSON only:\n"
                '{"next_node": "reader|action|final", "instruction": "..."}'
            ),
            expected_output="Valid router decision JSON.",
            verbose=verbose,
        )
        route = parse_router_decision(router_raw)
        next_node = route["next_node"]
        node_instruction = route["instruction"]
        history.append({"node": "router", "decision": route})

        if next_node != "final":
            worker_raw = run_single_task(
                agent=workers[next_node],
                description=(
                    f"User request:\n{query}\n\n"
                    f"Worker domain: {next_node}\n"
                    f"Router instruction: {node_instruction}\n\n"
                    "Use tools and return concrete tool-grounded evidence."
                ),
                expected_output=f"{next_node} worker result with evidence.",
                verbose=verbose,
            )
            history.append({"node": next_node, "result": worker_raw})
            final_input = f"Latest branch node: {next_node}\nLatest branch result:\n{worker_raw}"
        else:
            final_input = f"Router chose direct final.\nRouter instruction:\n{node_instruction}"

        final_raw = run_single_task(
            agent=final_agent,
            description=(
                f"Original user request:\n{query}\n\n"
                f"Graph history:\n{json.dumps(history, ensure_ascii=True)}\n\n"
                f"{final_input}\n\n"
                "Return strict JSON only:\n"
                '{"action": "route_to_router|output_to_user", "next_instruction": "...", "final_response": "..."}'
            ),
            expected_output="Valid final decision JSON.",
            verbose=verbose,
        )
        final_decision = parse_final_decision(final_raw)
        history.append({"node": "final", "decision": final_decision})

        if final_decision["action"] == "output_to_user":
            final_answer = final_decision["final_response"] or final_raw
            break

        router_instruction = final_decision["next_instruction"] or "Continue routing based on existing history."

    return final_answer or "Unable to complete within routing cycle limit."


def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    tmp_workspace = _create_tmp_workspace(base_dir)
    os.environ["PA_WORKSPACE_DIR"] = str(tmp_workspace)
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    mlflow.crewai.autolog()
    with mlflow.start_run(run_name=f"{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        result = run_router(args.query, verbose=args.verbose)
    print(result)


if __name__ == "__main__":
    main()