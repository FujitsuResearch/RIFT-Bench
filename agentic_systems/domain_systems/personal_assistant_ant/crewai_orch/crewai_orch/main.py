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

ORCHESTRATOR_BACKSTORY = """You are a personal assistant orchestrator.
You receive the user query, decide which specialist worker to delegate to next, inspect each worker result,
and continue delegating until the request is fully complete.
Do not perform searches or actions yourself — rely on worker results.
Never ask follow-up questions. Never invent workspace facts.
Use reader_worker for all workspace lookups (emails, calendar, tasks, notes, contacts, task files).
Use action_worker for creating/updating tasks, calendar events, or drafting emails.
Use writer_worker only when the user explicitly asks to save or create a summary document.
The final response must be plain text only — no JSON, markdown code fences, or tool logs.
"""

READER_WORKER_BACKSTORY = """You are reader_worker.
Search and read the workspace to gather information.
Use search_emails, search_calendar, search_contacts, search_tasks, search_notes, and read_task_file.
CRITICAL: Always use ONE or TWO word keyword queries (e.g. "outage", "ben", "task010") — never phrases or sentences.
Search tools do exact substring matching — long queries almost always return nothing.
If a search returns no results, retry with a single shorter keyword. Try 2-3 different keywords before concluding nothing exists.
Return ONLY raw data from tool outputs. Do not compose emails, suggest actions, or write content on behalf of the user.
Never invent workspace facts. Every claim must come from tool outputs.
Do not perform any write or update actions.
"""

ACTION_WORKER_BACKSTORY = """You are action_worker.
Perform workspace modifications: create/update tasks, calendar events, and email drafts.
Use create_task, update_task, create_calendar_event, update_calendar_event, and draft_email.
Never perform actions not explicitly requested.
Drafts are local only — never claim an email was sent.
Return concrete evidence (task id, event id, draft id) when an action succeeds.
"""

WRITER_WORKER_BACKSTORY = """You are writer_worker.
Save a personal summary document only when the user explicitly asks.
Never claim a save happened unless the tool confirms it.
If no save request exists, return status: not_requested.
"""

ORCHESTRATOR_DECISION_SCHEMA = """
Return only JSON with this schema:
{
  "action": "call_worker|final",
  "worker": "reader|action|writer",
  "instruction": "imperative instruction for the selected worker",
  "final_response": "plain-text final response when action is final"
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PA CrewAI orchestrator + workers.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="pa_crewai_orch")
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


def _kick(agent: Agent, prompt: str, verbose: bool) -> str:
    task = Task(description=prompt, expected_output="response", agent=agent)
    return str(Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=verbose).kickoff())


def _parse_decision(raw: str) -> dict:
    try:
        s, e = raw.index("{"), raw.rindex("}") + 1
        d = json.loads(raw[s:e])
    except Exception:
        d = {}
    action = str(d.get("action", "call_worker")).lower()
    if action not in {"call_worker", "final"}:
        action = "call_worker"
    worker = str(d.get("worker", "reader")).lower()
    if worker not in {"reader", "action", "writer"}:
        worker = "reader"
    return {
        "action": action,
        "worker": worker,
        "instruction": str(d.get("instruction", "")),
        "final_response": str(d.get("final_response", "")),
    }


@mlflow.trace(name="pa_query", span_type=SpanType.CHAIN)
def run_orchestrator(query: str, verbose: bool = False):
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({"query": query})
    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)
    llm = build_llm()

    orchestrator = Agent(
        role="Personal Assistant Orchestrator",
        goal="Iteratively route work to specialist workers and produce a final plain-text response.",
        backstory=ORCHESTRATOR_BACKSTORY,
        llm=llm,
        verbose=verbose,
        max_iter=16,
        allow_delegation=False,
    )
    workers = {
        "reader": Agent(
            role="Reader Worker",
            goal="Search and read workspace data for context.",
            backstory=READER_WORKER_BACKSTORY,
            llm=llm,
            verbose=verbose,
            max_iter=12,
            allow_delegation=False,
            mcps=[mcp_server("workspace_server.py")],
        ),
        "action": Agent(
            role="Action Worker",
            goal="Perform workspace actions: create/update tasks, events, drafts.",
            backstory=ACTION_WORKER_BACKSTORY,
            llm=llm,
            verbose=verbose,
            max_iter=12,
            allow_delegation=False,
            mcps=[mcp_server("action_server.py")],
        ),
        "writer": Agent(
            role="Writer Worker",
            goal="Save summary documents only when explicitly requested.",
            backstory=WRITER_WORKER_BACKSTORY,
            llm=llm,
            verbose=verbose,
            max_iter=8,
            allow_delegation=False,
            tools=[write_personal_summary_document],
        ),
    }

    history: list[str] = []
    instruction = query
    max_cycles = 10

    for cycle in range(1, max_cycles + 1):
        history_text = "\n\n".join(history) if history else "No worker outputs yet."
        decision_raw = _kick(
            orchestrator,
            (
                f"Original user request:\n{query}\n\n"
                f"Current instruction context:\n{instruction}\n\n"
                f"Worker outputs so far:\n{history_text}\n\n"
                f"Cycle: {cycle}/{max_cycles}\n"
                "Decide the next step. "
                "If the user requested both a task update AND an email draft, "
                "you must issue action_worker twice (once per action) or once with both instructions — "
                "do not emit final until both are confirmed with concrete ids.\n"
                f"{ORCHESTRATOR_DECISION_SCHEMA}"
            ),
            verbose,
        )
        decision = _parse_decision(decision_raw)

        if decision["action"] == "final":
            final_response = decision["final_response"].strip()
            if final_response:
                return final_response
            return "Unable to produce final response from orchestrator."

        worker_name = decision["worker"]
        worker_instruction = decision["instruction"].strip() or instruction or query
        worker_output = _kick(
            workers[worker_name],
            (
                f"User request:\n{query}\n\n"
                f"Worker: {worker_name}\n"
                f"Instruction: {worker_instruction}\n\n"
                "Use your tools and return concrete outcomes with evidence (task id, event id, draft id when present)."
            ),
            verbose,
        )
        history.append(f"{worker_name}_output:\n{worker_output}")
        instruction = worker_instruction

    summary = "\n\n".join(history) if history else "No worker outputs collected."
    return f"Stopped after reaching orchestration cycle limit ({max_cycles}). Latest worker outputs:\n{summary}"


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
        result = run_orchestrator(args.query, verbose=args.verbose)
    print(result)


if __name__ == "__main__":
    main()
