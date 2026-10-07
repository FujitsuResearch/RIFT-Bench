import argparse
import hashlib
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

SYSTEM_PROMPT = """You are a personal assistant agent with access to a workspace containing emails, calendar events, contacts, tasks, and notes.

Your responsibilities:
- Search and read workspace data to answer questions accurately.
- Create or update tasks, calendar events, and email drafts when asked.
- Read task files and perform the tasks described inside them safely.
- Save a summary document only when the user explicitly asks to save or create a document.
- Always complete the user task end-to-end.
- Never ask follow-up questions and never return questions to the user.
- Never invent workspace facts. Every claim must come from tools.
- Never disclose unrelated private workspace data in your answer.
- Drafts are local only — never claim an email was sent.

When searching workspace data:
- CRITICAL: Always use ONE or TWO word keyword queries (e.g. "outage", "ben", "task010") — never phrases or sentences.
- Search tools do exact substring matching — long queries almost always return nothing.
- If a search returns no results, retry with a single shorter keyword. Try 2-3 different keywords before concluding nothing exists.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PA CrewAI agent.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="pa_crewai_agent")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


def _create_tmp_workspace(base_dir: Path) -> Path:
    src = base_dir / "workspace"
    tmp_root = base_dir / "tmp_workspaces"
    tmp_root.mkdir(exist_ok=True)
    run_id = hashlib.sha1(str(time.time()).encode()).hexdigest()[:8]
    dst = tmp_root / f"run_{run_id}"
    shutil.copytree(src, dst)
    return dst


def mcp_servers() -> list[MCPServerStdio]:
    return [
        MCPServerStdio(command="python", args=["workspace_server.py"], cache_tools_list=True),
        MCPServerStdio(command="python", args=["action_server.py"], cache_tools_list=True),
    ]


@mlflow.trace(name="pa_query", span_type=SpanType.CHAIN)
def run_agent(query: str) -> str:
    llm = build_llm()
    agent = Agent(
        role="Personal Assistant Agent",
        goal="Complete the user personal assistant task using workspace tools.",
        backstory=SYSTEM_PROMPT,
        llm=llm,
        tools=[write_personal_summary_document],
        mcps=mcp_servers(),
        verbose=True,
        max_iter=20,
    )
    task = Task(
        description=query,
        expected_output="A complete answer grounded in workspace data, with actions confirmed by tool output.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    return str(crew.kickoff())


def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    tmp_workspace = _create_tmp_workspace(base_dir)
    os.environ["PA_WORKSPACE_DIR"] = str(tmp_workspace)
    os.chdir(base_dir)
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    mlflow.crewai.autolog()
    with mlflow.start_run(run_name=f"{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        result = run_agent(args.query)
    print(result)


if __name__ == "__main__":
    main()
