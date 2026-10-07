import argparse
import asyncio
import hashlib
import os
import shutil
import time
from pathlib import Path

import mlflow
try:
    import mlflow.langchain as mlflow_langchain
except Exception:
    mlflow_langchain = None
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from mlflow.entities import SpanType

from llm import LLM
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
    parser = argparse.ArgumentParser(description="Run the personal assistant LangGraph agent.")
    parser.add_argument("--query", type=str, required=True, help="User request.")
    parser.add_argument("--exp_name", type=str, default="pa_langraph_agent")
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


@mlflow.trace(name="pa_query", span_type=SpanType.AGENT)
async def run_agent(query: str, verbose: bool = False) -> str:
    base_dir = Path(__file__).resolve().parent
    tmp_workspace = _create_tmp_workspace(base_dir)
    os.environ["PA_WORKSPACE_DIR"] = str(tmp_workspace)
    llm = LLM()
    mcp_client = MultiServerMCPClient(build_mcp_config(base_dir, tmp_workspace))
    mcp_tools = await mcp_client.get_tools()
    tools = [*mcp_tools, write_personal_summary_document]
    agent = create_agent(model=llm.client, tools=tools, system_prompt=SYSTEM_PROMPT)
    response = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})
    messages = response.get("messages", [])
    if not messages:
        return "No response generated."
    if verbose:
        print("=== Agent Messages ===")
        for i, msg in enumerate(messages, start=1):
            role = getattr(msg, "type", None) or getattr(msg, "role", None) or msg.__class__.__name__
            content = getattr(msg, "content", str(msg))
            print(f"[{i}] {role}: {content}")
        print("=== End Messages ===")
    return str(messages[-1].content)


async def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    if mlflow_langchain is not None:
        mlflow_langchain.autolog(log_traces=True, silent=True)
    with mlflow.start_run():
        answer = await run_agent(args.query, verbose=args.verbose)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
