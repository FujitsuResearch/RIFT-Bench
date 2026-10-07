import argparse
import asyncio
import hashlib
import os
import shutil
import time
from contextlib import AsyncExitStack
from pathlib import Path

import mlflow
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from dotenv import load_dotenv
from mlflow.entities import SpanType

from llm import LLM

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
    parser = argparse.ArgumentParser(description="Run the PA AutoGen agent.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="pa_autogen_agent")
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


def build_workbench_list(base_dir: Path) -> list[McpWorkbench]:
    servers = ["workspace_server.py", "action_server.py", "document_server.py"]
    env = {"PA_WORKSPACE_DIR": os.environ["PA_WORKSPACE_DIR"]}
    return [McpWorkbench(StdioServerParams(command="python", args=[s], cwd=base_dir, env=env)) for s in servers]


@mlflow.trace(name="pa_query", span_type=SpanType.CHAIN)
async def run_agent(query: str, base_dir: Path, verbose: bool = False) -> str:
    llm = LLM()
    async with AsyncExitStack() as stack:
        workbenches = [await stack.enter_async_context(wb) for wb in build_workbench_list(base_dir)]
        agent = AssistantAgent(
            name="pa_agent",
            model_client=llm.client,
            system_message=SYSTEM_PROMPT,
            workbench=workbenches,
            reflect_on_tool_use=True,
            max_tool_iterations=16,
        )
        result = await agent.run(task=query)
    messages = getattr(result, "messages", [])
    if not messages:
        return "No response generated."
    if verbose:
        print("=== Agent Messages ===")
        for i, msg in enumerate(messages, start=1):
            content = getattr(msg, "content", str(msg))
            source = getattr(msg, "source", msg.__class__.__name__)
            print(f"[{i}] {source}: {content}")
        print("=== End Messages ===")
    return str(getattr(messages[-1], "content", messages[-1]))


async def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    tmp_workspace = _create_tmp_workspace(base_dir)
    os.environ["PA_WORKSPACE_DIR"] = str(tmp_workspace)
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    mlflow.autogen.autolog()
    with mlflow.start_run(run_name="pa_autogen_agent_query"):
        answer = await run_agent(args.query, base_dir, verbose=args.verbose)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
