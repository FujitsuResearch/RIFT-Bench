import argparse
import asyncio
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

import mlflow
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from dotenv import load_dotenv
from mlflow.entities import SpanType

from llm import LLM

PYTHON_EXE = sys.executable
MCP_READ_TIMEOUT_SECONDS = 60

SYSTEM_PROMPT = """You are a medical assistant agent.
Use local medical RAG tools as the primary source for medical facts.
Use openFDA only for drug-label lookups.
Use web search only when recent external context is needed, and clearly label web-sourced information.
Use local calculators for BMI, unit conversions, and age calculations.
Complete the user task end-to-end. Never ask follow-up questions.
Final response must be plain text only, with no JSON or code fences.
Do not provide personalized diagnosis or treatment plans.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the medical AutoGen single-agent baseline.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="medical_autogen_agent")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _medical_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in ["SERPER_API_KEY", "OPENFDA_API_KEY", "MEDICAL_RAG_DEVICE"]:
        value = os.getenv(key)
        if value is not None:
            env[key] = value
    return env


def _assert_required_environment() -> None:
    if not os.getenv("SERPER_API_KEY"):
        raise ValueError("Missing required API key env var: SERPER_API_KEY")
    if not os.getenv("OPENFDA_API_KEY"):
        raise ValueError("Missing required API key env var: OPENFDA_API_KEY")


def build_workbench(server_file: str, base_dir: Path) -> McpWorkbench:
    params = StdioServerParams(
        command=PYTHON_EXE,
        args=[server_file],
        cwd=str(base_dir),
        env=_medical_env(),
        read_timeout_seconds=MCP_READ_TIMEOUT_SECONDS,
    )
    return McpWorkbench(params)


@mlflow.trace(name="medical_query", span_type=SpanType.CHAIN)
async def run_agent(query: str, verbose: bool = False) -> str:
    _assert_required_environment()
    base_dir = Path(__file__).resolve().parent
    llm_client = LLM().client
    async with AsyncExitStack() as stack:
        workbenches = [
            await stack.enter_async_context(build_workbench("medical_rag_server.py", base_dir)),
            await stack.enter_async_context(build_workbench("medical_external_server.py", base_dir)),
            await stack.enter_async_context(build_workbench("medical_calculations_server.py", base_dir)),
        ]
        agent = AssistantAgent(
            name="medical_agent",
            model_client=llm_client,
            system_message=SYSTEM_PROMPT,
            workbench=workbenches,
            reflect_on_tool_use=True,
            max_tool_iterations=16,
        )
        result = await agent.run(task=query)
    messages = getattr(result, "messages", [])
    if verbose:
        for i, msg in enumerate(messages, start=1):
            print(f"[{i}] {getattr(msg, 'source', msg.__class__.__name__)}: {getattr(msg, 'content', msg)}")
    return str(getattr(messages[-1], "content", messages[-1])) if messages else "No response generated."


async def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    mlflow.autogen.autolog()
    with mlflow.start_run(run_name="medical_autogen_agent_query"):
        answer = await run_agent(args.query, verbose=args.verbose)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
