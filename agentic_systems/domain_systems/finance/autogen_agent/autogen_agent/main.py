import argparse
import asyncio
import hashlib
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

import mlflow
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams, StreamableHttpServerParams
from dotenv import load_dotenv
from mlflow.entities import SpanType

from llm import LLM

MARKET_MCP_URL = "https://mcp.financialdatasets.ai/api"
PYTHON_EXE = sys.executable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run finance AutoGen single-agent.")
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--exp_name", type=str, default="finance_autogen_agent")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _workbench_list(base_dir: Path) -> list[McpWorkbench]:
    key = os.getenv("FINANCIAL_DATASET_API_KEY")
    if not key:
        raise ValueError("Missing required API key env var: FINANCIAL_DATASET_API_KEY")
    return [
        McpWorkbench(StreamableHttpServerParams(url=MARKET_MCP_URL, headers={"X-API-KEY": key})),
        McpWorkbench(StdioServerParams(command=PYTHON_EXE, args=["portfolio_retrieval_server.py"], cwd=base_dir)),
        McpWorkbench(StdioServerParams(command=PYTHON_EXE, args=["portfolio_actions_server.py"], cwd=base_dir)),
        McpWorkbench(StdioServerParams(command=PYTHON_EXE, args=["portfolio_calculations_server.py"], cwd=base_dir)),
    ]


@mlflow.trace(name="finance_query", span_type=SpanType.CHAIN)
async def run_agent(query: str, verbose: bool = False) -> str:
    base_dir = Path(__file__).resolve().parent
    run_key = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
    os.environ["FINANCE_PORTFOLIO_RUN_KEY"] = run_key
    llm = LLM().client
    system_prompt = (
        "You are a finance agent. Use market/retrieval/actions/calculation tools as needed, "
        "complete the task end-to-end, and return plain text only. Never ask follow-up questions."
    )
    async with AsyncExitStack() as stack:
        workbenches = [await stack.enter_async_context(wb) for wb in _workbench_list(base_dir)]
        agent = AssistantAgent(
            name="finance_agent",
            model_client=llm,
            system_message=system_prompt,
            workbench=workbenches,
            reflect_on_tool_use=True,
            max_tool_iterations=12,
        )
        result = await agent.run(task=query)
    messages = getattr(result, "messages", [])
    if verbose:
        for i, msg in enumerate(messages, start=1):
            print(f"[{i}] {getattr(msg, 'source', 'msg')}: {getattr(msg, 'content', msg)}")
    return str(getattr(messages[-1], "content", messages[-1])) if messages else "No response generated."


async def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    mlflow.autogen.autolog()
    with mlflow.start_run(run_name="finance_autogen_agent_query"):
        answer = await run_agent(args.query, verbose=args.verbose)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
