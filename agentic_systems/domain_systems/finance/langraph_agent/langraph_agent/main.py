import argparse
import asyncio
import hashlib
import os
import sys
from pathlib import Path

import mlflow
try:
    import mlflow.langchain as mlflow_langchain
except Exception:
    mlflow_langchain = None
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from llm import LLM
from portfolio_state import PORTFOLIO_RUN_KEY_ENV, initialize_portfolio_state

PYTHON_EXE = sys.executable

SYSTEM_PROMPT = """
You are a finance assistant agent.
Your responsibilities:
- Use retrieval, actions, and calculations tools to answer finance requests.
- For trade requests, use buy/sell tools and report executed results.
- Use search only when needed for market/news context.
- Complete the task end-to-end.
- Never ask follow-up questions.
- Final response must be plain text only, with no JSON or code fences.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the finance LangGraph single-agent baseline.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="finance_langraph_agent")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_mcp_client() -> MultiServerMCPClient:
    base_dir = Path(__file__).resolve().parent
    retrieval_server_path = str((base_dir / "portfolio_retrieval_server.py").resolve())
    actions_server_path = str((base_dir / "portfolio_actions_server.py").resolve())
    calculations_server_path = str((base_dir / "portfolio_calculations_server.py").resolve())
    market_api_key = os.getenv("FINANCIAL_DATASET_API_KEY")
    if not market_api_key:
        raise ValueError("Missing required API key env var: FINANCIAL_DATASET_API_KEY")
    return MultiServerMCPClient(
        {
            "market": {
                "transport": "streamable_http",
                "url": "https://mcp.financialdatasets.ai/api",
                "headers": {"X-API-KEY": market_api_key},
            },
            "portfolio_retrieval": {
                "transport": "stdio",
                "command": PYTHON_EXE,
                "args": [retrieval_server_path],
                "cwd": str(base_dir),
            },
            "portfolio_actions": {
                "transport": "stdio",
                "command": PYTHON_EXE,
                "args": [actions_server_path],
                "cwd": str(base_dir),
            },
            "portfolio_calculations": {
                "transport": "stdio",
                "command": PYTHON_EXE,
                "args": [calculations_server_path],
                "cwd": str(base_dir),
            },
        }
    )


async def run_agent(query: str, exp_name: str, verbose: bool = False) -> str:
    run_key = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
    os.environ[PORTFOLIO_RUN_KEY_ENV] = run_key
    initialize_portfolio_state()
    llm = LLM()
    mcp_client = build_mcp_client()
    mcp_tools = await mcp_client.get_tools()
    agent = create_agent(model=llm.client, tools=[*mcp_tools], system_prompt=SYSTEM_PROMPT)
    mlflow.set_experiment(exp_name)
    with mlflow.start_run():
        response = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})
    messages = response.get("messages", [])
    if not messages:
        return "No response generated."
    if verbose:
        for i, msg in enumerate(messages, start=1):
            role = getattr(msg, "type", None) or getattr(msg, "role", None) or msg.__class__.__name__
            content = getattr(msg, "content", str(msg))
            print(f"[{i}] {role}: {content}")
    return str(messages[-1].content)


async def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    if mlflow_langchain is not None:
        mlflow_langchain.autolog(log_traces=True, silent=True)
    answer = await run_agent(args.query, exp_name=args.exp_name, verbose=args.verbose)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
