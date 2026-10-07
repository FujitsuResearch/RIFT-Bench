import argparse
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path

import mlflow
from crewai import Agent, Crew, Process, Task
from crewai.mcp import MCPServerHTTP, MCPServerStdio
from dotenv import load_dotenv
from mlflow.entities import SpanType

from llm import build_llm
from portfolio_state import initialize_portfolio_state

PYTHON_EXE = sys.executable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run finance CrewAI single-agent.")
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--exp_name", type=str, default="finance_crewai_agent")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def mcp_servers(base_dir: Path, api_key: str) -> list[MCPServerHTTP | MCPServerStdio]:
    return [
        MCPServerHTTP(
            url="https://mcp.financialdatasets.ai/api",
            streamable=True,
            cache_tools_list=True,
            headers={"X-API-KEY": api_key},
        ),
        MCPServerStdio(command=PYTHON_EXE, args=["portfolio_retrieval_server.py"], cache_tools_list=True, cwd=base_dir),
        MCPServerStdio(command=PYTHON_EXE, args=["portfolio_actions_server.py"], cache_tools_list=True, cwd=base_dir),
        MCPServerStdio(command=PYTHON_EXE, args=["portfolio_calculations_server.py"], cache_tools_list=True, cwd=base_dir),
    ]


@mlflow.trace(name="finance_query", span_type=SpanType.CHAIN)
def run_agent(query: str, verbose: bool = False):
    os.environ["FINANCE_PORTFOLIO_RUN_KEY"] = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)
    initialize_portfolio_state()
    key = os.getenv("FINANCIAL_DATASET_API_KEY")
    if not key:
        raise ValueError("Missing required API key env var: FINANCIAL_DATASET_API_KEY")
    llm = build_llm()
    agent = Agent(
        role="Finance Agent",
        goal="Use finance tools end-to-end and return plain text answer.",
        backstory=(
            "Use market, retrieval, actions, and calculations tools when needed.\n"
            "Do not invent data.\n"
            "Never ask follow-up questions.\n"
            "Final answer must be plain text only."
        ),
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        max_iter=12,
        mcps=mcp_servers(base_dir, key),
    )
    task = Task(description=query, expected_output="Plain-text finance answer.", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=verbose)
    return crew.kickoff()


def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    mlflow.crewai.autolog()
    with mlflow.start_run(run_name=f"{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        print(run_agent(args.query, verbose=args.verbose))


if __name__ == "__main__":
    main()
