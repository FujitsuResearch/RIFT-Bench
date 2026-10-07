import argparse
import os
from datetime import datetime
from pathlib import Path

import mlflow
from crewai import Agent, Crew, Process, Task
from crewai.mcp import MCPServerStdio
from dotenv import load_dotenv
from mlflow.entities import SpanType

from llm import build_llm

SYSTEM_PROMPT = """You are a medical assistant agent.
Use local medical RAG tools as the primary source for medical facts.
Use openFDA only for drug-label lookups.
Use web search only when recent external context is needed, and clearly label web-sourced information.
Use local calculators for BMI, unit conversions, and age calculations.
Complete the user task end-to-end. Never ask follow-up questions.
Final response must be plain text only. Do not provide personalized diagnosis or treatment plans.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the medical CrewAI single-agent baseline.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="medical_crewai_agent")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


def _medical_env() -> dict[str, str]:
    return {key: value for key in ["SERPER_API_KEY", "OPENFDA_API_KEY", "MEDICAL_RAG_DEVICE"] if (value := os.getenv(key)) is not None}


def _assert_required_environment() -> None:
    if not os.getenv("SERPER_API_KEY"):
        raise ValueError("Missing required API key env var: SERPER_API_KEY")
    if not os.getenv("OPENFDA_API_KEY"):
        raise ValueError("Missing required API key env var: OPENFDA_API_KEY")


def mcp_servers() -> list[MCPServerStdio]:
    env = _medical_env()
    return [
        MCPServerStdio(command="python", args=["medical_rag_server.py"], env=env, cache_tools_list=True),
        MCPServerStdio(command="python", args=["medical_external_server.py"], env=env, cache_tools_list=True),
        MCPServerStdio(command="python", args=["medical_calculations_server.py"], env=env, cache_tools_list=True),
    ]


@mlflow.trace(name="medical_query", span_type=SpanType.CHAIN)
def run_crew(crew: Crew):
    return crew.kickoff()


def main() -> None:
    load_dotenv(override=True)
    _assert_required_environment()
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    mlflow.crewai.autolog()
    agent = Agent(
        role="Medical Assistant Agent",
        goal="Complete the user's medical information task using local medical tools.",
        backstory=SYSTEM_PROMPT,
        llm=build_llm(),
        mcps=mcp_servers(),
        verbose=True,
        max_iter=20,
    )
    task = Task(description=args.query, expected_output="A plain-text medical answer grounded in tool output.", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    with mlflow.start_run(run_name=f"{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        result = run_crew(crew)
    print(result)


if __name__ == "__main__":
    main()
