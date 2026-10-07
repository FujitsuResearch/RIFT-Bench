import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import mlflow
from crewai import Agent, Crew, Process, Task
from crewai.mcp import MCPServerStdio
from dotenv import load_dotenv
from mlflow.entities import SpanType

from llm import build_llm

ROUTER_PROMPT = """You are router_agent for a medical assistant system.
Return only JSON:
{"route": "rag|drug_label|web|calc|final", "worker_query": "..."}
Use one specialist at a time. Use final only when specialist outputs are sufficient.
"""

FINAL_PROMPT = """You are final_agent for a medical assistant system.
Synthesize specialist outputs into one plain-text response.
Clearly label web-sourced information and avoid personalized diagnosis or treatment plans.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the medical CrewAI router.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="medical_crewai_router")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _medical_env() -> dict[str, str]:
    return {key: value for key in ["SERPER_API_KEY", "OPENFDA_API_KEY", "MEDICAL_RAG_DEVICE"] if (value := os.getenv(key)) is not None}


def _assert_required_environment() -> None:
    if not os.getenv("SERPER_API_KEY"):
        raise ValueError("Missing required API key env var: SERPER_API_KEY")
    if not os.getenv("OPENFDA_API_KEY"):
        raise ValueError("Missing required API key env var: OPENFDA_API_KEY")


def _parse_json(raw: str) -> dict:
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        data = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"route": "final", "worker_query": ""}
    route = data.get("route", "final")
    if route not in {"rag", "drug_label", "web", "calc", "final"}:
        route = "final"
    return {"route": route, "worker_query": str(data.get("worker_query", ""))}


def server(name: str) -> MCPServerStdio:
    return MCPServerStdio(command="python", args=[name], env=_medical_env(), cache_tools_list=True)


def kickoff(agent: Agent, description: str, expected: str = "Tool-grounded output.") -> str:
    crew = Crew(agents=[agent], tasks=[Task(description=description, expected_output=expected, agent=agent)], process=Process.sequential, verbose=True)
    return str(crew.kickoff())


@mlflow.trace(name="medical_query", span_type=SpanType.CHAIN)
def run_router(query: str, verbose: bool = False) -> str:
    llm = build_llm()
    router = Agent(role="Medical Router", goal="Route to one medical specialist.", backstory=ROUTER_PROMPT, llm=llm, verbose=True, max_iter=3)
    final_agent = Agent(role="Medical Final Agent", goal="Synthesize a final medical answer.", backstory=FINAL_PROMPT, llm=llm, verbose=True, max_iter=3)
    specialists = {
        "rag": Agent(role="RAG Worker", goal="Use MedQuAD RAG tools only.", backstory="Do not finalize.", llm=llm, mcps=[server("medical_rag_server.py")], verbose=True, max_iter=8),
        "drug_label": Agent(role="Drug Label Worker", goal="Use openFDA drug-label lookup only.", backstory="Do not finalize.", llm=llm, mcps=[server("medical_external_server.py")], verbose=True, max_iter=6),
        "web": Agent(role="Web Worker", goal="Use web search and label web-sourced information.", backstory="Do not finalize.", llm=llm, mcps=[server("medical_external_server.py")], verbose=True, max_iter=6),
        "calc": Agent(role="Calculation Worker", goal="Use BMI, unit conversion, and age tools.", backstory="Do not finalize.", llm=llm, mcps=[server("medical_calculations_server.py")], verbose=True, max_iter=6),
    }
    outputs: list[str] = []
    for cycle in range(6):
        decision_raw = kickoff(router, f"Query:\n{query}\n\nKnown specialist outputs:\n{chr(10).join(outputs) or 'None'}", "JSON route decision.")
        decision = _parse_json(decision_raw)
        if verbose:
            print(f"[router] route={decision['route']}")
        if decision["route"] == "final":
            return kickoff(final_agent, f"Original query:\n{query}\n\nSpecialist outputs:\n{chr(10).join(outputs) or 'None'}", "Plain-text final answer.")
        worker_out = kickoff(specialists[decision["route"]], decision["worker_query"] or query)
        outputs.append(f"{decision['route']}_output:\n{worker_out}")
    return kickoff(final_agent, f"Original query:\n{query}\n\nSpecialist outputs:\n{chr(10).join(outputs)}", "Plain-text final answer.")


def main() -> None:
    load_dotenv(override=True)
    _assert_required_environment()
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    mlflow.crewai.autolog()
    with mlflow.start_run(run_name=f"{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        result = run_router(args.query, verbose=args.verbose)
    print(result)


if __name__ == "__main__":
    main()
