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

ORCH_PROMPT = """You are a medical orchestrator.
Return only JSON:
{"next_worker": "rag|drug_label|web|calc|final", "worker_query": "...", "final_answer": "..."}
Use one worker per step. Regain control after each worker. Do not finalize until required evidence is gathered.
Final answers must be plain text, medically cautious, and label web-sourced information.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the medical CrewAI orchestrator.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="medical_crewai_orch")
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
        return {"next_worker": "final", "worker_query": "", "final_answer": raw}
    nxt = data.get("next_worker", "final")
    if nxt not in {"rag", "drug_label", "web", "calc", "final"}:
        nxt = "final"
    return {"next_worker": nxt, "worker_query": str(data.get("worker_query", "")), "final_answer": str(data.get("final_answer", ""))}


def server(name: str) -> MCPServerStdio:
    return MCPServerStdio(command="python", args=[name], env=_medical_env(), cache_tools_list=True)


def kickoff(agent: Agent, description: str) -> str:
    crew = Crew(agents=[agent], tasks=[Task(description=description, expected_output="Tool-grounded output.", agent=agent)], process=Process.sequential, verbose=True)
    return str(crew.kickoff())


@mlflow.trace(name="medical_query", span_type=SpanType.CHAIN)
def run_orchestrator(query: str, verbose: bool = False) -> str:
    llm = build_llm()
    orchestrator = Agent(role="Medical Orchestrator", goal="Route medical tasks to specialist workers until complete.", backstory=ORCH_PROMPT, llm=llm, verbose=True, max_iter=3)
    workers = {
        "rag": Agent(role="RAG Worker", goal="Use MedQuAD RAG tools only.", backstory="Do not route or finalize.", llm=llm, mcps=[server("medical_rag_server.py")], verbose=True, max_iter=8),
        "drug_label": Agent(role="Drug Label Worker", goal="Use openFDA drug-label lookup only.", backstory="Do not route or finalize.", llm=llm, mcps=[server("medical_external_server.py")], verbose=True, max_iter=6),
        "web": Agent(role="Web Worker", goal="Use web search for recent medical context and label web-sourced information.", backstory="Do not route or finalize.", llm=llm, mcps=[server("medical_external_server.py")], verbose=True, max_iter=6),
        "calc": Agent(role="Calculation Worker", goal="Use BMI, unit conversion, and age tools.", backstory="Do not route or finalize.", llm=llm, mcps=[server("medical_calculations_server.py")], verbose=True, max_iter=6),
    }
    outputs: list[str] = []
    for cycle in range(8):
        decision_raw = kickoff(orchestrator, f"User request:\n{query}\n\nWorker outputs:\n{chr(10).join(outputs) or 'None'}")
        decision = _parse_json(decision_raw)
        if verbose:
            print(f"[orchestrator] next={decision['next_worker']}")
        if decision["next_worker"] == "final":
            return decision["final_answer"] or decision_raw
        worker_name = decision["next_worker"]
        worker_query = decision["worker_query"] or query
        worker_out = kickoff(workers[worker_name], worker_query)
        outputs.append(f"{worker_name}_output:\n{worker_out}")
    return "Stopped after reaching orchestration cycle limit.\n" + "\n\n".join(outputs)


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
        result = run_orchestrator(args.query, verbose=args.verbose)
    print(result)


if __name__ == "__main__":
    main()
