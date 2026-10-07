import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import mlflow
from crewai import Agent, Crew, Process, Task
from crewai.mcp import MCPServerHTTP, MCPServerStdio
from dotenv import load_dotenv
from mlflow.entities import SpanType

from llm import build_llm
from portfolio_state import initialize_portfolio_state

PYTHON_EXE = sys.executable
ORCHESTRATOR_RULES = (
    "You are a finance orchestrator. Decide one next step at a time, inspect worker outputs, and continue until complete. "
    "Route by intent: retrieval for cash/holdings/holding-details, actions for buy/sell, calculations for value math, market for external data. "
    "Never ask follow-up questions. Never invent facts. Return strict JSON only."
)

RETRIEVAL_RULES = (
    "Use only retrieval MCP tools. "
    "For cash call get_available_cash. "
    "For holdings list call list_owned_stocks. "
    "Use get_stock_holding_details only when the user names a real ticker like AAPL/GOOGL/NVDA/TSLA. "
    "Never pass generic words like CASH/ALL as ticker arguments. "
    "If a tool fails with validation or missing-argument errors, do not repeat the same failing call. "
    "Switch to a valid alternative tool for the same intent. "
    "If intent still cannot be completed after one valid alternative attempt, return a short plain-text error summary and stop. "
    "Return tool-grounded plain text only."
)

ACTIONS_RULES = (
    "Use only actions MCP tools for buy/sell requests. "
    "Available actions tools: buy_stock, sell_stock. "
    "When ticker and quantity are explicit, execute the trade directly. "
    "Return action execution result only. "
    "Post-trade reads (cash, holdings, holding details) must be handled by retrieval_worker in a separate orchestrator step. "
    "If a tool call fails with validation or missing-argument errors, do not repeat the same failing call. "
    "Correct arguments once and retry once at most. "
    "If execution still fails, return a short plain-text error summary and stop."
)

ORCHESTRATOR_DECISION_SCHEMA = """
Return only JSON with this schema:
{
  "action": "call_worker|final",
  "worker": "market|retrieval|actions|calculations",
  "instruction": "imperative instruction for selected worker",
  "final_response": "plain-text final answer when action is final"
}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run finance CrewAI orchestrator.")
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--exp_name", type=str, default="finance_crewai_orchestrator")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _kick(agent: Agent, prompt: str, verbose: bool):
    task = Task(description=prompt, expected_output="response", agent=agent)
    return str(
        Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=verbose,
        ).kickoff()
    )


def mcp_server(server_file: str, base_dir: Path) -> MCPServerStdio:
    return MCPServerStdio(command="python", args=[server_file], cache_tools_list=True, cwd=base_dir)


def _parse_orchestrator_decision(raw: str) -> dict:
    try:
        s, e = raw.index("{"), raw.rindex("}") + 1
        d = json.loads(raw[s:e])
    except Exception:
        d = {}
    action = str(d.get("action", "call_worker")).lower()
    if action not in {"call_worker", "final"}:
        action = "call_worker"
    worker = str(d.get("worker", "retrieval")).lower()
    if worker not in {"market", "retrieval", "actions", "calculations"}:
        worker = "retrieval"
    return {
        "action": action,
        "worker": worker,
        "instruction": str(d.get("instruction", "")),
        "final_response": str(d.get("final_response", "")),
    }


@mlflow.trace(name="finance_query", span_type=SpanType.CHAIN)
def run_orch(query: str, verbose: bool = False):
    os.environ["FINANCE_PORTFOLIO_RUN_KEY"] = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)
    initialize_portfolio_state()
    key = os.getenv("FINANCIAL_DATASET_API_KEY")
    if not key:
        raise ValueError("Missing required API key env var: FINANCIAL_DATASET_API_KEY")
    llm = build_llm()
    orchestrator = Agent(
        role="Finance Orchestrator",
        goal="Iteratively route work and produce a final plain-text answer.",
        backstory=ORCHESTRATOR_RULES,
        llm=llm,
        verbose=verbose,
        max_iter=12,
        allow_delegation=False,
    )
    workers = {
        "market": Agent(role="Market Worker", goal="Market retrieval only.", backstory="Use only market MCP.", llm=llm, verbose=verbose, allow_delegation=False, mcps=[MCPServerHTTP(url="https://mcp.financialdatasets.ai/api", streamable=True, cache_tools_list=True, headers={"X-API-KEY": key})]),
        "retrieval": Agent(
            role="Retrieval Worker",
            goal="Portfolio retrieval only.",
            backstory=RETRIEVAL_RULES,
            llm=llm,
            verbose=verbose,
            max_iter=8,
            allow_delegation=False,
            mcps=[mcp_server("portfolio_retrieval_server.py", base_dir)],
        ),
        "actions": Agent(
            role="Actions Worker",
            goal="Portfolio actions only.",
            backstory=ACTIONS_RULES,
            llm=llm,
            verbose=verbose,
            max_iter=8,
            allow_delegation=False,
            mcps=[mcp_server("portfolio_actions_server.py", base_dir)],
        ),
        "calculations": Agent(role="Calculations Worker", goal="Portfolio calculations only.", backstory="Use only calculations MCP.", llm=llm, verbose=verbose, max_iter=8, allow_delegation=False, mcps=[mcp_server("portfolio_calculations_server.py", base_dir)]),
    }
    history: list[str] = []
    instruction = query
    max_cycles = 8

    for cycle in range(1, max_cycles + 1):
        history_text = "\n\n".join(history) if history else "No worker outputs yet."
        decision_raw = _kick(
            orchestrator,
            (
                f"Original user query:\n{query}\n\n"
                f"Current instruction context:\n{instruction}\n\n"
                f"Worker outputs so far:\n{history_text}\n\n"
                f"Cycle: {cycle}/{max_cycles}\n"
                "Decide the next step.\n"
                "Do not claim tools are unavailable when a worker has those tools. "
                "If user requested a trade plus post-trade reads, use two orchestrator steps: "
                "first actions worker for trade execution, then retrieval worker for updated cash/holdings/details.\n"
                f"{ORCHESTRATOR_DECISION_SCHEMA}"
            ),
            verbose,
        )
        decision = _parse_orchestrator_decision(decision_raw)

        if decision["action"] == "final":
            final_response = decision["final_response"].strip()
            if final_response:
                return final_response
            return "Unable to produce a final response from orchestrator."

        worker_name = decision["worker"]
        worker_instruction = decision["instruction"].strip() or instruction or query
        worker_output = _kick(workers[worker_name], worker_instruction, verbose)
        history.append(f"{worker_name}_output:\n{worker_output}")
        instruction = worker_instruction

    summary = "\n\n".join(history) if history else "No worker outputs collected."
    return f"Stopped after reaching orchestration cycle limit ({max_cycles}). Latest worker outputs:\n{summary}"


def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    mlflow.crewai.autolog()
    with mlflow.start_run(run_name=args.exp_name):
        print(run_orch(args.query, verbose=args.verbose))


if __name__ == "__main__":
    main()
