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
ROUTER_PROMPT = """You are a finance orchestration router.
Choose exactly one next_node from: market, retrieval, actions, calculations, final.

Routing policy:
- retrieval: portfolio state reads (cash, holdings list, position details)
- actions: buy/sell execution
- calculations: portfolio/position value math
- market: external market/news/price/fundamental data
- final: only after required worker outputs already exist in History

Hard rules:
- For queries about "available cash", "cash balance", "holdings", "positions", route to retrieval.
- Never claim tool limitations in the router output.
- Return strict JSON only: {"next_node":"...","instruction":"..."}.
"""

FINAL_PROMPT = """You are the finance finalizer.
Return strict JSON only:
{"action":"route_to_router|output_to_user","next_instruction":"...","final_response":"..."}

Hard rules:
- If required facts are missing from History, use action=route_to_router.
- Never output capability disclaimers like "I cannot retrieve user-specific financial data".
- For cash/holdings queries, output_to_user is allowed only after retrieval worker output includes cash and/or holdings facts.
- final_response must be plain text and grounded in worker outputs.
- If the latest worker output contains tool argument validation errors (for example missing required ticker),
  choose action=route_to_router and ask for a corrected retrieval instruction. Do not output a capability disclaimer.
"""

RETRIEVAL_WORKER_PROMPT = """Use only retrieval MCP tools.
Use these tools by intent:
- get_available_cash: cash balance requests
- list_owned_stocks: holdings list requests
- get_stock_holding_details(ticker): only when user asks about a specific ticker

Hard rules:
- Never pass generic tokens (ALL, CASH, HOLDINGS, PORTFOLIO) as ticker values.
- If the request asks for both cash and holdings, call get_available_cash and list_owned_stocks in the same run.
- Do not call get_stock_holding_details unless a specific ticker symbol is explicitly present in the user request or instruction.
- Never claim you cannot access portfolio data.
- Tool-error handling:
  1) If a tool returns argument validation errors or missing required fields, do not call that same tool again with the same arguments.
  2) Switch immediately to a valid alternative tool for the same user intent.
  3) If required intent cannot be satisfied after trying valid alternatives once, return a short plain-text error summary and stop.
- Return concise tool-grounded plain text only.
"""

RETRIEVAL_TASK_GUARD = """Execution guard for this task:
- If the request asks for available cash, call get_available_cash.
- If the request asks for holdings list/current holdings, call list_owned_stocks.
- If both are requested, call both tools in this run before answering.
- Do NOT call get_stock_holding_details unless a specific ticker symbol appears in the instruction.
- If a tool call fails due to missing ticker args, stop retrying that tool and proceed with valid no-arg retrieval tools.
"""

RETRIEVAL_TASK_STRICT_TEMPLATE = """You are executing a retrieval-only task.
Required behavior for this run:
1) Call get_available_cash
2) Call list_owned_stocks
3) Return plain text with both results

Forbidden for this run:
- Do not call get_stock_holding_details
- Do not ask follow-up questions
- Do not output capability disclaimers
- Do not retry the same failing tool call repeatedly

Task:
{task}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run finance CrewAI router.")
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--exp_name", type=str, default="finance_crewai_router")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _parse_json(raw: str, fallback: dict) -> dict:
    try:
        s, e = raw.index("{"), raw.rindex("}") + 1
        d = json.loads(raw[s:e])
        out = dict(fallback)
        out.update(d)
        return out
    except Exception:
        return fallback


def _kick(agent: Agent, prompt: str, verbose: bool):
    with mlflow.start_span(f"{agent.role}.kickoff", span_type=SpanType.AGENT) as span:
        span.set_inputs({"agent_role": agent.role, "prompt": prompt[:4000]})
        task = Task(description=prompt, expected_output="response", agent=agent)
        output = str(
            Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=verbose,
            ).kickoff()
        )
        span.set_outputs({"response_preview": output[:4000]})
        return output


def mcp_server(server_file: str, base_dir: Path) -> MCPServerStdio:
    return MCPServerStdio(command="python", args=[server_file], cache_tools_list=True, cwd=base_dir)


@mlflow.trace(name="finance_query", span_type=SpanType.CHAIN)
def run_router(query: str, verbose: bool = False):
    os.environ["FINANCE_PORTFOLIO_RUN_KEY"] = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)
    initialize_portfolio_state()
    key = os.getenv("FINANCIAL_DATASET_API_KEY")
    if not key:
        raise ValueError("Missing required API key env var: FINANCIAL_DATASET_API_KEY")
    llm = build_llm()

    router = Agent(role="Router Agent", goal="Select next node.", backstory=ROUTER_PROMPT, llm=llm, verbose=verbose, allow_delegation=False)
    final = Agent(role="Final Agent", goal="Route back or output final response.", backstory=FINAL_PROMPT, llm=llm, verbose=verbose, allow_delegation=False)
    workers = {
        "market": Agent(role="Market Worker", goal="Market retrieval only.", backstory="Use only market MCP.", llm=llm, verbose=verbose, allow_delegation=False, mcps=[MCPServerHTTP(url="https://mcp.financialdatasets.ai/api", streamable=True, cache_tools_list=True, headers={"X-API-KEY": key})]),
        "retrieval": Agent(role="Retrieval Worker", goal="Portfolio retrieval only.", backstory=RETRIEVAL_WORKER_PROMPT, llm=llm, verbose=verbose, allow_delegation=False, max_iter=4, mcps=[mcp_server("portfolio_retrieval_server.py", base_dir)]),
        "actions": Agent(role="Actions Worker", goal="Portfolio actions only.", backstory="Use only actions MCP.", llm=llm, verbose=verbose, allow_delegation=False, mcps=[mcp_server("portfolio_actions_server.py", base_dir)]),
        "calculations": Agent(role="Calculations Worker", goal="Portfolio calculations only.", backstory="Use only calculations MCP.", llm=llm, verbose=verbose, allow_delegation=False, mcps=[mcp_server("portfolio_calculations_server.py", base_dir)]),
    }

    history = []
    instruction = "Start."
    for i in range(10):
        with mlflow.start_span(f"router_iteration_{i+1}", span_type=SpanType.CHAIN) as iter_span:
            iter_span.set_inputs(
                {
                    "iteration": i + 1,
                    "instruction": instruction,
                    "history_len": len(history),
                }
            )
            r = _kick(
                router,
                (
                    f"Query:\n{query}\nHistory:\n{history}\nInstruction:\n{instruction}\n"
                    "Return JSON {\"next_node\":\"market|retrieval|actions|calculations|final\",\"instruction\":\"...\"}\n"
                    "For cash/holdings requests, instruction should explicitly say: "
                    "\"call get_available_cash and list_owned_stocks\"."
                ),
                verbose,
            )
            route = _parse_json(r, {"next_node": "retrieval", "instruction": query})
            node = str(route.get("next_node", "final")).lower()
            if node not in {"market", "retrieval", "actions", "calculations", "final"}:
                node = "final"
            history.append({"router": route})
            if node != "final":
                worker_instruction = route.get("instruction") or query
                if node == "retrieval":
                    worker_instruction = RETRIEVAL_TASK_STRICT_TEMPLATE.format(
                        task=f"{RETRIEVAL_TASK_GUARD}\n\n{worker_instruction}"
                    )
                w = _kick(workers[node], worker_instruction, verbose)
                history.append({node: w})
                latest = f"Latest worker: {node}\n{w}"
            else:
                latest = "Router chose final."
            fraw = _kick(
                final,
                (
                    f"Query:\n{query}\nHistory:\n{history}\n{latest}\n"
                    "Return JSON {\"action\":\"route_to_router|output_to_user\",\"next_instruction\":\"...\",\"final_response\":\"...\"}\n"
                    "If no tool-grounded cash/holdings facts are present yet, choose route_to_router."
                ),
                verbose,
            )
            fd = _parse_json(fraw, {"action": "route_to_router", "next_instruction": query, "final_response": ""})
            history.append({"final": fd})
            iter_span.set_outputs(
                {
                    "next_node": node,
                    "final_action": str(fd.get("action") or ""),
                    "history_len": len(history),
                }
            )
            if str(fd.get("action", "")).lower() == "output_to_user":
                return fd.get("final_response") or fraw
            instruction = fd.get("next_instruction") or "Continue."
    return "Unable to complete within routing cycle limit."


def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    mlflow.crewai.autolog()
    with mlflow.start_run(run_name=args.exp_name):
        print(run_router(args.query, verbose=args.verbose))


if __name__ == "__main__":
    main()
