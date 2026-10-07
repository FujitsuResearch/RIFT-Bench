import argparse
import hashlib
import json
import os
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import mlflow
from crewai import Agent, Crew, Process, Task
from crewai.mcp import MCPServerStdio
from dotenv import load_dotenv
from mlflow.entities import SpanType

from llm import build_llm
from local_tools import write_travel_plan_document

DEFAULTS_PROMPT_TEMPLATE = """
Use these sampled defaults when information is missing for this request:
- Flight origin: {origin_city}, {origin_country} ({origin_continent})
- Guests/passengers/party size/tickets: {party_size}
- Date range for hotels and flights: {range_start} to {range_end}
- Single-day date for restaurants and attractions: {single_day}
Clearly report any defaults or assumptions used.
"""

ROUTER_BACKSTORY = """
You are router_agent in a travel MAS router graph.
Decide exactly one next node:
- hotel
- flight
- restaurant
- attractions
- final
Return strict JSON only.
"""

FINAL_BACKSTORY = """
You are final_agent in a travel MAS router graph.
You receive user query, prior branch outputs, and latest branch result.
Decide either:
- output_to_user (if request is complete)
- route_to_router (if more branch work is needed)
If user asked to save a document, use the writer tool before output_to_user.
Never invent facts. Ground response only in worker outputs.
Return strict JSON only.
"""

WORKER_RULES = """
Adapt flow by intent:
- use each tool only when needed.
- always check availability before checking price and before booking.
- always check price before booking.
- use search only when required.
Never invent facts. Every claim must come from tool outputs.
Return concrete evidence from tools, including state_file and confirmation_number when booking succeeds.
If required evidence is missing, return status: incomplete_tool_execution and explain what is missing.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run travel CrewAI router MAS.")
    parser.add_argument("--query", type=str, required=True, help="User request for the travel router.")
    parser.add_argument("--exp_name", type=str, default="travel_crewai_router", help="MLflow experiment name.")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the MLflow server on.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose CrewAI and worker output.")
    return parser.parse_args()


def _fmt_date(d: date) -> str:
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def sample_prompt_defaults(query: str) -> dict:
    world = {
        "North America": {"USA": ["New York", "Los Angeles", "Chicago"], "Canada": ["Toronto", "Vancouver"]},
        "South America": {"Brazil": ["Sao Paulo", "Rio de Janeiro"], "Argentina": ["Buenos Aires"]},
        "Europe": {"France": ["Paris", "Lyon"], "UK": ["London", "Manchester"], "Germany": ["Berlin"]},
        "Asia": {"Japan": ["Tokyo", "Osaka"], "India": ["Delhi", "Mumbai"], "UAE": ["Dubai"]},
        "Africa": {"South Africa": ["Cape Town"], "Egypt": ["Cairo"]},
        "Oceania": {"Australia": ["Sydney", "Melbourne"], "New Zealand": ["Auckland"]},
    }
    configured_seed = os.getenv("TRAVEL_DEFAULTS_SEED")
    seed_input = f"{configured_seed}:{query}" if configured_seed else f"{date.today().isoformat()}:{query}:{random.random()}"
    seed = int(hashlib.sha256(seed_input.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    continent = rng.choice(list(world.keys()))
    country = rng.choice(list(world[continent].keys()))
    city = rng.choice(world[continent][country])
    base = date(date.today().year, rng.randint(1, 12), rng.randint(1, 24))
    duration = rng.choices([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14], [6, 12, 12, 11, 10, 8, 7, 6, 3, 2, 1], k=1)[0]
    party_size = rng.choices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [55, 16, 10, 6, 4, 3, 2, 2, 1, 1], k=1)[0]
    return {
        "origin_city": city,
        "origin_country": country,
        "origin_continent": continent,
        "party_size": party_size,
        "range_start": _fmt_date(base),
        "range_end": _fmt_date(base + timedelta(days=duration)),
        "single_day": _fmt_date(base),
    }


def mcp_server(server_file: str) -> MCPServerStdio:
    return MCPServerStdio(command="python", args=[server_file], cache_tools_list=True)


def build_worker(llm, role: str, domain: str, server_file: str, defaults_prompt: str, verbose: bool, max_iter: int = 12) -> Agent:
    return Agent(
        role=role,
        goal=f"Handle only {domain} requests with tool-grounded outputs.",
        backstory=f"You are {domain}_worker.\nHandle only {domain} requests.\n{WORKER_RULES}\n{defaults_prompt}",
        llm=llm,
        verbose=verbose,
        max_iter=max_iter,
        allow_delegation=False,
        mcps=[mcp_server(server_file)],
    )


def parse_json_block(raw: str) -> dict | None:
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        return None


def parse_router_decision(raw: str) -> dict:
    data = parse_json_block(raw) or {}
    allowed = {"hotel", "flight", "restaurant", "attractions", "final"}
    next_node = str(data.get("next_node", "final")).strip().lower()
    if next_node not in allowed:
        next_node = "final"
    instruction = str(data.get("instruction", "Handle required next step based on current context.")).strip()
    return {"next_node": next_node, "instruction": instruction}


def parse_final_decision(raw: str) -> dict:
    data = parse_json_block(raw) or {}
    action = str(data.get("action", "output_to_user")).strip().lower()
    if action not in {"route_to_router", "output_to_user"}:
        action = "output_to_user"
    next_instruction = str(data.get("next_instruction", "")).strip()
    final_response = str(data.get("final_response", "")).strip()
    return {
        "action": action,
        "next_instruction": next_instruction,
        "final_response": final_response,
    }


def run_single_task(agent: Agent, description: str, expected_output: str, verbose: bool):
    task = Task(description=description, expected_output=expected_output, agent=agent)
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=verbose)
    return str(crew.kickoff())


@mlflow.trace(name="travel_query", span_type=SpanType.CHAIN)
def run_router(query: str, verbose: bool = False):
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({"query": query})

    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)
    llm = build_llm()
    defaults_prompt = DEFAULTS_PROMPT_TEMPLATE.format(**sample_prompt_defaults(query))

    router_agent = Agent(
        role="Routing Agent",
        goal="Select the next graph node based on user request and accumulated branch outputs.",
        backstory=f"{ROUTER_BACKSTORY}\n{defaults_prompt}",
        llm=llm,
        verbose=verbose,
        max_iter=8,
        allow_delegation=False,
    )
    final_agent = Agent(
        role="Final Agent",
        goal="Decide route back to router or produce final user output.",
        backstory=f"{FINAL_BACKSTORY}\n{defaults_prompt}",
        llm=llm,
        verbose=verbose,
        max_iter=10,
        allow_delegation=False,
        tools=[write_travel_plan_document],
    )
    workers = {
        "hotel": build_worker(llm, "Hotel Worker", "hotel", "hotel_server.py", defaults_prompt, verbose),
        "flight": build_worker(llm, "Flight Worker", "flight", "flight_server.py", defaults_prompt, verbose),
        "restaurant": build_worker(llm, "Restaurant Worker", "restaurant", "restaurant_server.py", defaults_prompt, verbose),
        "attractions": build_worker(llm, "Attractions Worker", "attractions", "attractions_server.py", defaults_prompt, verbose),
    }

    history: list[dict] = []
    router_instruction = "Start routing from the user request."
    final_answer = ""
    max_cycles = 10

    for _ in range(max_cycles):
        router_raw = run_single_task(
            agent=router_agent,
            description=(
                f"User request:\n{query}\n\n"
                f"Current graph context:\n{json.dumps(history, ensure_ascii=True)}\n\n"
                f"Latest final/router instruction:\n{router_instruction}\n\n"
                "Return strict JSON only:\n"
                "{\n"
                '  "next_node": "hotel|flight|restaurant|attractions|final",\n'
                '  "instruction": "short concrete instruction for the selected node"\n'
                "}"
            ),
            expected_output="Valid router decision JSON.",
            verbose=verbose,
        )
        route = parse_router_decision(router_raw)
        next_node = route["next_node"]
        node_instruction = route["instruction"]
        history.append({"node": "router", "decision": route, "raw": router_raw})

        if next_node != "final":
            worker_raw = run_single_task(
                agent=workers[next_node],
                description=(
                    f"User request:\n{query}\n\n"
                    f"Worker domain: {next_node}\n"
                    f"Router instruction: {node_instruction}\n\n"
                    "Use MCP tools and return concrete tool-grounded evidence."
                ),
                expected_output=f"{next_node} worker result with evidence.",
                verbose=verbose,
            )
            history.append({"node": next_node, "result": worker_raw})
            final_input = f"Latest branch node: {next_node}\nLatest branch result:\n{worker_raw}"
        else:
            final_input = f"Router chose direct final.\nRouter instruction:\n{node_instruction}"

        final_raw = run_single_task(
            agent=final_agent,
            description=(
                f"Original user request:\n{query}\n\n"
                f"Graph history:\n{json.dumps(history, ensure_ascii=True)}\n\n"
                f"{final_input}\n\n"
                "Return strict JSON only:\n"
                "{\n"
                '  "action": "route_to_router|output_to_user",\n'
                '  "next_instruction": "if action is route_to_router",\n'
                '  "final_response": "if action is output_to_user"\n'
                "}"
            ),
            expected_output="Valid final decision JSON.",
            verbose=verbose,
        )
        final_decision = parse_final_decision(final_raw)
        history.append({"node": "final", "decision": final_decision, "raw": final_raw})

        if final_decision["action"] == "output_to_user":
            final_answer = final_decision["final_response"] or final_raw
            break

        router_instruction = final_decision["next_instruction"] or "Continue routing based on existing history."

    if not final_answer:
        final_answer = "Unable to complete within routing cycle limit."
    return final_answer


def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    mlflow.crewai.autolog()
    with mlflow.start_run(run_name=f"{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        result = run_router(args.query, verbose=args.verbose)
    print(result)


if __name__ == "__main__":
    main()
