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

ORCHESTRATOR_BACKSTORY = """
You are a travel orchestrator manager.
You receive the user query first, decide which specialist worker agent to delegate to, inspect each worker result,
and continue delegating until the request is complete.
Do not perform booking or search work yourself. Rely on worker results.
Never ask follow-up questions and never return questions to the user.
Use hotel_worker for hotel search, availability, price, and booking tasks.
Use flight_worker for flight search, availability, price, and booking tasks.
Use restaurant_worker for restaurant search, availability, price, and booking tasks.
Use attractions_worker for attraction/activity search, availability, price, and booking tasks.
Use writer_worker only when the user explicitly asks to save or create an itinerary document.
Do not claim a file was saved unless writer_worker confirms it with tool output.
The final response to the user must be plain text only, with no JSON, markdown code fences, tool logs, or worker transcripts.
Never ask follow-up questions.
"""

HOTEL_WORKER_BACKSTORY = """
You are hotel_worker.
Handle only hotel requests.
Adapt flow by intent:
- use each tool only when needed.
- always check availability before checking price and before booking.
- always check price before booking.
- use search only when required.
Never invent facts. Every claim must come from tool outputs.
Return concrete evidence from tools, including state_file and confirmation_number when booking succeeds.
If required evidence is missing, return status: incomplete_tool_execution and explain what is missing.
"""

FLIGHT_WORKER_BACKSTORY = """
You are flight_worker.
Handle only flight requests.
Adapt flow by intent:
- use each tool only when needed.
- always check availability before checking price and before booking.
- always check price before booking.
- use search only when required.
Never invent facts. Every claim must come from tool outputs.
Return concrete evidence from tools, including state_file and confirmation_number when booking succeeds.
If required evidence is missing, return status: incomplete_tool_execution and explain what is missing.
"""

RESTAURANT_WORKER_BACKSTORY = """
You are restaurant_worker.
Handle only restaurant requests.
Adapt flow by intent:
- use each tool only when needed.
- always check availability before checking price and before booking.
- always check price before booking.
- use search only when required.
Never invent facts. Every claim must come from tool outputs.
Return concrete evidence from tools, including state_file and confirmation_number when booking succeeds.
If required evidence is missing, return status: incomplete_tool_execution and explain what is missing.
"""

ATTRACTIONS_WORKER_BACKSTORY = """
You are attractions_worker.
Handle only attractions and activities requests.
Adapt flow by intent:
- use each tool only when needed.
- always check availability before checking price and before booking.
- always check price before booking.
- use search only when required.
Never invent facts. Every claim must come from tool outputs.
Return concrete evidence from tools, including state_file and confirmation_number when booking succeeds.
If required evidence is missing, return status: incomplete_tool_execution and explain what is missing.
"""

WRITER_WORKER_BACKSTORY = """
You are writer_worker.
Handle only itinerary document creation.
Write only when explicitly requested.
Never claim a save happened unless the tool confirms it.
If no save request exists, return status: not_requested.
"""

ROUTING_JSON_SCHEMA = """
Return only JSON with this schema:
{
  "selected_workers": ["hotel","flight","restaurant","attractions","writer"],
  "reasoning_summary": "short reason",
  "booking_requested": true,
  "worker_instructions": {
    "hotel": "instruction",
    "flight": "instruction",
    "restaurant": "instruction",
    "attractions": "instruction",
    "writer": "instruction"
  }
}
"""

LOOP_DECISION_SCHEMA = """
Return only JSON with this schema:
{
  "action": "call_worker|final",
  "worker": "hotel|flight|restaurant|attractions|writer",
  "worker_instruction": "imperative instruction for the selected worker",
  "final_response": "plain-text final response when action is final"
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run travel CrewAI orchestrator + worker agents.")
    parser.add_argument("--query", type=str, required=True, help="User request for the travel orchestrator.")
    parser.add_argument("--exp_name", type=str, default="travel_crewai_orchestrator", help="MLflow experiment name.")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the MLFlow server on.")
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


def build_hotel_worker(llm, defaults_prompt: str, verbose: bool) -> Agent:
    return Agent(
        role="Hotel Worker",
        goal="Execute hotel checks, price lookup, and bookings with tool-backed evidence.",
        backstory=f"{HOTEL_WORKER_BACKSTORY}\n{defaults_prompt}",
        llm=llm,
        verbose=verbose,
        max_iter=12,
        allow_delegation=False,
        mcps=[mcp_server("hotel_server.py")],
    )


def build_flight_worker(llm, defaults_prompt: str, verbose: bool) -> Agent:
    return Agent(
        role="Flight Worker",
        goal="Execute flight checks, price lookup, and bookings with tool-backed evidence.",
        backstory=f"{FLIGHT_WORKER_BACKSTORY}\n{defaults_prompt}",
        llm=llm,
        verbose=verbose,
        max_iter=12,
        allow_delegation=False,
        mcps=[mcp_server("flight_server.py")],
    )


def build_restaurant_worker(llm, defaults_prompt: str, verbose: bool) -> Agent:
    return Agent(
        role="Restaurant Worker",
        goal="Execute restaurant checks, price lookup, and bookings with tool-backed evidence.",
        backstory=f"{RESTAURANT_WORKER_BACKSTORY}\n{defaults_prompt}",
        llm=llm,
        verbose=verbose,
        max_iter=12,
        allow_delegation=False,
        mcps=[mcp_server("restaurant_server.py")],
    )


def build_attractions_worker(llm, defaults_prompt: str, verbose: bool) -> Agent:
    return Agent(
        role="Attractions Worker",
        goal="Execute attractions checks, price lookup, and bookings with tool-backed evidence.",
        backstory=f"{ATTRACTIONS_WORKER_BACKSTORY}\n{defaults_prompt}",
        llm=llm,
        verbose=verbose,
        max_iter=12,
        allow_delegation=False,
        mcps=[mcp_server("attractions_server.py")],
    )


def build_writer_worker(llm, defaults_prompt: str, verbose: bool) -> Agent:
    return Agent(
        role="Travel Plan Writer Worker",
        goal="Create itinerary documents only on explicit save requests.",
        backstory=f"{WRITER_WORKER_BACKSTORY}\n{defaults_prompt}",
        llm=llm,
        verbose=verbose,
        max_iter=8,
        allow_delegation=False,
        tools=[write_travel_plan_document],
    )


def parse_routing_decision(raw_text: str) -> dict:
    fallback = {
        "selected_workers": ["hotel"],
        "reasoning_summary": "Fallback routing used because routing JSON could not be parsed.",
        "booking_requested": False,
        "worker_instructions": {"hotel": "Handle only hotel parts of the user request with required tool checks."},
    }
    try:
        start = raw_text.index("{")
        end = raw_text.rindex("}") + 1
        data = json.loads(raw_text[start:end])
    except (ValueError, json.JSONDecodeError):
        return fallback

    allowed = {"hotel", "flight", "restaurant", "attractions", "writer"}
    selected = [w for w in data.get("selected_workers", []) if w in allowed]
    if not selected:
        selected = ["hotel"]
    instructions = data.get("worker_instructions", {})
    return {
        "selected_workers": selected,
        "reasoning_summary": str(data.get("reasoning_summary", "")),
        "booking_requested": bool(data.get("booking_requested", False)),
        "worker_instructions": {w: str(instructions.get(w, f"Handle {w} parts of the request.")) for w in selected},
    }


def _kick(agent: Agent, prompt: str, verbose: bool):
    task = Task(description=prompt, expected_output="response", agent=agent)
    return str(Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=verbose).kickoff())


def parse_loop_decision(raw_text: str) -> dict:
    try:
        start = raw_text.index("{")
        end = raw_text.rindex("}") + 1
        data = json.loads(raw_text[start:end])
    except (ValueError, json.JSONDecodeError):
        data = {}
    action = str(data.get("action", "call_worker")).lower()
    if action not in {"call_worker", "final"}:
        action = "call_worker"
    worker = str(data.get("worker", "hotel")).lower()
    if worker not in {"hotel", "flight", "restaurant", "attractions", "writer"}:
        worker = "hotel"
    return {
        "action": action,
        "worker": worker,
        "worker_instruction": str(data.get("worker_instruction", "")),
        "final_response": str(data.get("final_response", "")),
    }


@mlflow.trace(name="travel_query", span_type=SpanType.CHAIN)
def run_orchestrator(query: str, verbose: bool = False):
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({"query": query})
    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)
    llm = build_llm()
    defaults_prompt = DEFAULTS_PROMPT_TEMPLATE.format(**sample_prompt_defaults(query))
    hotel_worker = build_hotel_worker(llm, defaults_prompt, verbose)
    flight_worker = build_flight_worker(llm, defaults_prompt, verbose)
    restaurant_worker = build_restaurant_worker(llm, defaults_prompt, verbose)
    attractions_worker = build_attractions_worker(llm, defaults_prompt, verbose)
    writer_worker = build_writer_worker(llm, defaults_prompt, verbose)
    worker_agents_by_name = {
        "hotel": hotel_worker,
        "flight": flight_worker,
        "restaurant": restaurant_worker,
        "attractions": attractions_worker,
        "writer": writer_worker,
    }
    orchestrator = Agent(
        role="Travel Orchestrator",
        goal=(
            "Plan worker routing, review worker outputs, and return one complete final travel response."
        ),
        backstory=ORCHESTRATOR_BACKSTORY,
        llm=llm,
        verbose=verbose,
        max_iter=16,
        allow_delegation=False,
    )

    history: list[str] = []
    instruction = query
    max_cycles = 10
    for cycle in range(1, max_cycles + 1):
        outputs = "\n\n".join(history) if history else "No worker outputs yet."
        decision_raw = _kick(
            orchestrator,
            (
                f"Original user request:\n{query}\n\n"
                f"Current instruction context:\n{instruction}\n\n"
                f"Worker outputs so far:\n{outputs}\n\n"
                f"Cycle: {cycle}/{max_cycles}\n"
                "Decide the next step.\n"
                "If complete, set action=final and provide final_response.\n"
                f"{LOOP_DECISION_SCHEMA}"
            ),
            verbose,
        )
        decision = parse_loop_decision(decision_raw)
        if decision["action"] == "final":
            final_response = decision["final_response"].strip()
            if final_response:
                return final_response
            return "Unable to produce final response from orchestrator."

        worker_name = decision["worker"]
        worker_instruction = decision["worker_instruction"].strip() or instruction or query
        worker_agent = worker_agents_by_name[worker_name]
        worker_output = _kick(
            worker_agent,
            (
                f"User request:\n{query}\n\n"
                f"Worker: {worker_name}\n"
                f"Instruction: {worker_instruction}\n\n"
                "Use your tools and return concrete outcomes with evidence (state_file, price, confirmation_number when present)."
            ),
            verbose,
        )
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
    with mlflow.start_run(run_name=f"{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        result = run_orchestrator(args.query, verbose=args.verbose)
    print(result)


if __name__ == "__main__":
    main()
