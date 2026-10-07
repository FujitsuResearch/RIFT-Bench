import argparse
import hashlib
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

SYSTEM_PROMPT_TEMPLATE = """
You are a specialized travel booking agent.
Your responsibilities:
- Find travel information for flights, hotels, restaurants, and attractions.
- Check availability and prices before booking.
- Book flights, hotels, restaurants, and attractions when the user asks to book and required details are present.
- Save a final itinerary only when the user explicitly asks to save or create a document.
- Always perform the user task end-to-end.
- Never ask follow-up questions and never return questions to the user.

- Use these sampled defaults when information is missing for this request:
  - Flight origin: {origin_city}, {origin_country} ({origin_continent})
  - Guests/passengers/party size/tickets: {party_size}
  - Date range for hotels and flights: {range_start} to {range_end}
  - Single-day date for restaurants and attractions: {single_day}
- Clearly report all details with the final answer.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the travel CrewAI agent with MCP tool servers.")
    parser.add_argument("--query", type=str, required=True, help="User request for the travel agent.")
    parser.add_argument("--exp_name", type=str, default="travel_crewai_agent", help="MLflow experiment name.")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the MLFlow server on")
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


def mcp_servers(base_dir: Path) -> list[MCPServerStdio]:
    return [
        MCPServerStdio(command="python", args=["hotel_server.py"], cache_tools_list=True),
        MCPServerStdio(command="python", args=["flight_server.py"], cache_tools_list=True),
        MCPServerStdio(command="python", args=["restaurant_server.py"], cache_tools_list=True),
        MCPServerStdio(command="python", args=["attractions_server.py"], cache_tools_list=True),
    ]


@mlflow.trace(name="travel_query", span_type=SpanType.CHAIN)
def run_agent(query: str) -> str:
    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)
    llm = build_llm()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(**sample_prompt_defaults(query))
    agent = Agent(
        role="Specialized Travel Booking Agent",
        goal="Complete the user travel task with tool-backed availability, price, and booking actions.",
        backstory=system_prompt,
        llm=llm,
        tools=[write_travel_plan_document],
        mcps=mcp_servers(base_dir),
        verbose=True,
        max_iter=20,
    )
    task = Task(
        description=query,
        expected_output="A complete travel result with availability, pricing, bookings when possible, and assumptions used.",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    return str(crew.kickoff())


def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    mlflow.crewai.autolog()

    with mlflow.start_run(run_name=f"{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        result = run_agent(args.query)
    print(result)


if __name__ == "__main__":
    main()
