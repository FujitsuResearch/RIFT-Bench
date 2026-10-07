import argparse
import asyncio
import hashlib
import os
import random
from datetime import date, timedelta
from pathlib import Path

import mlflow
try:
    import mlflow.langchain as mlflow_langchain
except Exception:
    mlflow_langchain = None
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

from llm import LLM
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
    parser = argparse.ArgumentParser(description="Run the travel LangGraph agent with MCP tool servers.")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="User request for the travel agent.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="travel_langraph_agent",
        help="name of current experiments, for mlflow tracing.",
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to run the MLFlow server on"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print all returned agent messages before final output.",
    )
    return parser.parse_args()


def build_mcp_config(base_dir: Path) -> dict:
    return {
        "hotel": {
            "transport": "stdio",
            "command": "python",
            "args": ["hotel_server.py"],
            "cwd": str(base_dir),
        },
        "flight": {
            "transport": "stdio",
            "command": "python",
            "args": ["flight_server.py"],
            "cwd": str(base_dir),
        },
        "restaurant": {
            "transport": "stdio",
            "command": "python",
            "args": ["restaurant_server.py"],
            "cwd": str(base_dir),
        },
        "attractions": {
            "transport": "stdio",
            "command": "python",
            "args": ["attractions_server.py"],
            "cwd": str(base_dir),
        },
    }


def _fmt_date(d: date) -> str:
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def sample_prompt_defaults(query: str) -> dict:
    world = {
        "North America": {
            "USA": ["New York", "Los Angeles", "Chicago", "Miami", "Seattle"],
            "Canada": ["Toronto", "Vancouver", "Montreal"],
            "Mexico": ["Mexico City", "Cancun", "Guadalajara"],
        },
        "South America": {
            "Brazil": ["Sao Paulo", "Rio de Janeiro", "Brasilia"],
            "Argentina": ["Buenos Aires", "Cordoba"],
            "Colombia": ["Bogota", "Medellin"],
        },
        "Europe": {
            "France": ["Paris", "Lyon", "Nice"],
            "UK": ["London", "Manchester", "Edinburgh"],
            "Germany": ["Berlin", "Munich", "Frankfurt"],
            "Spain": ["Madrid", "Barcelona", "Seville"],
        },
        "Asia": {
            "Japan": ["Tokyo", "Osaka", "Kyoto"],
            "India": ["Delhi", "Mumbai", "Bengaluru"],
            "Singapore": ["Singapore"],
            "UAE": ["Dubai", "Abu Dhabi"],
        },
        "Africa": {
            "South Africa": ["Cape Town", "Johannesburg"],
            "Egypt": ["Cairo", "Alexandria"],
            "Kenya": ["Nairobi", "Mombasa"],
        },
        "Oceania": {
            "Australia": ["Sydney", "Melbourne", "Brisbane"],
            "New Zealand": ["Auckland", "Wellington"],
        },
    }

    configured_seed = os.getenv("TRAVEL_DEFAULTS_SEED")
    if configured_seed:
        seed_input = f"{configured_seed}:{query}"
    else:
        seed_input = f"{date.today().isoformat()}:{query}:{random.random()}"
    seed = int(hashlib.sha256(seed_input.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)

    continent = rng.choice(list(world.keys()))
    country = rng.choice(list(world[continent].keys()))
    city = rng.choice(world[continent][country])

    month = rng.randint(1, 12)
    start_day = rng.randint(1, 24)
    duration = rng.choices(
        population=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14],
        weights=[6, 12, 13, 12, 11, 9, 8, 6, 5, 4, 2, 1],
        k=1,
    )[0]
    base = date(date.today().year, month, start_day)
    end = base + timedelta(days=duration)

    party_size = rng.choices(
        population=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        weights=[55, 16, 10, 6, 4, 3, 2, 2, 1, 1],
        k=1,
    )[0]

    return {
        "origin_city": city,
        "origin_country": country,
        "origin_continent": continent,
        "party_size": party_size,
        "range_start": _fmt_date(base),
        "range_end": _fmt_date(end),
        "single_day": _fmt_date(base),
    }


async def run_agent(query: str, verbose: bool = False) -> str:
    base_dir = Path(__file__).resolve().parent
    llm = LLM()
    mcp_servers = build_mcp_config(base_dir)
    prompt_defaults = sample_prompt_defaults(query)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(**prompt_defaults)

    mcp_client = MultiServerMCPClient(mcp_servers)
    mcp_tools = await mcp_client.get_tools()
    tools = [*mcp_tools, write_travel_plan_document]
    agent = create_agent(model=llm.client, tools=tools, system_prompt=system_prompt)
    response = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})
    messages = response.get("messages", [])
    if not messages:
        return "No response generated."
    if verbose:
        print("=== Agent Messages ===")
        for i, msg in enumerate(messages, start=1):
            role = getattr(msg, "type", None) or getattr(msg, "role", None) or msg.__class__.__name__
            content = getattr(msg, "content", str(msg))
            print(f"[{i}] {role}: {content}")
        print("=== End Messages ===")
    return str(messages[-1].content)


async def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    if mlflow_langchain is not None:
        mlflow_langchain.autolog(log_traces=True, silent=True)
    with mlflow.start_run():
        answer = await run_agent(args.query, verbose=args.verbose)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
