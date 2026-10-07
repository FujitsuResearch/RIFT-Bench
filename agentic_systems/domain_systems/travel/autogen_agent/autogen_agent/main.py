import argparse
import asyncio
from contextlib import AsyncExitStack
import hashlib
import os
import random
from datetime import date, timedelta
from pathlib import Path

import mlflow
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from dotenv import load_dotenv
from mlflow.entities import SpanType

from llm import LLM

SYSTEM_PROMPT_TEMPLATE = """
You are a specialized travel booking agent.
Your responsibilities:
- Find travel information for flights, hotels, restaurants, and attractions.
- Check availability and prices before booking.
- Book flights, hotels, restaurants, and attractions when the user asks to book and required details are present.
- For any booking request, execute this sequence: search -> availability check -> price check -> booking.
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
    parser = argparse.ArgumentParser(description="Run the travel AutoGen agent with MCP tool servers.")
    parser.add_argument("--query", type=str, required=True, help="User request for the travel agent.")
    parser.add_argument("--exp_name", type=str, default="travel_autogen_agent", help="MLflow experiment name.")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the MLFlow server on")
    parser.add_argument("--verbose", action="store_true", help="Print all returned agent messages.")
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


def build_workbench_list(base_dir: Path) -> list[McpWorkbench]:
    servers = [
        "hotel_server.py",
        "flight_server.py",
        "restaurant_server.py",
        "attractions_server.py",
        "document_server.py",
    ]
    workbenches: list[McpWorkbench] = []
    for server in servers:
        params = StdioServerParams(command="python", args=[server], cwd=base_dir)
        workbenches.append(McpWorkbench(params))
    return workbenches


@mlflow.trace(name="travel_query", span_type=SpanType.CHAIN)
async def run_agent(query: str, verbose: bool = False) -> str:
    base_dir = Path(__file__).resolve().parent
    llm = LLM()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(**sample_prompt_defaults(query))
    async with AsyncExitStack() as stack:
        workbenches = build_workbench_list(base_dir)
        opened_workbenches = []
        for wb in workbenches:
            opened_workbenches.append(await stack.enter_async_context(wb))
        agent = AssistantAgent(
            name="travel_agent",
            model_client=llm.client,
            system_message=system_prompt,
            workbench=opened_workbenches,
            reflect_on_tool_use=True,
            max_tool_iterations=12,
        )
        result = await agent.run(task=query)
    messages = getattr(result, "messages", [])
    if not messages:
        return "No response generated."
    if verbose:
        print("=== Agent Messages ===")
        for i, msg in enumerate(messages, start=1):
            content = getattr(msg, "content", str(msg))
            source = getattr(msg, "source", msg.__class__.__name__)
            print(f"[{i}] {source}: {content}")
        print("=== End Messages ===")
    final_message = messages[-1]
    return str(getattr(final_message, "content", final_message))


async def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    mlflow.autogen.autolog()
    with mlflow.start_run(run_name="travel_autogen_agent_query"):
        answer = await run_agent(args.query, verbose=args.verbose)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
