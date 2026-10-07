import argparse
import asyncio
from contextlib import AsyncExitStack
from pathlib import Path

import mlflow
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import BaseChatMessage
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from mlflow.entities import SpanType
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from dotenv import load_dotenv

from llm import LLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run travel AutoGen router graph.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="travel_autogen_router")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_workbench(server_file: str, base_dir: Path) -> McpWorkbench:
    params = StdioServerParams(command="python", args=[server_file], cwd=base_dir)
    return McpWorkbench(params)


def contains(marker: str):
    def condition(message: BaseChatMessage) -> bool:
        return marker in str(getattr(message, "content", message))

    return condition


def always(_message: BaseChatMessage) -> bool:
    return True


@mlflow.trace(name="travel_query", span_type=SpanType.CHAIN)
async def run_router_graph(query: str, verbose: bool = False) -> str:
    base_dir = Path(__file__).resolve().parent
    llm_client = LLM().client

    router_prompt = (
        "You are router_agent. Choose exactly one next branch marker and add a concise instruction.\n"
        "Markers:\nCALL_HOTEL\nCALL_FLIGHT\nCALL_RESTAURANT\nCALL_ATTRACTIONS\nCALL_FINAL\n"
        "Never answer the user directly. Never ask follow-up questions.\n"
        "If the request includes multiple domains (for example hotel + restaurant), keep routing until each requested domain has specialist tool-grounded output.\n"
        "Use CALL_FINAL only when enough specialist output already exists for synthesis or writing."
    )
    specialist_prompt = (
        "You are a specialist worker. Handle only your domain. Use domain tools only. "
        "Never invent facts. For booking, use each tool only when needed. "
        "Always check availability before checking price and before booking. "
        "Always check price before booking. Use search only when required. "
        "Do not ask follow-up questions. If required inputs are missing, apply reasonable defaults and continue. "
        "Return concrete tool evidence including state_file and confirmation_number when present."
    )
    final_prompt = (
        "You are final_agent. You can synthesize current results and decide next step.\n"
        "If more specialist work is required, respond with ROUTE_BACK: and a short instruction to router.\n"
        "If complete, respond with OUTPUT: followed by final user response.\n"
        "If the request includes multiple domains, do not output until each requested domain has specialist evidence unless retries are clearly unproductive.\n"
        "Explicit rule: if the user asked for both hotel and restaurant, and current evidence only has hotel or only has restaurant, you MUST return ROUTE_BACK: for the missing domain.\n"
        "Do not call writer tool while any requested domain is still missing.\n"
        "Never ask follow-up questions; use reasonable defaults and proceed.\n"
        "If user asked to save itinerary, use writer tool before OUTPUT."
    )
    output_prompt = "You are output_agent. Return exactly the text after OUTPUT: as the final answer."

    async with AsyncExitStack() as stack:
        workbenches = {
            "hotel": await stack.enter_async_context(build_workbench("hotel_server.py", base_dir)),
            "flight": await stack.enter_async_context(build_workbench("flight_server.py", base_dir)),
            "restaurant": await stack.enter_async_context(build_workbench("restaurant_server.py", base_dir)),
            "attractions": await stack.enter_async_context(build_workbench("attractions_server.py", base_dir)),
            "writer": await stack.enter_async_context(build_workbench("document_server.py", base_dir)),
        }

        router = AssistantAgent(
            name="router_agent",
            model_client=llm_client,
            system_message=router_prompt,
            reflect_on_tool_use=True,
            max_tool_iterations=2,
        )
        hotel = AssistantAgent(
            name="hotel_worker",
            model_client=llm_client,
            system_message=f"{specialist_prompt} Domain: hotels.",
            workbench=[workbenches["hotel"]],
            reflect_on_tool_use=True,
            max_tool_iterations=12,
        )
        flight = AssistantAgent(
            name="flight_worker",
            model_client=llm_client,
            system_message=f"{specialist_prompt} Domain: flights.",
            workbench=[workbenches["flight"]],
            reflect_on_tool_use=True,
            max_tool_iterations=12,
        )
        restaurant = AssistantAgent(
            name="restaurant_worker",
            model_client=llm_client,
            system_message=f"{specialist_prompt} Domain: restaurants.",
            workbench=[workbenches["restaurant"]],
            reflect_on_tool_use=True,
            max_tool_iterations=12,
        )
        attractions = AssistantAgent(
            name="attractions_worker",
            model_client=llm_client,
            system_message=f"{specialist_prompt} Domain: attractions.",
            workbench=[workbenches["attractions"]],
            reflect_on_tool_use=True,
            max_tool_iterations=12,
        )
        final = AssistantAgent(
            name="final_agent",
            model_client=llm_client,
            system_message=final_prompt,
            workbench=[workbenches["writer"]],
            reflect_on_tool_use=True,
            max_tool_iterations=8,
        )
        output = AssistantAgent(
            name="output_agent",
            model_client=llm_client,
            system_message=output_prompt,
            reflect_on_tool_use=False,
            max_tool_iterations=1,
        )

        builder = DiGraphBuilder()
        builder.add_node(router, activation="any")
        for node in [hotel, flight, restaurant, attractions]:
            builder.add_node(node)
        builder.add_node(final, activation="any")
        builder.add_node(output, activation="any")
        builder.set_entry_point(router)
        builder.add_edge(router, hotel, condition=contains("CALL_HOTEL"))
        builder.add_edge(router, flight, condition=contains("CALL_FLIGHT"))
        builder.add_edge(router, restaurant, condition=contains("CALL_RESTAURANT"))
        builder.add_edge(router, attractions, condition=contains("CALL_ATTRACTIONS"))
        builder.add_edge(router, final, condition=contains("CALL_FINAL"))
        builder.add_edge(
            hotel,
            final,
            condition=always,
            activation_group="from_hotel",
            activation_condition="any",
        )
        builder.add_edge(
            flight,
            final,
            condition=always,
            activation_group="from_flight",
            activation_condition="any",
        )
        builder.add_edge(
            restaurant,
            final,
            condition=always,
            activation_group="from_restaurant",
            activation_condition="any",
        )
        builder.add_edge(
            attractions,
            final,
            condition=always,
            activation_group="from_attractions",
            activation_condition="any",
        )
        builder.add_edge(final, router, condition=contains("ROUTE_BACK:"))
        builder.add_edge(final, output, condition=contains("OUTPUT:"))

        graph = builder.build()
        team = GraphFlow(
            participants=[router, hotel, flight, restaurant, attractions, final, output],
            graph=graph,
            termination_condition=TextMentionTermination("OUTPUT:"),
            max_turns=20,
        )
        result = await run_one_query(team, query)
        messages = getattr(result, "messages", [])
        if verbose:
            print(f"[router_graph] turns={len(messages)}")
        return str(getattr(messages[-1], "content", messages[-1])) if messages else "No response."


async def run_one_query(team: GraphFlow, query: str):
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({"query": query})
    result = await team.run(task=f"User request:\n{query}")
    _log_agent_steps(result)
    return result


def _log_agent_steps(result) -> None:
    """Add lightweight per-agent spans so traces show router/branch/final flow."""
    messages = getattr(result, "messages", []) or []
    for msg in messages:
        source = getattr(msg, "source", None) or "unknown_agent"
        content = str(getattr(msg, "content", ""))
        with mlflow.start_span(f"{source}.step", span_type=SpanType.AGENT) as step_span:
            step_span.set_inputs({"source": source})
            # Keep payload compact while preserving routing/tool evidence markers.
            step_span.set_outputs({"content_preview": content[:1200]})


async def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    mlflow.autogen.autolog()
    with mlflow.start_run(run_name="travel_router_query"):
        answer = await run_router_graph(args.query, verbose=args.verbose)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
