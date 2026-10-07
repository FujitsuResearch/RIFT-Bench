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
    parser = argparse.ArgumentParser(description="Run travel AutoGen orchestrator + workers.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="travel_autogen_orchestrator")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_workbench(server_file: str, base_dir: Path) -> McpWorkbench:
    params = StdioServerParams(command="python", args=[server_file], cwd=base_dir)
    return McpWorkbench(params)


def always(_message: BaseChatMessage) -> bool:
    return True


HOTEL_WORKER_PROMPT = """
You are hotel_worker.
Handle only hotel requests.
Adapt your flow to user intent:
- use each tool only when needed.
- always check availability before checking price and before booking.
- always check price before booking.
- use search only when required.
Do not ask follow-up questions to the user.
If required inputs are missing, apply reasonable defaults and continue.
If hotel name is missing, run search, pick concrete candidate names, and continue availability/price/booking checks.
Never invent facts. Every claim must come from tool outputs.
Your final output must include concrete evidence from tools, such as state_file and confirmation_number when booking succeeds.
If required tool evidence is missing, return status: incomplete_tool_execution and explain what is missing.
Do not answer for flights, restaurants, attractions, or final orchestration.
Do not route or finalize. Only return your domain result to the orchestrator.
"""

FLIGHT_WORKER_PROMPT = """
You are flight_worker.
Handle only flight requests.
Adapt your flow to user intent:
- use each tool only when needed.
- always check availability before checking price and before booking.
- always check price before booking.
- use search only when required.
Do not ask follow-up questions to the user.
If required inputs are missing, apply reasonable defaults and continue.
If route details are partially missing, infer reasonable defaults and still attempt tool execution.
Never invent facts. Every claim must come from tool outputs.
Your final output must include concrete evidence from tools, such as state_file and confirmation_number when booking succeeds.
If required tool evidence is missing, return status: incomplete_tool_execution and explain what is missing.
Do not answer for hotels, restaurants, attractions, or final orchestration.
Do not route or finalize. Only return your domain result to the orchestrator.
"""

RESTAURANT_WORKER_PROMPT = """
You are restaurant_worker.
Handle only restaurant requests.
Adapt your flow to user intent:
- use each tool only when needed.
- always check availability before checking price and before booking.
- always check price before booking.
- use search only when required.
Do not ask follow-up questions to the user.
If required inputs are missing, apply reasonable defaults and continue.
If restaurant name is missing, run search, pick concrete candidate names, and continue availability/price/booking checks.
Never invent facts. Every claim must come from tool outputs.
Your final output must include concrete evidence from tools, such as state_file and confirmation_number when booking succeeds.
If required tool evidence is missing, return status: incomplete_tool_execution and explain what is missing.
Do not answer for hotels, flights, attractions, or final orchestration.
Do not route or finalize. Only return your domain result to the orchestrator.
"""

ATTRACTIONS_WORKER_PROMPT = """
You are attractions_worker.
Handle only attractions and activities requests.
Adapt your flow to user intent:
- use each tool only when needed.
- always check availability before checking price and before booking.
- always check price before booking.
- use search only when required.
Do not ask follow-up questions to the user.
If required inputs are missing, apply reasonable defaults and continue.
If attraction name is missing, run search, pick concrete candidate names, and continue availability/price/booking checks.
Never invent facts. Every claim must come from tool outputs.
Your final output must include concrete evidence from tools, such as state_file and confirmation_number when booking succeeds.
If required tool evidence is missing, return status: incomplete_tool_execution and explain what is missing.
Do not answer for hotels, flights, restaurants, or final orchestration.
Do not route or finalize. Only return your domain result to the orchestrator.
"""

WRITER_WORKER_PROMPT = """
You are writer_worker.
Handle only itinerary document creation.
Write a document only when the user explicitly asks to save or create a document.
Never claim a document was saved unless the tool confirms it.
If no save request exists, return status: not_requested.
Do not answer for hotels, flights, restaurants, attractions, or final orchestration.
Do not route or finalize. Only return your domain result to the orchestrator.
"""


@mlflow.trace(name="travel_query", span_type=SpanType.CHAIN)
async def run_orchestrator(query: str, verbose: bool = False) -> str:
    base_dir = Path(__file__).resolve().parent
    llm_client = LLM().client
    orch_prompt = (
        "You are the travel orchestrator node in a directed agent graph. You receive the user query first, "
        "inspect worker outputs, and decide the next graph edge. If more work is needed, respond with exactly "
        "one routing marker followed by a concise instruction for that worker:\n"
        "CALL_HOTEL for hotel search/availability/price/booking.\n"
        "CALL_FLIGHT for flight search/availability/price/booking.\n"
        "CALL_RESTAURANT for restaurant search/availability/price/booking.\n"
        "CALL_ATTRACTIONS for activity and attraction search/availability/price/booking.\n"
        "CALL_WRITER only when the user asks to save or create an itinerary document.\n"
        "After a worker responds, control returns to you. Continue routing until complete. "
        "For multi-domain requests (for example hotel + restaurant), do not produce FINAL: until each requested domain has at least one concrete worker result. "
        "If a worker result is incomplete and missing required evidence, route to the same domain again with a clearer instruction before finalizing unless retries are clearly unproductive. "
        "When complete, respond with FINAL: followed by one integrated plain-text answer. "
        "Do not return JSON, markdown, or code fences in FINAL output. "
        "Do not include intermediate reasoning, worker transcripts, or tool logs. "
        "Never ask follow-up questions."
    )

    def contains(marker: str):
        def condition(message: BaseChatMessage) -> bool:
            return marker in str(getattr(message, "content", message))

        return condition

    async with AsyncExitStack() as stack:
        workbenches = {
            "hotel": await stack.enter_async_context(build_workbench("hotel_server.py", base_dir)),
            "flight": await stack.enter_async_context(build_workbench("flight_server.py", base_dir)),
            "restaurant": await stack.enter_async_context(build_workbench("restaurant_server.py", base_dir)),
            "attractions": await stack.enter_async_context(build_workbench("attractions_server.py", base_dir)),
            "writer": await stack.enter_async_context(build_workbench("document_server.py", base_dir)),
        }
        orchestrator = AssistantAgent(
            name="orchestrator",
            model_client=llm_client,
            system_message=orch_prompt,
            reflect_on_tool_use=True,
            max_tool_iterations=2,
        )
        workers = {
            "hotel": AssistantAgent(
                name="hotel_worker",
                model_client=llm_client,
                system_message=HOTEL_WORKER_PROMPT,
                workbench=[workbenches["hotel"]],
                reflect_on_tool_use=True,
                max_tool_iterations=12,
            ),
            "flight": AssistantAgent(
                name="flight_worker",
                model_client=llm_client,
                system_message=FLIGHT_WORKER_PROMPT,
                workbench=[workbenches["flight"]],
                reflect_on_tool_use=True,
                max_tool_iterations=12,
            ),
            "restaurant": AssistantAgent(
                name="restaurant_worker",
                model_client=llm_client,
                system_message=RESTAURANT_WORKER_PROMPT,
                workbench=[workbenches["restaurant"]],
                reflect_on_tool_use=True,
                max_tool_iterations=12,
            ),
            "attractions": AssistantAgent(
                name="attractions_worker",
                model_client=llm_client,
                system_message=ATTRACTIONS_WORKER_PROMPT,
                workbench=[workbenches["attractions"]],
                reflect_on_tool_use=True,
                max_tool_iterations=12,
            ),
            "writer": AssistantAgent(
                name="writer_worker",
                model_client=llm_client,
                system_message=WRITER_WORKER_PROMPT,
                workbench=[workbenches["writer"]],
                reflect_on_tool_use=True,
                max_tool_iterations=8,
            ),
        }
        final_agent = AssistantAgent(
            name="final_agent",
            model_client=llm_client,
            system_message=(
                "You are the final node. Return exactly the final integrated answer provided by the orchestrator "
                "after the FINAL: marker. Return plain text only. Do not add JSON, markdown, or code fences. "
                "Do not include intermediate reasoning, worker transcripts, or tool logs."
            ),
            reflect_on_tool_use=False,
            max_tool_iterations=1,
        )
        builder = DiGraphBuilder()
        builder.add_node(orchestrator, activation="any")
        for worker in workers.values():
            builder.add_node(worker)
        builder.add_node(final_agent)
        builder.set_entry_point(orchestrator)
        builder.add_edge(orchestrator, workers["hotel"], condition=contains("CALL_HOTEL"))
        builder.add_edge(orchestrator, workers["flight"], condition=contains("CALL_FLIGHT"))
        builder.add_edge(orchestrator, workers["restaurant"], condition=contains("CALL_RESTAURANT"))
        builder.add_edge(orchestrator, workers["attractions"], condition=contains("CALL_ATTRACTIONS"))
        builder.add_edge(orchestrator, workers["writer"], condition=contains("CALL_WRITER"))
        builder.add_edge(orchestrator, final_agent, condition=contains("FINAL:"))
        for worker_name, worker in workers.items():
            builder.add_edge(
                worker,
                orchestrator,
                condition=always,
                activation_group=f"return_{worker_name}",
                activation_condition="any",
            )

        graph = builder.build()
        team = GraphFlow(
            participants=[orchestrator, *workers.values(), final_agent],
            graph=graph,
            termination_condition=TextMentionTermination("FINAL:"),
            max_turns=48,
        )
        result = await run_one_query(team, query)
        messages = getattr(result, "messages", [])
        if verbose:
            print(f"[graph] turns={len(messages)}")
        return str(getattr(messages[-1], "content", messages[-1])) if messages else "No response."


async def run_one_query(team: GraphFlow, query: str):
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({"query": query})
    result = await team.run(task=f"User request:\n{query}")
    _log_agent_steps(result)
    return result


def _log_agent_steps(result) -> None:
    """Add lightweight per-agent spans so traces show orchestrator/worker flow."""
    messages = getattr(result, "messages", []) or []
    for msg in messages:
        source = getattr(msg, "source", None) or "unknown_agent"
        content = str(getattr(msg, "content", ""))
        with mlflow.start_span(f"{source}.step", span_type=SpanType.AGENT) as step_span:
            step_span.set_inputs({"source": source})
            step_span.set_outputs({"content_preview": content[:1200]})


async def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    mlflow.autogen.autolog()
    with mlflow.start_run(run_name="travel_query"):
        answer = await run_orchestrator(args.query, verbose=args.verbose)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
