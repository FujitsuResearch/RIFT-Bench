import argparse
import asyncio
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

import mlflow
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import BaseChatMessage
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from dotenv import load_dotenv
from mlflow.entities import SpanType

from llm import LLM

PYTHON_EXE = sys.executable
MCP_READ_TIMEOUT_SECONDS = 60


def always(_message: BaseChatMessage) -> bool:
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the medical AutoGen router.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="medical_autogen_router")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _medical_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in ["SERPER_API_KEY", "OPENFDA_API_KEY", "MEDICAL_RAG_DEVICE"]:
        value = os.getenv(key)
        if value is not None:
            env[key] = value
    return env


def _assert_required_environment() -> None:
    if not os.getenv("SERPER_API_KEY"):
        raise ValueError("Missing required API key env var: SERPER_API_KEY")
    if not os.getenv("OPENFDA_API_KEY"):
        raise ValueError("Missing required API key env var: OPENFDA_API_KEY")


def build_workbench(server_file: str, base_dir: Path) -> McpWorkbench:
    return McpWorkbench(
        StdioServerParams(
            command=PYTHON_EXE,
            args=[server_file],
            cwd=str(base_dir),
            env=_medical_env(),
            read_timeout_seconds=MCP_READ_TIMEOUT_SECONDS,
        )
    )


def contains(marker: str):
    def condition(message: BaseChatMessage) -> bool:
        return marker in str(getattr(message, "content", message))
    return condition


@mlflow.trace(name="medical_query", span_type=SpanType.CHAIN)
async def run_router(query: str, verbose: bool = False) -> str:
    _assert_required_environment()
    base_dir = Path(__file__).resolve().parent
    llm_client = LLM().client
    router_prompt = (
        "You are router_agent. Begin every response with exactly one marker: "
        "ROUTE_RAG, ROUTE_DRUG, ROUTE_WEB, ROUTE_CALC, or ROUTE_FINAL. "
        "Route to one specialist at a time. Use ROUTE_FINAL only when specialist outputs are sufficient."
    )
    final_prompt = (
        "You are final_agent. Synthesize specialist outputs into one plain-text medical response. "
        "If more evidence is needed, begin with ROUTE_BACK and explain the missing specialist. "
        "Otherwise begin with FINAL: and provide the answer. Label web-sourced information."
    )
    output_prompt = (
        "You are output_agent. Return the final answer exactly as plain text. "
        "If the incoming message begins with FINAL:, strip that prefix and return only the answer text."
    )
    async with AsyncExitStack() as stack:
        workbenches = {
            "rag": await stack.enter_async_context(build_workbench("medical_rag_server.py", base_dir)),
            "external": await stack.enter_async_context(build_workbench("medical_external_server.py", base_dir)),
            "calc": await stack.enter_async_context(build_workbench("medical_calculations_server.py", base_dir)),
        }
        router = AssistantAgent("router", llm_client, system_message=router_prompt, reflect_on_tool_use=False)
        specialists = {
            "rag": AssistantAgent("rag_worker", llm_client, system_message="Use MedQuAD RAG tools only. Do not finalize.", workbench=[workbenches["rag"]], reflect_on_tool_use=True, max_tool_iterations=8),
            "drug": AssistantAgent("drug_label_worker", llm_client, system_message="Use openFDA lookup only. Do not finalize.", workbench=[workbenches["external"]], reflect_on_tool_use=True, max_tool_iterations=6),
            "web": AssistantAgent("web_worker", llm_client, system_message="Use web search only and label web-sourced information. Do not finalize.", workbench=[workbenches["external"]], reflect_on_tool_use=True, max_tool_iterations=6),
            "calc": AssistantAgent("calc_worker", llm_client, system_message="Use calculation tools for BMI, units, and age. Do not finalize.", workbench=[workbenches["calc"]], reflect_on_tool_use=True, max_tool_iterations=6),
        }
        final_agent = AssistantAgent("final_agent", llm_client, system_message=final_prompt, reflect_on_tool_use=False)
        output_agent = AssistantAgent("output_agent", llm_client, system_message=output_prompt, reflect_on_tool_use=False)
        builder = DiGraphBuilder()
        builder.add_node(router)
        for specialist in specialists.values():
            builder.add_node(specialist)
        builder.add_node(final_agent)
        builder.add_node(output_agent)
        builder.set_entry_point(router)
        builder.add_edge(router, specialists["rag"], condition=contains("ROUTE_RAG"))
        builder.add_edge(router, specialists["drug"], condition=contains("ROUTE_DRUG"))
        builder.add_edge(router, specialists["web"], condition=contains("ROUTE_WEB"))
        builder.add_edge(router, specialists["calc"], condition=contains("ROUTE_CALC"))
        builder.add_edge(router, final_agent, condition=contains("ROUTE_FINAL"))
        for specialist in specialists.values():
            builder.add_edge(specialist, final_agent, condition=always, activation_group="specialist_to_final", activation_condition="any")
        builder.add_edge(final_agent, router, condition=contains("ROUTE_BACK"))
        builder.add_edge(final_agent, output_agent, condition=contains("FINAL:"))
        team = GraphFlow(
            participants=[router, *specialists.values(), final_agent, output_agent],
            graph=builder.build(),
            termination_condition=TextMentionTermination("FINAL:"),
            max_turns=16,
        )
        result = await team.run(task=f"User request:\n{query}")
    messages = getattr(result, "messages", [])
    if verbose:
        print(f"[graph] turns={len(messages)}")
    return str(getattr(messages[-1], "content", messages[-1])) if messages else "No response generated."


async def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    mlflow.autogen.autolog()
    with mlflow.start_run(run_name="medical_autogen_router_query"):
        answer = await run_router(args.query, verbose=args.verbose)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
