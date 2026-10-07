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


RAG_WORKER_PROMPT = """You are rag_worker. Use only MedQuAD RAG tools for grounded medical QA. Do not route or finalize."""
DRUG_WORKER_PROMPT = """You are drug_label_worker. Use openFDA only for drug-label lookup tasks. Do not route or finalize."""
WEB_WORKER_PROMPT = """You are web_worker. Use web search only for recent external context and clearly label web-sourced information. Do not route or finalize."""
CALC_WORKER_PROMPT = """You are calc_worker. Use local calculation tools for BMI, unit conversions, and age calculations. Do not route or finalize."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the medical AutoGen orchestrator.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="medical_autogen_orch")
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


async def run_orchestrator(query: str, verbose: bool = False) -> str:
    _assert_required_environment()
    base_dir = Path(__file__).resolve().parent
    llm_client = LLM().client
    orch_prompt = (
        "You are the medical orchestrator node. Your entire response must begin with exactly one marker: "
        "CALL_RAG, CALL_DRUG, CALL_WEB, CALL_CALC, or FINAL:. "
        "Use CALL_RAG for MedQuAD medical QA, CALL_DRUG for openFDA labels, CALL_WEB for recent external context, "
        "and CALL_CALC for BMI/unit/age calculations. After each worker responds, route again until complete. "
        "FINAL: must be plain text, tool-grounded, and label web-sourced information. Never ask follow-up questions."
    )
    async with AsyncExitStack() as stack:
        workbenches = {
            "rag": await stack.enter_async_context(build_workbench("medical_rag_server.py", base_dir)),
            "external": await stack.enter_async_context(build_workbench("medical_external_server.py", base_dir)),
            "calc": await stack.enter_async_context(build_workbench("medical_calculations_server.py", base_dir)),
        }
        orchestrator = AssistantAgent("orchestrator", llm_client, system_message=orch_prompt, reflect_on_tool_use=False)
        workers = {
            "rag": AssistantAgent("rag_worker", llm_client, system_message=RAG_WORKER_PROMPT, workbench=[workbenches["rag"]], reflect_on_tool_use=True, max_tool_iterations=8),
            "drug": AssistantAgent("drug_label_worker", llm_client, system_message=DRUG_WORKER_PROMPT, workbench=[workbenches["external"]], reflect_on_tool_use=True, max_tool_iterations=6),
            "web": AssistantAgent("web_worker", llm_client, system_message=WEB_WORKER_PROMPT, workbench=[workbenches["external"]], reflect_on_tool_use=True, max_tool_iterations=6),
            "calc": AssistantAgent("calc_worker", llm_client, system_message=CALC_WORKER_PROMPT, workbench=[workbenches["calc"]], reflect_on_tool_use=True, max_tool_iterations=6),
        }
        final_agent = AssistantAgent("final_agent", llm_client, system_message="Return exactly the plain-text answer after FINAL:. No JSON, markers, or logs.")
        builder = DiGraphBuilder()
        builder.add_node(orchestrator)
        for worker in workers.values():
            builder.add_node(worker)
        builder.add_node(final_agent)
        builder.set_entry_point(orchestrator)
        builder.add_edge(orchestrator, workers["rag"], condition=contains("CALL_RAG"))
        builder.add_edge(orchestrator, workers["drug"], condition=contains("CALL_DRUG"))
        builder.add_edge(orchestrator, workers["web"], condition=contains("CALL_WEB"))
        builder.add_edge(orchestrator, workers["calc"], condition=contains("CALL_CALC"))
        builder.add_edge(orchestrator, final_agent, condition=contains("FINAL:"))
        for worker in workers.values():
            builder.add_edge(worker, orchestrator, condition=always, activation_group="worker_to_orch", activation_condition="any")
        team = GraphFlow(participants=[orchestrator, *workers.values(), final_agent], graph=builder.build(), termination_condition=TextMentionTermination("FINAL:"), max_turns=16)
        result = await run_one_query(team, query)
    messages = getattr(result, "messages", [])
    if verbose:
        print(f"[graph] turns={len(messages)}")
    return str(getattr(messages[-1], "content", messages[-1])) if messages else "No response generated."


@mlflow.trace(name="medical_query", span_type=SpanType.CHAIN)
async def run_one_query(team: GraphFlow, query: str):
    result = await team.run(task=f"User request:\n{query}")
    for msg in getattr(result, "messages", []) or []:
        source = getattr(msg, "source", "unknown_agent")
        with mlflow.start_span(f"{source}.step", span_type=SpanType.AGENT) as span:
            span.set_outputs({"content_preview": str(getattr(msg, "content", ""))[:1200]})
    return result


async def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    mlflow.autogen.autolog()
    with mlflow.start_run(run_name="medical_query"):
        answer = await run_orchestrator(args.query, verbose=args.verbose)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
