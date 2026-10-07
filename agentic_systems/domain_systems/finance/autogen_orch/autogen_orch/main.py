import argparse
import asyncio
import hashlib
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

import mlflow
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.messages import BaseChatMessage
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams, StreamableHttpServerParams
from dotenv import load_dotenv
from mlflow.entities import SpanType

from llm import LLM

MARKET_MCP_URL = "https://mcp.financialdatasets.ai/api"
PYTHON_EXE = sys.executable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run finance AutoGen orchestrator.")
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--exp_name", type=str, default="finance_autogen_orchestrator")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def contains(marker: str):
    def condition(message: BaseChatMessage) -> bool:
        return marker in str(getattr(message, "content", message))

    return condition


def always(_message: BaseChatMessage) -> bool:
    return True


@mlflow.trace(name="finance_query", span_type=SpanType.CHAIN)
async def run_orchestrator(query: str, verbose: bool = False) -> str:
    os.environ["FINANCE_PORTFOLIO_RUN_KEY"] = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
    base_dir = Path(__file__).resolve().parent
    llm = LLM().client
    key = os.getenv("FINANCIAL_DATASET_API_KEY")
    if not key:
        raise ValueError("Missing required API key env var: FINANCIAL_DATASET_API_KEY")

    async with AsyncExitStack() as stack:
        workbenches = {
            "market": await stack.enter_async_context(
                McpWorkbench(StreamableHttpServerParams(url=MARKET_MCP_URL, headers={"X-API-KEY": key}))
            ),
            "retrieval": await stack.enter_async_context(
                McpWorkbench(StdioServerParams(command=PYTHON_EXE, args=["portfolio_retrieval_server.py"], cwd=base_dir))
            ),
            "actions": await stack.enter_async_context(
                McpWorkbench(StdioServerParams(command=PYTHON_EXE, args=["portfolio_actions_server.py"], cwd=base_dir))
            ),
            "calculations": await stack.enter_async_context(
                McpWorkbench(StdioServerParams(command=PYTHON_EXE, args=["portfolio_calculations_server.py"], cwd=base_dir))
            ),
        }

        orchestrator = AssistantAgent(
            name="orchestrator",
            model_client=llm,
            system_message=(
                "You are finance orchestrator. Route work and regain control after each worker. "
                "Use exactly one marker per turn: CALL_MARKET/CALL_RETRIEVAL/CALL_ACTIONS/CALL_CALCULATIONS/FINAL:. "
                "FINAL output must be plain text only. "
                "Never ask follow-up questions. "
                "If the user provides explicit trade intent with ticker and quantity, route to CALL_ACTIONS and execute it directly; do not ask for confirmation. "
                "For buy/sell requests, do not produce FINAL: until both steps are done in order: "
                "1) CALL_ACTIONS to execute trade, 2) CALL_RETRIEVAL to fetch updated cash and affected holding details. "
                "After these two steps, produce FINAL: with the completed result only. "
                "Do not include offers like 'let me know if you want me to retrieve'."
            ),
            reflect_on_tool_use=True,
            max_tool_iterations=2,
        )
        workers = {
            "market": AssistantAgent("market_worker", llm, system_message="Use only market MCP tools.", workbench=[workbenches["market"]], reflect_on_tool_use=True, max_tool_iterations=8),
            "retrieval": AssistantAgent("retrieval_worker", llm, system_message="Use only portfolio retrieval MCP tools.", workbench=[workbenches["retrieval"]], reflect_on_tool_use=True, max_tool_iterations=8),
            "actions": AssistantAgent(
                "actions_worker",
                llm,
                system_message=(
                    "Use only portfolio actions MCP tools. "
                    "Execute requested buy/sell actions directly when ticker and quantity are present. "
                    "Do not ask for confirmation. "
                    "Return action execution result only. "
                    "Do not perform post-trade retrieval; that must be handled by retrieval_worker in a separate orchestrator step. "
                    "Do not invent results."
                ),
                workbench=[workbenches["actions"]],
                reflect_on_tool_use=True,
                max_tool_iterations=8,
            ),
            "calculations": AssistantAgent("calculations_worker", llm, system_message="Use only portfolio calculations MCP tools.", workbench=[workbenches["calculations"]], reflect_on_tool_use=True, max_tool_iterations=8),
        }
        final_agent = AssistantAgent(
            name="final_agent",
            model_client=llm,
            system_message="Return only the plain-text answer after FINAL: from orchestrator.",
            reflect_on_tool_use=False,
            max_tool_iterations=1,
        )

        b = DiGraphBuilder()
        b.add_node(orchestrator)
        for w in workers.values():
            b.add_node(w)
        b.add_node(final_agent)
        b.set_entry_point(orchestrator)
        b.add_edge(orchestrator, workers["market"], condition=contains("CALL_MARKET"))
        b.add_edge(orchestrator, workers["retrieval"], condition=contains("CALL_RETRIEVAL"))
        b.add_edge(orchestrator, workers["actions"], condition=contains("CALL_ACTIONS"))
        b.add_edge(orchestrator, workers["calculations"], condition=contains("CALL_CALCULATIONS"))
        b.add_edge(orchestrator, final_agent, condition=contains("FINAL:"))
        for w in workers.values():
            b.add_edge(w, orchestrator, condition=always)

        team = GraphFlow(
            participants=[orchestrator, *workers.values(), final_agent],
            graph=b.build(),
            termination_condition=TextMentionTermination("FINAL:"),
            max_turns=18,
        )
        result = await run_one_query(team, query)
        msgs = getattr(result, "messages", [])
        if verbose:
            print(f"[graph] turns={len(msgs)}")
        return str(getattr(msgs[-1], "content", msgs[-1])) if msgs else "No response."


async def run_one_query(team: GraphFlow, query: str):
    return await team.run(task=f"User request:\n{query}")


async def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    mlflow.autogen.autolog()
    with mlflow.start_run(run_name="finance_query"):
        print(await run_orchestrator(args.query, verbose=args.verbose))


if __name__ == "__main__":
    asyncio.run(main())
