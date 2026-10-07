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
from portfolio_state import initialize_portfolio_state

MARKET_MCP_URL = "https://mcp.financialdatasets.ai/api"
PYTHON_EXE = sys.executable


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run finance AutoGen router graph.")
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--exp_name", type=str, default="finance_autogen_router")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def contains(marker: str):
    def condition(message: BaseChatMessage) -> bool:
        return marker in str(getattr(message, "content", message))

    return condition


ROUTER_PROMPT = (
    "You are the finance router. Pick the next worker by returning exactly one marker and a short instruction.\n"
    "Markers: CALL_MARKET, CALL_RETRIEVAL, CALL_ACTIONS, CALL_CALCULATIONS, CALL_FINAL.\n"
    "Never answer the user directly. Never ask follow-up questions.\n"
    "If the user gives explicit trade intent with ticker and quantity, route to CALL_ACTIONS first.\n"
    "Use CALL_RETRIEVAL for cash, list holdings, and holding details.\n"
    "Use CALL_CALCULATIONS for position or portfolio value calculations.\n"
    "Use CALL_MARKET only for external market data requests.\n"
    "Use CALL_FINAL only when the conversation already contains all tool-grounded facts needed for the answer."
)

MARKET_PROMPT = (
    "You are the market worker. Use only market MCP tools. "
    "Return concise tool-grounded facts. Do not ask the user for data and do not invent market facts."
)

RETRIEVAL_PROMPT = (
    "You are the portfolio retrieval worker. Use only portfolio retrieval MCP tools. "
    "For cash requests, call get_available_cash. For holdings lists, call list_owned_stocks. "
    "For a specific holding, call get_stock_holding_details. "
    "If the instruction asks for multiple facts, call every required tool before answering. "
    "Return concise tool results only. Do not ask follow-up questions and do not invent portfolio facts."
)

ACTIONS_PROMPT = (
    "You are the portfolio actions worker. Use portfolio actions MCP tools for trades and portfolio retrieval MCP tools for post-trade facts. "
    "When ticker and quantity are present, execute buy_stock or sell_stock directly. "
    "Do not ask for confirmation, market price, cash balance, or current holdings; the tool owns that state. "
    "After a successful trade, satisfy every requested post-trade read before answering: "
    "for updated cash call get_available_cash, for holdings list call list_owned_stocks, "
    "and for affected ticker details call get_stock_holding_details. "
    "Return concise tool results covering the executed action and requested updated portfolio facts. "
    "Do not invent results."
)

CALCULATIONS_PROMPT = (
    "You are the portfolio calculations worker. Use only portfolio calculations MCP tools. "
    "Call the relevant calculation tool and return concise tool-grounded facts only. "
    "Do not ask follow-up questions and do not invent values."
)

FINAL_PROMPT = (
    "You are the finance finalizer. Decide whether more worker work is required.\n"
    "Return exactly one of:\n"
    "ROUTE_BACK: <instruction for the next missing worker>\n"
    "OUTPUT: <plain text answer>\n"
    "Never ask follow-up questions. Never use placeholders or assumptions.\n"
    "Every portfolio or market claim in OUTPUT must be grounded in prior worker/tool results.\n"
    "For buy/sell requests, do not output until an actions worker has executed the trade. "
    "If the user also asked for updated cash, holdings list, or holding details and those facts are missing after the trade, "
    "return ROUTE_BACK with a retrieval instruction for the missing facts.\n"
    "If required tool results are missing, return ROUTE_BACK rather than answering from general knowledge."
)


@mlflow.trace(name="finance_query", span_type=SpanType.CHAIN)
async def run_router_graph(query: str, verbose: bool = False) -> str:
    os.environ["FINANCE_PORTFOLIO_RUN_KEY"] = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
    initialize_portfolio_state()
    base_dir = Path(__file__).resolve().parent
    llm = LLM().client
    key = os.getenv("FINANCIAL_DATASET_API_KEY")
    if not key:
        raise ValueError("Missing required API key env var: FINANCIAL_DATASET_API_KEY")
    async with AsyncExitStack() as stack:
        wb_market = await stack.enter_async_context(
            McpWorkbench(StreamableHttpServerParams(url=MARKET_MCP_URL, headers={"X-API-KEY": key}))
        )
        wb_retrieval = await stack.enter_async_context(
            McpWorkbench(StdioServerParams(command=PYTHON_EXE, args=["portfolio_retrieval_server.py"], cwd=base_dir))
        )
        wb_actions = await stack.enter_async_context(
            McpWorkbench(StdioServerParams(command=PYTHON_EXE, args=["portfolio_actions_server.py"], cwd=base_dir))
        )
        wb_calcs = await stack.enter_async_context(
            McpWorkbench(StdioServerParams(command=PYTHON_EXE, args=["portfolio_calculations_server.py"], cwd=base_dir))
        )

        router = AssistantAgent("router_agent", llm, system_message=ROUTER_PROMPT, reflect_on_tool_use=False, max_tool_iterations=1)
        market = AssistantAgent("market_worker", llm, system_message=MARKET_PROMPT, workbench=[wb_market], reflect_on_tool_use=True, max_tool_iterations=8)
        retrieval = AssistantAgent("retrieval_worker", llm, system_message=RETRIEVAL_PROMPT, workbench=[wb_retrieval], reflect_on_tool_use=True, max_tool_iterations=8)
        actions = AssistantAgent("actions_worker", llm, system_message=ACTIONS_PROMPT, workbench=[wb_actions, wb_retrieval], reflect_on_tool_use=True, max_tool_iterations=10)
        calcs = AssistantAgent("calculations_worker", llm, system_message=CALCULATIONS_PROMPT, workbench=[wb_calcs], reflect_on_tool_use=True, max_tool_iterations=8)
        final = AssistantAgent(
            "final_agent",
            llm,
            system_message=FINAL_PROMPT,
            reflect_on_tool_use=False,
            max_tool_iterations=1,
        )
        output = AssistantAgent("output_agent", llm, system_message="Return exactly text after OUTPUT:", reflect_on_tool_use=False, max_tool_iterations=1)

        b = DiGraphBuilder()
        for n in [router, market, retrieval, actions, calcs, final, output]:
            b.add_node(n)
        b.set_entry_point(router)
        b.add_edge(router, market, condition=contains("CALL_MARKET"))
        b.add_edge(router, retrieval, condition=contains("CALL_RETRIEVAL"))
        b.add_edge(router, actions, condition=contains("CALL_ACTIONS"))
        b.add_edge(router, calcs, condition=contains("CALL_CALCULATIONS"))
        b.add_edge(router, final, condition=contains("CALL_FINAL"))
        for n in [market, retrieval, actions, calcs]:
            b.add_edge(n, final)
        b.add_edge(final, router, condition=contains("ROUTE_BACK:"))
        b.add_edge(final, output, condition=contains("OUTPUT:"))

        team = GraphFlow(
            participants=[router, market, retrieval, actions, calcs, final, output],
            graph=b.build(),
            termination_condition=TextMentionTermination("OUTPUT:"),
            max_turns=22,
        )
        result = await team.run(task=f"User request:\n{query}")
        msgs = getattr(result, "messages", [])
        if verbose:
            print(f"[router_graph] turns={len(msgs)}")
        return str(getattr(msgs[-1], "content", msgs[-1])) if msgs else "No response."


async def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    mlflow.autogen.autolog()
    with mlflow.start_run(run_name="finance_router_query"):
        print(await run_router_graph(args.query, verbose=args.verbose))


if __name__ == "__main__":
    asyncio.run(main())
