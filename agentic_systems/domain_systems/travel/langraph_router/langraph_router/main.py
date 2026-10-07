import argparse
import asyncio
import json
from pathlib import Path
from typing import Literal, TypedDict

import mlflow
try:
    import mlflow.langchain as mlflow_langchain
except Exception:
    mlflow_langchain = None
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph

from llm import LLM
from local_tools import write_travel_plan_document

RouteName = Literal["hotel", "flight", "restaurant", "attractions", "final", "output"]
ActionName = Literal["route_back", "output"]


class RouterGraphState(TypedDict, total=False):
    query: str
    route: RouteName
    worker_query: str
    worker_outputs: list[str]
    final_answer: str
    action: ActionName
    loops: int


ROUTER_PROMPT = """You are router_agent.
Choose one node: hotel|flight|restaurant|attractions|final.
Return only JSON: {"route":"...","worker_query":"..."}
Rules:
- Route to exactly one specialist when domain work is still needed.
- If query requests multiple domains, keep routing until each requested domain has at least one specialist output.
- Use final only when specialist outputs are sufficient to produce a complete user-facing answer.
- worker_query must be an imperative instruction for the selected specialist.
- Include known details from the original request in worker_query (location, date(s), party size, intent).
- If required details are missing, fill them with reasonable defaults in worker_query and state they are assumptions.
- Never ask the user follow-up questions in router_query.
"""

FINAL_PROMPT = """You are final_agent.
Return only JSON: {"action":"route_back|output","router_query":"...","final_answer":"..."}
Rules:
- If more specialist work is needed, return action=route_back with a concrete router_query instruction; keep final_answer empty.
- If you can answer the full request, return action=output with plain-text final_answer.
- Never return route/control JSON to the end user in final_answer.
- If query requested multiple domains, do not output until each requested domain has specialist coverage, unless further retries are clearly unproductive.
- Never ask the user follow-up questions; use reasonable defaults and proceed.
- For action=route_back, router_query must include known request details and fill missing operational fields with explicit assumptions.
- If user asks to save itinerary, call write_travel_plan_document before output.
final_answer requirements:
- Plain text only.
- Directly answers the user request.
- Includes concrete confirmations/evidence when available.
- If something failed, state that clearly in user-facing prose.
Never invent facts.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run travel LangGraph router graph.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="travel_langraph_router")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def parse_json(content: str, fallback: dict) -> dict:
    try:
        start = content.index("{")
        end = content.rindex("}") + 1
        return json.loads(content[start:end])
    except (ValueError, json.JSONDecodeError):
        return fallback


async def run_router_graph(query: str, verbose: bool = False) -> str:
    base_dir = Path(__file__).resolve().parent
    llm = LLM().client
    mcp_client = MultiServerMCPClient(
        {
            "hotel": {"transport": "stdio", "command": "python", "args": ["hotel_server.py"], "cwd": str(base_dir)},
            "flight": {"transport": "stdio", "command": "python", "args": ["flight_server.py"], "cwd": str(base_dir)},
            "restaurant": {"transport": "stdio", "command": "python", "args": ["restaurant_server.py"], "cwd": str(base_dir)},
            "attractions": {"transport": "stdio", "command": "python", "args": ["attractions_server.py"], "cwd": str(base_dir)},
        }
    )
    mcp_tools = await mcp_client.get_tools()
    hotel_tools = [t for t in mcp_tools if "hotel" in t.name]
    flight_tools = [t for t in mcp_tools if "flight" in t.name]
    restaurant_tools = [t for t in mcp_tools if "restaurant" in t.name]
    attractions_tools = [t for t in mcp_tools if "attraction" in t.name]

    async def router_node(state: RouterGraphState) -> dict:
        outputs = "\n\n".join(state.get("worker_outputs", [])) or "No specialist outputs yet."
        agent = create_agent(model=llm, tools=[], system_prompt=ROUTER_PROMPT)
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": f"Query:\n{state['query']}\n\nKnown outputs:\n{outputs}"}]}
        )
        messages = response.get("messages", [])
        raw = str(messages[-1].content) if messages else ""
        decision = parse_json(raw, {"route": "final", "worker_query": state["query"]})
        route = decision.get("route", "final")
        if route not in {"hotel", "flight", "restaurant", "attractions", "final"}:
            route = "final"
        if verbose:
            print(f"[router] route={route}")
        return {"route": route, "worker_query": str(decision.get("worker_query", state["query"]))}

    async def run_specialist(name: str, tools, state: RouterGraphState) -> dict:
        prompt = (
            f"You are {name}_worker. Handle only your domain. "
            "Choose workflow based on request. Use each tool only when needed. "
            "Always check availability before checking price and before booking. "
            "Always check price before booking. Use search only when required. "
            "Never invent facts. Include tool evidence (state_file/confirmation) when available. "
            "Do not ask the user follow-up questions. "
            "If required tool inputs are missing, infer reasonable defaults, proceed with tool calls, "
            "and clearly state assumptions in the output. "
            "Return incomplete_tool_execution only after at least one concrete tool attempt fails or required fields are truly unavailable from context."
        )
        agent = create_agent(model=llm, tools=tools, system_prompt=prompt)
        task = state.get("worker_query") or state["query"]
        resp = await agent.ainvoke({"messages": [{"role": "user", "content": f"User request:\n{task}"}]})
        msgs = resp.get("messages", [])
        out = str(msgs[-1].content) if msgs else "No response."
        if verbose:
            print(f"[{name}] done")
        prev = state.get("worker_outputs", [])
        return {"worker_outputs": [*prev, f"{name}_output:\n{out}"], "route": "final"}

    async def hotel_node(state: RouterGraphState) -> dict:
        return await run_specialist("hotel", hotel_tools, state)

    async def flight_node(state: RouterGraphState) -> dict:
        return await run_specialist("flight", flight_tools, state)

    async def restaurant_node(state: RouterGraphState) -> dict:
        return await run_specialist("restaurant", restaurant_tools, state)

    async def attractions_node(state: RouterGraphState) -> dict:
        return await run_specialist("attractions", attractions_tools, state)

    async def final_node(state: RouterGraphState) -> dict:
        loops = state.get("loops", 0) + 1
        outputs = "\n\n".join(state.get("worker_outputs", [])) or "No specialist outputs yet."
        agent = create_agent(model=llm, tools=[write_travel_plan_document], system_prompt=FINAL_PROMPT)
        resp = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Original query:\n{state['query']}\n\nSpecialist outputs:\n{outputs}",
                    }
                ]
            }
        )
        msgs = resp.get("messages", [])
        raw = str(msgs[-1].content) if msgs else ""
        decision = parse_json(raw, {"action": "output", "router_query": "", "final_answer": raw})
        action = decision.get("action", "output")
        if action not in {"route_back", "output"}:
            action = "output"
        router_query = str(decision.get("router_query", state["query"]))
        final_answer = str(decision.get("final_answer", "")).strip()
        if loops >= 8 and not final_answer:
            action = "output"
            final_answer = (
                "I could not complete all requested travel actions after repeated attempts. "
                "Here are the latest confirmed results from available tool outputs:\n\n"
                f"{outputs}"
            )
        if verbose:
            print(f"[final] action={action}")
        if action == "route_back":
            return {
                "action": "route_back",
                "query": router_query,
                "final_answer": "",
                "route": "router",
                "loops": loops,
            }
        return {
            "action": "output",
            "query": state["query"],
            "final_answer": final_answer or raw,
            "route": "output",
            "loops": loops,
        }

    async def output_node(state: RouterGraphState) -> dict:
        return {"final_answer": state.get("final_answer", "")}

    def route_from_router(state: RouterGraphState) -> RouteName:
        return state.get("route", "final")

    def route_from_final(state: RouterGraphState) -> str:
        return "router" if state.get("action") == "route_back" else "output"

    graph = StateGraph(RouterGraphState)
    graph.add_node("router", router_node)
    graph.add_node("hotel", hotel_node)
    graph.add_node("flight", flight_node)
    graph.add_node("restaurant", restaurant_node)
    graph.add_node("attractions", attractions_node)
    graph.add_node("final", final_node)
    graph.add_node("output", output_node)
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_from_router,
        {"hotel": "hotel", "flight": "flight", "restaurant": "restaurant", "attractions": "attractions", "final": "final"},
    )
    for name in ["hotel", "flight", "restaurant", "attractions"]:
        graph.add_edge(name, "final")
    graph.add_conditional_edges("final", route_from_final, {"router": "router", "output": "output"})
    graph.add_edge("output", END)

    app = graph.compile()
    result = await app.ainvoke({"query": query, "worker_outputs": [], "loops": 0})
    return result.get("final_answer") or "No response generated."


async def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    if mlflow_langchain is not None:
        mlflow_langchain.autolog(log_traces=True, silent=True)
    with mlflow.start_run():
        answer = await run_router_graph(args.query, verbose=args.verbose)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
