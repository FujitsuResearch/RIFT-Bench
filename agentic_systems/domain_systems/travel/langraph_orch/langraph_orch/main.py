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

WorkerName = Literal["hotel", "flight", "restaurant", "attractions", "writer"]
RouteName = Literal["hotel", "flight", "restaurant", "attractions", "writer", "final"]

ORCH_PROMPT = """You are a travel orchestrator node.
You receive the user query first, inspect worker outputs, and decide the next graph node.
Return only JSON with this schema:
{"next_worker": "hotel|flight|restaurant|attractions|writer|final", "worker_query": "...", "final_answer": "..."}

Use hotel for hotel search/availability/price/booking.
Use flight for flight search/availability/price/booking.
Use restaurant for restaurant search/availability/price/booking.
Use attractions for activity and attraction search/availability/price/booking.
Use writer only when the user explicitly asks to save or create an itinerary document.
If more work is needed, set final_answer to an empty string.
If the request is complete, set next_worker to "final" and provide final_answer as plain text only.
Do not put JSON, markdown, or code fences inside final_answer.
Do not include intermediate reasoning, worker transcripts, or tool logs in final_answer.
Never ask follow-up questions.
"""

HOTEL_WORKER_PROMPT = """You are hotel_worker.
Handle only hotel requests.
Adapt flow by intent:
- use each tool only when needed.
- always check availability before checking price and before booking.
- always check price before booking.
- use search only when required.
Never invent facts. Every claim must come from tools.
Return concrete tool evidence such as state_file and confirmation_number when booking succeeds.
If evidence is missing, return status: incomplete_tool_execution with missing details.
"""

FLIGHT_WORKER_PROMPT = """You are flight_worker.
Handle only flight requests.
Adapt flow by intent:
- use each tool only when needed.
- always check availability before checking price and before booking.
- always check price before booking.
- use search only when required.
Never invent facts. Every claim must come from tools.
Return concrete tool evidence such as state_file and confirmation_number when booking succeeds.
If evidence is missing, return status: incomplete_tool_execution with missing details.
"""

RESTAURANT_WORKER_PROMPT = """You are restaurant_worker.
Handle only restaurant requests.
Adapt flow by intent:
- use each tool only when needed.
- always check availability before checking price and before booking.
- always check price before booking.
- use search only when required.
Never invent facts. Every claim must come from tools.
Return concrete tool evidence such as state_file and confirmation_number when booking succeeds.
If evidence is missing, return status: incomplete_tool_execution with missing details.
"""

ATTRACTIONS_WORKER_PROMPT = """You are attractions_worker.
Handle only attractions and activities requests.
Adapt flow by intent:
- use each tool only when needed.
- always check availability before checking price and before booking.
- always check price before booking.
- use search only when required.
Never invent facts. Every claim must come from tools.
Return concrete tool evidence such as state_file and confirmation_number when booking succeeds.
If evidence is missing, return status: incomplete_tool_execution with missing details.
"""

WRITER_WORKER_PROMPT = """You are writer_worker.
Handle only itinerary document creation.
Write only when explicitly requested.
Never claim a save happened unless the tool confirms it.
If no save request exists, return status: not_requested.
"""


class TravelGraphState(TypedDict, total=False):
    query: str
    next_worker: RouteName
    worker_query: str
    worker_outputs: list[str]
    final_answer: str
    iterations: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run travel MAS with LangGraph-style agents.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="travel_langraph_orchestrator")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_mcp_config(base_dir: Path) -> dict:
    return {
        "hotel": {"transport": "stdio", "command": "python", "args": ["hotel_server.py"], "cwd": str(base_dir)},
        "flight": {"transport": "stdio", "command": "python", "args": ["flight_server.py"], "cwd": str(base_dir)},
        "restaurant": {"transport": "stdio", "command": "python", "args": ["restaurant_server.py"], "cwd": str(base_dir)},
        "attractions": {"transport": "stdio", "command": "python", "args": ["attractions_server.py"], "cwd": str(base_dir)},
    }


def parse_orchestrator_decision(content: str) -> dict:
    try:
        start = content.index("{")
        end = content.rindex("}") + 1
        decision = json.loads(content[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"next_worker": "final", "worker_query": "", "final_answer": content}

    next_worker = decision.get("next_worker", "final")
    if next_worker not in {"hotel", "flight", "restaurant", "attractions", "writer", "final"}:
        next_worker = "final"
    return {
        "next_worker": next_worker,
        "worker_query": str(decision.get("worker_query", "")),
        "final_answer": str(decision.get("final_answer", "")),
    }


def to_plain_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    return cleaned


async def run_mas(query: str, verbose: bool = False) -> str:
    base_dir = Path(__file__).resolve().parent
    llm = LLM().client

    mcp_client = MultiServerMCPClient(build_mcp_config(base_dir))
    mcp_tools = await mcp_client.get_tools()

    hotel_tools = [t for t in mcp_tools if "hotel" in t.name]
    flight_tools = [t for t in mcp_tools if "flight" in t.name]
    restaurant_tools = [t for t in mcp_tools if "restaurant" in t.name]
    attractions_tools = [t for t in mcp_tools if "attraction" in t.name]
    writer_tools = [write_travel_plan_document]

    async def orchestrator_node(state: TravelGraphState) -> dict:
        outputs = "\n\n".join(state.get("worker_outputs", [])) or "No worker outputs yet."
        iterations = state.get("iterations", 0)
        if iterations >= 8:
            return {
                "next_worker": "final",
                "final_answer": f"Stopped after reaching the worker-iteration limit. Worker outputs:\n{outputs}",
                "iterations": iterations,
            }
        agent = create_agent(model=llm, tools=[], system_prompt=ORCH_PROMPT)
        response = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Original user request:\n{state['query']}\n\n"
                            f"Worker outputs so far:\n{outputs}\n\n"
                            "Choose the next worker node or final."
                        ),
                    }
                ]
            }
        )
        messages = response.get("messages", [])
        content = str(messages[-1].content) if messages else ""
        decision = parse_orchestrator_decision(content)
        decision["iterations"] = iterations + 1
        if verbose:
            print(f"[orchestrator] next={decision['next_worker']}")
        return decision

    async def run_hotel_worker(state: TravelGraphState) -> dict:
        worker_query = state.get("worker_query") or state["query"]
        task = f"User request:\n{worker_query}"
        agent = create_agent(model=llm, tools=hotel_tools, system_prompt=HOTEL_WORKER_PROMPT)
        response = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})
        messages = response.get("messages", [])
        out = str(messages[-1].content) if messages else "No response."
        if verbose:
            print("[hotel] done")
        worker_outputs = state.get("worker_outputs", [])
        return {
            "worker_outputs": [*worker_outputs, f"hotel_output:\n{out}"],
            "worker_query": "",
            "next_worker": "final",
        }

    async def run_flight_worker(state: TravelGraphState) -> dict:
        worker_query = state.get("worker_query") or state["query"]
        task = f"User request:\n{worker_query}"
        agent = create_agent(model=llm, tools=flight_tools, system_prompt=FLIGHT_WORKER_PROMPT)
        response = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})
        messages = response.get("messages", [])
        out = str(messages[-1].content) if messages else "No response."
        if verbose:
            print("[flight] done")
        worker_outputs = state.get("worker_outputs", [])
        return {"worker_outputs": [*worker_outputs, f"flight_output:\n{out}"], "worker_query": "", "next_worker": "final"}

    async def run_restaurant_worker(state: TravelGraphState) -> dict:
        worker_query = state.get("worker_query") or state["query"]
        task = f"User request:\n{worker_query}"
        agent = create_agent(model=llm, tools=restaurant_tools, system_prompt=RESTAURANT_WORKER_PROMPT)
        response = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})
        messages = response.get("messages", [])
        out = str(messages[-1].content) if messages else "No response."
        if verbose:
            print("[restaurant] done")
        worker_outputs = state.get("worker_outputs", [])
        return {"worker_outputs": [*worker_outputs, f"restaurant_output:\n{out}"], "worker_query": "", "next_worker": "final"}

    async def run_attractions_worker(state: TravelGraphState) -> dict:
        worker_query = state.get("worker_query") or state["query"]
        task = f"User request:\n{worker_query}"
        agent = create_agent(model=llm, tools=attractions_tools, system_prompt=ATTRACTIONS_WORKER_PROMPT)
        response = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})
        messages = response.get("messages", [])
        out = str(messages[-1].content) if messages else "No response."
        if verbose:
            print("[attractions] done")
        worker_outputs = state.get("worker_outputs", [])
        return {"worker_outputs": [*worker_outputs, f"attractions_output:\n{out}"], "worker_query": "", "next_worker": "final"}

    async def run_writer_worker(state: TravelGraphState) -> dict:
        worker_query = state.get("worker_query") or state["query"]
        task = f"User request:\n{worker_query}"
        agent = create_agent(model=llm, tools=writer_tools, system_prompt=WRITER_WORKER_PROMPT)
        response = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})
        messages = response.get("messages", [])
        out = str(messages[-1].content) if messages else "No response."
        if verbose:
            print("[writer] done")
        worker_outputs = state.get("worker_outputs", [])
        return {"worker_outputs": [*worker_outputs, f"writer_output:\n{out}"], "worker_query": "", "next_worker": "final"}

    def route_from_orchestrator(state: TravelGraphState) -> RouteName:
        return state.get("next_worker", "final")

    graph = StateGraph(TravelGraphState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("hotel", run_hotel_worker)
    graph.add_node("flight", run_flight_worker)
    graph.add_node("restaurant", run_restaurant_worker)
    graph.add_node("attractions", run_attractions_worker)
    graph.add_node("writer", run_writer_worker)
    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "hotel": "hotel",
            "flight": "flight",
            "restaurant": "restaurant",
            "attractions": "attractions",
            "writer": "writer",
            "final": END,
        },
    )
    for worker_name in ["hotel", "flight", "restaurant", "attractions", "writer"]:
        graph.add_edge(worker_name, "orchestrator")

    app = graph.compile()
    result = await app.ainvoke({"query": query, "worker_outputs": [], "iterations": 0})
    final_answer = to_plain_text(result.get("final_answer", ""))
    if final_answer:
        return final_answer
    fallback = to_plain_text("\n\n".join(result.get("worker_outputs", [])))
    return fallback or "No response generated."


async def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    if mlflow_langchain is not None:
        mlflow_langchain.autolog(log_traces=True, silent=True)
    with mlflow.start_run():
        answer = await run_mas(args.query, verbose=args.verbose)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
