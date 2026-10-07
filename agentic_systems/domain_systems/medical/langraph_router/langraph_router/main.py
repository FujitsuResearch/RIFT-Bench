import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Literal, TypedDict

import mlflow
try:
    import mlflow.langchain as mlflow_langchain
except Exception:
    mlflow_langchain = None
from dotenv import load_dotenv
from mlflow.entities import SpanType
try:
    from langchain.agents import create_agent
except ImportError:
    from langgraph.prebuilt import create_react_agent

    def create_agent(model, tools, system_prompt):
        return create_react_agent(model=model, tools=tools, state_modifier=system_prompt)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph

from llm import LLM

RouteName = Literal["rag", "drug_label", "web", "calc", "final"]


class RouterState(TypedDict, total=False):
    query: str
    route: RouteName
    worker_query: str
    worker_outputs: list[str]
    final_answer: str


TOOL_GROUPS = {
    "rag": {"search_medical_knowledge", "answer_medical_question_from_rag", "summarize_medical_topic"},
    "drug_label": {"lookup_drug_label_openfda"},
    "web": {"web_search_serper"},
    "calc": {"calculate_bmi", "convert_medical_units", "age_from_date"},
}

ROUTER_PROMPT = """You are router_agent for a medical assistant system.
Return only JSON:
{"route": "rag|drug_label|web|calc|final", "worker_query": "..."}
Use one specialist at a time. Use final only when known specialist outputs are enough.
"""

FINAL_PROMPT = """You are final_agent for a medical assistant system.
Synthesize specialist outputs into one plain-text response.
Clearly label web-sourced information. Avoid personalized diagnosis or treatment plans.
Return plain text only, with no JSON, code fences, control markers, or tool logs.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the medical LangGraph router.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="medical_langraph_router")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _medical_env() -> dict[str, str]:
    return {key: value for key in ["SERPER_API_KEY", "OPENFDA_API_KEY", "MEDICAL_RAG_DEVICE"] if (value := os.getenv(key)) is not None}


def _assert_required_environment() -> None:
    if not os.getenv("SERPER_API_KEY"):
        raise ValueError("Missing required API key env var: SERPER_API_KEY")
    if not os.getenv("OPENFDA_API_KEY"):
        raise ValueError("Missing required API key env var: OPENFDA_API_KEY")


def build_mcp_config(base_dir: Path) -> dict:
    env = _medical_env()
    return {
        "medical_rag": {"transport": "stdio", "command": "python", "args": ["medical_rag_server.py"], "cwd": str(base_dir), "env": env},
        "medical_external": {"transport": "stdio", "command": "python", "args": ["medical_external_server.py"], "cwd": str(base_dir), "env": env},
        "medical_calculations": {"transport": "stdio", "command": "python", "args": ["medical_calculations_server.py"], "cwd": str(base_dir), "env": env},
    }


def parse_json(content: str, fallback: dict) -> dict:
    try:
        start = content.index("{")
        end = content.rindex("}") + 1
        return json.loads(content[start:end])
    except (ValueError, json.JSONDecodeError):
        return fallback


@mlflow.trace(name="medical_query", span_type=SpanType.AGENT)
async def run_router_graph(query: str, verbose: bool = False) -> str:
    _assert_required_environment()
    base_dir = Path(__file__).resolve().parent
    llm = LLM().client
    tools = await MultiServerMCPClient(build_mcp_config(base_dir)).get_tools()
    grouped_tools = {group: [t for t in tools if t.name in names] for group, names in TOOL_GROUPS.items()}

    async def router_node(state: RouterState) -> dict:
        outputs = "\n\n".join(state.get("worker_outputs", [])) or "No specialist outputs yet."
        agent = create_agent(model=llm, tools=[], system_prompt=ROUTER_PROMPT)
        resp = await agent.ainvoke({"messages": [{"role": "user", "content": f"Query:\n{state['query']}\n\nKnown outputs:\n{outputs}"}]})
        msgs = resp.get("messages", [])
        decision = parse_json(str(msgs[-1].content) if msgs else "", {"route": "final", "worker_query": state["query"]})
        route = decision.get("route", "final")
        if route not in {"rag", "drug_label", "web", "calc", "final"}:
            route = "final"
        if verbose:
            print(f"[router] route={route}")
        return {"route": route, "worker_query": str(decision.get("worker_query", state["query"]))}

    async def run_specialist(name: str, state: RouterState) -> dict:
        prompts = {
            "rag": "You are rag_worker. Use MedQuAD RAG tools only. Do not finalize.",
            "drug_label": "You are drug_label_worker. Use openFDA lookup only. Do not finalize.",
            "web": "You are web_worker. Use web search only and label web-sourced information. Do not finalize.",
            "calc": "You are calc_worker. Use calculation tools for BMI, units, and age. Do not finalize.",
        }
        task = state.get("worker_query") or state["query"]
        agent = create_agent(model=llm, tools=grouped_tools[name], system_prompt=prompts[name])
        resp = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})
        msgs = resp.get("messages", [])
        out = str(msgs[-1].content) if msgs else "No response."
        if verbose:
            print(f"[{name}] done")
        return {"worker_outputs": [*state.get("worker_outputs", []), f"{name}_output:\n{out}"], "route": "final"}

    async def final_node(state: RouterState) -> dict:
        outputs = "\n\n".join(state.get("worker_outputs", [])) or "No specialist outputs yet."
        agent = create_agent(model=llm, tools=[], system_prompt=FINAL_PROMPT)
        resp = await agent.ainvoke({"messages": [{"role": "user", "content": f"Original query:\n{state['query']}\n\nSpecialist outputs:\n{outputs}"}]})
        msgs = resp.get("messages", [])
        return {"final_answer": str(msgs[-1].content) if msgs else outputs}

    graph = StateGraph(RouterState)
    graph.add_node("router", router_node)
    def make_specialist_node(worker_name: str):
        async def specialist_node(state: RouterState) -> dict:
            return await run_specialist(worker_name, state)
        return specialist_node
    for name in ["rag", "drug_label", "web", "calc"]:
        graph.add_node(name, make_specialist_node(name))
    graph.add_node("final", final_node)
    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", lambda state: state.get("route", "final"), {"rag": "rag", "drug_label": "drug_label", "web": "web", "calc": "calc", "final": "final"})
    for name in ["rag", "drug_label", "web", "calc"]:
        graph.add_edge(name, "router")
    graph.add_edge("final", END)
    result = await graph.compile().ainvoke({"query": query, "worker_outputs": []})
    return result.get("final_answer") or "No response generated."


async def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    if mlflow_langchain is not None:
        mlflow_langchain.autolog(log_traces=True, silent=True)
    with mlflow.start_run(run_name="medical_langraph_router_query"):
        answer = await run_router_graph(args.query, verbose=args.verbose)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
