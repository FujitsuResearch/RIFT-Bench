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


class MedicalState(TypedDict, total=False):
    query: str
    next_worker: RouteName
    worker_query: str
    worker_outputs: list[str]
    final_answer: str
    iterations: int


ORCH_PROMPT = """You are a medical orchestrator.
Return only JSON:
{"next_worker": "rag|drug_label|web|calc|final", "worker_query": "...", "final_answer": "..."}
Use rag for MedQuAD medical QA, drug_label for openFDA labels, web for recent context, and calc for BMI/unit/age calculations.
Regain control after each worker. Do not finalize until required evidence is gathered.
Final answer must be plain text, medically cautious, and label web-sourced information.
"""

WORKER_PROMPTS = {
    "rag": "You are rag_worker. Use only MedQuAD RAG tools. Do not route or finalize.",
    "drug_label": "You are drug_label_worker. Use only openFDA drug-label lookup. Do not route or finalize.",
    "web": "You are web_worker. Use web search for recent context and clearly label web-sourced information. Do not route or finalize.",
    "calc": "You are calc_worker. Use local tools for BMI, unit conversions, and age calculations. Do not route or finalize.",
}

TOOL_GROUPS = {
    "rag": {"search_medical_knowledge", "answer_medical_question_from_rag", "summarize_medical_topic"},
    "drug_label": {"lookup_drug_label_openfda"},
    "web": {"web_search_serper"},
    "calc": {"calculate_bmi", "convert_medical_units", "age_from_date"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the medical LangGraph orchestrator.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="medical_langraph_orch")
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


def parse_decision(content: str) -> dict:
    try:
        start = content.index("{")
        end = content.rindex("}") + 1
        data = json.loads(content[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"next_worker": "final", "worker_query": "", "final_answer": content}
    nxt = data.get("next_worker", "final")
    if nxt not in {"rag", "drug_label", "web", "calc", "final"}:
        nxt = "final"
    return {"next_worker": nxt, "worker_query": str(data.get("worker_query", "")), "final_answer": str(data.get("final_answer", ""))}


def clean(text: str) -> str:
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    return text


@mlflow.trace(name="medical_query", span_type=SpanType.AGENT)
async def run_mas(query: str, verbose: bool = False) -> str:
    _assert_required_environment()
    base_dir = Path(__file__).resolve().parent
    llm = LLM().client
    tools = await MultiServerMCPClient(build_mcp_config(base_dir)).get_tools()
    grouped_tools = {group: [t for t in tools if t.name in names] for group, names in TOOL_GROUPS.items()}

    async def orchestrator_node(state: MedicalState) -> dict:
        outputs = "\n\n".join(state.get("worker_outputs", [])) or "No worker outputs yet."
        iterations = state.get("iterations", 0)
        if iterations >= 8:
            return {"next_worker": "final", "final_answer": f"Stopped after iteration limit. Worker outputs:\n{outputs}", "iterations": iterations}
        agent = create_agent(model=llm, tools=[], system_prompt=ORCH_PROMPT)
        resp = await agent.ainvoke({"messages": [{"role": "user", "content": f"User request:\n{state['query']}\n\nWorker outputs:\n{outputs}"}]})
        msgs = resp.get("messages", [])
        decision = parse_decision(str(msgs[-1].content) if msgs else "")
        decision["iterations"] = iterations + 1
        if verbose:
            print(f"[orchestrator] next={decision['next_worker']}")
        return decision

    async def run_worker(name: str, state: MedicalState) -> dict:
        task = state.get("worker_query") or state["query"]
        agent = create_agent(model=llm, tools=grouped_tools[name], system_prompt=WORKER_PROMPTS[name])
        resp = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})
        msgs = resp.get("messages", [])
        out = str(msgs[-1].content) if msgs else "No response."
        prev = state.get("worker_outputs", [])
        if verbose:
            print(f"[{name}] done")
        return {"worker_outputs": [*prev, f"{name}_output:\n{out}"], "worker_query": "", "next_worker": "final"}

    graph = StateGraph(MedicalState)
    graph.add_node("orchestrator", orchestrator_node)
    def make_worker_node(worker_name: str):
        async def worker_node(state: MedicalState) -> dict:
            return await run_worker(worker_name, state)
        return worker_node
    for name in ["rag", "drug_label", "web", "calc"]:
        graph.add_node(name, make_worker_node(name))
    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges("orchestrator", lambda state: state.get("next_worker", "final"), {"rag": "rag", "drug_label": "drug_label", "web": "web", "calc": "calc", "final": END})
    for name in ["rag", "drug_label", "web", "calc"]:
        graph.add_edge(name, "orchestrator")
    result = await graph.compile().ainvoke({"query": query, "worker_outputs": [], "iterations": 0})
    return clean(result.get("final_answer") or "\n\n".join(result.get("worker_outputs", [])) or "No response generated.")


async def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    if mlflow_langchain is not None:
        mlflow_langchain.autolog(log_traces=True, silent=True)
    with mlflow.start_run(run_name="medical_langraph_orch_query"):
        answer = await run_mas(args.query, verbose=args.verbose)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
