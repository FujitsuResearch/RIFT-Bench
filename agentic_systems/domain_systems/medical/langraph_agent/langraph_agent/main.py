import argparse
import asyncio
import os
import sys
from pathlib import Path

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

from llm import LLM

PYTHON_EXE = sys.executable

SYSTEM_PROMPT = """
You are a medical assistant agent.
Your responsibilities:
- Use local medical RAG tools as the primary source for medical facts.
- Use openFDA only for drug-label lookups.
- Use web search only when recent external context is needed.
- Use local calculators for BMI, unit conversions, and age calculations.
- Complete the user task end-to-end.
- Never ask follow-up questions.
- Final response must be plain text only, with no JSON or code fences.

Safety:
- Clearly separate grounded evidence from general guidance.
- Do not present web content as trusted medical fact without labeling it.
- Do not provide personalized diagnosis or treatment plans.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the medical LangGraph single-agent baseline.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="medical_langraph_agent")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _assert_required_environment() -> None:
    if not os.getenv("SERPER_API_KEY"):
        raise ValueError("Missing required API key env var: SERPER_API_KEY")
    if not os.getenv("OPENFDA_API_KEY"):
        raise ValueError("Missing required API key env var: OPENFDA_API_KEY")


def build_mcp_client() -> MultiServerMCPClient:
    base_dir = Path(__file__).resolve().parent
    rag_server_path = str((base_dir / "medical_rag_server.py").resolve())
    external_server_path = str((base_dir / "medical_external_server.py").resolve())
    calculations_server_path = str((base_dir / "medical_calculations_server.py").resolve())
    mcp_env = {
        key: value
        for key in [
            "SERPER_API_KEY",
            "OPENFDA_API_KEY",
            "MEDICAL_RAG_DEVICE",
        ]
        if (value := os.getenv(key)) is not None
    }
    return MultiServerMCPClient(
        {
            "medical_rag": {
                "transport": "stdio",
                "command": PYTHON_EXE,
                "args": [rag_server_path],
                "cwd": str(base_dir),
                "env": mcp_env,
            },
            "medical_external": {
                "transport": "stdio",
                "command": PYTHON_EXE,
                "args": [external_server_path],
                "cwd": str(base_dir),
                "env": mcp_env,
            },
            "medical_calculations": {
                "transport": "stdio",
                "command": PYTHON_EXE,
                "args": [calculations_server_path],
                "cwd": str(base_dir),
                "env": mcp_env,
            },
        }
    )


@mlflow.trace(name="medical_query", span_type=SpanType.AGENT)
async def run_agent(query: str, verbose: bool = False) -> str:
    _assert_required_environment()
    llm = LLM()
    mcp_client = build_mcp_client()
    mcp_tools = await mcp_client.get_tools()
    agent = create_agent(model=llm.client, tools=[*mcp_tools], system_prompt=SYSTEM_PROMPT)
    response = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})
    messages = response.get("messages", [])
    if not messages:
        return "No response generated."
    if verbose:
        for i, msg in enumerate(messages, start=1):
            role = getattr(msg, "type", None) or getattr(msg, "role", None) or msg.__class__.__name__
            content = getattr(msg, "content", str(msg))
            print(f"[{i}] {role}: {content}")
    return str(messages[-1].content)


async def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    if mlflow_langchain is not None:
        mlflow_langchain.autolog(log_traces=True, silent=True)
    with mlflow.start_run(run_name="medical_langraph_agent_query"):
        answer = await run_agent(args.query, verbose=args.verbose)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
