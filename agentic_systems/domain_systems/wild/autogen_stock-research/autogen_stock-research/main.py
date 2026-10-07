import argparse
import asyncio
import os
from typing import Any, Dict, List

import mlflow
from mlflow.entities import SpanType
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import Swarm
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()

DEFAULT_TASK = "Conduct market research for TSLA stock."
DEFAULT_EXPERIMENT_NAME = "stock_research"
DEFAULT_MLFLOW_URI = "http://localhost:5000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stock research with AutoGen and record the session in MLflow."
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="The stock-research request to send to the agent team.",
    )
    parser.add_argument(
        "--exp_name",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME),
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--mlflow_uri",
        default=os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI),
        help="MLflow tracking server URI.",
    )
    return parser.parse_args()


def _get_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    raise KeyError(f"Missing required environment variable. Tried: {', '.join(names)}")


def mlflow_setup(exp_name: str, mlflow_uri: str) -> None:
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(exp_name)

    if hasattr(mlflow, "openai") and hasattr(mlflow.openai, "autolog"):
        mlflow.openai.autolog()


@mlflow.trace(span_type=SpanType.TOOL)
async def get_stock_data(symbol: str) -> Dict[str, Any]:
    """Get stock market data for a given symbol."""
    return {
        "symbol": symbol,
        "price": 180.25,
        "volume": 1000000,
        "pe_ratio": 65.4,
        "market_cap": "700B",
    }


@mlflow.trace(span_type=SpanType.TOOL)
async def get_news(query: str) -> List[Dict[str, str]]:
    """Get recent news articles about a company."""
    return [
        {
            "title": "Tesla Expands Cybertruck Production",
            "date": "2024-03-20",
            "summary": "Tesla ramps up Cybertruck manufacturing capacity at Gigafactory Texas, aiming to meet strong demand.",
        },
        {
            "title": "Tesla FSD Beta Shows Promise",
            "date": "2024-03-19",
            "summary": "Latest Full Self-Driving beta demonstrates significant improvements in urban navigation and safety features.",
        },
        {
            "title": "Model Y Dominates Global EV Sales",
            "date": "2024-03-18",
            "summary": "Tesla's Model Y becomes best-selling electric vehicle worldwide, capturing significant market share.",
        },
    ]


@mlflow.trace(name="stock_research_workflow", span_type=SpanType.WORKFLOW)
async def run_team(task: str) -> Dict[str, Any]:
    trace_id = mlflow.get_active_trace_id()
    mlflow.update_current_trace(
        request_preview=task,
        metadata={"workflow": "stock_research"},
    )

    model_client = AzureOpenAIChatCompletionClient(
        azure_deployment=_get_env("AZURE_MODEL_NAME"),
        model=_get_env("AZURE_MODEL_NAME"),
        api_version=_get_env("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=_get_env("AZURE_OPENAI_API_KEY"),
    )

    try:
        planner = AssistantAgent(
            "planner",
            model_client=model_client,
            handoffs=["financial_analyst", "news_analyst", "writer"],
            system_message="""You are a research planning coordinator.
            Coordinate market research by delegating to specialized agents:
            - Financial Analyst: For stock data analysis
            - News Analyst: For news gathering and analysis
            - Writer: For compiling final report
            Always send your plan first, then handoff to an appropriate agent.
            Always handoff to a single agent at a time.
            Use TERMINATE when research is complete.""",
        )

        financial_analyst = AssistantAgent(
            "financial_analyst",
            model_client=model_client,
            handoffs=["planner"],
            tools=[get_stock_data],
            system_message="""You are a financial analyst.
            Analyze stock market data using the get_stock_data tool.
            Provide insights on financial metrics.
            Always handoff back to planner when analysis is complete.""",
        )

        news_analyst = AssistantAgent(
            "news_analyst",
            model_client=model_client,
            handoffs=["planner"],
            tools=[get_news],
            system_message="""You are a news analyst.
            Gather and analyze relevant news using the get_news tool.
            Summarize key market insights from news.
            Always handoff back to planner when analysis is complete.""",
        )

        writer = AssistantAgent(
            "writer",
            model_client=model_client,
            handoffs=["planner"],
            system_message="""You are a financial report writer.
            Compile research findings into clear, concise reports.
            Always handoff back to planner when writing is complete.""",
        )

        termination = TextMentionTermination("TERMINATE")
        research_team = Swarm(
            participants=[planner, financial_analyst, news_analyst, writer],
            termination_condition=termination,
        )
        events = []
        result = None
        with mlflow.start_span("run_swarm", span_type=SpanType.AGENT):
            async for event in research_team.run_stream(task=task):
                payload = event.model_dump() if hasattr(event, "model_dump") else str(event)
                events.append({"type": type(event).__name__, "payload": payload})
                result = event

        return {"answer": str(result), "events": events, "trace_id": trace_id}
    finally:
        await model_client.close()


def main() -> None:
    args = parse_args()
    mlflow_setup(args.exp_name, args.mlflow_uri)

    with mlflow.start_run(run_name="stock-research-session"):
        mlflow.log_params(
            {
                "task": args.task,
                "experiment_name": args.exp_name,
                "mlflow_uri": args.mlflow_uri,
                "model": _get_env("AZURE_MODEL_NAME", "AZURE_OPENAI_DEPLOYMENT"),
            }
        )
        out = asyncio.run(run_team(args.task))
        answer = out["answer"]
        trace_id = out.get("trace_id")
        mlflow.log_text(args.task, "task.txt")
        mlflow.log_text(answer, "stock_research.txt")
        mlflow.log_dict(out["events"], "agent_events.json")
        mlflow.flush_trace_async_logging()
        if trace_id and (trace := mlflow.get_trace(trace_id, silent=True, flush=True)):
            mlflow.log_text(trace.to_json(pretty=True), f"traces/{trace_id}/traces.json")
            mlflow.log_text(trace_id, "latest_trace_id.txt")

    print("\nFINAL ANSWER:\n")
    print(answer)


if __name__ == "__main__":
    main()
