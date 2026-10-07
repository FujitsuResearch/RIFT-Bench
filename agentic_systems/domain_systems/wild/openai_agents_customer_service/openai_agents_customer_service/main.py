from __future__ import annotations as _annotations

import argparse
import os
import asyncio
import contextlib
import random
import uuid
import mlflow
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel

from agents import (
    Agent,
    HandoffOutputItem,
    ItemHelpers,
    MessageOutputItem,
    RunContextWrapper,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
    function_tool,
    handoff,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents.extensions.models.litellm_model import LitellmModel

DEFAULT_EXPERIMENT_NAME = "openai_agents_customer_service"
DEFAULT_MLFLOW_URI = "http://localhost:5000"
DEFAULT_TASK = "Do you have wifi?"

### CONTEXT


class AirlineAgentContext(BaseModel):
    passenger_name: str | None = None
    confirmation_number: str | None = None
    seat_number: str | None = None
    flight_number: str | None = None


### TOOLS


@function_tool(
    name_override="faq_lookup_tool", description_override="Lookup frequently asked questions."
)
async def faq_lookup_tool(question: str) -> str:
    question_lower = question.lower()
    if any(
        keyword in question_lower
        for keyword in ["bag", "baggage", "luggage", "carry-on", "hand luggage", "hand carry"]
    ):
        return (
            "You are allowed to bring one bag on the plane. "
            "It must be under 50 pounds and 22 inches x 14 inches x 9 inches."
        )
    elif any(keyword in question_lower for keyword in ["seat", "seats", "seating", "plane"]):
        return (
            "There are 120 seats on the plane. "
            "There are 22 business class seats and 98 economy seats. "
            "Exit rows are rows 4 and 16. "
            "Rows 5-8 are Economy Plus, with extra legroom. "
        )
    elif any(
        keyword in question_lower
        for keyword in ["wifi", "internet", "wireless", "connectivity", "network", "online"]
    ):
        return "We have free wifi on the plane, join Airline-Wifi"
    return "I'm sorry, I don't know the answer to that question."


@function_tool
async def update_seat(
    context: RunContextWrapper[AirlineAgentContext], confirmation_number: str, new_seat: str
) -> str:
    """
    Update the seat for a given confirmation number.

    Args:
        confirmation_number: The confirmation number for the flight.
        new_seat: The new seat to update to.
    """
    # Update the context based on the customer's input
    context.context.confirmation_number = confirmation_number
    context.context.seat_number = new_seat
    # Ensure that the flight number has been set by the incoming handoff
    assert context.context.flight_number is not None, "Flight number is required"
    return f"Updated seat to {new_seat} for confirmation number {confirmation_number}"


### HOOKS


async def on_seat_booking_handoff(context: RunContextWrapper[AirlineAgentContext]) -> None:
    flight_number = f"FLT-{random.randint(100, 999)}"
    context.context.flight_number = flight_number


### AGENTS

faq_agent = Agent[AirlineAgentContext](
    name="FAQAgent",
    model=LitellmModel(
        model=f"azure/{os.getenv('AZURE_MODEL_NAME')}",  # Model from config.yaml
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),      # LiteLLM API key
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT")
    ),
    handoff_description="A helpful agent that can answer questions about the airline.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are an FAQ agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Identify the last question asked by the customer.
    2. Use the faq lookup tool to answer the question. Do not rely on your own knowledge.
    3. If you cannot answer the question, transfer back to the triage agent.""",
    tools=[faq_lookup_tool],
)

seat_booking_agent = Agent[AirlineAgentContext](
    name="SeatBookingAgent",
    model=LitellmModel(
        model=f"azure/{os.getenv('AZURE_MODEL_NAME')}",  # Model from config.yaml
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),      # LiteLLM API key
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT")
    ),
    handoff_description="A helpful agent that can update a seat on a flight.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a seat booking agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Ask for their confirmation number.
    2. Ask the customer what their desired seat number is.
    3. Use the update seat tool to update the seat on the flight.
    If the customer asks a question that is not related to the routine, transfer back to the triage agent. """,
    tools=[update_seat],
)

triage_agent = Agent[AirlineAgentContext](
    name="TriageAgent",
    model=LitellmModel(
        model=f"azure/{os.getenv('AZURE_MODEL_NAME')}",  # Model from config.yaml
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),      # LiteLLM API key
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT")
    ),
    handoff_description="A triage agent that can delegate a customer's request to the appropriate agent.",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} "
        "You are a helpful triaging agent. You can use your tools to delegate questions to other appropriate agents."
    ),
    handoffs=[
        faq_agent,
        handoff(agent=seat_booking_agent, on_handoff=on_seat_booking_handoff),
    ],
)

faq_agent.handoffs.append(triage_agent)
seat_booking_agent.handoffs.append(triage_agent)


### RUN


def _mlflow_span(name: str, attributes: dict[str, str] | None = None):
    """Create an MLflow trace span when tracing is available, else no-op."""
    start_span = getattr(mlflow, "start_span", None)
    if callable(start_span):
        return start_span(name=name, attributes=attributes or {})
    return contextlib.nullcontext()


def mlflow_setup(exp_name: str, mlflow_uri: str) -> None:
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(exp_name)
    if hasattr(mlflow, "openai") and hasattr(mlflow.openai, "autolog"):
        mlflow.openai.autolog()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--exp_name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--mlflow_uri", default=DEFAULT_MLFLOW_URI)
    return parser.parse_args()


async def main(task: str, exp_name: str, mlflow_uri: str):
    current_agent: Agent[AirlineAgentContext] = triage_agent
    input_items: list[TResponseInputItem] = []
    context = AirlineAgentContext()
    mlflow_setup(exp_name, mlflow_uri)

    # Use a short UUID as conversation/run identifier.
    conversation_id = uuid.uuid4().hex[:16]
    user_input = task

    with mlflow.start_run(run_name=f"openai-agents-session-{conversation_id}"):
        trace_id: str | None = None
        mlflow.log_params(
            {
                "task": user_input,
                "experiment_name": exp_name,
                "mlflow_uri": mlflow_uri,
                "model": os.getenv("AZURE_MODEL_NAME"),
                "conversation_id": conversation_id,
            }
        )

        while True:
            with _mlflow_span(
                "customer_service_turn", {"conversation_id": conversation_id}
            ):
                input_items.append({"content": user_input, "role": "user"})
                result = await Runner.run(current_agent, input_items, context=context)
                if hasattr(mlflow, "get_active_trace_id"):
                    trace_id = mlflow.get_active_trace_id()

                output_lines: list[str] = []
                for new_item in result.new_items:
                    agent_name = new_item.agent.name
                    if isinstance(new_item, MessageOutputItem):
                        text = ItemHelpers.text_message_output(new_item)
                        output_lines.append(f"{agent_name}: {text}")
                        print(f"{agent_name}: {text}")
                    elif isinstance(new_item, HandoffOutputItem):
                        msg = (
                            f"Handed off from {new_item.source_agent.name} "
                            f"to {new_item.target_agent.name}"
                        )
                        output_lines.append(msg)
                        print(msg)
                    elif isinstance(new_item, ToolCallItem):
                        msg = f"{agent_name}: Calling a tool"
                        output_lines.append(msg)
                        print(msg)
                    elif isinstance(new_item, ToolCallOutputItem):
                        msg = f"{agent_name}: Tool call output: {new_item.output}"
                        output_lines.append(msg)
                        print(msg)
                    else:
                        msg = f"{agent_name}: Skipping item: {new_item.__class__.__name__}"
                        output_lines.append(msg)
                        print(msg)

                mlflow.log_text(user_input, "task.txt")
                mlflow.log_text("\n".join(output_lines), "agent_output.txt")
                mlflow.log_metric("new_items_count", len(result.new_items))

                input_items = result.to_input_list()
                current_agent = result.last_agent
            break

        if hasattr(mlflow, "flush_trace_async_logging"):
            mlflow.flush_trace_async_logging()
        if trace_id and hasattr(mlflow, "get_trace"):
            trace = mlflow.get_trace(trace_id, silent=True, flush=True)
            if trace is not None:
                mlflow.log_text(trace.to_json(pretty=True), f"traces/{trace_id}/traces.json")
                mlflow.log_text(trace_id, "latest_trace_id.txt")

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.task, args.exp_name, args.mlflow_uri))
