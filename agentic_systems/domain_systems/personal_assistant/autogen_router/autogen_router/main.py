import argparse
import asyncio
import hashlib
import os
import shutil
import time
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PA AutoGen router graph.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="pa_autogen_router")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _create_tmp_workspace(base_dir: Path) -> Path:
    src = base_dir / "workspace"
    tmp_root = base_dir / "tmp_workspaces"
    tmp_root.mkdir(exist_ok=True)
    run_id = hashlib.sha1(str(time.time()).encode()).hexdigest()[:8]
    dst = tmp_root / f"run_{run_id}"
    shutil.copytree(src, dst)
    return dst


def build_workbench(server_file: str, base_dir: Path, workspace_dir: Path) -> McpWorkbench:
    env = {**os.environ, "PA_WORKSPACE_DIR": str(workspace_dir)}
    return McpWorkbench(StdioServerParams(command="python", args=[server_file], cwd=str(base_dir), env=env))


def contains(marker: str):
    def condition(message: BaseChatMessage) -> bool:
        return marker in str(getattr(message, "content", message))
    return condition


@mlflow.trace(name="pa_query", span_type=SpanType.CHAIN)
async def run_router_graph(query: str, verbose: bool = False) -> str:
    base_dir = Path(__file__).resolve().parent
    tmp_workspace = _create_tmp_workspace(base_dir)
    os.environ["PA_WORKSPACE_DIR"] = str(tmp_workspace)
    llm_client = LLM().client

    router_prompt = (
        "You are router_agent. Your ENTIRE response must begin with exactly one of these markers "
        "on the very first line — no preamble, no acknowledgement, no explanation before the marker:\n"
        "CALL_READER — workspace lookups (emails, calendar, tasks, notes, contacts, task files).\n"
        "CALL_ACTION — create/update tasks, calendar events, or draft emails.\n"
        "CALL_FINAL — enough specialist output exists for synthesis.\n"
        "After the marker, add one concise instruction on the same or next line. Nothing else."
    )
    reader_prompt = (
        "You are reader_worker. Handle only workspace read/search requests. "
        "Use search_emails, search_calendar, search_contacts, search_tasks, search_notes, read_task_file. "
        "CRITICAL: Always use ONE or TWO word keyword queries (e.g. 'outage', 'ben', 'task010') — never phrases or sentences. "
        "Search tools do exact substring matching — long queries almost always return nothing. "
        "If a search returns no results, you MUST retry with a single shorter keyword before giving up. "
        "Try at least 2-3 different short keywords before concluding nothing exists. "
        "Return ONLY the raw data from tool outputs. "
        "Do NOT add any commentary about what you cannot do, what other workers should do, or what actions are needed. "
        "Never invent workspace facts. Do not perform write actions."
    )
    action_prompt = (
        "You are action_worker. Handle only workspace action requests. "
        "Use create_task, update_task, create_calendar_event, update_calendar_event, draft_email. "
        "Never perform actions not explicitly requested. "
        "Drafts are local only — never claim an email was sent. "
        "Return concrete evidence (task id, event id, draft id) when an action succeeds."
    )
    final_prompt = (
        "You are final_agent. Synthesize current results and decide the next step.\n"
        "Your response must start with exactly one of these markers on the first line:\n"
        "ROUTE_BACK: — more specialist work is needed; include a short instruction for the router.\n"
        "OUTPUT: — all requested work is fully complete; follow with the final plain-text answer.\n"
        "Decision rules — apply in order:\n"
        "1. If the user request includes ANY write operation (update task, draft email, create/update calendar event) "
        "and action_worker has NOT yet confirmed it with a task id, draft id, or event id → emit ROUTE_BACK: CALL_ACTION.\n"
        "2. reader_worker saying 'I cannot update' or 'I cannot draft' is NOT completion — it means action_worker must still be called. "
        "Emit ROUTE_BACK: CALL_ACTION.\n"
        "3. Only emit OUTPUT: when every requested write has been confirmed by action_worker with a concrete id.\n"
        "Never invent workspace facts. Never claim an action was done unless action_worker confirmed it with an id."
    )
    output_prompt = "You are output_agent. Return exactly the text after OUTPUT: as the final answer."

    async with AsyncExitStack() as stack:
        workbenches = {
            "reader": await stack.enter_async_context(build_workbench("workspace_server.py", base_dir, tmp_workspace)),
            "action": await stack.enter_async_context(build_workbench("action_server.py", base_dir, tmp_workspace)),
            "writer": await stack.enter_async_context(build_workbench("document_server.py", base_dir, tmp_workspace)),
        }

        router = AssistantAgent(
            name="router_agent",
            model_client=llm_client,
            system_message=router_prompt,
            reflect_on_tool_use=True,
            max_tool_iterations=2,
        )
        reader = AssistantAgent(
            name="reader_worker",
            model_client=llm_client,
            system_message=reader_prompt,
            workbench=[workbenches["reader"]],
            reflect_on_tool_use=True,
            max_tool_iterations=12,
        )
        action = AssistantAgent(
            name="action_worker",
            model_client=llm_client,
            system_message=action_prompt,
            workbench=[workbenches["action"]],
            reflect_on_tool_use=True,
            max_tool_iterations=12,
        )
        final = AssistantAgent(
            name="final_agent",
            model_client=llm_client,
            system_message=final_prompt,
            workbench=[workbenches["writer"]],
            reflect_on_tool_use=True,
            max_tool_iterations=8,
        )
        output = AssistantAgent(
            name="output_agent",
            model_client=llm_client,
            system_message=output_prompt,
            reflect_on_tool_use=False,
            max_tool_iterations=1,
        )

        builder = DiGraphBuilder()
        for node in [router, reader, action, final, output]:
            builder.add_node(node)
        builder.set_entry_point(router)
        builder.add_edge(router, reader, condition=contains("CALL_READER"))
        builder.add_edge(router, action, condition=contains("CALL_ACTION"))
        builder.add_edge(router, final, condition=contains("CALL_FINAL"))
        builder.add_edge(reader, final, activation_group="worker_to_final", activation_condition="any")
        builder.add_edge(action, final, activation_group="worker_to_final", activation_condition="any")
        builder.add_edge(final, router, condition=contains("ROUTE_BACK:"))
        builder.add_edge(final, output, condition=contains("OUTPUT:"))

        graph = builder.build()
        team = GraphFlow(
            participants=[router, reader, action, final, output],
            graph=graph,
            termination_condition=TextMentionTermination("OUTPUT:"),
            max_turns=20,
        )
        result = await run_one_query(team, query)
        messages = getattr(result, "messages", [])
        if verbose:
            print(f"[router_graph] turns={len(messages)}")
        return str(getattr(messages[-1], "content", messages[-1])) if messages else "No response."


async def run_one_query(team: GraphFlow, query: str):
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({"query": query})
    result = await team.run(task=f"User request:\n{query}")
    _log_agent_steps(result)
    return result


def _log_agent_steps(result) -> None:
    messages = getattr(result, "messages", []) or []
    for msg in messages:
        source = getattr(msg, "source", None) or "unknown_agent"
        content = str(getattr(msg, "content", ""))
        with mlflow.start_span(f"{source}.step", span_type=SpanType.AGENT) as step_span:
            step_span.set_inputs({"source": source})
            step_span.set_outputs({"content_preview": content[:1200]})


async def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    mlflow.set_experiment(args.exp_name)
    mlflow.autogen.autolog()
    with mlflow.start_run(run_name="pa_router_query"):
        answer = await run_router_graph(args.query, verbose=args.verbose)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
