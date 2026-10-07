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


def always(_message: BaseChatMessage) -> bool:
    return True
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from dotenv import load_dotenv
from mlflow.entities import SpanType

from llm import LLM

READER_WORKER_PROMPT = """
You are reader_worker.
Handle only workspace read and search requests.
Use search_emails, search_calendar, search_contacts, search_tasks, search_notes, and read_task_file.
CRITICAL: Always use ONE or TWO word keyword queries (e.g. "outage", "ben", "task010") — never phrases or sentences.
Search tools do exact substring matching — long queries almost always return nothing.
If a search returns no results, you MUST retry with a single shorter keyword before giving up.
Try at least 2-3 different short keywords (e.g. "outage", then "ben", then "server") before concluding nothing exists.
Return ONLY raw data from tool outputs. Do not compose emails, suggest actions, or write content on behalf of the user.
Never invent workspace facts. Every claim must come from tool outputs.
Do not perform any write or update actions.
Do not answer for action tasks, writer tasks, or final orchestration.
"""

ACTION_WORKER_PROMPT = """
You are action_worker.
Handle only workspace action requests: create/update tasks, calendar events, and email drafts.
Use create_task, update_task, create_calendar_event, update_calendar_event, and draft_email.
Never perform actions not explicitly requested.
Drafts are local only — never claim an email was sent.
Return concrete evidence (task id, event id, draft id) when an action succeeds.
Do not answer for reader tasks, writer tasks, or final orchestration.
"""

WRITER_WORKER_PROMPT = """
You are writer_worker.
Handle only summary document creation.
Write a document only when the user explicitly asks to save or create a document.
Never claim a document was saved unless the tool confirms it.
If no save request exists, return status: not_requested.
Do not answer for reader tasks, action tasks, or final orchestration.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PA AutoGen orchestrator.")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default="pa_autogen_orch")
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
    params = StdioServerParams(command="python", args=[server_file], cwd=str(base_dir), env=env)
    return McpWorkbench(params)


@mlflow.trace(name="pa_query", span_type=SpanType.CHAIN)
async def run_orchestrator(query: str, verbose: bool = False) -> str:
    base_dir = Path(__file__).resolve().parent
    tmp_workspace = _create_tmp_workspace(base_dir)
    os.environ["PA_WORKSPACE_DIR"] = str(tmp_workspace)
    llm_client = LLM().client
    orch_prompt = (
        "You are the personal assistant orchestrator node in a directed agent graph. "
        "You receive the user query first, inspect worker outputs, and decide the next graph edge. "
        "Your ENTIRE response must begin with exactly one of these markers on the very first line — "
        "no preamble, no acknowledgement, no explanation before the marker:\n"
        "CALL_READER — workspace lookups (emails, calendar, tasks, notes, contacts, task files).\n"
        "CALL_ACTION — create/update tasks, calendar events, or draft emails.\n"
        "CALL_WRITER — only when the user asks to save or create a summary document.\n"
        "FINAL: — when all requested work is complete; follow with one integrated plain-text answer.\n"
        "After the marker, add one concise instruction on the same or next line. "
        "After a worker responds, control returns to you — continue routing until all steps are done. "
        "Finding information via CALL_READER is NOT completion when writes are also requested — "
        "you must issue CALL_ACTION before FINAL:. "
        "Do not include JSON, markdown, code fences, intermediate reasoning, or tool logs in FINAL: output. "
        "Never ask follow-up questions."
    )

    def contains(marker: str):
        def condition(message: BaseChatMessage) -> bool:
            return marker in str(getattr(message, "content", message))
        return condition

    async with AsyncExitStack() as stack:
        workbenches = {
            "reader": await stack.enter_async_context(build_workbench("workspace_server.py", base_dir, tmp_workspace)),
            "action": await stack.enter_async_context(build_workbench("action_server.py", base_dir, tmp_workspace)),
            "writer": await stack.enter_async_context(build_workbench("document_server.py", base_dir, tmp_workspace)),
        }
        orchestrator = AssistantAgent(
            name="orchestrator",
            model_client=llm_client,
            system_message=orch_prompt,
            reflect_on_tool_use=True,
            max_tool_iterations=2,
        )
        workers = {
            "reader": AssistantAgent(
                name="reader_worker",
                model_client=llm_client,
                system_message=READER_WORKER_PROMPT,
                workbench=[workbenches["reader"]],
                reflect_on_tool_use=True,
                max_tool_iterations=12,
            ),
            "action": AssistantAgent(
                name="action_worker",
                model_client=llm_client,
                system_message=ACTION_WORKER_PROMPT,
                workbench=[workbenches["action"]],
                reflect_on_tool_use=True,
                max_tool_iterations=12,
            ),
            "writer": AssistantAgent(
                name="writer_worker",
                model_client=llm_client,
                system_message=WRITER_WORKER_PROMPT,
                workbench=[workbenches["writer"]],
                reflect_on_tool_use=True,
                max_tool_iterations=8,
            ),
        }
        final_agent = AssistantAgent(
            name="final_agent",
            model_client=llm_client,
            system_message=(
                "You are the final node. Return exactly the final integrated answer provided by the orchestrator "
                "after the FINAL: marker. Return plain text only. Do not add JSON, markdown, or code fences. "
                "Do not include intermediate reasoning, worker transcripts, or tool logs."
            ),
            reflect_on_tool_use=False,
            max_tool_iterations=1,
        )
        builder = DiGraphBuilder()
        builder.add_node(orchestrator)
        for worker in workers.values():
            builder.add_node(worker)
        builder.add_node(final_agent)
        builder.set_entry_point(orchestrator)
        builder.add_edge(orchestrator, workers["reader"], condition=contains("CALL_READER"))
        builder.add_edge(orchestrator, workers["action"], condition=contains("CALL_ACTION"))
        builder.add_edge(orchestrator, workers["writer"], condition=contains("CALL_WRITER"))
        builder.add_edge(orchestrator, final_agent, condition=contains("FINAL:"))
        for worker in workers.values():
            builder.add_edge(worker, orchestrator, condition=always, activation_group="worker_to_orch", activation_condition="any")

        graph = builder.build()
        team = GraphFlow(
            participants=[orchestrator, *workers.values(), final_agent],
            graph=graph,
            termination_condition=TextMentionTermination("FINAL:"),
            max_turns=16,
        )
        result = await run_one_query(team, query)
        messages = getattr(result, "messages", [])
        if verbose:
            print(f"[graph] turns={len(messages)}")
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
    with mlflow.start_run(run_name="pa_query"):
        answer = await run_orchestrator(args.query, verbose=args.verbose)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
