import os
import sys
import argparse
import json
import mlflow
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
import asyncio
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore

from langchain_core.runnables import RunnableConfig
import uuid
from typing import Optional
from langgraph.store.base import BaseStore
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def setup_paths():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_directory)
    grandparent_dir = os.path.dirname(parent_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    if grandparent_dir not in sys.path:
        sys.path.append(grandparent_dir)


setup_paths()

from agent_scripts.langraph_react_tools import *

# from langraph_react_tools import *


async def main():
    load_dotenv(override=True)
    mlflow.langchain.autolog()

    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
        model=os.getenv("AZURE_MODEL_NAME"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
        temperature=0.0,
    )

    parser = argparse.ArgumentParser(
        description="Run the ReAct agent with a custom user task."
    )

    parser.add_argument(
        "--task",
        type=str,
        required=False,
        help="The user task to pass to the agent.",
        default="Generate 5 distinct random integers between 1 and 10 inclusive, sorted ascending.",
    )
    parser.add_argument("--task_file", type=str, default=None)
    parser.add_argument(
        "--tool_file",
        type=str,
        default="./resources/langraph_react_tools_list.json",
        help="Path to the JSON file containing tool names.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="langraph_react_partial_random_list",
        help="name of current experiments, for mlflow tracing.",
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to run the MLFlow server on"
    )
    args = parser.parse_args()
    mlflow.set_tracking_uri(f"http://localhost:{args.port}")
    # Load tool names from file
    with open(args.tool_file, "r") as f:
        tool_names = json.load(f)
    tool_list = []
    for name in tool_names:
        if name in globals():
            tool_list.append(globals()[name])
        else:
            print(f"⚠️ Warning: Tool '{name}' not found in MCP or globals.")

    # Build and run agent
    # checkpointer = MemorySaver()  # short-term memory
    cfg = RunnableConfig(configurable={"thread_id": str(uuid.uuid4())})
    # store = InMemoryStore()  # long-term memory

    def make_memory_tools(store: BaseStore):
        def remember(user_id: str, note: str) -> str:
            """
            Save a memory (note) for a specific user_id into the vector store.
            Returns "Saved." after storing.
            """
            ns = ("users", user_id, "notes")
            store.put(ns, key=str(uuid.uuid4()), value={"text": note})

            return "Saved."

        def recall(user_id: str, query: Optional[str] = None) -> str:
            """
            Retrieve stored memories for a user_id.
            If a query is provided, performs semantic/full-text search (limit 5).
            If no query is provided, lists recent stored notes (limit 20).
            Returns concatenated notes or 'No memories found.'.
            """
            ns = ("users", user_id, "notes")
            if query:
                hits = store.search(ns, query=query, limit=5)
                vals = [h.value["text"] for h in hits]
            else:
                vals = [item.value["text"] for item in store.search(ns, limit=20)]
            return "\n".join(vals) if vals else "No memories found."

        # Async wrappers for memory tools
        async def remember_tool_async(user_id: str, note: str) -> str:
            return await asyncio.to_thread(remember, user_id, note)

        async def recall_tool_async(user_id: str, query: Optional[str] = None) -> str:
            return await asyncio.to_thread(recall, user_id, query)

        # ----- Tool Objects -----
        remember_tool = StructuredTool(
            name="remember_tool",
            description=(
                "Save important information about the user for future conversations. "
                "Use this when details about the user are provided and it is important to store them."
                "The details could be the user name, preferences, job, hobbies, or any personal details. "
                "This memory persists across all conversations. "
                "Input: user_id='user' (string) and note (string containing what to remember)."
            ),
            args_schema=RememberInput,
            func=remember,
            coroutine=remember_tool_async,
        )

        recall_tool = StructuredTool(
            name="recall_tool",
            description=(
                "Retrieve previously saved information about the user. "
                "Use this when you need to recall facts about the user that were saved earlier, "
                "or when the user asks 'what do you know about me?' or 'recall my [information]'. "
                "You can optionally provide a search query to filter specific memories. "
                "Input: user_id='user' (string) and optional query (string to search for specific information)."
            ),
            args_schema=RecallInput,
            func=recall,
            coroutine=recall_tool_async,
        )

        return [remember_tool, recall_tool]

    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        async with AsyncSqliteStore.from_conn_string("store.db") as store:

            tool_list += make_memory_tools(store)
            # print(tool_list)

            # Build and run agent
            agent = create_agent(
                llm, tools=tool_list, checkpointer=checkpointer, store=store
            )
            user_tasks = []
            if args.task_file:
                template_query_mapping = json.load(open(args.task_file))
                for task_template in template_query_mapping:
                    user_tasks.extend(task_template["queries"])
                # print(len(user_tasks))
                # print(type(user_tasks[0]))
                # input()
            else:
                user_tasks = [args.task]

            mlflow.set_experiment(args.exp_name)
            # Start manual MLflow run
            with mlflow.start_run() as run:
                for user_task in tqdm(user_tasks):
                    # Run the agent as usual
                    print(f"Task: {user_task[:50]}...")
                    final_response = None
                    async for response in agent.astream(
                        {
                            "input": user_task,
                            "agent": {},
                            "messages": [{"role": "user", "content": user_task}],
                        },
                        cfg,
                        stream_mode="values",
                    ):
                        response["messages"][-1].pretty_print()
                        final_response = response

                    # print("Response:", final_response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
