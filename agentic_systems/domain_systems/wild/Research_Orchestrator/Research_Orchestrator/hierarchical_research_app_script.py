import getpass
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, List, Dict, Optional, Literal, Any, Iterable
from functools import partial
import mlflow

from langchain_community.document_loaders import WebBaseLoader
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from typing_extensions import TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.prebuilt import create_react_agent


# --- Environment Setup ---
def _set_if_undefined(var: str, is_secret: bool = True):
    env_val = os.environ.get(var)
    if not env_val:
        if is_secret:
            os.environ[var] = getpass.getpass(f"Please provide your {var}: ")
        else:
            os.environ[var] = input(f"Please provide your {var}: ")


# Load from .env if available, then check/set
from dotenv import load_dotenv

load_dotenv(override=True)

_set_if_undefined("AZURE_API_KEY")
_set_if_undefined("AZURE_API_BASE", is_secret=False)
_set_if_undefined("AZURE_API_VERSION")
_set_if_undefined("AZURE_MODEL_NAME", is_secret=False)
_set_if_undefined("TAVILY_API_KEY")
if not os.environ.get("USER_AGENT"):
    os.environ["USER_AGENT"] = (
        "hierarchical-research-app/0.1 (+https://example.com; contact@example.com)"
    )


# --- Tools Definition ---
tavily_tool = TavilySearch(max_results=5)


@tool
def scrape_webpages(urls: List[str]) -> str:
    """
    Scrape the content of the given list of URLs and return the combined text content.
    Each document is wrapped in a <Document> tag with its title as metadata.
    """
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )


_TEMP_DIRECTORY = TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)


def _slugify(value: str, max_length: int = 50) -> str:
    """
    Lightweight slugifier for run names to keep MLflow happy.
    """
    safe = "".join(ch if ch.isalnum() else "-" for ch in value)
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-")[:max_length] or "run"


def setup_mlflow_logging(
    experiment_name: str = "hierarchical_research",
    tracking_uri: Optional[str] = None,
):
    """
    Configure MLflow logging with LangGraph autologging enabled.
    """
    if not tracking_uri:
        default_dir = Path(__file__).parent / "mlruns"
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", f"file://{default_dir}")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    try:
        import mlflow.langgraph as mlflow_langgraph

        mlflow_langgraph.autolog()
    except ImportError:
        try:
            import mlflow.langchain as mlflow_langchain

            mlflow_langchain.autolog()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: MLflow autologging unavailable: {exc}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Warning: Failed to enable mlflow.langgraph.autolog: {exc}")


FINAL_OUTPUT_INSTRUCTIONS = (
    "Produce the final research report text only. Do not add prefaces, meta commentary, "
    "follow-up offers, or prompts to the user. If you create files, use 'outline.txt' "
    "for outlines and 'report.txt' for drafts; respond with the report content only."
)


def _default_outline_name(name: Optional[str]) -> str:
    return name or "outline.txt"


@tool
def create_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str, "File path to save the outline."] = "outline.txt",
) -> Annotated[str, "Path of the saved outline file."]:
    """
    Create an outline from a list of points and save it to the specified file.
    Returns the path of the saved outline file.
    """
    file_name = _default_outline_name(file_name)
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"Outline saved to {file_name}"


@tool
def read_document(
    file_name: Annotated[str, "File path to read the document from."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
) -> str:
    """
    Read lines from a document file between the specified start and end lines.
    Returns the extracted text.
    """
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    if start is None:
        start = 0
    return "\n".join(lines[start:end])


@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document."] = "report.txt",
) -> Annotated[str, "Path of the saved document file."]:
    """
    Write the provided content to the specified file.
    Returns the path of the saved document file.
    """
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.write(content)
    return f"Document saved to {file_name}"


@tool
def edit_document(
    file_name: Annotated[str, "Path of the document to be edited."],
    inserts: Annotated[
        Dict[int, str],
        "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
    ],
) -> Annotated[str, "Path of the edited document file."]:
    """
    Edit a document by inserting text at specified line numbers.
    Returns the path of the edited document file.
    """
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    sorted_inserts = sorted(inserts.items())
    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)
    return f"Document edited and saved to {file_name}"


repl = PythonREPL()

DOC_WRITER_MEMORY: List[str] = []


@tool
def doc_writer_remember(
    note: Annotated[str, "Note to store for the document writer."],
) -> str:
    """
    Store a note in the doc writer's private memory.
    """
    DOC_WRITER_MEMORY.append(note)
    return "Saved."


@tool
def doc_writer_recall(
    query: Annotated[
        Optional[str], "Optional substring filter; if empty returns all notes."
    ] = None,
) -> str:
    """
    Retrieve stored notes from the doc writer's private memory.
    """
    if not DOC_WRITER_MEMORY:
        return "No memories found."
    if not query:
        return "\n".join(DOC_WRITER_MEMORY)
    query_lower = query.lower()
    hits = [n for n in DOC_WRITER_MEMORY if query_lower in n.lower()]
    return "\n".join(hits) if hits else "No matching memories found."


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """
    Execute the provided Python code in a REPL environment and return the output.
    Returns the code executed and its stdout.
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"


# --- State Definition ---
class State(MessagesState):
    next: str


# --- LLM Instantiation ---
try:
    llm = AzureChatOpenAI(
        temperature=0,
        azure_deployment=os.environ["AZURE_MODEL_NAME"],
        api_key=os.environ["AZURE_API_KEY"],
        azure_endpoint=os.environ["AZURE_API_BASE"],
        api_version=os.environ["AZURE_API_VERSION"],
    )
    print("AzureChatOpenAI initialized successfully.")
except Exception as e:
    print(f"Error initializing AzureChatOpenAI: {e}")
    print(
        "Please ensure AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, OPENAI_API_VERSION, and AZURE_OPENAI_CHAT_DEPLOYMENT_NAME are correctly set."
    )
    exit()


# Shared system_info for LLM access
class SharedSystemInfo:
    def __init__(self, llm_instance):
        self.system_info = {"llm": llm_instance}


shared = SharedSystemInfo(llm)


# --- Supervisor Node Factory ---
def make_supervisor_node(llm_supervisor: BaseChatModel, members: list[str]) -> callable:
    options = ["FINISH"] + members
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    class Router(TypedDict):
        next: Literal[*options]

    structured_llm_supervisor = llm_supervisor.with_structured_output(Router)

    def supervisor_node_fn(state: State) -> Command[Literal[*members, "__end__"]]:
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        response = structured_llm_supervisor.invoke(messages)
        goto = response["next"]
        if goto == "FINISH":
            goto = END
        return Command(goto=goto, update={"next": goto})

    return supervisor_node_fn


# --- Agent and Node Definitions (ALL nodes are defined globally) ---

# Research Team Agents & Nodes
search_agent_runnable = create_react_agent(llm, tools=[tavily_tool])
web_scraper_agent_runnable = create_react_agent(llm, tools=[scrape_webpages])


def search_node(state: State) -> Command[Literal["research_team_supervisor_node"]]:
    result = search_agent_runnable.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="search")
            ]
        },
        goto="research_team_supervisor_node",
    )


def web_scraper_node(state: State) -> Command[Literal["research_team_supervisor_node"]]:
    result = web_scraper_agent_runnable.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="web_scraper")
            ]
        },
        goto="research_team_supervisor_node",
    )


# Document Writing Team Agents & Nodes
doc_writer_agent_runnable = create_react_agent(
    llm,
    tools=[
        write_document,
        edit_document,
        read_document,
        doc_writer_remember,
        doc_writer_recall,
    ],
    prompt=(
        "You can read, write and edit documents based on note-taker's outlines. "
        "Do not ask follow-up questions. Always pick deterministic filenames: "
        "use 'report.txt' for full drafts and 'outline.txt' for outlines when needed. "
        "When delivering the final result, return only the report text with no preamble, "
        "no meta commentary, and no follow-up offers. Do not include thank-yous."
    ),
)
note_taking_agent_runnable = create_react_agent(
    llm,
    tools=[create_outline, read_document],
    prompt=(
        "You can read documents and create outlines for the document writer. "
        "Do not ask follow-up questions. Always save outlines to 'outline.txt'."
    ),
)
chart_generating_agent_runnable = create_react_agent(
    llm, tools=[read_document, python_repl_tool]
)


def doc_writer_node(
    state: State,
) -> Command[Literal["doc_writing_team_supervisor_node"]]:
    result = doc_writer_agent_runnable.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="doc_writer")
            ]
        },
        goto="doc_writing_team_supervisor_node",
    )


def note_taking_node(
    state: State,
) -> Command[Literal["doc_writing_team_supervisor_node"]]:
    result = note_taking_agent_runnable.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="note_taker")
            ]
        },
        goto="doc_writing_team_supervisor_node",
    )


def chart_generating_node(
    state: State,
) -> Command[Literal["doc_writing_team_supervisor_node"]]:
    result = chart_generating_agent_runnable.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content, name="chart_generator"
                )
            ]
        },
        goto="doc_writing_team_supervisor_node",
    )


# Supervisor Function Definitions
research_team_supervisor_fn = make_supervisor_node(llm, ["search", "web_scraper"])
doc_writing_team_supervisor_fn = make_supervisor_node(
    llm, ["doc_writer", "note_taker", "chart_generator"]
)
top_level_supervisor_fn = make_supervisor_node(llm, ["research_team", "writing_team"])


# Subgraph-calling nodes defined globally, accepting graphs as arguments
def call_research_team_node(
    state: State, research_graph: StateGraph
) -> Command[Literal["top_level_supervisor_node"]]:
    last_supervisor_message = state["messages"][-1]
    response = research_graph.invoke({"messages": [last_supervisor_message]})
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response["messages"][-1].content, name="research_team"
                )
            ]
        },
        goto="top_level_supervisor_node",
    )


def call_paper_writing_team_node(
    state: State, paper_writing_graph: StateGraph
) -> Command[Literal["top_level_supervisor_node"]]:
    last_supervisor_message = state["messages"][-1]
    response = paper_writing_graph.invoke({"messages": [last_supervisor_message]})
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response["messages"][-1].content, name="writing_team"
                )
            ]
        },
        goto="top_level_supervisor_node",
    )


# --- Entry point for run_analysis_AIR.py ---
def get_graph_runnable(query: Any = None, entry_point_hint: Any = None):
    """
    Dynamically builds and compiles the graph on each call to ensure
    function patches from the adapter are applied.
    """
    # --- Research Team Graph Compilation ---
    research_builder = StateGraph(State)
    research_builder.add_node(
        "research_team_supervisor_node", research_team_supervisor_fn
    )
    research_builder.add_node("search", search_node)
    research_builder.add_node("web_scraper", web_scraper_node)
    research_builder.add_edge(START, "research_team_supervisor_node")
    research_builder.add_conditional_edges(
        "research_team_supervisor_node",
        lambda x: x["next"],
        {"search": "search", "web_scraper": "web_scraper", END: END},
    )
    research_graph = research_builder.compile()

    # --- Document Writing Team Graph Compilation ---
    paper_writing_builder = StateGraph(State)
    paper_writing_builder.add_node(
        "doc_writing_team_supervisor_node", doc_writing_team_supervisor_fn
    )
    paper_writing_builder.add_node("doc_writer", doc_writer_node)
    paper_writing_builder.add_node("note_taker", note_taking_node)
    paper_writing_builder.add_node("chart_generator", chart_generating_node)
    paper_writing_builder.add_edge(START, "doc_writing_team_supervisor_node")
    paper_writing_builder.add_conditional_edges(
        "doc_writing_team_supervisor_node",
        lambda x: x["next"],
        {
            "doc_writer": "doc_writer",
            "note_taker": "note_taker",
            "chart_generator": "chart_generator",
            END: END,
        },
    )
    paper_writing_graph = paper_writing_builder.compile()

    # --- Top-Level Graph Building ---
    super_builder = StateGraph(State)
    super_builder.add_node("top_level_supervisor_node", top_level_supervisor_fn)

    # Use functools.partial to bind the compiled graphs to the global node functions
    bound_research_node = partial(
        call_research_team_node, research_graph=research_graph
    )
    bound_writing_node = partial(
        call_paper_writing_team_node, paper_writing_graph=paper_writing_graph
    )

    super_builder.add_node("research_team", bound_research_node)
    super_builder.add_node("writing_team", bound_writing_node)

    super_builder.add_edge(START, "top_level_supervisor_node")
    super_builder.add_conditional_edges(
        "top_level_supervisor_node",
        lambda x: x["next"],
        {"research_team": "research_team", "writing_team": "writing_team", END: END},
    )
    super_graph = super_builder.compile()

    # Return the newly compiled graph and the initial state template
    initial_state_template = {"messages": [], "next": ""}
    return super_graph, initial_state_template


def run_hierarchical_research_query(
    query: str, recursion_limit: int = 150
) -> str:
    """
    Run the hierarchical research graph for a single query and return the final answer.
    """
    super_graph, _ = get_graph_runnable()
    final_state = super_graph.invoke(
        {"messages": [SystemMessage(content=FINAL_OUTPUT_INSTRUCTIONS), HumanMessage(content=query)]},
        {"recursion_limit": recursion_limit},
    )
    messages = final_state.get("messages", [])
    if not messages:
        return ""
    raw = messages[-1].content if hasattr(messages[-1], "content") else str(
        messages[-1]
    )
    return _clean_final_answer(raw)


def _clean_final_answer(text: str) -> str:
    """
    Remove common meta/offers/thank-you lines from the model output.
    """
    banned_phrases = [
        "if you'd like",
        "let me know",
        "i can expand",
        "if you’d like",
        "thank you",
        "thanks for",
        "just let me know",
    ]
    lines = []
    for line in text.splitlines():
        lower = line.strip().lower()
        if any(phrase in lower for phrase in banned_phrases):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    return cleaned or text.strip()


def run_multiple_queries(
    queries: Iterable[str], recursion_limit: int = 150
) -> List[str]:
    """
    Run the hierarchical research graph for multiple queries.
    """
    return [
        run_hierarchical_research_query(query, recursion_limit)
        for query in queries
    ]


def run_query_with_mlflow(
    query: str,
    recursion_limit: int = 150,
) -> str:
    """
    Wrapper that optionally logs the query/answer to MLflow.
    """
    mlflow_run = mlflow.start_run(run_name=_slugify(query))
    mlflow.log_param("query", query)
    try:
        answer = run_hierarchical_research_query(query, recursion_limit)
        try:
            mlflow.log_text(answer, artifact_file="answer.txt")
        except Exception:
            temp_path = WORKING_DIRECTORY / "answer.txt"
            temp_path.write_text(answer)
            mlflow.log_artifact(temp_path)
        # Log any generated artifacts (outline/report) created in the working directory.
        for item in WORKING_DIRECTORY.iterdir():
            try:
                mlflow.log_artifact(item)
            except Exception:
                pass
        return answer
    finally:
        if mlflow_run:
            mlflow.end_run()


if __name__ == "__main__":
    example_query = "Research AI agents and write a brief report about them."
    print(run_hierarchical_research_query(example_query))
