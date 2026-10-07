"""Advanced example showing explicit executor usage and custom grep tool."""

import argparse
import contextlib
import os
import shlex
from collections.abc import Sequence

import mlflow
from dotenv import load_dotenv
from pydantic import Field, SecretStr

from openhands.sdk import (
    LLM,
    Action,
    Agent,
    Conversation,
    Event,
    ImageContent,
    LLMConvertibleEvent,
    Observation,
    TextContent,
    ToolDefinition,
    get_logger,
)
from openhands.sdk.tool import (
    Tool,
    ToolExecutor,
    register_tool,
)
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import (
    TerminalAction,
    TerminalExecutor,
    TerminalTool,
)

load_dotenv()

logger = get_logger(__name__)

DEFAULT_EXPERIMENT_NAME = "openhands_grep_example"
DEFAULT_MLFLOW_URI = "http://localhost:5000"
DEFAULT_TASK_1 = (
    "Hello! Can you use the grep tool to find all files "
    "containing the word 'class' in this project, then create a summary file listing them? "
    "Use the pattern 'class' to search and include only Python files with '*.py'."
)
DEFAULT_TASK_2 = "Great! Now delete that file."

# --- Action / Observation ---


class GrepAction(Action):
    pattern: str = Field(description="Regex to search for")
    path: str = Field(
        default=".", description="Directory to search (absolute or relative)"
    )
    include: str | None = Field(
        default=None, description="Optional glob to filter files (e.g. '*.py')"
    )


class GrepObservation(Observation):
    matches: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    count: int = 0

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if not self.count:
            return [TextContent(text="No matches found.")]
        files_list = "\n".join(f"- {f}" for f in self.files[:20])
        sample = "\n".join(self.matches[:10])
        more = "\n..." if self.count > 10 else ""
        ret = (
            f"Found {self.count} matching lines.\n"
            f"Files:\n{files_list}\n"
            f"Sample:\n{sample}{more}"
        )
        return [TextContent(text=ret)]


# --- Executor ---


class GrepExecutor(ToolExecutor[GrepAction, GrepObservation]):
    def __init__(self, terminal: TerminalExecutor):
        self.terminal: TerminalExecutor = terminal

    def __call__(
        self, action: GrepAction, conversation=None
    ) -> GrepObservation:  # noqa: ARG002
        root = os.path.abspath(action.path)
        pat = shlex.quote(action.pattern)
        root_q = shlex.quote(root)

        # Use grep -r; add --include when provided
        if action.include:
            inc = shlex.quote(action.include)
            cmd = f"grep -rHnE --include {inc} {pat} {root_q} 2>/dev/null | head -100"
        else:
            cmd = f"grep -rHnE {pat} {root_q} 2>/dev/null | head -100"

        result = self.terminal(TerminalAction(command=cmd))

        matches: list[str] = []
        files: set[str] = set()

        # grep returns exit code 1 when no matches; treat as empty
        output_text = result.text

        if output_text.strip():
            for line in output_text.strip().splitlines():
                matches.append(line)
                # Expect "path:line:content" — take the file part before first ":"
                file_path = line.split(":", 1)[0]
                if file_path:
                    files.add(os.path.abspath(file_path))

        return GrepObservation(matches=matches, files=sorted(files), count=len(matches))


# Tool description
_GREP_DESCRIPTION = """Fast content search tool.
* Searches file contents using regular expressions
* Supports full regex syntax (eg. "log.*Error", "function\\s+\\w+", etc.)
* Filter files by pattern with the include parameter (eg. "*.js", "*.{ts,tsx}")
* Returns matching file paths sorted by modification time.
* Only the first 100 results are returned. Consider narrowing your search with stricter regex patterns or provide path parameter if you need more results.
* Use this tool when you need to find files containing specific patterns
* When you are doing an open ended search that may require multiple rounds of globbing and grepping, use the Agent tool instead
"""  # noqa: E501


# --- Tool Definition ---


class GrepTool(ToolDefinition[GrepAction, GrepObservation]):
    """A custom grep tool that searches file contents using regular expressions."""

    @classmethod
    def create(
        cls, conv_state, terminal_executor: TerminalExecutor | None = None
    ) -> Sequence[ToolDefinition]:
        """Create GrepTool instance with a GrepExecutor.

        Args:
            conv_state: Conversation state to get working directory from.
            terminal_executor: Optional terminal executor to reuse. If not provided,
                         a new one will be created.

        Returns:
            A sequence containing a single GrepTool instance.
        """
        if terminal_executor is None:
            terminal_executor = TerminalExecutor(
                working_dir=conv_state.workspace.working_dir
            )
        grep_executor = GrepExecutor(terminal_executor)

        return [
            cls(
                description=_GREP_DESCRIPTION,
                action_type=GrepAction,
                observation_type=GrepObservation,
                executor=grep_executor,
            )
        ]


# Configure LLM
api_key = os.getenv("AZURE_OPENAI_API_KEY")
assert api_key is not None, "LLM_API_KEY environment variable is not set."
model = f"azure/{os.getenv("AZURE_MODEL_NAME")}"
base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
llm = LLM(
    usage_id="agent",
    model=model,
    base_url=base_url,
    api_key=SecretStr(api_key),
)

# Tools - demonstrating both simplified and advanced patterns
cwd = os.getcwd()


def _make_bash_and_grep_tools(conv_state) -> list[ToolDefinition]:
    """Create terminal and custom grep tools sharing one executor."""

    terminal_executor = TerminalExecutor(working_dir=conv_state.workspace.working_dir)
    # terminal_tool = terminal_tool.set_executor(executor=terminal_executor)
    terminal_tool = TerminalTool.create(conv_state, executor=terminal_executor)[0]

    # Use the GrepTool.create() method with shared terminal_executor
    grep_tool = GrepTool.create(conv_state, terminal_executor=terminal_executor)[0]

    return [terminal_tool, grep_tool]



tools = [
    Tool(name=FileEditorTool.name),
    Tool(name="BashAndGrepToolSet"),
]

# Agent
agent = Agent(llm=llm, tools=tools)

llm_messages = []  # collect raw LLM messages


def conversation_callback(event: Event):
    if isinstance(event, LLMConvertibleEvent):
        llm_messages.append(event.to_llm_message())


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
    parser.add_argument("--task_1", default=DEFAULT_TASK_1)
    parser.add_argument("--task_2", default=DEFAULT_TASK_2)
    parser.add_argument("--exp_name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--mlflow_uri", default=DEFAULT_MLFLOW_URI)
    return parser.parse_args()

conversation = Conversation(
    agent=agent, callbacks=[conversation_callback], workspace=cwd
)
args = parse_args()
task_1 = args.task_1
task_2 = args.task_2

exp_name = args.exp_name
mlflow_uri = args.mlflow_uri
mlflow_setup(exp_name, mlflow_uri)
register_tool("BashAndGrepToolSet", _make_bash_and_grep_tools)

with mlflow.start_run(run_name="openhands-grep-session"):
    trace_id: str | None = None
    mlflow.log_params(
        {
            "task_1": task_1,
            "task_2": task_2,
            "experiment_name": exp_name,
            "mlflow_uri": mlflow_uri,
            "model": os.getenv("AZURE_MODEL_NAME"),
        }
    )

    with _mlflow_span("conversation_run_1", {"task": "grep_and_summarize"}):
        conversation.send_message(task_1)
        conversation.run()
        if hasattr(mlflow, "get_active_trace_id"):
            trace_id = mlflow.get_active_trace_id()

    with _mlflow_span("conversation_run_2", {"task": "delete_summary"}):
        conversation.send_message(task_2)
        conversation.run()
        if hasattr(mlflow, "get_active_trace_id"):
            trace_id = mlflow.get_active_trace_id()

    llm_messages_text = "\n\n".join(
        f"Message {i}: {str(message)}" for i, message in enumerate(llm_messages)
    )
    mlflow.log_text(task_1, "task_1.txt")
    mlflow.log_text(task_2, "task_2.txt")
    mlflow.log_text(llm_messages_text, "llm_messages.txt")
    mlflow.log_metric("llm_message_count", len(llm_messages))
    mlflow.log_metric("example_cost", llm.metrics.accumulated_cost)

    if hasattr(mlflow, "flush_trace_async_logging"):
        mlflow.flush_trace_async_logging()
    if trace_id and hasattr(mlflow, "get_trace"):
        trace = mlflow.get_trace(trace_id, silent=True, flush=True)
        if trace is not None:
            mlflow.log_text(trace.to_json(pretty=True), f"traces/{trace_id}/traces.json")
            mlflow.log_text(trace_id, "latest_trace_id.txt")

print("=" * 100)
print("Conversation finished. Got the following LLM messages:")
for i, message in enumerate(llm_messages):
    print(f"Message {i}: {str(message)[:200]}")

# Report cost
cost = llm.metrics.accumulated_cost
print(f"EXAMPLE_COST: {cost}")
