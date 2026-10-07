import os
from urllib.parse import urlparse

from crewai import Agent, LLM
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from typing import List, Tuple

from tools.mcp_tools import mcp_local
from tools.code_interpreter import InterpreterTool


MCPS = [mcp_local]
interpreter_tool = InterpreterTool()

def _azure_root(url: str | None) -> str | None:
    if not url:
        return None

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url.rstrip("/")

    return f"{parsed.scheme}://{parsed.netloc}"


_azure_model = os.getenv('AZURE_DEPLOYMENT_NAME') or os.getenv('AZURE_MODEL_NAME')
_azure_endpoint_root = _azure_root(os.getenv("AZURE_OPENAI_ENDPOINT"))
llm = LLM(
    model=f"azure/{_azure_model}",
    api_version=os.getenv("AZURE_API_VERSION"),
    api_key=os.getenv("AZURE_API_KEY"),
    endpoint=f"{_azure_endpoint_root}/openai/deployments/{_azure_model}" if _azure_endpoint_root else None,
    temperature=0.0,
)


class CodeAct(Agent):
    # Reasoning enabled to allow the agent to think step by step before answering as int ReAct.
    def __init__(
        self,
        role: str = "CodeAct Research & Execution Agent",
        goal: Tuple[str, ...] = (
            "Plan with short scratchpad steps and use tools when helpful."
            "Return a concise final answer that starts with 'FINAL:'"
        ),
        backstory: str = "You are a pragmatic assistant that writes and executes code to answer user questions.",
        tools: List = [interpreter_tool],
        llm: LLM = llm,
        reasoning: bool = True,
        mcps: List = MCPS,
        verbose: bool = True,
        max_iter: int = 8,
    ):
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tools,
            verbose=verbose,
            llm=llm,
            max_iter=max_iter,
            reasoning=reasoning,
            mcps=mcps,
        )

    def get_server_arguments_from_mcp(
        self, mcp, excluded_arguments: List[str] = ["tool_filter", "cache_tools_list"]
    ):
        arguments = mcp.model_dump()
        for arg in excluded_arguments:
            if arg in arguments:
                arguments.pop(arg, None)

        # Add any additional required arguments here
        if "command" in arguments:
            server_params = StdioServerParameters(
                command=arguments.pop("command"),
                args=arguments.pop("args", []),
                env=arguments.pop("env", None),
            )
        else:
            is_streamable = arguments.pop("streamable", False)
            if is_streamable:
                arguments["transport"] = "streamable-http"
            server_params = arguments

        return server_params

    def get_all_tools(self):
        base_tools = [tool.name for tool in self.tools]
        mcp_tools = []

        for mcp in self.mcps:
            try:
                # Wrap the actual MCP server instance
                server_arguments = self.get_server_arguments_from_mcp(mcp)
                with MCPServerAdapter(server_arguments, connect_timeout=90) as tools:
                    for tool in tools:
                        mcp_tools.append(tool.name)
            except Exception as e:
                print(f"Error fetching tools from MCP server {mcp}: {e}")

        return base_tools + mcp_tools
