from langchain.agents import create_agent
from langchain.agents import middleware as mw
from langchain.agents.middleware import (
    before_agent,
    before_model,
    after_agent,
    after_model,
    AgentState,
    AgentMiddleware,
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    hook_config
)
# from tools import tool1, tool2
from langchain.agents.middleware import HumanInTheLoopMiddleware
from dotenv import load_dotenv
import os
from langgraph.func import entrypoint, task
from langchain_openai import AzureChatOpenAI
from langgraph.graph import MessagesState
from langchain.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.types import Command
from typing import Callable
from langchain_core.messages import BaseMessage
from langgraph.runtime import Runtime
import types
from typing import Callable, Any
from typing_extensions import Literal
from pydantic import BaseModel, Field
import inspect
import ast
from langchain.tools.tool_node import ToolCallRequest


import importlib.util
from inspect import getmembers, isfunction

load_dotenv(override=True)


llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
    model=os.getenv("AZURE_MODEL_NAME"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_API_KEY"),
    temperature=0.0,
)

# *************************************************************************
#                   UTIL FUNCTIONS USED BY CustomMiddleWare
# *************************************************************************

def get_top_level_function_names(file_path: str) -> list[str]:
    """
    Returns names of top-level functions (defined directly in module body).
    Excludes class methods and nested functions.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    ] 

def get_tools_dict(fixed_tools_file_path: str, dynamic_tools_file_path: str):
    """Return the dict of tools. Each entry is of the form 'name':<tool handle> """
    
    names = set(get_top_level_function_names(fixed_tools_file_path))
    spec = importlib.util.spec_from_file_location("dynamic_module", fixed_tools_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fixed_tools_dict={
        name: obj
        for name, obj in vars(module).items()
        if name in names and inspect.isfunction(obj)
    }

    names = set(get_top_level_function_names(dynamic_tools_file_path))
    spec = importlib.util.spec_from_file_location("dynamic_module", dynamic_tools_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dynamic_tools_dict={
        name: obj
        for name, obj in vars(module).items()
        if name in names and inspect.isfunction(obj)
    }

    # merge these two dicts together
    full_tool_dict = fixed_tools_dict | dynamic_tools_dict
    return full_tool_dict 


def get_tools_list(fixed_tools_file_path: str, dynamic_tools_file_path: str):
    """Return the list of tools. Each item in the list is just the tool handle."""

    fixed_tools=[]
    names = set(get_top_level_function_names(fixed_tools_file_path))
    spec = importlib.util.spec_from_file_location("dynamic_module", fixed_tools_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name, obj in vars(module).items():
        if name in names and inspect.isfunction(obj):
            fixed_tools += [obj]

    dynamic_tools=[]
    names = set(get_top_level_function_names(dynamic_tools_file_path))
    spec = importlib.util.spec_from_file_location("dynamic_module", dynamic_tools_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name, obj in vars(module).items():
        if name in names and inspect.isfunction(obj):
            dynamic_tools += [obj]

    tools = fixed_tools + dynamic_tools

    return tools

# *************************************************************************
#                   GET FUNCTIONS USED BY main.py
# *************************************************************************

def get_all_tools():
    fixed_tools_file_path = "all_tools.py"
    dynamic_tools_file_path = "scratchpad.py"

    return get_tools_dict(fixed_tools_file_path,dynamic_tools_file_path)    

def createAgent():
    userQueryAgent = create_agent(
        model=llm,
        middleware=[CustomMiddleware()],
    )
    return userQueryAgent    

def get_model_used():
    return llm

# *************************************************************************
#        MAIN MIDDLEWARE Class USED BY ReAct Agent FOR FINER CONTROL
# *************************************************************************


class CustomMiddleware(AgentMiddleware):

    def before_agent(self, state, runtime) -> dict[str, Any] | None:
        # print(f"Before Model\nState: {state}\nRuntime: {runtime}")
        return None

    def before_model(self, state: AgentState, runtime: Runtime):
        print(f"[LOG] Before model call ({len(state['messages'])} messages)")
        print(f"\n [LOG] Before model call messages={state['messages']}")
        return None        


    def wrap_model_call(self, request, handler):
        print(f"Tools before: {request.tools}")

        fixed_tools_file_path = "all_tools.py"
        dynamic_tools_file_path = "scratchpad.py"

        tools = get_tools_list(fixed_tools_file_path,dynamic_tools_file_path)
       
        request_new = request.override(tools=tools)
        print(f"Tools after: {request_new.tools}\n")
        print(f"request_new = {request_new}")
        return handler(request_new)


    def after_model(self,state,runtime)-> dict[str, Any] | None:
       
        print("after model")        

        fixed_tools_file_path = "all_tools.py"
        dynamic_tools_file_path = "scratchpad.py"
        tools_list = get_tools_dict(fixed_tools_file_path,dynamic_tools_file_path)

        messages = state["messages"]
        last_message = messages[-1]

        while last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                toolFnHandle = tools_list[tool_name]

                toolFnInputs = tool_call["args"]

                inputList=list(toolFnInputs.values())

                # if len(inputList)==1:
                #     inputToTool=inputList[0]
                #     observation=toolFnHandle(inputToTool)
                # else:
                #     observation=toolFnHandle(*inputList)

                observation=toolFnHandle(*inputList)
                observation=f'{observation}'
                state["messages"].append(ToolMessage(content=observation, name=tool_name, tool_call_id=tool_call["id"]))
                
            print(state['messages'])
            state["messages"] += [llm.invoke(state["messages"])]
            messages = state["messages"]
            last_message = messages[-1]        

        return state


    
    def after_agent(self, state, runtime) -> dict[str, Any] | None:
        # print(f'\n\nafter_agent state = {state}')
        # print(f"\nRuntime:{runtime}\n\n")
        messages = state["messages"]
        last_message = messages[-1]        
       
            
        return state








