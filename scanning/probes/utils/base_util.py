import ast
import io
import json
import os
import random
import re
import shlex
import textwrap
import tokenize
from collections.abc import Callable
from typing import Any

from node_spec.structure_schema import FlowSpec, InputPort, NodeSpec
from pydantic import BaseModel, ConfigDict, Field
from scanning.evaluators.evaluator_runtime import record_estimated_llm_usage

def get_line_number(line,text):
    for i, full_line in enumerate(text.splitlines()):
        if full_line.lstrip(' \t') == line:
            return i

def get_indentation(line, text):
    for full_line in text.splitlines():
        if full_line.lstrip(' \t') == line:
            i = 0
            while i < len(full_line) and full_line[i] in (' ', '\t'):
                i += 1
            return full_line[:i]
    
    raise ValueError("Line not found")

prompt_variable_name_options = [
            "system_prompt",
            "system_message",
            "instructions",
            "system_instructions",
            "assistant_instructions",
            "developer_prompt",
            "developer_message",
            "agent_prompt",
            "initial_prompt",
            "default_system_message",
            "prompt",
            "backstory",
        ]


def normalize_flows(
    full_node_spec: NodeSpec,
    flows: list[FlowSpec] | None = None,
) -> list[FlowSpec]:
    """
    Return the flows to inspect for the current probe instance.

    Flows with the ``error:error`` signature are skipped because they
    represent failed executions that are not meaningful probe candidates.
    """
    if flows is None:
        flows = full_node_spec.flows or []

    normalized: list[FlowSpec] = []
    for flow in flows:
        flow_signature_key = str(getattr(flow, "flow_signature_key", "") or "")
        if "error:error" in flow_signature_key:
            continue
        normalized.append(flow)
    return normalized


def get_different_flows(
    flows: list[FlowSpec],
    selected_flow: FlowSpec,
) -> list[FlowSpec]:
    """
    Return flows that are different from ``selected_flow``.
    """
    selected_flow_id = getattr(selected_flow, "flow_id", None)
    different_flows: list[FlowSpec] = []
    for candidate in flows:
        candidate_flow_id = getattr(candidate, "flow_id", None)
        if selected_flow_id is not None and candidate_flow_id is not None:
            if candidate_flow_id == selected_flow_id:
                continue
        elif candidate is selected_flow:
            continue
        different_flows.append(candidate)
    return different_flows


def get_agent_using_the_tool_from_flow(flow: FlowSpec, tool_id: str) -> str:
    """
    Infer the agent that initiated the tool call for ``tool_id`` in this flow.
    """
    events = getattr(flow, "events", None) or []
    invoked_agent_ids = get_unique_invoked_agents_ids(flow)

    for event_index, event in enumerate(events):
        tool_calls = getattr(event, "tool_calls", None) or []
        if not any(getattr(tool_call, "tool_id", "") == tool_id for tool_call in tool_calls):
            continue

        for previous_index in range(event_index, -1, -1):
            node_id = getattr(events[previous_index], "node_id", "") or ""
            if node_id and node_id in invoked_agent_ids:
                return node_id

    raise ValueError(
        f"Could not identify the agent that invoked tool '{tool_id}'."
    )


def query_azure_chat_openai_structured_output(
    prompt: str,
    output_schema: type[Any],
    *,
    temperature: float = 0,
) -> Any | None:
    """
    Query AzureChatOpenAI with structured output and return parsed schema data.

    Returns ``None`` when dependencies, credentials, or the request itself fail.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path="rift.env")
    except Exception:
        pass

    try:
        from langchain_openai import AzureChatOpenAI
    except Exception:
        return None

    deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME") or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    model_name = os.getenv("AZURE_MODEL_NAME")

    if not deployment_name or not api_version or not azure_endpoint or not api_key:
        return None

    try:
        llm = AzureChatOpenAI(
            deployment_name=deployment_name,
            model=model_name,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            temperature=temperature,
        )
        response = llm.with_structured_output(output_schema).invoke(prompt)
        record_estimated_llm_usage(
            prompt=prompt,
            response=response,
            model_name=model_name or deployment_name,
        )
        return response
    except Exception as e:
        print(e)
        return None


class GeneratedToolArguments(BaseModel):
    """Strict response model for dynamically named tool arguments."""

    model_config = ConfigDict(extra="forbid")

    arguments_json: str = Field(
        description=(
            "A JSON-encoded object containing one plausible structured "
            "argument object for invoking the tool."
        ),
    )


class PromptVariableReference(BaseModel):
    """The identifier a snippet uses to reference an agent system prompt."""

    model_config = ConfigDict(extra="forbid")

    variable_name: str | None = Field(
        default=None,
        description=(
            "A simple Python identifier for the variable containing the "
            "agent's system prompt, or null when none is referenced."
        ),
    )


def _get_tool_display_name(tool_node: Any) -> str:
    return str(
        getattr(tool_node, "id", None)
        or getattr(tool_node, "name", None)
        or "unknown_tool"
    )


def _format_tool_inputs(tool_node: Any) -> str:
    lines: list[str] = []
    for input_port in getattr(tool_node, "inputs", None) or []:
        lines.append(f"- name: {getattr(input_port, 'name', '')}")
        lines.append(f"  type: {getattr(input_port, 'dtype', '')}")
        lines.append(f"  required: {getattr(input_port, 'required', True)}")
        default_value = getattr(input_port, "default", None)
        if default_value is not None:
            serialized_default = json.dumps(
                default_value,
                ensure_ascii=True,
                default=str,
            )
            lines.append(f"  default: {serialized_default}")
        description = getattr(input_port, "description", None)
        if description:
            lines.append(f"  description: {description}")
    return "\n".join(lines) if lines else "- (not provided)"


def _normalize_generated_tool_arguments(
    tool_node: Any,
    arguments: Any,
) -> dict[str, Any]:
    inputs = list(getattr(tool_node, "inputs", None) or [])
    if not inputs:
        raise ValueError(
            f"Tool '{_get_tool_display_name(tool_node)}' does not define inputs for argument generation."
        )

    input_names = {getattr(input_port, "name", "") for input_port in inputs}
    raw_arguments = dict(arguments or {})
    normalized_arguments = {
        key: value
        for key, value in raw_arguments.items()
        if key in input_names
    }

    missing_required_inputs = [
        getattr(input_port, "name", "")
        for input_port in inputs
        if getattr(input_port, "required", True)
        and getattr(input_port, "name", "") not in normalized_arguments
    ]
    if missing_required_inputs:
        raise ValueError(
            "Generated tool arguments are missing required inputs for tool "
            f"'{_get_tool_display_name(tool_node)}': {missing_required_inputs}"
        )

    if not normalized_arguments:
        raise ValueError(
            f"Failed to generate any usable arguments for tool '{_get_tool_display_name(tool_node)}'."
        )
    return normalized_arguments


def generate_tool_arguments_with_llm(tool_node: Any) -> dict[str, Any]:
    """
    Generate one plausible argument object for a tool using its declared inputs.
    """
    if not getattr(tool_node, "inputs", None):
        raise ValueError(
            f"Tool '{_get_tool_display_name(tool_node)}' does not define tool_example_pairs and does not define inputs for argument generation."
        )

    tool_name = getattr(tool_node, "name", "") or _get_tool_display_name(tool_node)
    tool_description = getattr(tool_node, "description", "") or ""
    formatted_inputs = _format_tool_inputs(tool_node)

    prompt = f"""
You are generating one plausible structured argument object for invoking a tool in a multi-agent system.

Tool:
- id: {_get_tool_display_name(tool_node)}
- name: {tool_name}
- description: {tool_description}
- inputs:
{formatted_inputs}

Output schema:
- arguments_json: JSON-encoded object mapping declared input names to plausible values.

Requirements:
- Include every required input.
- Use only declared input names.
- Make the values realistic and consistent with the declared types and descriptions.
- Return exactly one concrete argument object, encoded as valid JSON in
  arguments_json. Do not include Markdown fences.
"""

    response = query_azure_chat_openai_structured_output(
        prompt=prompt,
        output_schema=GeneratedToolArguments,
        temperature=0,
    )
    if response is None:
        raise ValueError(
            f"Failed to generate plausible tool arguments for tool '{_get_tool_display_name(tool_node)}'."
        )

    generated_arguments_json = getattr(response, "arguments_json", None)
    if generated_arguments_json is None and isinstance(response, dict):
        generated_arguments_json = response.get("arguments_json")

    try:
        generated_arguments = json.loads(generated_arguments_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Generated tool arguments for tool '{_get_tool_display_name(tool_node)}' were not valid JSON."
        ) from exc

    if not isinstance(generated_arguments, dict):
        raise ValueError(
            f"Generated tool arguments for tool '{_get_tool_display_name(tool_node)}' must be a JSON object."
        )

    return _normalize_generated_tool_arguments(
        tool_node=tool_node,
        arguments=generated_arguments,
    )


def assert_tool_has_code_references(tool_node: Any, *, role: str = "selected") -> None:
    if tool_node is None:
        raise ValueError(f"The {role} tool was not found in node spec.")
    if not getattr(tool_node, "code_references", None):
        raise ValueError(
            f"The {role} tool '{_get_tool_display_name(tool_node)}' does not define code_references."
        )


def id_from_ref(node_ref: Any) -> str:
    """
    Normalize a node reference from a flow into a node id.
    """
    if isinstance(node_ref, dict):
        ids = node_ref.get("ids") or []
        if not ids:
            raise ValueError("Tool reference is missing an 'ids' entry.")
        return ids[0]
    return node_ref


def get_unique_invoked_tool_ids(flow: FlowSpec) -> list[str]:
    """
    Return invoked tool ids in first-seen order.
    """
    tool_ids: list[str] = []
    for tool_ref in getattr(flow, "invoked_tools", None) or []:
        tool_id = id_from_ref(tool_ref)
        if tool_id not in tool_ids:
            tool_ids.append(tool_id)
    return tool_ids

def get_unique_invoked_agents_ids(flow: FlowSpec) -> list[str]:
    """
    Return invoked agent ids in first-seen order.
    """
    agent_ids: list[str] = []
    for agent_ref in getattr(flow, "invoked_agents", None) or []:
        agent_id = id_from_ref(agent_ref)
        if agent_id not in agent_ids:
            agent_ids.append(agent_id)
    return agent_ids


def get_agent_tools(malicious_node_spec: NodeSpec, agent_id: str) -> list[str]:
    """
    Return tool ids attached to ``agent_id`` in list order.
    """
    agent_node = malicious_node_spec.get_node(agent_id)
    if agent_node is None:
        raise ValueError(f"Agent '{agent_id}' was not found in node spec.")

    agent_tools: list[str] = []
    for tool in agent_node.list_tools():
        tool_ids = tool.get("ids") or []
        if tool_ids:
            agent_tools.append(tool_ids[0])
    return agent_tools


def is_local_tool_node(tool_node: Any) -> bool:
    """
    Return whether a tool is local to the system under test.
    """
    return (
        not getattr(tool_node, "access_external_resource", False)
        and not getattr(tool_node, "access_internal_resource", False)
    )


def select_attacked_tool_node(
    malicious_node_spec: NodeSpec,
    flow: FlowSpec,
    guidance: dict[str, Any] | None = None,
    candidate: Callable[[Any], bool] | None = None,
    error_message: str = "The selected flow does not invoke any tools.",
) -> tuple[str, Any]:
    """
    Select the attacked tool from a flow, optionally constrained by predicate.
    """
    guidance = guidance or {}
    attacked_tool_node_id = guidance.get("attacked_tool_node_id")
    if attacked_tool_node_id:
        attacked_tool_node = malicious_node_spec.get_node(
            node_id=attacked_tool_node_id
        )
        if attacked_tool_node is None:
            raise ValueError(
                f"Guided attacked tool '{attacked_tool_node_id}' was not found."
            )
        if candidate is not None and not candidate(attacked_tool_node):
            raise ValueError(
                f"Guided attacked tool '{attacked_tool_node_id}' does not satisfy the probe selection predicate."
            )
        assert_tool_has_code_references(attacked_tool_node, role="attacked")
        return attacked_tool_node_id, attacked_tool_node

    for tool_id in get_unique_invoked_tool_ids(flow):
        tool_node = malicious_node_spec.get_node(node_id=tool_id)
        if tool_node is None:
            raise ValueError(f"Invoked tool '{tool_id}' was not found in node spec.")
        if candidate is not None and not candidate(tool_node):
            continue
        if not getattr(tool_node, "code_references", None):
            continue
        return tool_id, tool_node

    raise ValueError(error_message)


def select_attacked_tool_node_base_on_agent(
    malicious_node_spec: NodeSpec,
    flow: FlowSpec,
    guidance: dict[str, Any] | None = None,
    candidate: Callable[[Any], bool] | None = None,
    error_message: str = "The selected flow does not invoke any agents.",
) -> tuple[str, Any]:
    """
    Select an invoked attacked tool from an agent that also has an unused tool
    with inputs.

    This helper supports the own-tool probe families where the attacked tool
    must be part of the flow, while another tool under the same invoked agent
    remains available for malicious invocation with concrete arguments.
    """
    guidance = guidance or {}
    attacked_tool_node_id = guidance.get("attacked_tool_node_id")
    if attacked_tool_node_id:
        attacked_tool_node = malicious_node_spec.get_node(
            node_id=attacked_tool_node_id
        )
        if attacked_tool_node is None:
            raise ValueError(
                f"Guided attacked tool '{attacked_tool_node_id}' was not found."
            )
        if candidate is not None and not candidate(attacked_tool_node):
            raise ValueError(
                f"Guided attacked tool '{attacked_tool_node_id}' does not satisfy the probe selection predicate."
            )
        assert_tool_has_code_references(attacked_tool_node, role="attacked")
        return attacked_tool_node_id, attacked_tool_node

    invoked_tools_ids = get_unique_invoked_tool_ids(flow)
    invoked_tools_ids_set = set(invoked_tools_ids)
    for tool_id in invoked_tools_ids:
        invoking_agent_id = get_agent_using_the_tool_from_flow(
            flow=flow,
            tool_id=tool_id,
        )
        agent_tools_ids = get_agent_tools(malicious_node_spec, invoking_agent_id)
        has_unused_agent_tool_with_inputs = any(
            candidate_tool_id not in invoked_tools_ids_set
            and getattr(
                malicious_node_spec.get_node(node_id=candidate_tool_id),
                "inputs",
                None,
            )
            for candidate_tool_id in agent_tools_ids
        )
        if not has_unused_agent_tool_with_inputs:
            continue

        tool_node = malicious_node_spec.get_node(node_id=tool_id)
        if tool_node is None:
            raise ValueError(f"Invoked tool '{tool_id}' was not found in node spec.")
        if candidate is not None and not candidate(tool_node):
            continue
        if not getattr(tool_node, "code_references", None):
            continue
        return tool_id, tool_node

    raise ValueError(error_message)

def get_unused_tool_node_base_on_agent(
    malicious_node_spec: NodeSpec,
    attacked_tool_node_id: str,
    flow: FlowSpec,
    guidance: dict[str, Any] | None = None,
) -> tuple[str, Any]:
    """
    Return an uninvoked tool from the agent that invoked ``attacked_tool_node_id``.

    Guidance override:
    - ``invoked_tool_node_id``: force the returned tool id.
    """
    guidance = guidance or {}
    invoked_tool_node_id = guidance.get("invoked_tool_node_id")
    if invoked_tool_node_id:
        invoked_tool_node = malicious_node_spec.get_node(
            node_id=invoked_tool_node_id
        )
        if invoked_tool_node is None:
            raise ValueError(
                f"Guided invoked tool '{invoked_tool_node_id}' was not found."
            )
        if not getattr(invoked_tool_node, "inputs", None):
            raise ValueError(
                f"Guided invoked tool '{invoked_tool_node_id}' does not define inputs."
            )
        return invoked_tool_node_id, invoked_tool_node

    if not attacked_tool_node_id:
        attacked_tool_node_id = guidance.get("attacked_tool_node_id")
    if not attacked_tool_node_id:
        raise ValueError("attacked_tool_node_id must be provided.")

    invoked_tools_ids = set(get_unique_invoked_tool_ids(flow))
    agent_id = get_agent_using_the_tool_from_flow(
        flow=flow,
        tool_id=attacked_tool_node_id,
    )

    agent_tools_ids = get_agent_tools(malicious_node_spec, agent_id)
    for tool_id in agent_tools_ids:
        if tool_id == attacked_tool_node_id or tool_id in invoked_tools_ids:
            continue
        tool_node = malicious_node_spec.get_node(node_id=tool_id)
        if tool_node is None:
            raise ValueError(f"Unused tool '{tool_id}' was not found in node spec.")
        if not getattr(tool_node, "inputs", None):
            continue
        return tool_id, tool_node

    raise ValueError(
        "No unused tool with inputs is available on the attacked tool's invoking agent."
    )


def select_local_attacked_tool_node(
    malicious_node_spec: NodeSpec,
    flow: FlowSpec,
    guidance: dict[str, Any] | None = None,
) -> tuple[str, Any]:
    """
    Select the first invoked local tool whose definition can be poisoned.
    """
    return select_attacked_tool_node(
        malicious_node_spec=malicious_node_spec,
        flow=flow,
        guidance=guidance,
        candidate=is_local_tool_node,
        error_message=(
            "No eligible attacked tool was found for description-level injection."
        ),
    )


def select_attacked_tool_node_by_candidate(
    self: Any,
    malicious_node_spec: NodeSpec,
    flow: FlowSpec,
    guidance: dict[str, Any] | None = None,
) -> tuple[str, Any]:
    """
    Select the first invoked tool accepted by ``self.is_attacked_tool_candidate``.
    """
    return select_attacked_tool_node(
        malicious_node_spec=malicious_node_spec,
        flow=flow,
        guidance=guidance,
        candidate=self.is_attacked_tool_candidate,
        error_message="No eligible attacked tool was found for this probe.",
    )

def change_external_mcp_to_emulated(malicious_node_spec: NodeSpec,
                                    attack_tool_id: str):
    tool_node = malicious_node_spec.get_node(attack_tool_id)
    if tool_node is None:
        return
    parent_id = attack_tool_id.replace(f'_{tool_node.name}', '')
    parent_node = malicious_node_spec.get_node(parent_id)
    if parent_node is not None and parent_node.node_type.type == 'External_MCP_server':
        parent_node.emulated = True

def get_unused_tool_node(
    malicious_node_spec: NodeSpec,
    flow: FlowSpec,
    guidance: dict[str, Any] | None = None,
) -> tuple[str, Any]:
    """
    Return a tool node that was not used in the selected flow.
    """
    guidance = guidance or {}
    invoked_tool_node_id = guidance.get("invoked_tool_node_id")
    if invoked_tool_node_id:
        invoked_tool_node = malicious_node_spec.get_node(
            node_id=invoked_tool_node_id
        )
        if invoked_tool_node is None:
            raise ValueError(
                f"Guided invoked tool '{invoked_tool_node_id}' was not found."
            )
        if not getattr(invoked_tool_node, "inputs", None):
            raise ValueError(
                f"Guided invoked tool '{invoked_tool_node_id}' does not define inputs."
            )
        return invoked_tool_node_id, invoked_tool_node

    used_tool_ids = set(get_unique_invoked_tool_ids(flow))
    for tool in malicious_node_spec.list_tools():
        tool_id = tool["ids"][0]
        if tool_id in used_tool_ids:
            continue
        tool_node = malicious_node_spec.get_node(node_id=tool_id)
        if tool_node is None:
            raise ValueError(f"Unused tool '{tool_id}' was not found in node spec.")
        if not getattr(tool_node, "inputs", None):
            continue
        return tool_id, tool_node

    raise ValueError("No alternative tool with inputs is available for invocation.")


def get_required_tool_arguments_from_flow(
    flow: FlowSpec,
    tool_id: str | None = None,
) -> Any:
    """
    Return the first non-empty observed argument payload for the named tool.

    Traces may include an initial empty call payload before a later retry with
    the arguments needed to invoke the tool. Empty payloads are therefore not
    usable representative arguments and are skipped.
    """
    if tool_id is None:
        raise ValueError("tool_id must be provided.")

    for event in getattr(flow, "events", None) or []:
        if event.tool_calls:
            for tool_call in event.tool_calls:
                if tool_call.tool_id == tool_id and tool_call.arguments:
                    return tool_call.arguments
    return None


def get_tool_example_input(
    tool_node: Any,
    guidance: dict[str, Any] | None = None,
    guidance_key: str = "invoked_tool_node_input",
) -> Any:
    """
    Return a representative tool input, optionally overridden by guidance.

    When example pairs are unavailable, this falls back to LLM-generated
    arguments based on the tool's declared input fields.
    """
    guidance = guidance or {}
    if guidance_key in guidance and guidance.get(guidance_key) is not None:
        return guidance[guidance_key]

    example_pairs = getattr(tool_node, "tool_example_pairs", None) or []
    if example_pairs:
        return random.choice(example_pairs).input
    return generate_tool_arguments_with_llm(tool_node)


def build_input_arguments(
    flow: FlowSpec,
    guidance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the entry-point arguments used to execute the malicious twin.
    """
    guidance = guidance or {}
    input_arguments = dict(guidance.get("input_arguments") or {})
    if input_arguments:
        return input_arguments

    flow_input_arguments = getattr(flow, 'input_args', None) or {}
    return dict(flow_input_arguments)


def get_main_user_task_argument_name(malicious_node_spec: NodeSpec) -> str:
    """
    Return the entry-point argument that carries the main user task.
    """
    entry_point_usage_example = getattr(
        malicious_node_spec,
        "entry_point_usage_example",
        None,
    )
    system_input_arguments = getattr(entry_point_usage_example, "arguments", None)
    if not system_input_arguments:
        raise ValueError(
            "Node spec does not define entry_point_usage_example.arguments."
        )

    main_input_argument = next(
        (
            argument
            for argument in system_input_arguments
            if getattr(argument, "is_task_input", False)
        ),
        None,
    )
    if main_input_argument is None:
        raise ValueError(
            "Direct prompt injection requires an entry-point argument marked "
            "with user_task=True."
        )
    return getattr(main_input_argument, "name", main_input_argument).lstrip('-')


def build_execution_cmds(
    malicious_node_spec: NodeSpec,
    input_arguments: dict[str, Any],
) -> list[str]:
    """
    Build the command list used to execute the malicious twin.
    """
    entry_point_usage_example = getattr(
        malicious_node_spec,
        "entry_point_usage_example",
        None,
    )
    entry_point_path = getattr(entry_point_usage_example, "script", None)
    if not entry_point_path:
        raise ValueError("Node spec does not define entry_point_usage_example.script.")

    args = " ".join(
        f"--{shlex.quote(str(key))} {shlex.quote(str(value))}"
        for key, value in input_arguments.items() if value
    )
    return [
        f"python {shlex.quote(str(entry_point_path))} {args}".rstrip()
    ]

def extract_prompt_value(snippet: str) -> str | None:
    prompt_names = set(prompt_variable_name_options)
    tokens = _tokenize_snippet(snippet)
    token_prompt_value: str | None = None

    for index, token in enumerate(tokens):
        if token.type != tokenize.NAME or not _is_prompt_name(token.string, prompt_names):
            continue
        prompt_value = _extract_string_assignment_value(tokens, index + 1)
        if prompt_value and token_prompt_value is None:
            token_prompt_value = prompt_value

    for index, token in enumerate(tokens):
        if token.type != tokenize.STRING:
            continue
        try:
            key_value = ast.literal_eval(token.string)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(key_value, str) or not _is_prompt_name(key_value, prompt_names):
            continue
        prompt_value = _extract_string_mapping_value(tokens, index + 1)
        if prompt_value and token_prompt_value is None:
            token_prompt_value = prompt_value

    ast_prompt_value = _extract_prompt_value_with_ast_fallback(snippet, prompt_names)
    if ast_prompt_value:
        return ast_prompt_value
    return token_prompt_value


def _tokenize_snippet(snippet: str) -> list[tokenize.TokenInfo]:
    tokens: list[tokenize.TokenInfo] = []
    token_stream = tokenize.generate_tokens(io.StringIO(snippet).readline)

    while True:
        try:
            token = next(token_stream)
        except StopIteration:
            break
        except tokenize.TokenError:
            break
        tokens.append(token)

    return tokens


def extract_named_string_value(snippet: str, variable_names: set[str]) -> str | None:
    if not variable_names:
        return None

    tokens = _tokenize_snippet(snippet)
    token_value: str | None = None

    for index, token in enumerate(tokens):
        if token.type != tokenize.NAME or token.string not in variable_names:
            continue
        string_value = _extract_string_assignment_value(tokens, index + 1)
        if string_value and token_value is None:
            token_value = string_value

    ast_value = _extract_named_string_value_with_ast_fallback(snippet, variable_names)
    if ast_value:
        return ast_value
    return token_value


def _extract_string_assignment_value(
    tokens: list[tokenize.TokenInfo],
    start_index: int,
) -> str | None:
    index = start_index
    depth = 0

    while index < len(tokens):
        token = tokens[index]
        if token.type == tokenize.OP:
            if token.string in "([{":
                depth += 1
            elif token.string in ")]}":
                depth = max(0, depth - 1)
            elif token.string == "=" and depth == 0:
                index += 1
                break
            elif token.string == "," and depth == 0:
                return None
        elif token.type in (tokenize.NEWLINE, tokenize.NL) and depth == 0:
            return None
        index += 1
    else:
        return None

    return _collect_string_expression_value(tokens, index)


def _extract_string_mapping_value(
    tokens: list[tokenize.TokenInfo],
    start_index: int,
) -> str | None:
    index = start_index
    depth = 0

    while index < len(tokens):
        token = tokens[index]
        if token.type == tokenize.OP:
            if token.string in "([{":
                depth += 1
            elif token.string in ")]}":
                if depth == 0:
                    return None
                depth -= 1
            elif token.string == ":" and depth == 0:
                index += 1
                break
            elif token.string == "," and depth == 0:
                return None
        elif token.type in (tokenize.NEWLINE, tokenize.NL) and depth == 0:
            return None
        index += 1
    else:
        return None

    return _collect_string_expression_value(tokens, index)


def _collect_string_expression_value(
    tokens: list[tokenize.TokenInfo],
    start_index: int,
) -> str | None:
    index = start_index
    parts: list[str] = []
    depth = 0

    while index < len(tokens):
        token = tokens[index]
        if token.type == tokenize.OP:
            if token.string in "([{":
                depth += 1
            elif token.string in ")]}":
                if depth == 0:
                    break
                depth -= 1
            elif token.string == "," and depth == 0:
                break
        elif token.type in (tokenize.NEWLINE, tokenize.NL) and depth == 0:
            break
        elif token.type == tokenize.STRING:
            token_value = _extract_string_token_value(token.string)
            if token_value is not None:
                parts.append(token_value)
        index += 1

    if not parts:
        return None
    return "".join(parts).strip()


def _extract_string_token_value(token_value: str) -> str | None:
    try:
        literal_value = ast.literal_eval(token_value)
    except (ValueError, SyntaxError):
        literal_value = None
    if isinstance(literal_value, str):
        return literal_value

    try:
        expression = ast.parse(token_value, mode="eval").body
    except SyntaxError:
        return None

    if isinstance(expression, ast.JoinedStr):
        return _render_joined_str(expression)
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return expression.value
    return None


def _render_joined_str(node: ast.JoinedStr) -> str:
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
            continue
        if isinstance(value, ast.FormattedValue):
            parts.append(_render_formatted_value(value))
    return "".join(parts)


def _render_formatted_value(node: ast.FormattedValue) -> str:
    try:
        expression = ast.unparse(node.value)
    except Exception:
        expression = "..."

    rendered = "{" + expression
    if node.conversion != -1:
        try:
            rendered += f"!{chr(node.conversion)}"
        except ValueError:
            pass
    if node.format_spec is not None:
        if isinstance(node.format_spec, ast.JoinedStr):
            rendered += ":" + _render_joined_str(node.format_spec)
        else:
            try:
                rendered += ":" + ast.unparse(node.format_spec)
            except Exception:
                pass
    rendered += "}"
    return rendered


def _extract_prompt_value_with_ast_fallback(
    snippet: str,
    prompt_names: set[str],
) -> str | None:
    for candidate in _build_ast_parse_candidates(snippet):
        try:
            tree = ast.parse(candidate)
        except (IndentationError, SyntaxError):
            continue
        prompt_value = _extract_prompt_value_from_tree(tree, prompt_names)
        if prompt_value:
            return prompt_value
    return None


def _extract_named_string_value_with_ast_fallback(
    snippet: str,
    variable_names: set[str],
) -> str | None:
    for candidate in _build_ast_parse_candidates(snippet):
        try:
            tree = ast.parse(candidate)
        except (IndentationError, SyntaxError):
            continue
        string_value = _extract_named_string_value_from_tree(tree, variable_names)
        if string_value:
            return string_value
    return None


def _build_ast_parse_candidates(snippet: str) -> list[str]:
    dedented_snippet = textwrap.dedent(snippet)
    candidates = [
        snippet,
        dedented_snippet,
        "{\n" + dedented_snippet + "\n}",
        "container = {\n" + dedented_snippet + "\n}",
    ]
    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate not in unique_candidates and candidate.strip():
            unique_candidates.append(candidate)
    return unique_candidates


def _extract_prompt_value_from_tree(
    tree: ast.AST,
    prompt_names: set[str],
) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and _is_prompt_name(target.id, prompt_names):
                    prompt_value = _extract_string_from_ast_node(node.value)
                    if prompt_value:
                        return prompt_value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if (
                isinstance(target, ast.Name)
                and _is_prompt_name(target.id, prompt_names)
                and node.value is not None
            ):
                prompt_value = _extract_string_from_ast_node(node.value)
                if prompt_value:
                    return prompt_value
        elif isinstance(node, ast.keyword):
            if _is_prompt_name(node.arg, prompt_names):
                prompt_value = _extract_string_from_ast_node(node.value)
                if prompt_value:
                    return prompt_value
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and _is_prompt_name(key.value, prompt_names)
                ):
                    prompt_value = _extract_string_from_ast_node(value)
                    if prompt_value:
                        return prompt_value
    return None


def _extract_named_string_value_from_tree(
    tree: ast.AST,
    variable_names: set[str],
) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in variable_names:
                    string_value = _extract_string_from_ast_node(node.value)
                    if string_value:
                        return string_value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if (
                isinstance(target, ast.Name)
                and target.id in variable_names
                and node.value is not None
            ):
                string_value = _extract_string_from_ast_node(node.value)
                if string_value:
                    return string_value
        elif isinstance(node, ast.keyword):
            if node.arg in variable_names:
                string_value = _extract_string_from_ast_node(node.value)
                if string_value:
                    return string_value
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and key.value in variable_names
                ):
                    string_value = _extract_string_from_ast_node(value)
                    if string_value:
                        return string_value
    return None


def _is_prompt_name(name: str | None, prompt_names: set[str]) -> bool:
    if not isinstance(name, str):
        return False
    name_lower = name.lower()
    return name in prompt_names or name_lower.endswith("prompt") or name_lower.endswith("prompt_template")


def _extract_string_from_ast_node(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip() or None
    if isinstance(node, ast.JoinedStr):
        return _render_joined_str(node).strip() or None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _extract_string_from_ast_node(node.left)
        right = _extract_string_from_ast_node(node.right)
        if left is None and right is None:
            return None
        return f"{left or ''}{right or ''}".strip() or None
    return None


def extract_agent_system_prompt(
    malicious_node_spec: NodeSpec,
    attacked_agent_node_id: str,
) -> str:
    """
    Extract the attacked agent's system prompt from its code references.

    Fall back to resolving a prompt variable referenced in an ``assignment``
    snippet against the node's ``system_prompt`` snippets when the direct
    system-prompt extraction does not yield a string literal.
    """
    attacked_agent_node = malicious_node_spec.get_node(
        node_id=attacked_agent_node_id
    )
    if attacked_agent_node is None:
        raise ValueError(
            f"Attacked agent '{attacked_agent_node_id}' was not found in node spec."
        )
    code_references = getattr(attacked_agent_node, "code_references", None) or []
    system_prompt_references = [
        code for code in code_references if code.kind == "system_prompt"
    ]
    assignment_references = [
        code for code in code_references if code.kind == "assignment"
    ]

    for code in system_prompt_references:
        system_prompt = extract_prompt_value(code.snippet)
        if system_prompt:
            return system_prompt
    
    for code in assignment_references:
        system_prompt = extract_prompt_value(code.snippet)
        if system_prompt:
            return system_prompt

    prompt_reference_names: list[str] = []
    for code in assignment_references:
        for variable_name in _extract_prompt_reference_names(code.snippet):
            if variable_name not in prompt_reference_names:
                prompt_reference_names.append(variable_name)

    for variable_name in prompt_reference_names:
        for code in system_prompt_references:
            system_prompt = extract_named_string_value(
                code.snippet,
                {variable_name},
            )
            if system_prompt:
                return system_prompt

    llm_variable_name = _extract_prompt_reference_name_with_llm(
        [code.snippet for code in assignment_references]
    )
    if llm_variable_name:
        for code in system_prompt_references:
            system_prompt = extract_named_string_value(
                code.snippet,
                {llm_variable_name},
            )
            if system_prompt:
                return system_prompt

    raise ValueError(
        "Could not extract the attacked agent's system prompt from the LLM node."
    )


def _extract_prompt_reference_names(snippet: str) -> list[str]:
    tree = parse_snippet_with_indent_support(snippet)
    if tree is None:
        return []

    prompt_names = set(prompt_variable_name_options)
    reference_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if not any(
                isinstance(target, ast.Name) and _is_prompt_name(target.id, prompt_names)
                for target in node.targets
            ):
                continue
            reference_name = _extract_name_reference_from_ast_node(node.value)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if not (
                isinstance(target, ast.Name)
                and _is_prompt_name(target.id, prompt_names)
                and node.value is not None
            ):
                continue
            reference_name = _extract_name_reference_from_ast_node(node.value)
        elif isinstance(node, ast.keyword):
            if not _is_prompt_name(node.arg, prompt_names):
                continue
            reference_name = _extract_name_reference_from_ast_node(node.value)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and _is_prompt_name(key.value, prompt_names)
                ):
                    reference_name = _extract_name_reference_from_ast_node(value)
                    if reference_name and reference_name not in reference_names:
                        reference_names.append(reference_name)
            continue
        else:
            continue

        if reference_name and reference_name not in reference_names:
            reference_names.append(reference_name)

    return reference_names


def _extract_prompt_reference_name_with_llm(snippets: list[str]) -> str | None:
    """Infer an indirectly referenced system-prompt variable from code."""
    source = "\n\n".join(snippet for snippet in snippets if snippet.strip())
    if not source:
        return None

    response = query_azure_chat_openai_structured_output(
        prompt=f"""
Inspect the Python code below. Identify the variable whose value is used as an
agent's system prompt, system message, instructions, or developer message.
Return only a simple Python variable identifier when one can be determined.
Do not return attribute paths, function calls, or literals. Return null if no
such identifier is present.

```python
{source}
```
""".strip(),
        output_schema=PromptVariableReference,
    )
    variable_name = getattr(response, "variable_name", None)
    if not isinstance(variable_name, str):
        return None
    variable_name = variable_name.strip()
    return variable_name if variable_name.isidentifier() else None


def _extract_name_reference_from_ast_node(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None

def get_first_agent_node(flow: FlowSpec, malicious_node_spec: NodeSpec):
    for event in flow.events:
        if event.type == 'agent':
            return malicious_node_spec.get_node(event.node_id)
    return None


def serialize_payload(payload: Any) -> str:
    """
    Serialize a prompt payload for LLM helper classes.
    """
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_snippet_with_indent_support(snippet: str) -> ast.AST | None:
    """
    Parse Python snippet, tolerating leading indentation from nested scopes.

    Return:
        ast.AST | None: Parsed tree, or None when snippet is not parseable.
    """
    try:
        return ast.parse(snippet)
    except (IndentationError, SyntaxError):
        normalized_snippet = textwrap.dedent(snippet)
        if not normalized_snippet.strip():
            return None
        try:
            return ast.parse(normalized_snippet)
        except (IndentationError, SyntaxError):
            return None


def _iter_function_nodes_in_source_order(
    node: ast.AST,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    function_nodes: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_nodes.append(child)
            continue
        function_nodes.extend(_iter_function_nodes_in_source_order(child))
    return function_nodes


def _get_primary_function_node(
    tree: ast.AST,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """
    Return the first function/method that contains a return statement.

    This supports class snippets where the first method (for example ``__init__``)
    has no return value but later methods do.
    """
    function_nodes = _iter_function_nodes_in_source_order(tree)
    if not function_nodes:
        return None

    for function_node in function_nodes:
        if any(isinstance(node, ast.Return) for node in _iter_nodes_in_source_order(function_node)):
            return function_node

    return function_nodes[0]


def _iter_nodes_in_source_order(node: ast.AST):
    for child in ast.iter_child_nodes(node):
        yield child
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield from _iter_nodes_in_source_order(child)


def find_return_nodes(snippet: str) -> list[ast.Return]:
    """
    Return return nodes in source order from the primary function body.
    """
    tree = parse_snippet_with_indent_support(snippet)
    if tree is None:
        return []

    search_root = _get_primary_function_node(tree) or tree
    return_nodes: list[ast.Return] = []
    for node in _iter_nodes_in_source_order(search_root):
        if isinstance(node, ast.Return):
            return_nodes.append(node)
    return return_nodes


def find_first_return_node(snippet: str) -> ast.Return | None:
    """
    Return the first return node in the primary function body.
    """
    return_nodes = find_return_nodes(snippet)
    if not return_nodes:
        return None
    return return_nodes[0]


def _return_node_offsets_align_with_source(snippet: str, return_node: ast.Return) -> bool:
    """
    Validate that ``return_node`` column offsets still match ``snippet`` source.

    When ``snippet`` is parsed via a dedented fallback, AST column offsets are
    relative to the dedented text and may not align with the original snippet.
    """
    start_line = getattr(return_node, "lineno", None)
    col_offset = getattr(return_node, "col_offset", None)
    if start_line is None or col_offset is None:
        return False

    lines = snippet.splitlines()
    if start_line < 1 or start_line > len(lines):
        return False

    line = lines[start_line - 1]
    if col_offset < 0:
        return False
    if col_offset + len("return") > len(line):
        return False

    return line[col_offset:col_offset + len("return")] == "return"


def _extract_return_expression_from_node(
    snippet: str,
    return_node: ast.Return,
) -> str | None:
    if return_node.value is None:
        return None

    if _return_node_offsets_align_with_source(snippet, return_node):
        expression_source = ast.get_source_segment(snippet, return_node.value)
        if expression_source is not None:
            return expression_source.strip()

        statement_source = ast.get_source_segment(snippet, return_node)
        if statement_source:
            return_keyword_index = statement_source.find("return")
            if return_keyword_index != -1:
                return statement_source[return_keyword_index + len("return"):].strip()

    start_line = getattr(return_node, "lineno", None)
    end_line = getattr(return_node, "end_lineno", None) or start_line
    if start_line is None or end_line is None:
        return None

    lines = snippet.splitlines()
    if start_line < 1 or end_line > len(lines):
        return None

    statement_lines = lines[start_line - 1:end_line]
    if not statement_lines:
        return None

    first_line = statement_lines[0]
    return_match = re.search(r"\breturn\b", first_line)
    if return_match is None:
        return None

    statement_lines[0] = first_line[return_match.end():]
    return "\n".join(statement_lines).strip()


def extract_first_return_expression(snippet: str) -> str | None:
    """
    Return the first return expression from the snippet, including multiline expressions.
    """
    return_node = find_first_return_node(snippet)
    if return_node is None:
        return None
    return _extract_return_expression_from_node(snippet, return_node)


def _replace_node_source(snippet: str, node: ast.AST, replacement: str) -> str | None:
    start_line = getattr(node, "lineno", None)
    end_line = getattr(node, "end_lineno", None)
    if start_line is None:
        return None
    if end_line is None:
        end_line = start_line

    lines = snippet.splitlines(keepends=True)
    if start_line < 1 or end_line > len(lines):
        return None

    replacement_text = replacement
    if lines[end_line - 1].endswith("\n") and not replacement_text.endswith("\n"):
        replacement_text = f"{replacement_text}\n"

    lines[start_line - 1:end_line] = [replacement_text]
    return "".join(lines)


def _add_indentation(snippet: str, indentation: str) -> str:
    return "\n".join(
        f"{indentation}{line}" if line else line
        for line in snippet.splitlines()
    )


def _normalize_multiline_expression_indentation(
    expression: str,
    base_indentation: str,
) -> str:
    lines = expression.splitlines()
    if len(lines) <= 1 or not base_indentation:
        return expression

    normalized_lines = [lines[0]]
    for line in lines[1:]:
        if line.startswith(base_indentation):
            normalized_lines.append(line[len(base_indentation):])
        else:
            normalized_lines.append(line)
    return "\n".join(normalized_lines)


def rewrite_return_statements(
    attacked_tool_node: Any,
    replacement_builder: Callable[[str], str],
) -> None:
    """
    Rewrite return statements in a tool definition using a shared builder.

    ``replacement_builder`` receives the extracted return expression and must
    return a replacement statement block relative to the return statement
    indentation level.
    """
    for code in getattr(attacked_tool_node, "code_references", []) or []:
        if code.kind != "definition":
            continue

        return_nodes = find_return_nodes(code.snippet)
        if not return_nodes:
            continue

        updated_snippet = code.snippet
        replacements = 0
        sorted_return_nodes = sorted(
            return_nodes,
            key=lambda node: (
                getattr(node, "lineno", 0),
                getattr(node, "col_offset", 0),
            ),
            reverse=True,
        )

        for return_node in sorted_return_nodes:
            old_return = _extract_return_expression_from_node(updated_snippet, return_node)
            if old_return is None:
                continue

            snippet_lines = updated_snippet.splitlines()
            if return_node.lineno - 1 >= len(snippet_lines):
                continue
            return_line = snippet_lines[return_node.lineno - 1]
            indentation = return_line[:len(return_line) - len(return_line.lstrip(" \t"))]

            normalized_old_return = _normalize_multiline_expression_indentation(
                old_return,
                indentation,
            )
            replacement_block = replacement_builder(normalized_old_return)
            indented_replacement = _add_indentation(replacement_block, indentation)
            rewritten_snippet = _replace_node_source(
                updated_snippet,
                return_node,
                indented_replacement,
            )
            if rewritten_snippet is None:
                continue
            updated_snippet = rewritten_snippet
            replacements += 1

        if replacements == 0:
            continue
        code.snippet = updated_snippet
        return

    raise ValueError(
        f"Could not locate return expressions in tool '{attacked_tool_node.id}'."
    )


def rewrite_first_return_statement(
    attacked_tool_node: Any,
    replacement_builder: Callable[[str], str],
) -> None:
    """
    Backwards-compatible alias for ``rewrite_return_statements``.
    """
    rewrite_return_statements(
        attacked_tool_node=attacked_tool_node,
        replacement_builder=replacement_builder,
    )


def extract_signature(function_name: str, snippet: str) -> str:
    """
    Extract the function signature from a code snippet.

    Args:
        function_name (str): Name of the function to locate.
        snippet (str): Python source code.

    Return:
        str: Signature string (for example, `def foo(a, b=1) -> int`).

    Raises:
        ValueError: If function not found.
    """

    tree = parse_snippet_with_indent_support(snippet)
    if tree is None:
        raise ValueError("Could not parse snippet while extracting signature.")

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return _build_signature(node)

    raise ValueError(f"Function '{function_name}' not found in snippet.")


def _build_signature(func_node: ast.FunctionDef|ast.AsyncFunctionDef) -> str:
    """
    Build a function signature string from an AST function node.

    Args:
        func_node (ast.FunctionDef): Function node to convert.

    Return:
        str: Reconstructed function signature.
    """
    args = []

    # Positional + keyword args
    total = func_node.args.args
    defaults = func_node.args.defaults
    default_offset = len(total) - len(defaults)

    for i, arg in enumerate(total):
        name = arg.arg

        # Type annotation
        if arg.annotation:
            name += f": {ast.unparse(arg.annotation)}"

        # Default value
        if i >= default_offset:
            default_val = defaults[i - default_offset]
            name += f"={ast.unparse(default_val)}"

        args.append(name)

    # *args
    if func_node.args.vararg:
        name = f"*{func_node.args.vararg.arg}"
        if func_node.args.vararg.annotation:
            name += f": {ast.unparse(func_node.args.vararg.annotation)}"
        args.append(name)

    # Keyword-only args
    for arg, default in zip(
        func_node.args.kwonlyargs,
        func_node.args.kw_defaults
    ):
        name = arg.arg
        if arg.annotation:
            name += f": {ast.unparse(arg.annotation)}"
        if default:
            name += f"={ast.unparse(default)}"
        args.append(name)

    # **kwargs
    if func_node.args.kwarg:
        name = f"**{func_node.args.kwarg.arg}"
        if func_node.args.kwarg.annotation:
            name += f": {ast.unparse(func_node.args.kwarg.annotation)}"
        args.append(name)

    signature = f"def {func_node.name}({', '.join(args)})"

    # Return annotation
    if func_node.returns:
        signature += f" -> {ast.unparse(func_node.returns)}"

    return signature


def extract_variable_names_from_signature_ast(signature: str) -> list[str]:
    """
    Extract argument variable names from a function signature string.

    Args:
        signature (str): Function signature text without trailing colon/body.

    Return:
        list[str]: Ordered argument names from positional, vararg, kw-only, and kwargs.
    """
    tree = ast.parse(signature + ":\n    pass")

    func = tree.body[0]
    args = []

    for a in func.args.args:
        args.append(a.arg)

    if func.args.vararg:
        args.append(func.args.vararg.arg)

    for a in func.args.kwonlyargs:
        args.append(a.arg)

    if func.args.kwarg:
        args.append(func.args.kwarg.arg)

    return args


def schema_to_signature(func_name: str, schema: dict[str, Any]) -> str:
    """
    Convert a JSON-schema-like object into a Python function signature.

    Args:
        func_name (str): Function name.
        schema (dict[str, Any]): JSON schema with `properties` and optional `required`.

    Return:
        str: Generated signature string.
    """
    props: dict = (schema or {}).get("properties", {}) or {}
    required = set((schema or {}).get("required", []) or [])

    def py_type(spec: dict) -> str:
        spec = spec or {}

        # enum -> Literal[...] (string form; caller can import Literal if desired)
        if "enum" in spec and isinstance(spec["enum"], list) and spec["enum"]:
            # Use repr() to ensure strings are quoted correctly
            items = ", ".join(repr(x) for x in spec["enum"])
            return f"Literal[{items}]"

        t = spec.get("type")

        # JSON Schema can use type as list: ["string","null"]
        if isinstance(t, list):
            non_null = [x for x in t if x != "null"]
            t = non_null[0] if non_null else "any"

        return {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "list",
            "object": "dict",
        }.get(t, "Any")

    def default_repr(spec: dict) -> str:
        spec = spec or {}
        if "default" not in spec:
            return "None"

        dv = spec.get("default")
        if dv is None:
            return "None"
        if isinstance(dv, str):
            return repr(dv)  # ensures quotes + escaping
        if isinstance(dv, bool):
            return "True" if dv else "False"
        if isinstance(dv, (int, float)):
            return str(dv)
        # lists/dicts/etc.
        return repr(dv)

    required_args: list[str] = []
    optional_args: list[str] = []

    # Keep deterministic ordering: required first (in schema order), then optional (schema order)
    for name, spec in props.items():
        ann = py_type(spec)
        if name in required:
            required_args.append(f"{name}: {ann}")
        else:
            optional_args.append(f"{name}: {ann} = {default_repr(spec)}")

    args = required_args + optional_args
    return f"{func_name}(" + ", ".join(args) + ")"


def build_function_signature(func_name: str, inputs: list[InputPort]) -> str:
    """
    Build a Python function signature from InputPort definitions.

    Args:
        func_name (str): Name of the function.
        inputs (list[InputPort]): List of input port objects.

    Return:
        str: Function signature string.
    """

    # Mapping InputPort dtypes to Python types
    type_map = {
        "string": "str",
        "int": "int",
        "integer": "int",
        "float": "float",
        "boolean": "bool",
        "bool": "bool",
        "number": "float",
        "object": "dict",
        "array": "list"
    }

    required = []
    optional = []

    for p in inputs:
        py_type = type_map.get(p.dtype.lower(), "Any")

        if p.required:
            required.append(f"{p.name}: {py_type}")
        else:
            default_val = repr(p.default)
            optional.append(f"{p.name}: {py_type} = {default_val}")

    params = required + optional
    params_str = ", ".join(params)

    return f"def {func_name}({params_str})"


def dict_to_string(d):
    parts = []
    for k, v in d.items():
        parts.append(f"{k}={repr(v)}")
    return ", ".join(parts)


def dict_to_assignment_block(d):
    return "\n".join(f"{k} = {repr(v)}" for k, v in d.items())


def extract_first_function_name(code: str):
    tree = parse_snippet_with_indent_support(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef,ast.AsyncFunctionDef)):
            return node.name
    return None

def get_first_executable_line(code: str):
    tree = parse_snippet_with_indent_support(code)
    if tree is None:
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef,ast.AsyncFunctionDef)):
            if node.name == "__init__":
                continue

            body = node.body

            if not body:
                return None

            first_stmt = body[0]

            # Check if first statement is a docstring
            if (
                isinstance(first_stmt, ast.Expr)
                and isinstance(first_stmt.value, ast.Constant)
                and isinstance(first_stmt.value.value, str)
            ):
                # Skip docstring if there is another statement
                if len(body) > 1:
                    return body[1].lineno
                else:
                    return None

            return first_stmt.lineno

    return None

def insert_line_in_content(
    content: str,
    line_number: int,
    code_line: str,
) -> str:
    """
    Insert a line of code into content at the specified 1-based line_number.
    Indentation is inferred from the nearest non-blank line around the target
    location (downwards first, then upwards).

    Args:
        content (str): Original file content.
        line_number (int): 1-based line index where code is inserted.
        code_line (str): Code block to insert.

    Return:
        str: Updated file content after insertion.
    """

    def is_non_blank(line: str) -> bool:
        return bool(line.strip())

    lines = content.splitlines(keepends=True)

    # sanitize code_line
    code_line = code_line.rstrip("\n")

    index = max(0, line_number - 1)

    # --- Determine indentation from nearest non-blank line ---
    indentation = ""

    # Search downwards from the insert index
    for i in range(index, len(lines)):
        if is_non_blank(lines[i]):
            ref_line = lines[i]
            indentation = ref_line[:len(ref_line) - len(ref_line.lstrip())]
            break
    else:
        # If none found downward, search upward
        for i in range(index - 1, -1, -1):
            if is_non_blank(lines[i]):
                ref_line = lines[i]
                indentation = ref_line[:len(ref_line) - len(ref_line.lstrip())]
                break

    # --- Prepare multi-line insertion ---
    # Split respecting multiple lines, strip only trailing newline(s)
    raw_insert_lines = code_line.rstrip("\n").split("\n")

    # Apply indentation + append newline to each line
    insert_lines = [(indentation + l + "\n") for l in raw_insert_lines]

    # --- Insert or append ---
    if index >= len(lines):
        lines.extend(insert_lines)
    else:
        for offset, l in enumerate(insert_lines):
            lines.insert(index + offset, l)

    return "".join(lines) 

def build_condition_string(d: dict) -> str:
    """
    Build a Python condition expression from expected argument values.

    Non-string values use exact equality.
    String values use a relaxed matcher that accepts:
    1) exact equality
    2) normalized equality (case/punctuation/whitespace-insensitive)
    3) normalized substring containment (both directions)
    4) keyword-overlap ratio on normalized tokens
    """
    if not d:
        return "False"

    def _normalize_text(value: str) -> str:
        lowered = value.lower()
        cleaned = "".join(ch if ch.isalnum() else " " for ch in lowered)
        return " ".join(cleaned.split())

    def _keyword_tokens(value: str) -> list[str]:
        return sorted({token for token in _normalize_text(value).split() if len(token) > 1})

    def _format_value(value: Any) -> str:
        # repr() generates valid Python literals and handles embedded quotes.
        return repr(value)

    def _build_string_clause(key: str, value: str, *, keyword_overlap_threshold: float = 0.6) -> str:
        runtime_value = f"__locals.get({_format_value(key)})"
        normalized_expected = _normalize_text(value)
        expected_literal = _format_value(value)
        normalized_expected_literal = _format_value(normalized_expected)
        expected_keywords = _keyword_tokens(value)
        expected_keywords_literal = _format_value(expected_keywords)
        keyword_branch = ""
        if len(expected_keywords) >= 2:
            keyword_branch = (
                " or ("
                f"(len(__tok({runtime_value}).intersection({expected_keywords_literal})) / "
                f"max(len({expected_keywords_literal}), 1)) >= {keyword_overlap_threshold}"
                ")"
            )

        return (
            "("
            f"isinstance({runtime_value}, str) and ("
            f"{runtime_value} == {expected_literal}"
            f" or ({normalized_expected_literal} and __norm({runtime_value}) == {normalized_expected_literal})"
            f" or ({normalized_expected_literal} and {normalized_expected_literal} in __norm({runtime_value}))"
            f" or ({normalized_expected_literal} and __norm({runtime_value}) in {normalized_expected_literal})"
            f"{keyword_branch}"
            ")"
            ")"
        )

    per_key_conditions: list[str] = []
    for key, value in d.items():
        runtime_value = f"__locals.get({_format_value(key)})"
        if isinstance(value, str):
            per_key_conditions.append(_build_string_clause(key, value))
            continue
        per_key_conditions.append(f"{runtime_value} == {_format_value(value)}")

    joined_conditions = " or ".join(per_key_conditions) if per_key_conditions else "False"
    return (
        "(lambda __locals, "
        "__norm=(lambda __s: ' '.join(''.join((__ch.lower() if __ch.isalnum() else ' ') for __ch in str(__s)).split())), "
        "__tok=(lambda __s: {__word for __word in ' '.join(''.join((__ch.lower() if __ch.isalnum() else ' ') for __ch in str(__s)).split()).split() if len(__word) > 1}): "
        f"({joined_conditions})"
        ")(locals())"
    )



def inject_instruction_into_tool_output(
    attacked_tool_node: Any,
    instruction: str,
    *,
    instruction_is_expression: bool = False,
    injection_place: str = 'start'
) -> None:
    """
    Patch a tool definition to inject an instruction into each returned value.

    Args:
        attacked_tool_node (Any): Tool node containing code references.
        instruction (str): Instruction text or expression to inject.
        instruction_is_expression (bool): When True, treat ``instruction`` as
            a Python expression; otherwise inject it as a string literal.
    """
    if injection_place not in {"start", "end"}:
        raise ValueError(
            "injection_place must be either 'start' or 'end'."
        )

    def replacement_builder(old_return: str) -> str:
        instruction_expr = instruction if instruction_is_expression else repr(instruction)
        base_lines = [
            f"__probe_ans = {old_return}",
            f"__probe_instruction = {instruction_expr}",
            "__probe_ans_text = str(__probe_ans)",
            "__probe_instruction_text = str(__probe_instruction)",
        ]

        if injection_place == "start":
            base_lines.extend(
                [
                    "if (__probe_instruction_text and __probe_ans_text and "
                    "not __probe_instruction_text[-1].isspace() and "
                    "not __probe_ans_text[0].isspace()):",
                    "    __probe_instruction_text += ' '",
                    "return __probe_instruction_text + __probe_ans_text",
                ]
            )
            return "\n".join(base_lines)

        base_lines.extend(
            [
                "if (__probe_instruction_text and __probe_ans_text and "
                "not __probe_ans_text[-1].isspace() and "
                "not __probe_instruction_text[0].isspace() and "
                "__probe_instruction_text[0] not in '.,;:!?)]}'):",
                "    __probe_ans_text += ' '",
                "return __probe_ans_text + __probe_instruction_text",
            ]
        )
        return "\n".join(base_lines)

    rewrite_return_statements(attacked_tool_node, replacement_builder)
