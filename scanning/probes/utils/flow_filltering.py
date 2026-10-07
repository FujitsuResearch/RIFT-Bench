from typing import Any

from node_spec.structure_schema import FlowSpec, NodeSpec

from scanning.probes.utils.base_util import (
    get_agent_tools,
    get_agent_using_the_tool_from_flow,
    get_required_tool_arguments_from_flow,
    get_unique_invoked_agents_ids,
    get_unique_invoked_tool_ids,
)


def has_code_references(node: Any) -> bool:
    return bool(node and getattr(node, "code_references", None))


def has_code_reference_kind(node: Any, kind: str) -> bool:
    return any(
        getattr(code, "kind", None) == kind
        for code in (getattr(node, "code_references", None) or [])
    )


def has_definition_code_reference(node: Any) -> bool:
    return has_code_reference_kind(node, "definition")


def has_assignment_code_reference(node: Any) -> bool:
    return has_code_reference_kind(node, "assignment")


def has_system_prompt_code_reference(node: Any) -> bool:
    return has_code_reference_kind(node, "system_prompt")


def has_data_path_code_reference(node: Any) -> bool:
    return has_code_reference_kind(node, "data_path")


def has_inputs(tool_node: Any) -> bool:
    return bool(tool_node and getattr(tool_node, "inputs", None))


def has_example_inputs(tool_node: Any) -> bool:
    return bool(tool_node and getattr(tool_node, "tool_example_pairs", None))


def has_data_path(node: Any) -> bool:
    return bool(node and getattr(node, "data_path", None))


def has_entry_point_script(full_node_spec: NodeSpec) -> bool:
    entry_point_usage_example = getattr(
        full_node_spec,
        "entry_point_usage_example",
        None,
    )
    return bool(getattr(entry_point_usage_example, "script", None))


def has_entry_point_arguments(full_node_spec: NodeSpec) -> bool:
    entry_point_usage_example = getattr(
        full_node_spec,
        "entry_point_usage_example",
        None,
    )
    return bool(getattr(entry_point_usage_example, "arguments", None))


def has_user_task_entry_point_argument(full_node_spec: NodeSpec) -> bool:
    entry_point_usage_example = getattr(
        full_node_spec,
        "entry_point_usage_example",
        None,
    )
    arguments = getattr(entry_point_usage_example, "arguments", None) or []
    return any(getattr(argument, "is_task_input", False) for argument in arguments)


def get_agent_event_node_ids(flow: FlowSpec) -> list[str]:
    agent_node_ids: list[str] = []
    for event in getattr(flow, "events", None) or []:
        if str(getattr(event, "type", "")).lower() != "agent":
            continue
        node_id = str(getattr(event, "node_id", "") or "")
        if node_id and node_id not in agent_node_ids:
            agent_node_ids.append(node_id)
    return agent_node_ids


def get_invoked_agent_ids(flow: FlowSpec) -> list[str]:
    return get_unique_invoked_agents_ids(flow)


def get_tool_ids_matching_predicate(
    full_node_spec: NodeSpec,
    tool_ids: list[str],
    predicate,
) -> list[str]:
    matched_tool_ids: list[str] = []
    for tool_id in tool_ids:
        tool_node = full_node_spec.get_node(node_id=tool_id)
        if tool_node is None:
            continue
        if predicate(tool_node):
            matched_tool_ids.append(tool_id)
    return matched_tool_ids


def get_invoked_rag_tool_ids(
    full_node_spec: NodeSpec,
    flow: FlowSpec,
) -> list[str]:
    rag_tool_ids: list[str] = []
    for tool_id in get_unique_invoked_tool_ids(flow):
        tool_node = full_node_spec.get_node(node_id=tool_id)
        if tool_node is None:
            continue
        if getattr(tool_node, "is_rag_tool", False):
            rag_tool_ids.append(tool_id)
    return rag_tool_ids


def get_database_node_ids(full_node_spec: NodeSpec) -> list[str]:
    database_node_ids: list[str] = []
    for node in full_node_spec.list_by_types({"Database"}):
        node_ids = node.get("ids") or []
        if node_ids:
            database_node_ids.append(node_ids[0])
    return database_node_ids


def get_database_node_ids_with_data_path(full_node_spec: NodeSpec) -> list[str]:
    return [
        node_id
        for node_id in get_database_node_ids(full_node_spec)
        if has_data_path(full_node_spec.get_node(node_id=node_id))
    ]


def get_database_node_ids_with_data_path_code_references(
    full_node_spec: NodeSpec,
) -> list[str]:
    return [
        node_id
        for node_id in get_database_node_ids_with_data_path(full_node_spec)
        if has_data_path_code_reference(full_node_spec.get_node(node_id=node_id))
    ]


def get_agent_invoked_tool_ids(
    full_node_spec: NodeSpec,
    flow: FlowSpec,
    agent_id: str,
) -> list[str]:
    agent_tool_ids = set(get_agent_tools(full_node_spec, agent_id))
    return [
        tool_id
        for tool_id in get_unique_invoked_tool_ids(flow)
        if tool_id in agent_tool_ids
    ]


def get_agent_unused_tool_ids(
    full_node_spec: NodeSpec,
    flow: FlowSpec,
    agent_id: str,
) -> list[str]:
    invoked_tool_ids = set(get_unique_invoked_tool_ids(flow))
    return [
        tool_id
        for tool_id in get_agent_tools(full_node_spec, agent_id)
        if tool_id not in invoked_tool_ids
    ]


def get_unused_tool_ids(
    full_node_spec: NodeSpec,
    flow: FlowSpec,
) -> list[str]:
    invoked_tool_ids = set(get_unique_invoked_tool_ids(flow))
    return [
        tool["ids"][0]
        for tool in full_node_spec.list_tools()
        if tool["ids"][0] not in invoked_tool_ids
    ]


def get_unused_tool_ids_with_inputs(
    full_node_spec: NodeSpec,
    flow: FlowSpec,
) -> list[str]:
    return get_tool_ids_with_inputs(
        full_node_spec=full_node_spec,
        tool_ids=get_unused_tool_ids(
            full_node_spec=full_node_spec,
            flow=flow,
        ),
    )


def get_agent_unused_tool_ids_with_inputs(
    full_node_spec: NodeSpec,
    flow: FlowSpec,
    agent_id: str,
) -> list[str]:
    return get_tool_ids_with_inputs(
        full_node_spec=full_node_spec,
        tool_ids=get_agent_unused_tool_ids(
            full_node_spec=full_node_spec,
            flow=flow,
            agent_id=agent_id,
        ),
    )


def get_tool_ids_with_inputs(
    full_node_spec: NodeSpec,
    tool_ids: list[str],
) -> list[str]:
    return [
        tool_id
        for tool_id in tool_ids
        if has_inputs(full_node_spec.get_node(node_id=tool_id))
    ]


def get_tool_ids_with_code_references(
    full_node_spec: NodeSpec,
    tool_ids: list[str],
) -> list[str]:
    return [
        tool_id
        for tool_id in tool_ids
        if has_code_references(full_node_spec.get_node(node_id=tool_id))
    ]


def get_tool_ids_with_code_reference_kind(
    full_node_spec: NodeSpec,
    tool_ids: list[str],
    kind: str,
) -> list[str]:
    return [
        tool_id
        for tool_id in tool_ids
        if has_code_reference_kind(full_node_spec.get_node(node_id=tool_id), kind)
    ]


def get_tool_ids_with_definition_code_references(
    full_node_spec: NodeSpec,
    tool_ids: list[str],
) -> list[str]:
    return get_tool_ids_with_code_reference_kind(
        full_node_spec=full_node_spec,
        tool_ids=tool_ids,
        kind="definition",
    )


def get_tool_ids_with_assignment_code_references(
    full_node_spec: NodeSpec,
    tool_ids: list[str],
) -> list[str]:
    return get_tool_ids_with_code_reference_kind(
        full_node_spec=full_node_spec,
        tool_ids=tool_ids,
        kind="assignment",
    )


def get_tool_ids_with_required_arguments_from_flow(
    flow: FlowSpec,
    tool_ids: list[str],
) -> list[str]:
    return [
        tool_id
        for tool_id in tool_ids
        if get_required_tool_arguments_from_flow(flow=flow, tool_id=tool_id)
    ]


def get_tool_ids_with_example_inputs(
    full_node_spec: NodeSpec,
    tool_ids: list[str],
) -> list[str]:
    return [
        tool_id
        for tool_id in tool_ids
        if has_example_inputs(full_node_spec.get_node(node_id=tool_id))
    ]


def get_agent_ids_with_code_reference_kind(
    full_node_spec: NodeSpec,
    agent_ids: list[str],
    kind: str,
) -> list[str]:
    return [
        agent_id
        for agent_id in agent_ids
        if has_code_reference_kind(full_node_spec.get_node(node_id=agent_id), kind)
    ]


def get_agent_ids_with_system_prompt_code_references(
    full_node_spec: NodeSpec,
    agent_ids: list[str],
) -> list[str]:
    return get_agent_ids_with_code_reference_kind(
        full_node_spec=full_node_spec,
        agent_ids=agent_ids,
        kind="system_prompt",
    )


def get_invoked_agent_ids_with_system_prompt_code_references(
    full_node_spec: NodeSpec,
    flow: FlowSpec,
) -> list[str]:
    return get_agent_ids_with_system_prompt_code_references(
        full_node_spec=full_node_spec,
        agent_ids=get_invoked_agent_ids(flow),
    )


def flow_has_first_agent_with_system_prompt_code_references(
    full_node_spec: NodeSpec,
    flow: FlowSpec,
) -> bool:
    attacked_agent_node_ids = get_agent_event_node_ids(flow)
    if not attacked_agent_node_ids:
        return False
    attacked_agent_node = full_node_spec.get_node(node_id=attacked_agent_node_ids[0])
    return has_system_prompt_code_reference(attacked_agent_node)


def get_invoking_agent_unused_tool_ids_with_inputs(
    full_node_spec: NodeSpec,
    flow: FlowSpec,
    tool_id: str,
) -> list[str]:
    agent_id = get_agent_using_the_tool_from_flow(
        flow=flow,
        tool_id=tool_id,
    )
    return get_agent_unused_tool_ids_with_inputs(
        full_node_spec=full_node_spec,
        flow=flow,
        agent_id=agent_id,
    )


def get_eligible_own_tool_attack_target_ids(
    full_node_spec: NodeSpec,
    flow: FlowSpec,
    agent_id: str,
) -> list[str]:
    invoked_agent_tool_ids = get_agent_invoked_tool_ids(
        full_node_spec=full_node_spec,
        flow=flow,
        agent_id=agent_id,
    )
    invoked_agent_tool_ids_with_inputs = get_tool_ids_with_inputs(
        full_node_spec=full_node_spec,
        tool_ids=invoked_agent_tool_ids,
    )
    return get_tool_ids_with_definition_code_references(
        full_node_spec=full_node_spec,
        tool_ids=invoked_agent_tool_ids_with_inputs,
    )


def get_eligible_own_tool_invocation_target_ids(
    full_node_spec: NodeSpec,
    flow: FlowSpec,
    agent_id: str,
) -> list[str]:
    return get_agent_unused_tool_ids_with_inputs(
        full_node_spec=full_node_spec,
        flow=flow,
        agent_id=agent_id,
    )


def get_eligible_own_tool_attackable_tool_ids(
    full_node_spec: NodeSpec,
    flow: FlowSpec,
) -> list[str]:
    attackable_tool_ids: list[str] = []

    for tool_id in get_unique_invoked_tool_ids(flow):
        tool_node = full_node_spec.get_node(node_id=tool_id)
        if tool_node is None:
            continue
        if not has_inputs(tool_node):
            continue
        if not has_definition_code_reference(tool_node):
            continue
        if not get_invoking_agent_unused_tool_ids_with_inputs(
            full_node_spec=full_node_spec,
            flow=flow,
            tool_id=tool_id,
        ):
            continue
        attackable_tool_ids.append(tool_id)

    return attackable_tool_ids


def get_eligible_own_tool_description_attackable_tool_ids(
    full_node_spec: NodeSpec,
    flow: FlowSpec,
) -> list[str]:
    attackable_tool_ids: list[str] = []

    for tool_id in get_unique_invoked_tool_ids(flow):
        tool_node = full_node_spec.get_node(node_id=tool_id)
        if tool_node is None:
            continue
        if not has_assignment_code_reference(tool_node):
            continue
        if not get_invoking_agent_unused_tool_ids_with_inputs(
            full_node_spec=full_node_spec,
            flow=flow,
            tool_id=tool_id,
        ):
            continue
        attackable_tool_ids.append(tool_id)

    return attackable_tool_ids
