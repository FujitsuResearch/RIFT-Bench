from collections import defaultdict
from typing import Any
from copy import deepcopy
from .coding import *


def regroup_dict(data: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """
    Regroup tool entries into regular, local MCP, and external MCP buckets.

    Args:
        data (dict[str, list[dict[str, Any]]]): Flat tool mapping keyed by source prefix.

    Return:
        dict[str, Any]: Regrouped tool mapping with `regular`, `local`, and `external` keys.
    """
    result = {
        "regular": [],
        "local": {},
        "external": {},
    }

    for key, value in data.items():
        if key.startswith("local_"):
            suffix = key[len("local_") :]
            result["local"][suffix] = value
        elif key.startswith("external_"):
            suffix = key[len("external_") :]
            result["external"][suffix] = value
        else:
            result["regular"].extend(value)

    return result


def get_all_tools(node: Any) -> dict[str, Any]:
    """
    Collect tool metadata from a node graph.

    Args:
        node (Any): Root node specification object.

    Return:
        dict[str, Any]: Tool metadata grouped by regular, local MCP, and external MCP scopes.
    """
    tools = defaultdict(list)

    def get_all_tools_rec(node: Any, parent_node: str = "general") -> None:
        if node.node_type.type == "Tool":
            if node.duplicates.exists:
                current_tools_ids = [i["id"] for i in tools[parent_node]]
                instances_ids = [i["id"] for i in node.duplicates.instances]
                if set(current_tools_ids) & set(instances_ids):
                    return
                
            entry = {"id": node.id, "name": node.metadata.get('component_handle', sanitize_function_name(node.name))}
            if node.description:
                entry["desc"] = node.description
            assignment_ref = None
            definition_ref = None
            for tool_code_ref in node.code_references or []:
                if tool_code_ref.kind == "assignment":
                    assignment_ref = tool_code_ref
                if tool_code_ref.kind == "definition":
                    definition_ref = tool_code_ref
            if assignment_ref:
                entry["file_path"] = assignment_ref.file
                entry["assignment_lines"] = normalize_line_numbers(assignment_ref.line)
                entry["assignment_snippet"] = assignment_ref.snippet
            if definition_ref:
                entry["definition_file_path"] = definition_ref.file
                entry["definition_lines"] = normalize_line_numbers(definition_ref.line)
                entry["definition_snippet"] = definition_ref.snippet
            if entry.get("file_path", None) is None and not assignment_ref and definition_ref:
                new_assignment_ref = deepcopy(definition_ref)
                new_assignment_ref.kind = "assignment"
                node.code_references.append(new_assignment_ref)
                entry["file_path"] = definition_ref.file
                entry["assignment_lines"] = normalize_line_numbers(definition_ref.line)
                entry["assignment_snippet"] = definition_ref.snippet

            inputs = None
            outputs = None
            tool_example_pairs = None
            if node.inputs:
                inputs = node.inputs
            if node.outputs:
                outputs = node.outputs
            if node.tool_example_pairs:
                tool_example_pairs = node.tool_example_pairs

            entry["inputs"] = inputs
            entry["outputs"] = outputs
            entry["tool_example_pairs"] = tool_example_pairs

            tools[parent_node].append(entry)

        if node.nodes:
            for inner_node in node.nodes:
                get_all_tools_rec(inner_node)

        if node.tool_list:
            parent_node = node.id
            for tool in node.tool_list:
                if node.node_type.type == "Local_MCP_server":
                    get_all_tools_rec(tool, f"local_{parent_node}")
                elif node.node_type.type == "External_MCP_server":
                    get_all_tools_rec(tool, f"external_{parent_node}")
                else:
                    get_all_tools_rec(tool, parent_node)

    get_all_tools_rec(node)
    tools = regroup_dict(tools)
    return tools


def update_nodespec(node_spec: Any, file_path: str, file_content: str) -> None:
    """
    Update line-number references in a node specification after file modifications.

    Args:
        node_spec (Any): Node specification object to update.
        file_path (str): Updated file path in the project.
        file_content (str): Full updated file content.
    """
    nodes_to_update = node_spec.get_nodes_with_snipets_from_given_file(file_path)
    for node_id in nodes_to_update:
        node = node_spec.get_node(node_id)
        if node.name == 'add':
            pass
        for i, ref in enumerate(node.code_references):
            if ref.file == file_path:
                new_line = find_snippet_lines(file_content, ref.snippet) 
                if new_line == None:
                    continue
                ref.line = new_line
        if node.emulated_code_references:
            for i, ref in enumerate(node.emulated_code_references):
                if ref.file == file_path:
                    new_line = find_snippet_lines(file_content, ref.snippet) 
                    if new_line == None:
                        continue
                    ref.line = new_line
