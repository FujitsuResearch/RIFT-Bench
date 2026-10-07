from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_SUPPORTED_SCHEMA_CALLS = {
    "NodeSpec",
    "NodeType",
    "CodeReference",
    "InputPort",
    "OutputPort",
    "Edge",
    "Connection",
    "RequiredKeys",
    "Duplication",
    "FrameworkType",
    "AgentType",
    "SystemType",
    "LLMConfig",
    "ToolIOPair",
    "UsageExample",
    "ExampleArgument",
}

_CALL_ARG_NAMES: Dict[str, List[str]] = {
    "NodeSpec": [
        "name",
        "id",
        "is_graph",
        "emulated",
        "node_type",
        "description",
        "code_execution",
        "code_references",
        "inputs",
        "outputs",
        "external_connections",
        "framework",
        "agency_level",
        "flows",
        "entry_point_usage_example",
        "agent_type",
        "system_type",
        "llm_config",
        "system_prompt",
        "user_prompt_template",
        "routing_logic",
        "tool_list",
        "tool_example_pairs",
        "nodes",
        "internal_edges",
        "metadata",
    ],
    "NodeType": ["type", "other_description"],
    "CodeReference": ["kind", "other_kind_description", "file", "line", "snippet"],
    "InputPort": ["name", "dtype", "description", "required", "default"],
    "OutputPort": ["name", "dtype", "description", "output_kind"],
    "Edge": ["from_", "to", "condition", "description"],
    "Connection": ["parent", "in_", "out"],
    "RequiredKeys": ["enabled", "keys"],
    "Duplication": ["exists", "ids"],
    "FrameworkType": ["framework", "other_description"],
    "AgentType": ["type", "other_description"],
    "SystemType": ["type", "other_description"],
    "LLMConfig": ["provider", "class_name", "model_name", "temperature"],
    "ToolIOPair": ["input", "output"],
    "UsageExample": ["script", "arguments"],
    "ExampleArgument": ["name", "type", "example", "is_task_input"],
}


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _parse_ast_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.List):
        return [_parse_ast_value(x) for x in node.elts]
    if isinstance(node, ast.Tuple):
        return [_parse_ast_value(x) for x in node.elts]
    if isinstance(node, ast.Set):
        return [_parse_ast_value(x) for x in node.elts]
    if isinstance(node, ast.Dict):
        out: Dict[Any, Any] = {}
        for key, val in zip(node.keys, node.values):
            out[_parse_ast_value(key)] = _parse_ast_value(val)
        return out
    if isinstance(node, ast.Call):
        fname = _call_name(node.func)
        if fname in _SUPPORTED_SCHEMA_CALLS:
            out: Dict[str, Any] = {}
            arg_names = _CALL_ARG_NAMES.get(fname, [])
            for idx, arg in enumerate(node.args):
                if idx < len(arg_names):
                    out[arg_names[idx]] = _parse_ast_value(arg)
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                out[kw.arg] = _parse_ast_value(kw.value)
            if fname == "NodeSpec":
                out.setdefault("code_references", [])
                out.setdefault("inputs", [])
                out.setdefault("outputs", [])
                out["__is_nodespec__"] = True
            return out
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _load_nodes_with_vars_from_python(path: Path, list_name: str = "ALL_NODES") -> Tuple[List[str], List[Dict[str, Any]]]:
    source = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=str(path))

    nodes_by_var: Dict[str, Dict[str, Any]] = {}
    list_expr: Optional[ast.AST] = None
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == list_name for target in stmt.targets):
                list_expr = stmt.value
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                continue
            target = stmt.targets[0].id
            if isinstance(stmt.value, ast.Call) and _call_name(stmt.value.func) == "NodeSpec":
                parsed = _parse_ast_value(stmt.value)
                if isinstance(parsed, dict):
                    parsed.pop("__is_nodespec__", None)
                    nodes_by_var[target] = parsed
            continue

        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            target = stmt.target.id
            if target == list_name and stmt.value is not None:
                list_expr = stmt.value
            if isinstance(stmt.value, ast.Call) and _call_name(stmt.value.func) == "NodeSpec":
                parsed = _parse_ast_value(stmt.value)
                if isinstance(parsed, dict):
                    parsed.pop("__is_nodespec__", None)
                    nodes_by_var[target] = parsed

    var_order: List[str] = []
    nodes: List[Dict[str, Any]] = []
    if isinstance(list_expr, ast.List):
        for elt in list_expr.elts:
            if isinstance(elt, ast.Name):
                var = elt.id
                node = nodes_by_var.get(var)
                if node is not None:
                    var_order.append(var)
                    nodes.append(dict(node))
            elif isinstance(elt, ast.Call) and _call_name(elt.func) == "NodeSpec":
                parsed = _parse_ast_value(elt)
                if isinstance(parsed, dict):
                    parsed.pop("__is_nodespec__", None)
                    var_order.append(f"node_{len(var_order):03d}")
                    nodes.append(parsed)

    if not nodes:
        for var in sorted(nodes_by_var.keys()):
            var_order.append(var)
            nodes.append(dict(nodes_by_var[var]))
    return var_order, nodes


def load_nodes_by_var(path: Path) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    var_order, nodes = _load_nodes_with_vars_from_python(path, list_name="ALL_NODES")
    if not var_order:
        var_order, nodes = _load_nodes_with_vars_from_python(path, list_name="MAIN_GRAPH_NODES")
    by_var: Dict[str, Dict[str, Any]] = {}
    for var, node in zip(var_order, nodes):
        if isinstance(node, dict):
            by_var[var] = dict(node)
    return var_order, by_var
