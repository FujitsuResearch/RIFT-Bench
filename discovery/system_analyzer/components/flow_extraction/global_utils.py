#!/usr/bin/env python3
"""Shared helpers for modular components."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from .model_client import call_model
from .rag.context_request import (
    _apply_need_filters,
    _set_retriever_k,
    format_context,
    merge_aggregated_context,
    run_task_with_rag,
    try_parse_context_request,
)

retrieval_instructions = (
    "When returning context_request, produce a focused needs[] list where each need resolves exactly one missing fact. "
    "Use query as short literal evidence tokens only (exact symbol names, callee names, argument keys, return-field names, config keys, route/path fragments). "
    "Do not include assignment/definition syntax in query text (=, :=, def, class, import, from, lambda). "
    "Put scope constraints only in filters.path_contains and filters.file_ext, not in query. "
    "Avoid duplicate or near-duplicate queries across rounds. "
    "If a fact is still unresolved, refine that same need by adding one discriminative literal token (do not broaden to a new vague query). "
    "Prefer multiple precise needs over one broad query, and set per-need k only as high as necessary."
)

def to_int_or_none(val: Any) -> int | None:
    try:
        return int(val)
    except Exception:
        return None


def drop_var_refs_from_internal_edges(node: Dict[str, Any], drop_var: str) -> int:
    cur = node.get("internal_edges")
    if not isinstance(cur, list):
        return 0
    nxt: List[Dict[str, Any]] = []
    removed = 0
    for e in cur:
        if not isinstance(e, dict):
            continue
        frm = str(e.get("from_") or "").strip()
        to = str(e.get("to") or "").strip()
        if frm == drop_var or to == drop_var:
            removed += 1
            continue
        nxt.append(e)
    node["internal_edges"] = nxt
    return removed


def drop_var_refs_from_external_connections(node: Dict[str, Any], drop_var: str) -> bool:
    ec = node.get("external_connections")
    if not isinstance(ec, dict):
        return False
    changed = False
    for key in ("in_", "out"):
        vals = ec.get(key)
        if isinstance(vals, str):
            vals = [vals]
        if not isinstance(vals, list):
            continue
        old = [x for x in vals if isinstance(x, str)]
        new_vals = [x for x in old if x != drop_var]
        if len(new_vals) != len(old):
            ec[key] = new_vals
            changed = True
    node["external_connections"] = ec
    return changed


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def parse_json_loose(text: str) -> Dict[str, Any]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def json_loose(text: str) -> Dict[str, Any]:
    return parse_json_loose(text)


def append_rag_log(log_path: Path, event: str, payload: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("\n" + "-" * 80 + "\n")
        fh.write(f"event: {event}\n")
        fh.write(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        fh.write("\n")


def to_json_compatible(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for key, value in obj.items():
            out[str(key)] = to_json_compatible(value)
        return out
    if isinstance(obj, list):
        return [to_json_compatible(item) for item in obj]
    if isinstance(obj, tuple):
        return [to_json_compatible(item) for item in obj]
    if isinstance(obj, set):
        normalized = [to_json_compatible(item) for item in obj]
        return sorted(normalized, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, default=str))
    if isinstance(obj, Path):
        return str(obj)
    return obj


def canonical_json_text(obj: Any) -> str:
    return json.dumps(
        to_json_compatible(obj),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def cache_payload_matches(payload_path: Path, payload: Dict[str, Any]) -> bool:
    if not payload_path.exists():
        return False
    try:
        cached = json.loads(payload_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return False
    return canonical_json_text(cached) == canonical_json_text(payload)


def unique_json_items(values: List[Any]) -> List[Any]:
    out: List[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def norm(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def best_var_match(name: str, var_to_node: Dict[str, Dict[str, Any]]) -> str | None:
    if name in var_to_node:
        return name
    needle = norm(name)
    if not needle:
        return None
    best = None
    best_ratio = 0.0
    for var, node in var_to_node.items():
        candidates = [var]
        node_name = node.get("name")
        if isinstance(node_name, str):
            candidates.append(node_name)
        for cand in candidates:
            cand_norm = norm(cand)
            if not cand_norm:
                continue
            ratio = difflib.SequenceMatcher(None, needle, cand_norm).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best = var
    return best


def primary_file(node: Dict[str, Any]) -> str:
    refs = node.get("code_references")
    if isinstance(refs, list) and refs:
        first = refs[0]
        if isinstance(first, dict):
            file_path = first.get("file")
            if isinstance(file_path, str):
                return file_path
    return ""


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


def call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def parse_ast_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.List):
        return [parse_ast_value(x) for x in node.elts]
    if isinstance(node, ast.Tuple):
        return [parse_ast_value(x) for x in node.elts]
    if isinstance(node, ast.Set):
        # Keep downstream payloads JSON-serializable.
        return [parse_ast_value(x) for x in node.elts]
    if isinstance(node, ast.Dict):
        out: Dict[Any, Any] = {}
        for key, val in zip(node.keys, node.values):
            out[parse_ast_value(key)] = parse_ast_value(val)
        return out
    if isinstance(node, ast.Call):
        fname = call_name(node.func)
        if fname in _SUPPORTED_SCHEMA_CALLS:
            out: Dict[str, Any] = {}
            arg_names = _CALL_ARG_NAMES.get(fname, [])
            for idx, arg in enumerate(node.args):
                if idx < len(arg_names):
                    out[arg_names[idx]] = parse_ast_value(arg)
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                out[kw.arg] = parse_ast_value(kw.value)
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


def load_nodes_with_vars_from_python(path: Path, list_name: str = "ALL_NODES") -> Tuple[List[str], List[Dict[str, Any]]]:
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
            if isinstance(stmt.value, ast.Call) and call_name(stmt.value.func) == "NodeSpec":
                parsed = parse_ast_value(stmt.value)
                if isinstance(parsed, dict):
                    parsed.pop("__is_nodespec__", None)
                    nodes_by_var[target] = parsed
            continue

        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            target = stmt.target.id
            if target == list_name and stmt.value is not None:
                list_expr = stmt.value
            if isinstance(stmt.value, ast.Call) and call_name(stmt.value.func) == "NodeSpec":
                parsed = parse_ast_value(stmt.value)
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
            elif isinstance(elt, ast.Call) and call_name(elt.func) == "NodeSpec":
                parsed = parse_ast_value(elt)
                if isinstance(parsed, dict):
                    parsed.pop("__is_nodespec__", None)
                    var_order.append(f"node_{len(var_order):03d}")
                    nodes.append(parsed)

    if not nodes:
        for var in sorted(nodes_by_var.keys()):
            var_order.append(var)
            nodes.append(dict(nodes_by_var[var]))
    return var_order, nodes


def load_nodes_list_from_python(path: Path, list_name: str = "ALL_NODES") -> List[Dict[str, Any]]:
    _, nodes = load_nodes_with_vars_from_python(path, list_name=list_name)
    return nodes


def load_nodes_by_var(path: Path) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    var_order, nodes = load_nodes_with_vars_from_python(path, list_name="ALL_NODES")
    if not var_order:
        var_order, nodes = load_nodes_with_vars_from_python(path, list_name="MAIN_GRAPH_NODES")
    by_var: Dict[str, Dict[str, Any]] = {}
    for var, node in zip(var_order, nodes):
        if isinstance(node, dict):
            by_var[var] = dict(node)
    return var_order, by_var


def resolve_root_var(root_node: Dict[str, Any], nodes_by_var: Dict[str, Dict[str, Any]]) -> str | None:
    if not isinstance(root_node, dict) or not nodes_by_var:
        return None
    root_name = root_node.get("name")
    root_file = primary_file(root_node)
    if isinstance(root_name, str) and root_name:
        for var, node in nodes_by_var.items():
            if str(node.get("name") or "") != root_name:
                continue
            if root_file:
                node_file = primary_file(node)
                if node_file and str(Path(node_file).resolve()) == str(Path(root_file).resolve()):
                    return var
            else:
                return var
    if isinstance(root_name, str) and root_name:
        return best_var_match(root_name, nodes_by_var)
    return None


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_root_node_json(path: Path, *, allow_flat_object: bool = False) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = read_json(path)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    root_node = obj.get("root_node")
    if isinstance(root_node, dict):
        return root_node
    if allow_flat_object:
        return obj
    return {}


def copy_nodespec_schema_to_out_dir(out_dir: Path, schema_src: Path | None = None) -> Path | None:
    src = schema_src or Path("modular_imp/NodeSpec_schema.py").resolve()
    if not src.exists():
        return None
    dst = out_dir / "NodeSpec_schema.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def normalize_code_references_add(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        ref = dict(item)
        if not isinstance(ref.get("file"), str) or not ref.get("file"):
            continue
        line_start = ref.get("line_start")
        line_end = ref.get("line_end")
        if isinstance(line_start, int) and isinstance(line_end, int):
            ref["line"] = [line_start, line_end] if line_start != line_end else line_start
        elif isinstance(line_start, int):
            ref["line"] = line_start
        out.append(ref)
    return out


def read_snippet(file_path: str, line_val: Any) -> str:
    try:
        lines = Path(file_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    start: Optional[int] = None
    end: Optional[int] = None
    if isinstance(line_val, (list, tuple)) and len(line_val) == 2:
        start, end = int(line_val[0]), int(line_val[1])
    elif isinstance(line_val, int):
        start = end = int(line_val)
    if start is None or end is None:
        return ""
    start = max(1, start)
    end = min(len(lines), end)
    if start > end:
        return ""
    return "\n".join(lines[start - 1 : end])


def extract_var_name(item: Any) -> str:
    if isinstance(item, dict) and isinstance(item.get("__var__"), str):
        return item["__var__"]
    if isinstance(item, str):
        return item
    return ""


def as_var(item: Any) -> str | None:
    value = extract_var_name(item)
    if isinstance(value, str) and value:
        return value
    return None


def build_parent_child_maps(
    var_order: List[str],
    nodes_by_var: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    children_by_parent: Dict[str, Set[str]] = {v: set() for v in var_order}
    parents_by_child: Dict[str, Set[str]] = {v: set() for v in var_order}
    for parent in var_order:
        node = nodes_by_var.get(parent, {})
        for field in ("nodes", "tool_list"):
            vals = node.get(field)
            if not isinstance(vals, list):
                continue
            for item in vals:
                child = as_var(item)
                if child and child in parents_by_child and child != parent:
                    children_by_parent[parent].add(child)
                    parents_by_child[child].add(parent)
    return (
        {k: sorted(v) for k, v in children_by_parent.items()},
        {k: sorted(v) for k, v in parents_by_child.items()},
    )


def build_graph_adjacency_undirected(
    var_order: List[str],
    nodes_by_var: Dict[str, Dict[str, Any]],
) -> Dict[str, Set[str]]:
    adj: Dict[str, Set[str]] = {v: set() for v in var_order}
    valid = set(var_order)

    children_by_parent, _ = build_parent_child_maps(var_order, nodes_by_var)
    for pvar, children in children_by_parent.items():
        if pvar not in valid:
            continue
        for cvar in children:
            if cvar in valid and cvar != pvar:
                adj[pvar].add(cvar)
                adj[cvar].add(pvar)

    for pvar in var_order:
        node = nodes_by_var.get(pvar, {})
        for edge in (node.get("internal_edges") if isinstance(node.get("internal_edges"), list) else []):
            if not isinstance(edge, dict):
                continue
            frm = str(edge.get("from_") or "").strip()
            to = str(edge.get("to") or "").strip()
            if frm in valid and to in valid and frm != to:
                adj[frm].add(to)
                adj[to].add(frm)

        conn = node.get("external_connections") if isinstance(node.get("external_connections"), dict) else {}
        for key in ("in_", "out"):
            vals = conn.get(key)
            if isinstance(vals, str):
                vals = [vals]
            if not isinstance(vals, list):
                continue
            for x in vals:
                if isinstance(x, str) and x in valid and x != pvar:
                    adj[pvar].add(x)
                    adj[x].add(pvar)

    return adj


def root_connected_set(
    *,
    root_var: str,
    var_order: List[str],
    nodes_by_var: Dict[str, Dict[str, Any]],
) -> Set[str]:
    if not root_var or root_var not in set(var_order):
        return set()
    adj = build_graph_adjacency_undirected(var_order, nodes_by_var)
    seen: Set[str] = set()
    stack = [root_var]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in adj.get(cur, set()):
            if nxt not in seen:
                stack.append(nxt)
    return seen


def node_type_name(node: Dict[str, Any]) -> str:
    nt = node.get("node_type")
    if isinstance(nt, dict) and isinstance(nt.get("type"), str):
        return nt["type"]
    return ""


def add_child_ref(parent: Dict[str, Any], child_var: str, field: str = "nodes") -> bool:
    if field == "nodes":
        nt = parent.get("node_type")
        ptype = nt.get("type") if isinstance(nt, dict) else ""
        md = parent.get("metadata")
        if (
            ptype == "Deterministic_controller"
            and isinstance(md, dict)
            and bool(md.get("synthetic_controller_injected"))
        ):
            # Never attach node children to synthetic controllers.
            parent["nodes"] = []
            return False
    vals = parent.get(field)
    if not isinstance(vals, list):
        vals = []
    normalized = [extract_var_name(x) for x in vals]
    normalized = [x for x in normalized if x]
    if child_var in normalized:
        parent[field] = sorted(set(normalized))
        return False
    normalized.append(child_var)
    parent[field] = sorted(set(normalized))
    return True


def remove_child_ref(parent: Dict[str, Any], child_var: str) -> bool:
    changed = False
    for field in ("nodes", "tool_list"):
        vals = parent.get(field)
        if not isinstance(vals, list):
            continue
        old_vals = [extract_var_name(x) for x in vals]
        new_vals = sorted({x for x in old_vals if x and x != child_var})
        if new_vals != sorted({x for x in old_vals if x}):
            changed = True
        parent[field] = new_vals
    return changed


def detach_var_from_parents(
    *,
    var_order: List[str],
    nodes_by_var: Dict[str, Dict[str, Any]],
    child_var: str,
    parent_vars: Iterable[str] | None = None,
    drop_internal_edges: bool = True,
    drop_external_connections: bool = True,
) -> Dict[str, Any]:
    targets = list(parent_vars) if parent_vars is not None else list(var_order)
    links_removed = 0
    internal_edges_removed = 0
    external_refs_removed = 0
    touched_parents: Set[str] = set()

    for p in targets:
        if p not in nodes_by_var:
            continue
        pnode = nodes_by_var[p]
        if remove_child_ref(pnode, child_var):
            links_removed += 1
            touched_parents.add(p)
        if drop_internal_edges:
            internal_edges_removed += int(drop_var_refs_from_internal_edges(pnode, child_var) or 0)
        if drop_external_connections:
            external_refs_removed += int(drop_var_refs_from_external_connections(pnode, child_var) or 0)

    return {
        "parents": sorted(touched_parents),
        "parent_links_removed": int(links_removed),
        "internal_edges_removed": int(internal_edges_removed),
        "external_refs_removed": int(external_refs_removed),
        "edge_refs_removed": int(internal_edges_removed + external_refs_removed),
    }


def write_nodes_output(
    out_path: Path,
    var_order: List[str],
    nodes_by_var: Dict[str, Dict[str, Any]],
    report: Dict[str, Any],
    report_var_name: str,
) -> None:
    out_vars = [v for v in var_order if v in nodes_by_var]
    header = (
        "from NodeSpec_schema import NodeSpec, NodeType, CodeReference, InputPort, OutputPort, Edge, Connection, "
        "LLMConfig, RequiredKeys, AgentType, SystemType, FrameworkType, UsageExample, ExampleArgument\n\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header, encoding="utf-8")
    with out_path.open("a", encoding="utf-8") as fh:
        for var in out_vars:
            node = dict(nodes_by_var[var])
            node.pop("__var__", None)
            fh.write(render_nodespec_assignment(var, node))
            fh.write("\n\n")
        fh.write("ALL_NODES = [\n")
        for var in out_vars:
            fh.write(f"    {var},\n")
        fh.write("]\n\n")
        fh.write(f"{report_var_name} = ")
        fh.write(render_obj(report))
        fh.write("\n")


def render_obj(value: Any, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, dict) and "__var__" in value:
        return str(value["__var__"])
    if isinstance(value, dict) and "__call__" in value:
        cname = str(value.get("__call__") or "")
        parts: List[str] = []
        for k, val in value.items():
            if k == "__call__":
                continue
            if val is None:
                continue
            parts.append(f"{k}={render_obj(val, indent=indent + 4)}")
        return f"{cname}({', '.join(parts)})"
    if isinstance(value, str):
        if "\n" not in value:
            return repr(value)
        parts = [json.dumps(line) for line in value.splitlines(True)]
        inner = "\n".join(f"{pad}    {part}" for part in parts)
        return "(\n" + inner + f"\n{pad})"
    if value is None or isinstance(value, (int, float, bool)):
        return repr(value)
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        inner = ",\n".join(f"{pad}    {render_obj(item, indent + 4)}" for item in value)
        return "[\n" + inner + f"\n{pad}]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        inner = ",\n".join(f"{pad}    {repr(k)}: {render_obj(v, indent + 4)}" for k, v in value.items())
        return "{\n" + inner + f"\n{pad}" + "}"
    return repr(value)


def render_nodespec_assignment(var: str, node: Dict[str, Any]) -> str:
    preferred = [
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
        "required_keys",
        "duplicates",
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
    ]
    keys = [k for k in preferred if k in node] + [k for k in node.keys() if k not in preferred]
    lines: List[str] = [f"{var} = NodeSpec("]
    for key in keys:
        value = node.get(key)
        if value is None:
            lines.append(f"    {key}=None,")
            continue
        lines.append(f"    {key}={render_obj(value, indent=4)},")
    lines.append(")")
    return "\n".join(lines)


def render_multiline_string(text: str, indent: int) -> str:
    if "\n" not in text:
        return repr(text)
    pad = " " * indent
    parts = [json.dumps(line) for line in text.splitlines(True)]
    inner = "\n".join(f"{pad}    {part}" for part in parts)
    return "(\n" + inner + f"\n{pad})"


def safe_file_tag(text: str) -> str:
    tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text or "").strip("._")
    return tag or "call"


def safe_name(value: str, max_len: int = 96) -> str:
    out = []
    for ch in value:
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out).strip("_")
    while "__" in s:
        s = s.replace("__", "_")
    s = s or "item"
    limit = max(16, int(max_len))
    if len(s) <= limit:
        return s
    digest = hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
    head = s[: max(1, limit - 13)].rstrip("._-")
    return f"{head}_{digest}"


def slugify(name: str) -> str:
    chars = []
    for ch in name.lower():
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("_")
    slug = "".join(chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "node"


def slug_var_name(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", text.strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "node"
    if s[0].isdigit():
        s = f"n_{s}"
    return s


def next_var(base_name: str, existing_vars: Set[str]) -> str:
    base = slug_var_name(base_name)
    if base not in existing_vars:
        return base
    i = 2
    while f"{base}_{i}" in existing_vars:
        i += 1
    return f"{base}_{i}"


def identity_key(name: Optional[str], file_path: Optional[str]) -> str:
    return f"{name or ''}||{file_path or ''}"


def load_identity_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    return {}


def save_identity_map(path: Path, mapping: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_var_name(name: Optional[str], file_path: Optional[str], used: Set[str], mapping: Dict[str, str]) -> str:
    key = identity_key(name, file_path)
    base = slugify(name or "node")
    var = mapping.get(key, base)
    if isinstance(name, str) and " " in name and "_" not in var:
        compact_base = base.replace("_", "")
        if var == compact_base:
            var = base
    if var in used:
        if file_path:
            file_tag = slugify(Path(file_path).name)
            if file_tag:
                var = f"{base}_{file_tag}"
        if var in used:
            idx = 2
            while f"{var}_{idx}" in used:
                idx += 1
            var = f"{var}_{idx}"
    used.add(var)
    mapping[key] = var
    return var


def _normalize_line_value(val: Any) -> Any:
    """
    Normalize code reference line field to a safe canonical shape:
    - int -> int
    - [start, end] or (start, end) with numeric entries -> [int, int]
    - everything else -> None
    """
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, (list, tuple)) and len(val) == 2:
        a, b = val[0], val[1]
        if isinstance(a, bool) or isinstance(b, bool):
            return None
        try:
            return [int(a), int(b)]
        except Exception:
            return None
    return None


def line_key(val: Any) -> Tuple[int, int]:
    norm = _normalize_line_value(val)
    if isinstance(norm, list) and len(norm) == 2:
        return int(norm[0]), int(norm[1])
    if isinstance(norm, int):
        return norm, norm
    return 0, 0


def sort_code_references(refs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _key(ref: Dict[str, Any]) -> Tuple[str, str, Tuple[int, int]]:
        kind = str(ref.get("kind") or "")
        file_path = str(ref.get("file") or "")
        line = line_key(ref.get("line"))
        return kind, file_path, line

    normalized: List[Dict[str, Any]] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        item = dict(ref)
        item["line"] = _normalize_line_value(item.get("line"))
        normalized.append(item)
    return sorted(normalized, key=_key)


def sort_ports(ports: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted([port for port in ports if isinstance(port, dict)], key=lambda p: str(p.get("name") or ""))


def sort_var_refs(items: Iterable[Any]) -> List[Any]:
    def _key(item: Any) -> str:
        if isinstance(item, dict) and "__var__" in item:
            return str(item["__var__"])
        if isinstance(item, dict) and "name" in item:
            return str(item["name"])
        if isinstance(item, str):
            return item
        return ""

    return sorted(list(items), key=_key)


def sort_node_dicts(nodes: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _key(node: Dict[str, Any]) -> Tuple[str, str, str]:
        name = str(node.get("name") or "")
        file_path = primary_file(node)
        node_type = ""
        nt = node.get("node_type")
        if isinstance(nt, dict):
            node_type = str(nt.get("type") or "")
        return name, file_path, node_type

    out: List[Dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        item = dict(node)
        if isinstance(item.get("code_references"), list):
            item["code_references"] = sort_code_references(item["code_references"])
        if isinstance(item.get("inputs"), list):
            item["inputs"] = sort_ports(item["inputs"])
        if isinstance(item.get("outputs"), list):
            item["outputs"] = sort_ports(item["outputs"])
        if isinstance(item.get("nodes"), list):
            item["nodes"] = sort_var_refs(item["nodes"])
        if isinstance(item.get("tool_list"), list):
            item["tool_list"] = sort_var_refs(item["tool_list"])
        out.append(item)
    return sorted(out, key=_key)


def validate_minimal_nodes(nodes: Iterable[Any], required_fields: Iterable[str]) -> None:
    req = list(required_fields)
    for idx, node in enumerate(nodes, start=1):
        if isinstance(node, dict):
            for field in req:
                if field not in node:
                    raise ValueError(f"Node {idx} missing required field: {field}")
        else:
            for field in req:
                if not hasattr(node, field):
                    raise ValueError(f"Node {idx} missing required attr: {field}")


def call_with_cache_text(
    payload: Dict[str, Any],
    *,
    payload_path: Path,
    raw_path: Path,
    model: str,
    refresh_raw: bool,
    retriever: Any | None = None,
    rag_max_rounds: int = 6,
    use_first_round_context_request_hint: bool = False,
) -> Tuple[str, str]:
    safe_payload = to_json_compatible(payload)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(
        json.dumps(safe_payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    if raw_path.exists() and not refresh_raw and cache_payload_matches(payload_path, safe_payload):
        text = raw_path.read_text(encoding="utf-8", errors="replace")
        return text, "cache"

    if retriever is not None:
        rag_log_path = raw_path.parent / f"{raw_path.name}.rag.log"
        text = run_task_with_rag(
            safe_payload,
            model,
            retriever,
            max_rounds=rag_max_rounds,
            debug_log_path=rag_log_path,
            use_first_round_context_request_hint=bool(use_first_round_context_request_hint),
        )
    else:
        text = call_model(safe_payload, model)
    raw_path.write_text(text, encoding="utf-8")
    return text, "model"


def call_with_cache(
    payload: Dict[str, Any],
    *,
    payload_path: Path,
    raw_path: Path,
    model: str,
    refresh_raw: bool,
    retriever: Any | None = None,
    rag_max_rounds: int = 6,
    use_first_round_context_request_hint: bool = False,
) -> Tuple[Dict[str, Any], str]:
    def _looks_like_refusal(text: str) -> bool:
        s = str(text or "").strip().lower()
        if not s:
            return True
        refusal_markers = (
            "i'm sorry",
            "i am sorry",
            "cannot assist",
            "can't assist",
            "cannot help with",
            "can't help with",
            "unable to assist",
            "unable to help",
            "must refuse",
            "cannot comply",
            "can't comply",
        )
        return any(m in s for m in refusal_markers)

    def _is_usable_json_obj(obj: Any) -> bool:
        return isinstance(obj, dict) and len(obj) > 0

    text, source = call_with_cache_text(
        payload,
        payload_path=payload_path,
        raw_path=raw_path,
        model=model,
        refresh_raw=refresh_raw,
        retriever=retriever,
        rag_max_rounds=rag_max_rounds,
        use_first_round_context_request_hint=use_first_round_context_request_hint,
    )
    parsed = json_loose(text)
    if _is_usable_json_obj(parsed) and not _looks_like_refusal(text):
        return parsed, source

    # Global hardening: if the cached/first response is refusal-like or non-JSON/empty JSON,
    # force fresh attempts to reduce brittle single-shot failures.
    attempts = 2
    last_text = text
    last_obj = parsed if isinstance(parsed, dict) else {}
    for _ in range(attempts):
        retry_text, _retry_source = call_with_cache_text(
            payload,
            payload_path=payload_path,
            raw_path=raw_path,
            model=model,
            refresh_raw=True,
            retriever=retriever,
            rag_max_rounds=rag_max_rounds,
            use_first_round_context_request_hint=use_first_round_context_request_hint,
        )
        retry_obj = json_loose(retry_text)
        last_text = retry_text
        last_obj = retry_obj if isinstance(retry_obj, dict) else {}
        if _is_usable_json_obj(last_obj) and not _looks_like_refusal(last_text):
            return last_obj, "model"

    return last_obj, "model"


def run_subprocess(cmd: List[str], *, repo_root: Path) -> None:
    print(f"\n[run] $ {' '.join(cmd)}")
    env = os.environ.copy()
    current_pp = env.get("PYTHONPATH", "")
    repo_root_str = str(repo_root)
    env["PYTHONPATH"] = (
        f"{repo_root_str}{os.pathsep}{current_pp}" if current_pp else repo_root_str
    )
    p = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True, env=env)
    if p.returncode != 0:
        if p.stdout:
            print("\n[stdout]\n" + p.stdout)
        if p.stderr:
            print("\n[stderr]\n" + p.stderr)
        raise subprocess.CalledProcessError(p.returncode, cmd, output=p.stdout, stderr=p.stderr)


def extract_context_request_needs(req: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not isinstance(req, dict):
        return []
    parsed_needs = [x for x in (req.get("needs") if isinstance(req.get("needs"), list) else []) if isinstance(x, dict)]
    if parsed_needs:
        return parsed_needs
    query = str(req.get("query") or "").strip()
    if not query:
        return []
    try:
        k_val = int(req.get("k", 8))
    except Exception:
        k_val = 8
    return [{"query": query, "k": k_val, "filters": {"file_ext": [], "path_contains": []}}]


def retrieve_context_for_needs(
    *,
    parsed_needs: List[Dict[str, Any]],
    retriever: Any,
    seen_need_keys: Set[Tuple[str, str, str, int]] | None = None,
    key_prefix: str = "context_request_need",
) -> Tuple[str, List[Dict[str, Any]]]:
    context = ""
    history_batch: List[Dict[str, Any]] = []

    for need in parsed_needs:
        if not isinstance(need, dict):
            continue
        need_query = str(need.get("query") or "").strip()
        if not need_query:
            continue
        filters = need.get("filters") if isinstance(need.get("filters"), dict) else {}
        try:
            need_k = int(need.get("k", 8))
        except Exception:
            need_k = 8
        need_k = max(1, min(50, need_k))

        if seen_need_keys is not None:
            filter_key = json.dumps(filters, ensure_ascii=False, sort_keys=True, default=str)
            dedupe_key = (key_prefix, need_query.lower(), filter_key, need_k)
            if dedupe_key in seen_need_keys:
                continue
            seen_need_keys.add(dedupe_key)

        applied_k = _set_retriever_k(retriever, need_k)
        docs_raw = retriever.get_relevant_documents(need_query)
        docs_filtered = _apply_need_filters(docs_raw, filters)
        docs = docs_filtered[:applied_k]
        ctx_piece = format_context(docs)
        if ctx_piece:
            context = merge_aggregated_context(context, ctx_piece) if context else ctx_piece
        history_batch.append(
            {
                "query": need_query,
                "k": applied_k,
                "docs": len(docs) if isinstance(docs, list) else None,
                "raw_docs": len(docs_raw) if isinstance(docs_raw, list) else None,
                "filtered_docs": len(docs_filtered) if isinstance(docs_filtered, list) else None,
                "filters": filters,
            }
        )

    return context, history_batch


def run_context_rounds(
    *,
    initial_payload: Dict[str, Any],
    model: str,
    retriever: Any,
    max_rounds: int = 6,
    call_model_fn: Callable[[Dict[str, Any], str], str] = call_model,
    parse_response_fn: Callable[[str], Dict[str, Any]] = json_loose,
    on_round_start: Callable[[int, Dict[str, Any]], None] | None = None,
    on_round_response: Callable[[int, Dict[str, Any], Dict[str, Any]], None] | None = None,
    on_context_retrieved: Callable[[int, Dict[str, Any], List[Dict[str, Any]], str, Dict[str, Any]], None] | None = None,
    stop_after_response: Callable[[int, Dict[str, Any], Dict[str, Any]], bool] | None = None,
) -> Tuple[Dict[str, Any], str, Dict[str, Any], Dict[str, Any]]:
    rounds = max(1, int(max_rounds))
    work = dict(initial_payload)
    work.setdefault("retrieved_evidence_context", None)
    work.setdefault("retrieval_history", [])
    if "additional_context" not in work:
        work["additional_context"] = work.get("retrieved_evidence_context")

    last_text = ""
    last_obj: Dict[str, Any] = {}
    retrieval_count = 0
    rounds_executed = 0

    for i in range(1, rounds + 1):
        rounds_executed = i
        if on_round_start is not None:
            on_round_start(i, work)

        text = call_model_fn(work, model)
        last_text = text
        obj = parse_response_fn(text)
        last_obj = obj if isinstance(obj, dict) else {}

        if on_round_response is not None:
            on_round_response(i, last_obj, work)

        if stop_after_response is not None and bool(stop_after_response(i, last_obj, work)):
            break

        req = try_parse_context_request(last_obj)
        parsed_needs = extract_context_request_needs(req if isinstance(req, dict) else None)
        if not parsed_needs:
            break

        ctx, hist = retrieve_context_for_needs(
            parsed_needs=parsed_needs,
            retriever=retriever,
            seen_need_keys=None,
        )
        if not hist:
            break

        retrieval_count += len(hist)
        existing_ctx = str(work.get("retrieved_evidence_context") or "").strip()
        merged = merge_aggregated_context(existing_ctx, ctx) if ctx else existing_ctx
        work["retrieved_evidence_context"] = merged
        work["additional_context"] = merged

        history_entry = {
            "round": i,
            "request_id": (req or {}).get("request_id") if isinstance(req, dict) else None,
            "query": " ; ".join(str(x.get("query") or "") for x in parsed_needs if isinstance(x, dict)),
            "k": int(hist[0].get("k", 8)) if hist else 8,
            "needs": parsed_needs,
            "source": (req or {}).get("source") if isinstance(req, dict) else None,
            "retrieved_context": ctx,
        }
        rh = work.get("retrieval_history") if isinstance(work.get("retrieval_history"), list) else []
        work["retrieval_history"] = [*rh, history_entry]

        if on_context_retrieved is not None:
            on_context_retrieved(i, req if isinstance(req, dict) else {}, hist, ctx, work)

    meta = {
        "rounds": rounds_executed,
        "retrievals": retrieval_count,
        "history_len": len(work.get("retrieval_history") if isinstance(work.get("retrieval_history"), list) else []),
    }
    return last_obj, last_text, meta, work


__all__ = [
    "call_model",
    "run_task_with_rag",
    "cache_payload_matches",
    "parse_json_loose",
    "json_loose",
    "append_rag_log",
    "load_env_file",
    "load_nodes_by_var",
    "load_nodes_list_from_python",
    "load_nodes_with_vars_from_python",
    "primary_file",
    "render_nodespec_assignment",
    "resolve_root_var",
    "safe_name",
    "read_json",
    "read_root_node_json",
    "copy_nodespec_schema_to_out_dir",
    "normalize_code_references_add",
    "read_snippet",
    "extract_var_name",
    "as_var",
    "build_parent_child_maps",
    "build_graph_adjacency_undirected",
    "root_connected_set",
    "node_type_name",
    "add_child_ref",
    "remove_child_ref",
    "write_nodes_output",
    "unique_json_items",
    "safe_file_tag",
    "slugify",
    "slug_var_name",
    "next_var",
    "identity_key",
    "load_identity_map",
    "save_identity_map",
    "resolve_var_name",
    "sort_node_dicts",
    "validate_minimal_nodes",
    "render_multiline_string",
    "call_with_cache_text",
    "call_with_cache",
    "run_subprocess",
    "extract_context_request_needs",
    "retrieve_context_for_needs",
    "run_context_rounds",
]
