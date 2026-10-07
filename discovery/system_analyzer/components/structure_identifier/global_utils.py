#!/usr/bin/env python3
"""Shared helpers for modular components."""

import ast
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

try:
    from .model_client import call_model
    from .rag.context_request import (
        extract_context_request_needs,
        merge_aggregated_context,
        run_task_with_rag,
        try_parse_context_request,
    )
except ImportError:
    from model_client import call_model
    from rag.context_request import (
        extract_context_request_needs,
        merge_aggregated_context,
        run_task_with_rag,
        try_parse_context_request,
    )

# Shared text blocks used in many prompts
agentic_terminology_instructions = (
    "Terminology (mandatory):\n"
    "- Agentic Component: a runtime unit with its own operational role in the agentic system "
    "(for example System, Agent, LLM instance, Tool, MCP server/server, Database, major custom runtime component).\n"
    "- Support Artifact: non-component artifact used to define/configure/instantiate an Agentic Component "
    "(for example wrappers, adapters, config payloads, prompts/templates, schemas, constants, helper functions, list-builders).\n"
    "- Code Object: function/class/variable/module in code. Code objects are evidence only; decisions are about Agentic Components."
)

guidance_summary_instructions = (
    "`guidance_summary` provides high-level direction only; use it to guide search focus, "
    "but never treat it as direct evidence and never let it replace concrete code evidence."
)

context_request_format_template = (
    "Context request format:\n"
    "{\n"
    "  \"context_request\": {\n"
    "    \"request_id\": \"<<REQUEST_ID>>\",\n"
    "    \"needs\": [\n"
    "      {\n"
    "        \"kind\": \"call_flow|symbol_definition|concept\",\n"
    "        \"query\": \"<specific missing evidence>\",\n"
    "        \"k\": 8,\n"
    "        \"filters\": {\"file_ext\": [\".py\", \".json\"], \"path_contains\": []}\n"
    "      }\n"
    "    ]\n"
    "  }\n"
    "}"
)

vars_fields_context_request_format_template = (
    "Context request format:\n"
    "{\n"
    "  \"context_request\": {\n"
    "    \"request_id\": \"<<REQUEST_ID>>\",\n"
    "    \"vars_fields\": {\n"
    "      \"<target_var>\": [\"<field1>\", \"<field2>\"]\n"
    "    },\n"
    "    \"why\": \"<short reason for missing information>\"\n"
    "  }\n"
    "}"
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
    """ returns the value as an integer or None"""
    try:
        return int(val)
    except Exception:
        return None


def drop_var_refs_from_internal_edges(node: Dict[str, Any], drop_var: str) -> int:
    """Drop internal edges whose `from_` or `to` endpoint matches `drop_var`."""
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
    """Drop external connections whose `in_` or `out` endpoint matches `drop_var`."""
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
    """Load simple KEY=VALUE entries from a .env-style file into `os.environ` when unset."""
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
    """Parse a JSON object from raw text, tolerating surrounding non-JSON text when possible."""
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


def to_json_compatible(obj: Any) -> Any:
    """Recursively convert values into JSON-safe shapes using strings and lists where needed."""
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
    """Serialize an object into a stable normalized JSON string for equality checks and deduping."""
    return json.dumps(
        to_json_compatible(obj),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def cache_payload_matches(payload_path: Path, payload: Dict[str, Any]) -> bool:
    """Return whether a saved payload file is semantically equal to the current payload."""
    if not payload_path.exists():
        return False
    try:
        cached = json.loads(payload_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return False
    return canonical_json_text(cached) == canonical_json_text(payload)


def unique_json_items(values: List[Any]) -> List[Any]:
    """Return items with JSON-equivalent duplicates removed while preserving first-seen order."""
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
    """Normalize text for loose name matching by lowercasing and keeping only alphanumerics."""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def best_var_match(name: str, var_to_node: Dict[str, Dict[str, Any]]) -> str | None:
    """Return the closest variable key match for a name using normalized fuzzy comparison."""
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
    """Return the first available code-reference file path for a node, or an empty string."""
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
    """Return the simple callable name from an AST function expression when available."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def parse_ast_value(node: ast.AST) -> Any:
    """Convert supported AST value expressions into plain Python data structures."""
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
    """Load node dicts from a Python file and preserve the original variable ordering used to build the node list."""
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
    """Thin wrapper for load_nodes_with_vars_from_python"""
    _, nodes = load_nodes_with_vars_from_python(path, list_name=list_name)
    return nodes


def load_nodes_by_var(path: Path) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Load emitted nodes from a Python nodes file and index them by their variable names."""
    var_order, nodes = load_nodes_with_vars_from_python(path, list_name="ALL_NODES")
    if not var_order:
        var_order, nodes = load_nodes_with_vars_from_python(path, list_name="MAIN_GRAPH_NODES")
    by_var: Dict[str, Dict[str, Any]] = {}
    for var, node in zip(var_order, nodes):
        if isinstance(node, dict):
            by_var[var] = dict(node)
    return var_order, by_var


def resolve_root_var(root_node: Dict[str, Any], nodes_by_var: Dict[str, Dict[str, Any]]) -> str | None:
    """Find the emitted variable name that best matches the chosen root node by name and file."""
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
    """Read and parse a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_root_node_json(path: Path, *, allow_flat_object: bool = False) -> Dict[str, Any]:
    """Read a root-node JSON artifact and return the embedded root node object when present."""
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


def validate_json_artifact(path: Path, required_keys: Iterable[str] | None = None) -> Dict[str, Any]:
    """Validate that a JSON artifact exists, parses, and contains any required top-level keys."""
    result: Dict[str, Any] = {
        "ok": False,
        "kind": "json",
        "path": str(path),
        "reason": "",
        "details": {},
    }
    if not path.exists():
        result["reason"] = "missing_file"
        return result
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        result["reason"] = "invalid_json"
        result["details"] = {"error": str(exc)}
        return result
    if not isinstance(obj, dict):
        result["reason"] = "json_not_object"
        result["details"] = {"type": type(obj).__name__}
        return result
    required = [str(key) for key in (required_keys or []) if str(key)]
    missing = [key for key in required if key not in obj]
    if missing:
        result["reason"] = "missing_required_keys"
        result["details"] = {"missing_keys": missing}
        return result
    result["ok"] = True
    result["reason"] = "ok"
    result["details"] = {
        "required_keys": required,
        "present_keys": sorted(str(key) for key in obj.keys()),
    }
    return result


def validate_root_node_artifact(path: Path, *, allow_flat_object: bool = False) -> Dict[str, Any]:
    """Validate that a root-node JSON artifact contains a usable root-node object."""
    result: Dict[str, Any] = {
        "ok": False,
        "kind": "root_json",
        "path": str(path),
        "reason": "",
        "details": {
            "allow_flat_object": bool(allow_flat_object),
        },
    }
    json_result = validate_json_artifact(path)
    if not json_result.get("ok"):
        result["reason"] = str(json_result.get("reason") or "invalid_json_artifact")
        result["details"] = dict(json_result.get("details") or {})
        result["details"]["allow_flat_object"] = bool(allow_flat_object)
        return result
    root_node = read_root_node_json(path, allow_flat_object=allow_flat_object)
    if not isinstance(root_node, dict) or not root_node:
        result["reason"] = "missing_root_node"
        result["details"]["root_node_type"] = type(root_node).__name__
        return result
    result["ok"] = True
    result["reason"] = "ok"
    result["details"]["root_keys"] = sorted(str(key) for key in root_node.keys())
    return result


def validate_nodes_py_artifact(path: Path) -> Dict[str, Any]:
    """Validate that a Python nodes artifact loads into a non-empty node graph."""
    result: Dict[str, Any] = {
        "ok": False,
        "kind": "nodes_py",
        "path": str(path),
        "reason": "",
        "details": {},
    }
    if not path.exists():
        result["reason"] = "missing_file"
        return result
    try:
        var_order, nodes_by_var = load_nodes_by_var(path)
    except Exception as exc:
        result["reason"] = "nodes_load_failed"
        result["details"] = {"error": str(exc)}
        return result
    if not isinstance(var_order, list) or not isinstance(nodes_by_var, dict):
        result["reason"] = "nodes_invalid_container_types"
        result["details"] = {
            "var_order_type": type(var_order).__name__,
            "nodes_by_var_type": type(nodes_by_var).__name__,
        }
        return result
    if not var_order or not nodes_by_var:
        result["reason"] = "empty_nodes"
        result["details"] = {
            "var_count": len(var_order),
            "node_count": len(nodes_by_var),
        }
        return result
    missing_vars = [var for var in var_order if var not in nodes_by_var]
    if missing_vars:
        result["reason"] = "missing_nodes_for_vars"
        result["details"] = {"missing_vars": missing_vars[:20]}
        return result
    result["ok"] = True
    result["reason"] = "ok"
    result["details"] = {
        "var_count": len(var_order),
        "node_count": len(nodes_by_var),
        "sample_vars": var_order[:10],
    }
    return result


def copy_nodespec_schema_to_out_dir(out_dir: Path, schema_src: Path | None = None) -> Path | None:
    """Copy the NodeSpec schema file into an output directory and return the destination path."""
    src = schema_src or (Path(__file__).resolve().parent / "NodeSpec_schema.py").resolve()
    if not src.exists():
        return None
    dst = out_dir / "NodeSpec_schema.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def normalize_code_references_add(raw: Any) -> List[Dict[str, Any]]:
    """Normalize added code-reference objects into the line field shape used elsewhere in the pipeline."""
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
    """Return the file snippet at a 1-based line or line range, or an empty string when invalid."""
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
    """Return a variable name from either a `{'__var__': ...}` record or a plain string."""
    if isinstance(item, dict) and isinstance(item.get("__var__"), str):
        return item["__var__"]
    if isinstance(item, str):
        return item
    return ""


def as_var(item: Any) -> str | None:
    """Return a child/reference var name when present, otherwise `None`."""
    value = extract_var_name(item)
    if isinstance(value, str) and value:
        return value
    return None


def build_parent_child_maps(
    var_order: List[str],
    nodes_by_var: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Build parent-to-children and child-to-parents maps from `nodes` and `tool_list` refs."""
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
    """Build an undirected adjacency map using only `nodes` and `tool_list` parent-child links."""
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

    return adj


def root_connected_set(
    *,
    root_var: str,
    var_order: List[str],
    nodes_by_var: Dict[str, Dict[str, Any]],
) -> Set[str]:
    """Return the vars connected to `root_var` through `nodes` and `tool_list` links only."""
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
    """Return the normalized string node type from a node dictionary."""
    nt = node.get("node_type")
    if isinstance(nt, dict) and isinstance(nt.get("type"), str):
        return nt["type"]
    return ""


def add_child_ref(parent: Dict[str, Any], child_var: str, field: str = "nodes") -> bool:
    """Attach a child var to a parent under `nodes` or `tool_list`, enforcing parent-type constraints."""
    nt = parent.get("node_type")
    ptype = nt.get("type") if isinstance(nt, dict) else ""

    if field == "nodes":
        md = parent.get("metadata")
        if (
            ptype == "Deterministic_controller"
        ):
            # Never attach node children to controllers.
            parent["nodes"] = []
            return False

    if field == "tool_list" and ptype not in {"Local_MCP_server", "External_MCP_server"}:
        parent["tool_list"] = []
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
    """Remove one child var from a parent's `nodes` and `tool_list` refs."""
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
    drop_internal_edges: bool = True
) -> Dict[str, Any]:
    """Remove a child var reference from the selected parents and optionally clean related edges/connections."""

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


    # Return the shared decision data for the caller-specific mutation step.
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
    """Write emitted nodes plus a stage report into a Python output file."""
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


def load_guidance_summary(out_dir: Path) -> str:
    """Load the saved guidance summary string from the stage output directory."""
    path = out_dir / "system_guidance.json"
    if not path.exists():
        return ""
    try:
        obj = read_json(path)
    except Exception:
        return ""
    if isinstance(obj.get("guidance"), dict):
        return str((obj.get("guidance") or {}).get("summary") or "")
    return str(obj.get("summary") or obj.get("guidance_summary") or "")


def _generate_graph_summary(
    *,
    raw_dir: Path,
    summary_generation_prompt: str,
    summary_node_catalog: List[Dict[str, Any]],
    nodespec_fields: List[str],
    model: str,
    refresh_raw: bool,
    rag_max_rounds: int,
    nodes_by_var: Dict[str, Dict[str, Any]],
    var_order: List[str],
    guidance_summary: str,
) -> Dict[str, Any]:
    """Generate a graph-level summary with iterative per-node field follow-ups."""

    def _children_in_nodes(node: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        vals = node.get("nodes")
        if isinstance(vals, list):
            for item in vals:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict) and isinstance(item.get("__var__"), str):
                    out.append(str(item["__var__"]))
        return sorted(set(x for x in out if x))

    def _children_in_tool_list(node: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        vals = node.get("tool_list")
        if isinstance(vals, list):
            for item in vals:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict) and isinstance(item.get("__var__"), str):
                    out.append(str(item["__var__"]))
        return sorted(set(x for x in out if x))

    def _parse_summary_vars_fields_request(obj: Dict[str, Any]) -> Dict[str, List[str]] | None:
        if not isinstance(obj, dict):
            return None
        req = obj.get("context_request")
        if not isinstance(req, dict):
            return None
        vars_fields = req.get("vars_fields")
        if not isinstance(vars_fields, dict):
            return None
        out: Dict[str, List[str]] = {}
        for var, fields in vars_fields.items():
            v = str(var or "").strip()
            if not v:
                continue
            if isinstance(fields, list):
                norm_fields = [str(f or "").strip() for f in fields if str(f or "").strip()]
            else:
                norm_fields = [str(fields or "").strip()] if str(fields or "").strip() else []
            if norm_fields:
                out[v] = norm_fields
        return out or None

    def _load_or_call_model_for_summary_round(
        *,
        payload: Dict[str, Any],
        payload_path: Path,
        raw_path: Path,
        model: str,
        refresh_raw: bool,
    ) -> Dict[str, Any]:
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        if raw_path.exists() and not refresh_raw:
            return parse_json_loose(raw_path.read_text(encoding="utf-8", errors="replace"))
        resp = call_model(payload, model)
        raw_path.write_text(resp or "", encoding="utf-8")
        return parse_json_loose(resp or "")

    def _node_fields_response_for_vars(
        *,
        vars_fields: Dict[str, List[str]],
        nodes_by_var: Dict[str, Dict[str, Any]],
        var_order: List[str],
        allowed_fields: Set[str],
    ) -> Dict[str, Any]:
        children_by_parent, parents_by_child = build_parent_child_maps(var_order, nodes_by_var)
        out: Dict[str, Any] = {"vars_fields_response": {}}
        for var, fields in vars_fields.items():
            node = nodes_by_var.get(var, {})
            row: Dict[str, Any] = {}
            for field in fields:
                f = str(field or "").strip()
                if not f or f not in allowed_fields:
                    continue
                if f == "parent_vars":
                    row[f] = sorted([p for p in (parents_by_child.get(var) or []) if isinstance(p, str)])
                elif f == "node_children":
                    row[f] = _children_in_nodes(node)
                elif f == "tool_children":
                    row[f] = _children_in_tool_list(node)
                elif f == "all_children":
                    row[f] = children_by_parent.get(var, [])
                else:
                    row[f] = json.loads(json.dumps(node.get(f), ensure_ascii=False, default=str))
            out["vars_fields_response"][var] = row
        return out

    def _run_nodespec_summary_loop(
        *,
        payload_base: Dict[str, Any],
        issue_dir: Path,
        stem: str,
        model: str,
        refresh_raw: bool,
        max_rounds: int,
        nodes_by_var: Dict[str, Dict[str, Any]],
        var_order: List[str],
        nodespec_fields: List[str],
    ) -> Dict[str, Any]:
        issue_dir.mkdir(parents=True, exist_ok=True)
        allowed_fields: Set[str] = set(nodespec_fields) | {"parent_vars", "node_children", "tool_children", "all_children"}
        history: List[Dict[str, Any]] = []
        retrieved_context: Dict[str, Any] | None = None
        last_obj: Dict[str, Any] = {}

        for i in range(1, max(1, int(max_rounds)) + 1):
            payload = dict(payload_base)
            payload["additional_context"] = {
                "retrieved_context": retrieved_context,
                "context_request_history": history,
            }
            obj = _load_or_call_model_for_summary_round(
                payload=payload,
                payload_path=issue_dir / f"{stem}.hop{i}.payload.json",
                raw_path=issue_dir / f"{stem}.hop{i}.txt",
                model=model,
                refresh_raw=refresh_raw,
            )
            if not isinstance(obj, dict):
                obj = {}
            last_obj = obj
            req = _parse_summary_vars_fields_request(obj)
            if not req:
                break
            response_obj = _node_fields_response_for_vars(
                vars_fields=req,
                nodes_by_var=nodes_by_var,
                var_order=var_order,
                allowed_fields=allowed_fields,
            )
            retrieved_context = response_obj
            history.append({"round": i, "requested": req, "response": response_obj})

        if isinstance(last_obj, dict):
            last_obj["_summary_context_history"] = history
        return last_obj

    try:
        from .graph_correctness.prompts import SYSTEM_SUMMARY_GENERATION_PROMPT
    except ImportError:
        from graph_correctness.prompts import SYSTEM_SUMMARY_GENERATION_PROMPT

    summary_dir = raw_dir / "summary_generation"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_decision = _run_nodespec_summary_loop(
        payload_base={
            "prompt": SYSTEM_SUMMARY_GENERATION_PROMPT,
            "summary_node_catalog": summary_node_catalog,
            "nodespec_class_fields": nodespec_fields,
        },
        issue_dir=summary_dir,
        stem="system_summary",
        model=model,
        refresh_raw=refresh_raw,
        max_rounds=rag_max_rounds,
        nodes_by_var=nodes_by_var,
        var_order=var_order,
        nodespec_fields=nodespec_fields,
    )
    inferred_summary = str(summary_decision.get("summary") or "").strip() or str(guidance_summary or "").strip()
    summary_history = (
        summary_decision.get("_summary_context_history")
        if isinstance(summary_decision.get("_summary_context_history"), list)
        else []
    )
    return {
        "summary": inferred_summary,
        "summary_history": summary_history,
        "summary_decision": summary_decision,
    }


def render_obj(value: Any, indent: int = 0) -> str:
    """Render nested Python data into source-code text used in emitted node artifact files."""
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
    """Render a node dictionary as a Python `NodeSpec` assignment statement."""
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
    """Render a Python triple-quoted multiline string with the requested indentation."""
    if "\n" not in text:
        return repr(text)
    pad = " " * indent
    parts = [json.dumps(line) for line in text.splitlines(True)]
    inner = "\n".join(f"{pad}    {part}" for part in parts)
    return "(\n" + inner + f"\n{pad})"


def safe_file_tag(text: str) -> str:
    """Convert arbitrary text into a filesystem-safe short tag for filenames."""
    tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", text or "").strip("._")
    return tag or "call"


def safe_name(value: str, max_len: int = 96) -> str:
    """Return a filesystem-safe label from arbitrary text, truncating long values with a hash suffix."""
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
    """Convert text into a simple underscore-separated slug."""
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
    """Convert a display name into a safe lowercase Python-style variable slug."""
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", text.strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "node"
    if s[0].isdigit():
        s = f"n_{s}"
    return s


def next_var(base_name: str, existing_vars: Set[str]) -> str:
    """Return a unique slug-style variable name for `base_name`, avoiding names already in `existing_vars`."""
    base = slug_var_name(base_name)
    if base not in existing_vars:
        return base
    i = 2
    while f"{base}_{i}" in existing_vars:
        i += 1
    return f"{base}_{i}"


def identity_key(name: Optional[str], file_path: Optional[str]) -> str:
    """Build a stable string key from a name and file path pair for persisted identity maps."""
    return f"{name or ''}||{file_path or ''}"


def load_identity_map(path: Path) -> Dict[str, str]:
    """Load the persisted node identity map from disk, or return an empty mapping."""
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
    """Persist the node identity map to disk as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_var_name(name: Optional[str], file_path: Optional[str], used: Set[str], mapping: Dict[str, str]) -> str:
    """Resolve a stable, collision-free Python variable name for a node and persist it in the identity map."""
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
    """Convert a stored line value into a sortable `(start, end)` tuple."""
    norm = _normalize_line_value(val)
    if isinstance(norm, list) and len(norm) == 2:
        return int(norm[0]), int(norm[1])
    if isinstance(norm, int):
        return norm, norm
    return 0, 0


def sort_code_references(refs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort code references into a stable order by kind, file, and line."""
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
    """Sort port dictionaries by their `name` field for deterministic output."""
    return sorted([port for port in ports if isinstance(port, dict)], key=lambda p: str(p.get("name") or ""))


def sort_var_refs(items: Iterable[Any]) -> List[Any]:
    """Sort variable-reference-like items by their variable or name field."""
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
    """Sort node dictionaries into a stable order by name, type, and description."""
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
    """Raise if any node is missing required fields or attributes."""
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
    """Return cached or fresh raw model text for a payload, optionally through the RAG loop."""
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
    """Return a parsed JSON object from cached or fresh model output, retrying on refusals or unusable results."""
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
    parsed = parse_json_loose(text)
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
        retry_obj = parse_json_loose(retry_text)
        last_text = retry_text
        last_obj = retry_obj if isinstance(retry_obj, dict) else {}
        if _is_usable_json_obj(last_obj) and not _looks_like_refusal(last_text):
            return last_obj, "model"

    return last_obj, "model"


def run_shared_graph_candidate_decision(
    *,
    candidate_var: str,
    candidate_node: Dict[str, Any],
    root_var: str | None,
    graph_catalog: List[Dict[str, Any]],
    nodes_by_var: Dict[str, Dict[str, Any]],
    decision_dir: Path,
    model: str,
    refresh_raw: bool,
    retriever: Any | None,
    rag_max_rounds: int,
    guidance_summary: str,
    representation_shortlist_prompt: str,
    representation_pair_prompt: str,
    missing_node_decision_prompt: str,
    adding_node_parent_decision_prompt: str,
    allowed_match_vars: Set[str] | None = None,
    allowed_parent_vars: Set[str] | None = None,
    fallback_parent_vars: List[str] | None = None,
    allow_parent_fallback: bool = False,
    filter_rep_list: bool = True,
) -> Dict[str, Any]:
    """Run shortlist, represented-check, missing-node, and parent-approval decisions for one candidate node."""
    valid_graph_vars = {
        entry.get("var")
        for entry in graph_catalog
        if isinstance(entry, dict) and isinstance(entry.get("var"), str)
    }
    match_scope = set(allowed_match_vars) if allowed_match_vars is not None else set(valid_graph_vars)
    parent_scope = set(allowed_parent_vars) if allowed_parent_vars is not None else set(valid_graph_vars)

    shortlist_payload = {
        "prompt": representation_shortlist_prompt,
        "candidate_var": candidate_var,
        "candidate_node": candidate_node,
        "graph_catalog": graph_catalog,
        "guidance_summary": guidance_summary,
        "retrieved_context": None,
    }
    short_parsed_raw, short_src = call_with_cache(
        payload=shortlist_payload,
        payload_path=decision_dir / "representation_shortlist.payload.json",
        raw_path=decision_dir / "representation_shortlist.txt",
        model=model,
        refresh_raw=refresh_raw,
        retriever=retriever,
        rag_max_rounds=rag_max_rounds,
    )
    short_parsed = short_parsed_raw if isinstance(short_parsed_raw, dict) else parse_json_loose("")
    shortlist_output = dict(short_parsed) if isinstance(short_parsed, dict) else {}
    plausible_matches = [
        value
        for value in (short_parsed.get("plausible_matches") if isinstance(short_parsed.get("plausible_matches"), list) else [])
        if isinstance(value, str)
        and value in nodes_by_var
        and value in valid_graph_vars
        and value in match_scope
        and (not filter_rep_list or value != candidate_var)
    ]

    represented_vars: List[str] = []
    represented_reasons: Dict[str, str] = {}
    pair_checks: List[Dict[str, Any]] = []
    pair_src = "skipped"
    for graph_var in plausible_matches:
        pair_payload = {
            "prompt": representation_pair_prompt,
            "candidate_var": candidate_var,
            "candidate_node": candidate_node,
            "graph_var": graph_var,
            "graph_node": nodes_by_var.get(graph_var, {}),
            "guidance_summary": guidance_summary,
            "retrieved_context": None,
        }
        safe_graph_var = slugify(graph_var) or "candidate"
        pair_parsed_raw, pair_src = call_with_cache(
            payload=pair_payload,
            payload_path=decision_dir / f"representation_pair_{safe_graph_var}.payload.json",
            raw_path=decision_dir / f"representation_pair_{safe_graph_var}.txt",
            model=model,
            refresh_raw=refresh_raw,
            retriever=retriever,
            rag_max_rounds=rag_max_rounds,
        )
        pair_parsed = pair_parsed_raw if isinstance(pair_parsed_raw, dict) else parse_json_loose("")
        is_represented = bool(pair_parsed.get("is_represented"))
        pair_reason = str(pair_parsed.get("reason") or "")
        pair_checks.append({"graph_var": graph_var, "is_represented": is_represented, "reason": pair_reason})
        if is_represented:
            represented_vars.append(graph_var)
            represented_reasons[graph_var] = pair_reason

    missing_decision_output: Dict[str, Any] | None = None
    need_to_attach = False
    suggested_parent: str | None = None
    reason = represented_reasons.get(represented_vars[0], "") if represented_vars else ""
    miss_dec_src = "skipped"
    parent_src = "skipped"
    parent_decisions: List[Dict[str, Any]] = []
    approved_parents: List[str] = []

    if not represented_vars:
        miss_payload = {
            "prompt": missing_node_decision_prompt,
            "candidate_var": candidate_var,
            "candidate_node": candidate_node,
            "root_var": root_var,
            "graph_catalog": graph_catalog,
            "guidance_summary": guidance_summary,
            "retrieved_context": None,
        }
        miss_parsed_raw, miss_dec_src = call_with_cache(
            payload=miss_payload,
            payload_path=decision_dir / "missing_decision.payload.json",
            raw_path=decision_dir / "missing_decision.txt",
            model=model,
            refresh_raw=refresh_raw,
            retriever=retriever,
            rag_max_rounds=rag_max_rounds,
        )
        miss_parsed = miss_parsed_raw if isinstance(miss_parsed_raw, dict) else parse_json_loose("")
        missing_decision_output = dict(miss_parsed) if isinstance(miss_parsed, dict) else {}
        need_to_attach = bool(miss_parsed.get("need_to_attach"))
        suggested_parent = miss_parsed.get("attach_parent_var") if isinstance(miss_parsed.get("attach_parent_var"), str) else None
        if suggested_parent and (suggested_parent not in valid_graph_vars or suggested_parent not in parent_scope):
            suggested_parent = None
        reason = str(miss_parsed.get("reason") or "")

        if need_to_attach:
            parent_candidates: List[str] = []
            if suggested_parent and suggested_parent in nodes_by_var and suggested_parent in parent_scope:
                parent_candidates.append(suggested_parent)
            if allow_parent_fallback:
                for parent_var in fallback_parent_vars or []:
                    if (
                        isinstance(parent_var, str)
                        and parent_var in nodes_by_var
                        and parent_var in valid_graph_vars
                        and parent_var in parent_scope
                        and (not filter_rep_list or parent_var != candidate_var)
                        and parent_var not in parent_candidates
                    ):
                        parent_candidates.append(parent_var)

            for parent_var in parent_candidates:
                parent_payload = {
                    "prompt": adding_node_parent_decision_prompt,
                    "candidate_var": candidate_var,
                    "candidate_node": candidate_node,
                    "root_var": root_var,
                    "graph_catalog": graph_catalog,
                    "parent_var": parent_var,
                    "parent_node": nodes_by_var.get(parent_var, {}),
                    "missing_decision_prior": {
                        "need_to_attach": bool(need_to_attach),
                        "attach_parent_var": suggested_parent,
                        "reason": reason,
                    },
                    "guidance_summary": guidance_summary,
                    "retrieved_context": None,
                }
                safe_parent = slugify(parent_var) or "parent"
                parent_parsed_raw, parent_src = call_with_cache(
                    payload=parent_payload,
                    payload_path=decision_dir / f"adding_parent_decision_{safe_parent}.payload.json",
                    raw_path=decision_dir / f"adding_parent_decision_{safe_parent}.txt",
                    model=model,
                    refresh_raw=refresh_raw,
                    retriever=retriever,
                    rag_max_rounds=rag_max_rounds,
                )
                parent_parsed = parent_parsed_raw if isinstance(parent_parsed_raw, dict) else parse_json_loose("")
                should_attach = bool(parent_parsed.get("attach_to_parent"))
                parent_decisions.append({
                    "parent_var": parent_var,
                    "attach_to_parent": should_attach,
                    "reason": str(parent_parsed.get("reason") or ""),
                })
                if should_attach:
                    approved_parents.append(parent_var)
                    if not allow_parent_fallback:
                        break

    return {
        "shortlist_output": shortlist_output,
        "plausible_matches": plausible_matches,
        "pair_checks": pair_checks,
        "represented_vars": represented_vars,
        "represented_reasons": represented_reasons,
        "represented_by_var": represented_vars[0] if represented_vars else None,
        "missing_decision_output": missing_decision_output,
        "need_to_attach": need_to_attach,
        "suggested_parent": suggested_parent,
        "reason": reason,
        "parent_decisions": parent_decisions,
        "approved_parents": approved_parents,
        "sources": {
            "representation_shortlist": short_src,
            "representation_pair": pair_src,
            "missing_decision": miss_dec_src,
            "adding_parent_decision": parent_src,
        },
    }


def run_subprocess(cmd: List[str], *, repo_root: Path) -> None:
    """Run one subprocess from the repository root with repo-local `PYTHONPATH` and raise on failure."""
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


__all__ = [
    "call_model",
    "run_task_with_rag",
    "cache_payload_matches",
    "parse_json_loose",
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
    "extract_context_request_needs"
]
