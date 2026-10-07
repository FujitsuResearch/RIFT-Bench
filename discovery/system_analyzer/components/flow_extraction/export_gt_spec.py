#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from pydantic import ValidationError

try:
    from .global_utils import extract_var_name, load_nodes_by_var
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parents[4]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from discovery.system_analyzer.components.flow_extraction.global_utils import (
        extract_var_name,
        load_nodes_by_var,
    )

SHAREABLE_RUNTIME_TYPES = {"Tool", "LLM", "Database", "Local_MCP_server", "External_MCP_server"}


def _node_type_name(node: Dict[str, Any]) -> str:
    nt = node.get("node_type")
    if isinstance(nt, dict):
        return str(nt.get("type") or "")
    if isinstance(nt, str):
        return nt
    return ""


def _children_map(var_order: List[str], nodes_by_var: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    children: Dict[str, Set[str]] = {v: set() for v in var_order}
    parents: Dict[str, Set[str]] = {v: set() for v in var_order}
    for pvar in var_order:
        pnode = nodes_by_var.get(pvar, {})
        for fld in ("nodes", "tool_list"):
            vals = pnode.get(fld)
            if not isinstance(vals, list):
                continue
            for x in vals:
                cvar = extract_var_name(x)
                if cvar and cvar in parents and cvar != pvar:
                    children[pvar].add(cvar)
                    parents[cvar].add(pvar)
    return ({k: sorted(v) for k, v in children.items()}, {k: sorted(v) for k, v in parents.items()})


def _pick_root_var(var_order: List[str], nodes_by_var: Dict[str, Dict[str, Any]]) -> str:
    children, parents = _children_map(var_order, nodes_by_var)

    def descendants_count(v: str) -> int:
        seen: Set[str] = set()
        stack = list(children.get(v, []))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(children.get(cur, []))
        return len(seen)

    roots = [v for v in var_order if len(parents.get(v, [])) == 0] or list(var_order)

    def score(v: str) -> Tuple[int, int, int]:
        n = nodes_by_var.get(v, {})
        nt = _node_type_name(n)
        kind = 0
        if nt == "System":
            kind = 4
        elif nt == "Agent":
            kind = 3
        elif nt in {"Local_MCP_server", "External_MCP_server"}:
            kind = 2
        elif nt == "LLM":
            kind = 1
        return (kind, 1 if bool(n.get("is_graph")) else 0, descendants_count(v))

    return max(roots, key=score) if roots else ""


def _norm(s: str | None) -> str:
    return "".join(ch for ch in str(s or "").lower() if ch.isalnum())


def _as_conn_list(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    conns = node.get("external_connections")
    if isinstance(conns, dict):
        conns = [conns]
    if not isinstance(conns, list):
        return []
    return [c for c in conns if isinstance(c, dict)]


def _split_multi_parent_shareables(var_order: List[str], nodes_by_var: Dict[str, Dict[str, Any]]) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    out_order = list(var_order)
    out_nodes = {k: deepcopy(v) for k, v in nodes_by_var.items()}

    parent_identity: Dict[str, Set[str]] = {}
    for pvar in out_order:
        pnode = out_nodes.get(pvar, {})
        pnm = str(pnode.get("name") or "")
        parent_identity[pvar] = {_norm(pvar), _norm(pnm)}

    split_map: Dict[str, Dict[str, str]] = {}  # child_var -> {parent_var: cloned_var}
    used_vars: Set[str] = set(out_order)

    for cvar in list(out_order):
        cnode = out_nodes.get(cvar, {})
        ctype = _node_type_name(cnode)
        if ctype not in SHAREABLE_RUNTIME_TYPES:
            continue
        conns = _as_conn_list(cnode)
        if not conns:
            continue

        parent_vars_for_conn: Set[str] = set()
        for c in conns:
            raw_parent = str(c.get("parent") or "").strip()
            if not raw_parent:
                continue
            nparent = _norm(raw_parent)
            for pvar, keys in parent_identity.items():
                if nparent in keys:
                    parent_vars_for_conn.add(pvar)
        if len(parent_vars_for_conn) <= 1:
            continue

        per_parent_clone: Dict[str, str] = {}
        for pvar in sorted(parent_vars_for_conn):
            base = f"{cvar}__for__{pvar}"
            new_var = base
            i = 2
            while new_var in used_vars:
                new_var = f"{base}_{i}"
                i += 1
            used_vars.add(new_var)
            cloned = deepcopy(cnode)
            pkeys = parent_identity.get(pvar, set())
            kept: List[Dict[str, Any]] = []
            for c in conns:
                cp = str(c.get("parent") or "").strip()
                if not cp:
                    continue
                if _norm(cp) in pkeys:
                    kept.append(deepcopy(c))
            cloned["external_connections"] = kept if kept else None
            out_nodes[new_var] = cloned
            out_order.append(new_var)
            per_parent_clone[pvar] = new_var

        split_map[cvar] = per_parent_clone

    # Rewire parent children refs to the per-parent clone.
    for pvar in list(out_order):
        pnode = out_nodes.get(pvar, {})
        for fld in ("nodes", "tool_list"):
            vals = pnode.get(fld)
            if not isinstance(vals, list):
                continue
            new_vals: List[Any] = []
            changed = False
            for x in vals:
                cvar = extract_var_name(x)
                if not cvar or cvar not in split_map:
                    new_vals.append(x)
                    continue
                target = split_map[cvar].get(pvar)
                if target:
                    new_vals.append(target)
                    changed = True
                else:
                    new_vals.append(x)
            if changed:
                pnode[fld] = new_vals

    # Drop original split vars if no longer referenced by any parent.
    referenced: Set[str] = set()
    for pvar in out_order:
        pnode = out_nodes.get(pvar, {})
        for fld in ("nodes", "tool_list"):
            vals = pnode.get(fld)
            if not isinstance(vals, list):
                continue
            for x in vals:
                cvar = extract_var_name(x)
                if cvar:
                    referenced.add(cvar)

    for orig_var in split_map.keys():
        if orig_var in referenced:
            continue
        out_nodes.pop(orig_var, None)
        if orig_var in out_order:
            out_order.remove(orig_var)

    return out_order, out_nodes


def _expand_tree(root_var: str, nodes_by_var: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    def _rewrite_ref(value: Any, rename_map: Dict[str, str]) -> Any:
        if isinstance(value, str):
            return rename_map.get(value, value)
        if isinstance(value, list):
            return [rename_map.get(x, x) if isinstance(x, str) else x for x in value]
        return value

    def _finalize_rename_map(raw_map: Dict[str, Set[str]]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for old, new_set in raw_map.items():
            if len(new_set) == 1:
                out[old] = next(iter(new_set))
        return out

    def walk(v: str, stack: Set[str]) -> Dict[str, Any]:
        if v in stack:
            return {}
        nxt = set(stack)
        nxt.add(v)
        node = deepcopy(nodes_by_var.get(v, {}))
        ntype = _node_type_name(node)
        metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        is_synth_controller = (
            ntype == "Deterministic_controller"
            and bool(metadata.get("synthetic_controller_for"))
        )
        if is_synth_controller:
            node["name"] = v
        local_rename_raw: Dict[str, Set[str]] = {}
        for fld in ("nodes", "tool_list"):
            vals = node.get(fld)
            out: List[Dict[str, Any]] = []
            if isinstance(vals, list):
                for x in vals:
                    cvar = extract_var_name(x)
                    if cvar and cvar in nodes_by_var:
                        src_child = nodes_by_var.get(cvar, {})
                        old_child_name = str(src_child.get("name") or "")
                        child = walk(cvar, nxt)
                        if isinstance(child, dict) and child:
                            new_child_name = str(child.get("name") or "")
                            if old_child_name and new_child_name and old_child_name != new_child_name:
                                local_rename_raw.setdefault(old_child_name, set()).add(new_child_name)
                            out.append(child)
            node[fld] = out

        rename_map = _finalize_rename_map(local_rename_raw)
        if rename_map:
            edges = node.get("internal_edges")
            if isinstance(edges, list):
                for e in edges:
                    if not isinstance(e, dict):
                        continue
                    if isinstance(e.get("from_"), str):
                        e["from_"] = _rewrite_ref(e.get("from_"), rename_map)
                    if isinstance(e.get("to"), str):
                        e["to"] = _rewrite_ref(e.get("to"), rename_map)

            for fld in ("nodes", "tool_list"):
                vals = node.get(fld)
                if not isinstance(vals, list):
                    continue
                for ch in vals:
                    if not isinstance(ch, dict):
                        continue
                    conns = ch.get("external_connections")
                    if isinstance(conns, dict):
                        conns = [conns]
                    if not isinstance(conns, list):
                        continue
                    for c in conns:
                        if not isinstance(c, dict):
                            continue
                        if isinstance(c.get("parent"), str):
                            c["parent"] = _rewrite_ref(c.get("parent"), rename_map)
                        c["in_"] = _rewrite_ref(c.get("in_"), rename_map)
                        c["out"] = _rewrite_ref(c.get("out"), rename_map)
        return node

    return walk(root_var, set())


def _get_parent_and_key(root: Any, loc: Tuple[Any, ...]) -> Tuple[Any, Any]:
    cur = root
    for part in loc[:-1]:
        if isinstance(part, int) and isinstance(cur, list) and 0 <= part < len(cur):
            cur = cur[part]
        elif isinstance(part, str) and isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None, None
    return cur, loc[-1] if loc else None


def _get_value_at(root: Any, loc: Tuple[Any, ...]) -> Any:
    cur = root
    for part in loc:
        if isinstance(part, int) and isinstance(cur, list) and 0 <= part < len(cur):
            cur = cur[part]
        elif isinstance(part, str) and isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _set_value_at(root: Any, loc: Tuple[Any, ...], value: Any) -> bool:
    parent, key = _get_parent_and_key(root, loc)
    if parent is None:
        return False
    if isinstance(key, int) and isinstance(parent, list) and 0 <= key < len(parent):
        parent[key] = value
        return True
    if isinstance(key, str) and isinstance(parent, dict):
        parent[key] = value
        return True
    return False


def _apply_other_fix(root: Dict[str, Any], err: Dict[str, Any]) -> bool:
    msg = str(err.get("msg") or "")
    loc = tuple(err.get("loc") or ())
    if "Input should be" not in msg or "'other'" not in msg or not loc:
        return False

    invalid = _get_value_at(root, loc)
    invalid_s = str(invalid) if invalid not in (None, "") else "unknown"

    # common enum field fixes
    if loc[-1] in {"framework", "type", "kind"}:
        if _set_value_at(root, loc, "other"):
            container_loc = loc[:-1]
            if loc[-1] == "kind":
                _set_value_at(root, container_loc + ("other_kind_description",), invalid_s)
            else:
                _set_value_at(root, container_loc + ("other_description",), invalid_s)
            return True

    # list/set element in `type` (e.g. agent_type.type[0], system_type.type[1])
    if len(loc) >= 2 and isinstance(loc[-1], int) and loc[-2] == "type":
        container_loc = loc[:-2]
        if _set_value_at(root, container_loc + ("type",), ["other"]):
            _set_value_at(root, container_loc + ("other_description",), invalid_s)
            return True
    return False


def _apply_tool_example_fix(root: Dict[str, Any], err: Dict[str, Any]) -> bool:
    msg = str(err.get("msg") or "")
    loc = tuple(err.get("loc") or ())
    if "Tool node" not in msg:
        return False
    if ("example.input" not in msg) and ("tool_example_pairs" not in msg):
        return False
    if not loc:
        return False
    node_obj = _get_value_at(root, loc)
    if isinstance(node_obj, dict):
        node_obj["tool_example_pairs"] = []
        return True
    parent, key = _get_parent_and_key(root, loc)
    if isinstance(parent, list) and isinstance(key, int) and 0 <= key < len(parent) and isinstance(parent[key], dict):
        parent[key]["tool_example_pairs"] = []
        return True
    return False


def _coerce_flow_ids_to_ints(root: Dict[str, Any]) -> bool:
    """
    Normalize flows[*].flow_id to int to satisfy FlowSpec.flow_id schema.

    Rules:
    - int stays int
    - digit-only strings -> int(value)
    - other strings -> stable deterministic int via md5 prefix
    - None/invalid -> deterministic fallback based on position
    """
    changed = False

    def _to_int(value: Any, fallback_key: str) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            s = value.strip()
            if s.isdigit():
                return int(s)
            # Stable deterministic 32-bit positive int from string.
            h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
            return int(h, 16)
        h = hashlib.md5(fallback_key.encode("utf-8")).hexdigest()[:8]
        return int(h, 16)

    def walk(node: Any, path: str = "root") -> None:
        nonlocal changed
        if isinstance(node, dict):
            flows = node.get("flows")
            if isinstance(flows, list):
                for i, fl in enumerate(flows):
                    if not isinstance(fl, dict):
                        continue
                    old = fl.get("flow_id")
                    new = _to_int(old, f"{path}.flows[{i}]")
                    if old != new:
                        fl["flow_id"] = new
                        changed = True
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, x in enumerate(node):
                walk(x, f"{path}[{i}]")

    walk(root)
    return changed


def _drop_empty_frameworks(root: Any) -> bool:
    changed = False
    if isinstance(root, dict):
        if root.get("framework") == {}:
            root["framework"] = None
            changed = True
        for v in root.values():
            changed = _drop_empty_frameworks(v) or changed
    elif isinstance(root, list):
        for x in root:
            changed = _drop_empty_frameworks(x) or changed
    return changed


def _validate_with_auto_fix(root_dict: Dict[str, Any]) -> Dict[str, Any]:
    from NodeSpec_schema import NodeSpecTop  # type: ignore

    work = deepcopy(root_dict)
    _drop_empty_frameworks(work)
    _coerce_flow_ids_to_ints(work)
    for _ in range(8):
        try:
            closed = NodeSpecTop.model_validate(work, context={"run_root_pass": True})
            return closed.model_dump(mode="python")
        except ValidationError as exc:
            changed = False
            for e in exc.errors():
                changed = _apply_tool_example_fix(work, e) or changed
                changed = _apply_other_fix(work, e) or changed
            changed = _drop_empty_frameworks(work) or changed
            changed = _coerce_flow_ids_to_ints(work) or changed
            if not changed:
                raise
    # final strict attempt
    closed = NodeSpecTop.model_validate(work, context={"run_root_pass": True})
    return closed.model_dump(mode="python")


def _load_project_root(use_case_out_dir: Path) -> str | None:
    root_record_path = use_case_out_dir / "root_record.json"
    if not root_record_path.exists():
        return None
    try:
        payload = json.loads(root_record_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    project_root = payload.get("project_root")
    if not isinstance(project_root, str) or not project_root.strip():
        return None
    return project_root.rstrip("/")


def _normalize_string_path_refs(value: str, project_root: str) -> str:
    prefix = f"{project_root}/"
    if prefix in value:
        return value.replace(prefix, "")
    return value


def _normalize_paths_in_obj(obj: Any, project_root: str) -> Any:
    if isinstance(obj, str):
        return _normalize_string_path_refs(obj, project_root)
    if isinstance(obj, list):
        return [_normalize_paths_in_obj(x, project_root) for x in obj]
    if isinstance(obj, dict):
        return {k: _normalize_paths_in_obj(v, project_root) for k, v in obj.items()}
    return obj


def _normalize_code_refs_and_flows_paths(root_dump: Dict[str, Any], project_root: str | None) -> Dict[str, Any]:
    if not project_root:
        return root_dump

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "code_references" in node:
                node["code_references"] = _normalize_paths_in_obj(node.get("code_references"), project_root)
            if "flows" in node:
                node["flows"] = _normalize_paths_in_obj(node.get("flows"), project_root)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for x in node:
                walk(x)

    walk(root_dump)
    return root_dump


def _render_value(value: Any, *, indent: int, key_hint: str | None = None) -> str:
    pad = " " * indent
    class_by_field = {
        "node_type": "NodeType",
        "required_keys": "RequiredKeys",
        "duplicates": "Duplication",
        "framework": "FrameworkType",
        "agent_type": "AgentType",
        "system_type": "SystemType",
        "llm_config": "LLMConfig",
        "entry_point_usage_example": "UsageExample",
    }
    list_class_by_field = {
        "inputs": "InputPort",
        "outputs": "OutputPort",
        "code_references": "CodeReference",
        "external_connections": "Connection",
        "internal_edges": "Edge",
        "tool_example_pairs": "ToolIOPair",
    }

    def _render_class_call(class_name: str, payload: Dict[str, Any], *, indent_inner: int) -> str:
        ip = " " * indent_inner
        parts: List[str] = []
        for k, v in payload.items():
            if k == "__call__":
                continue
            if v is None:
                continue
            parts.append(f"{k}={_render_value(v, indent=indent_inner + 4, key_hint=k)}")
        if not parts:
            return f"{class_name}()"
        if len(parts) == 1 and "\n" not in parts[0]:
            return f"{class_name}({parts[0]})"
        return f"{class_name}(\n{ip}    " + f",\n{ip}    ".join(parts) + f",\n{ip})"

    if isinstance(value, str):
        return repr(value)
    if value is None or isinstance(value, (int, float, bool)):
        return repr(value)
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        # For graph children, render nested NodeSpec constructors.
        if key_hint in {"nodes", "tool_list"}:
            parts = ",\n".join(f"{pad}    {_render_nodespec_obj(x, indent=indent + 4)}" for x in value)
        elif key_hint in list_class_by_field:
            cls = list_class_by_field[key_hint]
            parts = ",\n".join(
                f"{pad}    {_render_class_call(cls, x, indent_inner=indent + 4) if isinstance(x, dict) else _render_value(x, indent=indent + 4)}"
                for x in value
            )
        else:
            parts = ",\n".join(f"{pad}    {_render_value(x, indent=indent + 4)}" for x in value)
        return "[\n" + parts + f"\n{pad}]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        if "__call__" in value and isinstance(value.get("__call__"), str):
            return _render_class_call(str(value["__call__"]), value, indent_inner=indent)
        if key_hint in class_by_field:
            return _render_class_call(class_by_field[key_hint], value, indent_inner=indent)
        parts = []
        for k, v in value.items():
            if v is None:
                continue
            parts.append(f"{pad}    {repr(k)}: {_render_value(v, indent=indent + 4)}")
        if not parts:
            return "{}"
        return "{\n" + ",\n".join(parts) + f"\n{pad}" + "}"
    return repr(value)


def _render_nodespec_obj(node: Dict[str, Any], *, indent: int) -> str:
    if not isinstance(node, dict):
        return _render_value(node, indent=indent)

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
    pad = " " * indent
    lines: List[str] = ["NodeSpec("]
    for k in keys:
        v = node.get(k)
        if v is None:
            continue
        lines.append(f"{pad}    {k}={_render_value(v, indent=indent + 4, key_hint=k)},")
    lines.append(f"{pad})")
    return "\n".join(lines)


def _render_output_script(*, use_case_name: str, root_dump: Dict[str, Any]) -> str:
    json_name = f"{use_case_name}_spec.json"
    spec_txt = _render_nodespec_obj(root_dump, indent=0)
    return (
        "from __future__ import annotations\n\n"
        "from pathlib import Path\n\n"
        "try:\n"
        "    from NodeSpec_schema import (\n"
        "        AgentType,\n"
        "        CodeReference,\n"
        "        Connection,\n"
        "        Duplication,\n"
        "        Edge,\n"
        "        FrameworkType,\n"
        "        InputPort,\n"
        "        LLMConfig,\n"
        "        NodeSpec,\n"
        "        NodeSpecTop,\n"
        "        NodeType,\n"
        "        OutputPort,\n"
        "        RequiredKeys,\n"
        "        SystemType,\n"
        "        ToolIOPair,\n"
        "        UsageExample,\n"
        "    )\n"
        "except ModuleNotFoundError:\n"
        "    import sys\n"
        "    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / \"modular_imp\"))\n"
        "    from NodeSpec_schema import (\n"
        "        AgentType,\n"
        "        CodeReference,\n"
        "        Connection,\n"
        "        Duplication,\n"
        "        Edge,\n"
        "        FrameworkType,\n"
        "        InputPort,\n"
        "        LLMConfig,\n"
        "        NodeSpec,\n"
        "        NodeSpecTop,\n"
        "        NodeType,\n"
        "        OutputPort,\n"
        "        RequiredKeys,\n"
        "        SystemType,\n"
        "        ToolIOPair,\n"
        "        UsageExample,\n"
        "    )\n\n"
        f"agent_spec = {spec_txt}\n\n"
        "agent_spec = NodeSpecTop.model_validate(\n"
        "    agent_spec.model_dump(mode=\"python\"),\n"
        "    context={\"run_root_pass\": True},\n"
        ")\n\n"
        f"agent_spec.to_json(path=str(Path(__file__).with_name(\"{json_name}\")))\n"
        f"loaded_agent_spec = NodeSpec.from_json(path=str(Path(__file__).with_name(\"{json_name}\")))\n"
        "\n"
        "print(f\"\\n Tools: {loaded_agent_spec.list_tools()}\")\n"
        "print(f\"\\n Agents: {loaded_agent_spec.list_agents()}\")\n"
        "print(f\"\\n LLMs: {loaded_agent_spec.list_llms()}\")\n"
        "print(f\"\\n Systems: {loaded_agent_spec.list_systems()}\")\n"
        "print(f\"\\n MCP servers: {loaded_agent_spec.list_mcp_servers()}\")\n"
        "print(f\"\\n Structured Description: {loaded_agent_spec.get_desc()}\")\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Export GT NodeSpec script from a use_case output final_nodespec.py")
    ap.add_argument("--use_case_out_dir", required=True, help="Path containing final_nodespec.py")
    ap.add_argument("--use_case_name", required=True, help="Name used for output file: <use_case_name>_spec.py")
    ap.add_argument("--final_nodespec", default="final_nodespec.py", help="Final nodespec filename in use_case_out_dir")
    ap.add_argument("--gt_dir", default=None, help="Output gt_specs directory. Default: <repo_root>/gt_specs")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    use_case_out_dir = Path(args.use_case_out_dir).resolve()
    final_nodespec_path = (use_case_out_dir / Path(args.final_nodespec).name).resolve()
    if not final_nodespec_path.exists():
        raise FileNotFoundError(f"final_nodespec file not found: {final_nodespec_path}")

    gt_dir = Path(args.gt_dir).resolve() if args.gt_dir else (repo_root / "gt_specs")
    gt_dir.mkdir(parents=True, exist_ok=True)
    out_py = gt_dir / f"{args.use_case_name}_spec.py"

    var_order, nodes_by_var = load_nodes_by_var(final_nodespec_path)
    if not var_order or not nodes_by_var:
        raise RuntimeError(f"No nodes parsed from: {final_nodespec_path}")
    var_order, nodes_by_var = _split_multi_parent_shareables(var_order, nodes_by_var)
    root_var = _pick_root_var(var_order, nodes_by_var)
    if not root_var:
        raise RuntimeError("Could not resolve root var from final_nodespec.")
    root_dict = _expand_tree(root_var, nodes_by_var)
    project_root = _load_project_root(use_case_out_dir)

    sys.path.insert(0, str(repo_root / "modular_imp"))
    root_dump = _validate_with_auto_fix(root_dict)
    root_dump = _normalize_code_refs_and_flows_paths(root_dump, project_root)

    out_py.write_text(_render_output_script(use_case_name=args.use_case_name, root_dump=root_dump), encoding="utf-8")
    print(f"[export_gt_spec] wrote: {out_py}")


if __name__ == "__main__":
    main()
