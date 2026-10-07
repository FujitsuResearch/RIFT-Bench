from __future__ import annotations
import re
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple
from pydantic import ValidationError
from node_spec.structure_schema import NodeSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MAX_CONTENT_CHARS = 1000
GENERIC_AGENT_RUNTIME_NAMES_NORM = {
    "assistant",
    "agent",
    "ai",
    "model",
    "crewkickoff",
    "teamquery",
    "litellmcompletion",
}


def _norm_name_token(s: str) -> str:
    return "".join(ch for ch in str(s or "").lower() if ch.isalnum())


def _is_generic_agent_runtime_name(name: str) -> bool:
    return _norm_name_token(name) in GENERIC_AGENT_RUNTIME_NAMES_NORM


def _safe_name(value: Any, fallback: str = "unknown") -> str:
    s = str(value or "").strip()
    return s if s else fallback


def _norm_event_name(event_type: str, name: str) -> str:
    et = str(event_type or "").strip().lower()
    nm = str(name or "").strip()
    if et == "user":
        return "user:user"
    if et == "system":
        return "system:system"
    if et == "error":
        return "error:error"
    return f"{et}:{nm or 'unknown'}"


def _build_flow_signature(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    ordered: List[str] = []
    seen: set[str] = set()
    unique_components: List[str] = []

    def _append(tok: str) -> None:
        if not tok:
            return
        if not ordered or ordered[-1] != tok:
            ordered.append(tok)
        if tok not in seen:
            seen.add(tok)
            unique_components.append(tok)

    i = 0
    while i < len(events):
        e = events[i]
        if not isinstance(e, dict):
            i += 1
            continue
        et = str(e.get("type") or "").strip().lower()
        if et not in {"user", "system", "agent", "tool", "error"}:
            i += 1
            continue

        if et == "agent":
            tcs = e.get("tool_calls")
            tc_names: List[str] = []
            if isinstance(tcs, list):
                for tc in tcs:
                    if not isinstance(tc, dict):
                        continue
                    tc_names.append(_safe_name(tc.get("name"), "unknown"))
            agent_name = _safe_name(e.get("name"), "unknown")
            if tc_names:
                group = "+".join(sorted(tc_names, key=lambda x: x.lower()))
                _append(f"agent:{agent_name}.tool_call:{group}")
            else:
                _append(_norm_event_name("agent", agent_name))
            i += 1
            continue

        if et == "tool":
            names: List[str] = []
            while i < len(events):
                ee = events[i]
                if not isinstance(ee, dict) or str(ee.get("type") or "").strip().lower() != "tool":
                    break
                names.append(_safe_name(ee.get("name"), "unknown"))
                i += 1
            _append(f"tool:{'+'.join(sorted(names, key=lambda x: x.lower()))}")
            continue

        _append(_norm_event_name(et, _safe_name(e.get("name"), "unknown")))
        i += 1

    signature_key = "|".join(ordered)
    signature_id = hashlib.md5(signature_key.encode("utf-8")).hexdigest() if signature_key else "empty_flow"
    return {
        "flow_signature_ordered": ordered,
        "flow_signature_key": signature_key,
        "flow_signature_id": signature_id,
        "flow_unique_components": unique_components,
    }


def _truncate_content_text(value: Any, max_chars: int = MAX_CONTENT_CHARS) -> str:
    s = str(value or "")
    if len(s) <= max_chars:
        return s
    return s[:max_chars]


def _iter_dicts(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_dicts(v)


def _first_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                txt = str(item.get("text") or item.get("content") or "").strip()
                if txt:
                    parts.append(txt)
        if parts:
            return "\n".join(parts)
    if isinstance(value, dict):
        return str(value.get("text") or value.get("content") or "").strip()
    return ""


def _extract_user_query_from_raw_trace(trace_obj: Dict[str, Any]) -> str:
    for d in _iter_dicts(trace_obj):
        role = str(d.get("role") or d.get("type") or "").strip().lower()
        if role != "user":
            continue
        txt = _first_text(d.get("content"))
        if txt:
            return txt
        txt = _first_text(d.get("text"))
        if txt:
            return txt
    return ""


def _find_query_in_generated_manifest(flow_root: Path, trace_file_name: str) -> str:
    p = flow_root / "generated_trace_manifest.json"
    if not p.exists():
        return ""
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ""
    rows = obj.get("generated_trace_rows") if isinstance(obj, dict) else None
    if not isinstance(rows, list):
        return ""
    for row in rows:
        if not isinstance(row, dict):
            continue
        tp = str(row.get("trace_path") or "")
        if tp and Path(tp).name == trace_file_name:
            q = str(row.get("query") or "").strip()
            if q:
                return q
    return ""


def _find_query_in_all_queries_manifest(flow_root: Path, trace_file_name: str) -> str:
    p = flow_root / "all_queries_manifest.json"
    if not p.exists():
        return ""
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ""
    rows = obj.get("all_queries") if isinstance(obj, dict) else None
    if not isinstance(rows, list):
        return ""
    for row in rows:
        if not isinstance(row, dict):
            continue
        tp = str(row.get("existing_trace_path") or "")
        if tp and Path(tp).name == trace_file_name:
            q = str(row.get("query") or "").strip()
            if q:
                return q
    return ""


def _resolve_user_query(parsed_obj: Dict[str, Any], parsed_path: Path) -> str:
    trace_file_name = str(parsed_obj.get("trace_file") or "").strip()

    flow_root = parsed_path.parent.parent.parent
    if flow_root.name == "flow_query_generation":
        q = _find_query_in_generated_manifest(flow_root, trace_file_name)
        if q:
            return q
        q = _find_query_in_all_queries_manifest(flow_root, trace_file_name)
        if q:
            return q

    if trace_file_name:
        raw_trace = parsed_path.parent.parent / "traces" / trace_file_name
        if raw_trace.exists():
            try:
                raw_obj = json.loads(raw_trace.read_text(encoding="utf-8"))
                q = _extract_user_query_from_raw_trace(raw_obj)
                if q:
                    return q
            except Exception:
                pass

    events = parsed_obj.get("events") if isinstance(parsed_obj.get("events"), list) else []
    for e in events:
        if not isinstance(e, dict):
            continue
        if str(e.get("type") or "").strip().lower() == "user":
            c = str(e.get("content") or "").strip()
            if c:
                return c
    return ""


def _find_default_nodespec(parsed_path: Path) -> Optional[Path]:
    cur = parsed_path.resolve()
    for parent in [cur.parent] + list(cur.parents):
        m = re.search(r"use_case_(\d+)_outputs_\d+$", parent.name)
        if m:
            uc = m.group(1)
            spec_cand = Path("use_cases") / f"use_case_{uc}" / f"use_case_{uc}_spec.py"
            if spec_cand.exists():
                return spec_cand.resolve()
        spec_hits = sorted(parent.glob("*_spec.py"))
        if spec_hits:
            return spec_hits[0]
        for name in ("nodespec.py", "final_nodespec.py"):
            cand = parent / name
            if cand.exists():
                return cand
    return None


def _load_nodespec_schema_module(nodespec_path: Path):
    try:
        return importlib.import_module("modular_imp.NodeSpec_schema")
    except Exception as exc:
        raise RuntimeError(
            f"Failed importing modular_imp.NodeSpec_schema while loading {nodespec_path}: {exc}"
        ) from exc


def _load_nodespec_top(nodespec_path: Path):
    nodespec_path = nodespec_path.resolve()
    ns_mod = _load_nodespec_schema_module(nodespec_path)
    sys.modules["NodeSpec_schema"] = ns_mod
    parent = str(nodespec_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    mod_name = f"_flow_nodespec_runtime_{abs(hash(str(nodespec_path.resolve())))}"
    spec = importlib.util.spec_from_file_location(mod_name, str(nodespec_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading nodespec file: {nodespec_path}")
    mod = importlib.util.module_from_spec(spec)
    # Some generated final_nodespec.py files reference schema classes that are
    # not fully imported in the file header; preload all schema symbols.
    for k in dir(ns_mod):
        if not k.startswith("__") and k not in mod.__dict__:
            mod.__dict__[k] = getattr(ns_mod, k)
    spec.loader.exec_module(mod)

    node_spec_top = getattr(mod, "NodeSpecTop", None) or getattr(ns_mod, "NodeSpecTop", None)

    roots: List[Any] = []
    for attr in ("ALL_NODES", "MAIN_GRAPH_NODES"):
        arr = getattr(mod, attr, None)
        if isinstance(arr, list):
            for n in arr:
                if bool(getattr(n, "is_graph", False)):
                    roots.append(n)
    if not roots:
        for v in mod.__dict__.values():
            if bool(getattr(v, "is_graph", False)):
                roots.append(v)
    if not roots:
        raise RuntimeError(f"No graph root NodeSpec found in {nodespec_path}")

    def _root_score(n: Any) -> Tuple[int, int]:
        ntype = str(getattr(getattr(n, "node_type", None), "type", "") or "")
        kind = 0
        if ntype == "System":
            kind = 3
        elif ntype == "Agent":
            kind = 2
        elif ntype in {"Local_MCP_server", "External_MCP_server"}:
            kind = 1
        child_count = len(getattr(n, "nodes", None) or []) + len(getattr(n, "tool_list", None) or [])
        return (kind, child_count)

    root = max(roots, key=_root_score)
    if node_spec_top is not None and hasattr(root, "model_dump"):
        root_dump = root.model_dump()
        try:
            return node_spec_top.model_validate(root_dump, context={"run_root_pass": True})
        except ValidationError:
            # Mirror export_gt_spec.py fallback path for malformed connection refs.
            from discovery.system_analyzer.components.flow_extraction.global_utils import load_nodes_by_var
            from discovery.system_analyzer.components.flow_extraction.export_gt_spec import (
                _expand_tree,
                _pick_root_var,
                _split_multi_parent_shareables,
                _validate_with_auto_fix,
            )

            var_order, nodes_by_var = load_nodes_by_var(nodespec_path)
            if not var_order or not nodes_by_var:
                raise
            var_order, nodes_by_var = _split_multi_parent_shareables(var_order, nodes_by_var)
            root_var = _pick_root_var(var_order, nodes_by_var)
            if not root_var:
                raise
            rebuilt_root = _expand_tree(root_var, nodes_by_var)
            fixed_dump = _validate_with_auto_fix(rebuilt_root)
            return node_spec_top.model_validate(fixed_dump, context={"run_root_pass": True})
    else:
        # Mirror export_gt_spec.py fallback path for malformed connection refs.
        from discovery.system_analyzer.components.flow_extraction.global_utils import load_nodes_by_var
        from discovery.system_analyzer.components.flow_extraction.export_gt_spec import (
            _expand_tree,
            _pick_root_var,
            _split_multi_parent_shareables,
            _validate_with_auto_fix,
        )

        var_order, nodes_by_var = load_nodes_by_var(nodespec_path)
        if not var_order or not nodes_by_var:
            raise
        var_order, nodes_by_var = _split_multi_parent_shareables(var_order, nodes_by_var)
        root_var = _pick_root_var(var_order, nodes_by_var)
        if not root_var:
            raise
        rebuilt_root = _expand_tree(root_var, nodes_by_var)
        fixed_dump = _validate_with_auto_fix(rebuilt_root)
        if node_spec_top is None:
            raise RuntimeError("NodeSpecTop missing from loaded schema context")
        return node_spec_top.model_validate(fixed_dump, context={"run_root_pass": True})


def _parse_nodespec_index(nodespec_path: Path) -> Dict[str, Any]:
    try:
        top = _load_nodespec_top(nodespec_path)

        nodes_by_id: Dict[str, Dict[str, Any]] = {}
        agent_name_to_ids: Dict[str, List[str]] = {}
        tool_name_to_ids: Dict[str, List[str]] = {}
        tool_ids_by_parent_and_name: Dict[Tuple[str, str], List[str]] = {}

        def walk(node: Any, parent_id: str) -> None:
            nid = str(getattr(node, "id", "") or "")
            name = str(getattr(node, "name", "") or "")
            ntype = str(getattr(getattr(node, "node_type", None), "type", "") or "")
            if not nid or not name:
                return

            inputs_obj = getattr(node, "inputs", None) or []
            input_names: List[str] = []
            for ip in inputs_obj:
                nm = str(getattr(ip, "name", "") or "").strip()
                if nm:
                    input_names.append(nm)

            nodes_by_id[nid] = {
                "id": nid,
                "name": name,
                "type": ntype,
                "parent_id": parent_id,
                "inputs": input_names,
                "entry_point_usage_example": getattr(node, "entry_point_usage_example", None),
            }
            if ntype == "Agent":
                agent_name_to_ids.setdefault(name, []).append(nid)
            if ntype == "Tool":
                tool_name_to_ids.setdefault(name, []).append(nid)
                tool_ids_by_parent_and_name.setdefault((parent_id, name), []).append(nid)

            for ch in (getattr(node, "nodes", None) or []):
                walk(ch, nid)
            for ch in (getattr(node, "tool_list", None) or []):
                walk(ch, nid)

        walk(top, "")

        root_agent_ids: List[str] = []
        for nid, nd in nodes_by_id.items():
            if str(nd.get("type") or "") == "Agent" and (
                str(nd.get("parent_id") or "") == "" or str(nd.get("parent_id") or "") == str(getattr(top, "id", ""))
            ):
                root_agent_ids.append(nid)

        return {
            "nodes_by_id": nodes_by_id,
            "agent_name_to_ids": agent_name_to_ids,
            "tool_name_to_ids": tool_name_to_ids,
            "tool_ids_by_parent_and_name": tool_ids_by_parent_and_name,
            "root_agent_ids": root_agent_ids,
            "top_id": str(getattr(top, "id", "") or ""),
            "top_entry_point_usage_example": getattr(top, "entry_point_usage_example", None),
        }
    except Exception:
        # Tolerant fallback for legacy nodespec variants where strict NodeSpec
        # validation/import fails (e.g., string child refs in `nodes`).
        from discovery.system_analyzer.components.flow_extraction.global_utils import load_nodes_by_var

        var_order, var_nodes = load_nodes_by_var(nodespec_path)
        if not var_order:
            return {
                "nodes_by_id": {},
                "agent_name_to_ids": {},
                "tool_name_to_ids": {},
                "tool_ids_by_parent_and_name": {},
                "root_agent_ids": [],
                "top_id": "",
                "top_entry_point_usage_example": None,
            }

        all_vars = set(var_order)
        child_parent_var: Dict[str, str] = {}
        referenced_children: set[str] = set()

        for pvar in var_order:
            pnode = var_nodes.get(pvar) or {}
            for fld in ("nodes", "tool_list"):
                vals = pnode.get(fld)
                if not isinstance(vals, list):
                    continue
                for ch in vals:
                    if isinstance(ch, str) and ch in all_vars:
                        referenced_children.add(ch)
                        if ch not in child_parent_var:
                            child_parent_var[ch] = pvar

        top_candidates = [v for v in var_order if v not in referenced_children]
        if not top_candidates:
            top_candidates = list(var_order)

        def _type_of(v: str) -> str:
            n = var_nodes.get(v) or {}
            nt = n.get("node_type") if isinstance(n.get("node_type"), dict) else {}
            return str(nt.get("type") or "")

        def _is_graph(v: str) -> bool:
            n = var_nodes.get(v) or {}
            return bool(n.get("is_graph", False))

        graph_tops = [v for v in top_candidates if _is_graph(v)]
        if graph_tops:
            top_candidates = graph_tops
        preferred = ["System", "Agent", "Local_MCP_server", "External_MCP_server"]
        top_var = top_candidates[0]
        for t in preferred:
            hit = [v for v in top_candidates if _type_of(v) == t]
            if hit:
                top_var = hit[0]
                break

        var_to_id: Dict[str, str] = {}
        nodes_by_id: Dict[str, Dict[str, Any]] = {}
        for v in var_order:
            n = var_nodes.get(v) or {}
            nid = str(n.get("id") or "").strip() or v
            var_to_id[v] = nid

        agent_name_to_ids: Dict[str, List[str]] = {}
        tool_name_to_ids: Dict[str, List[str]] = {}
        tool_ids_by_parent_and_name: Dict[Tuple[str, str], List[str]] = {}

        for v in var_order:
            n = var_nodes.get(v) or {}
            name = str(n.get("name") or "").strip()
            if not name:
                continue
            nid = var_to_id.get(v, v)
            pvar = child_parent_var.get(v, "")
            parent_id = var_to_id.get(pvar, "") if pvar else ""
            ntype = _type_of(v)

            inputs_obj = n.get("inputs") if isinstance(n.get("inputs"), list) else []
            input_names: List[str] = []
            for ip in inputs_obj:
                if isinstance(ip, dict):
                    nm = str(ip.get("name") or "").strip()
                else:
                    nm = str(getattr(ip, "name", "") or "").strip()
                if nm:
                    input_names.append(nm)

            nodes_by_id[nid] = {
                "id": nid,
                "name": name,
                "type": ntype,
                "parent_id": parent_id,
                "inputs": input_names,
                "entry_point_usage_example": n.get("entry_point_usage_example"),
            }
            if ntype == "Agent":
                agent_name_to_ids.setdefault(name, []).append(nid)
            if ntype == "Tool":
                tool_name_to_ids.setdefault(name, []).append(nid)
                tool_ids_by_parent_and_name.setdefault((parent_id, name), []).append(nid)

        top_id = var_to_id.get(top_var, top_var)
        root_agent_ids: List[str] = []
        for nid, nd in nodes_by_id.items():
            if str(nd.get("type") or "") == "Agent" and (
                str(nd.get("parent_id") or "") == "" or str(nd.get("parent_id") or "") == top_id
            ):
                root_agent_ids.append(nid)

        top_node = var_nodes.get(top_var) or {}
        return {
            "nodes_by_id": nodes_by_id,
            "agent_name_to_ids": agent_name_to_ids,
            "tool_name_to_ids": tool_name_to_ids,
            "tool_ids_by_parent_and_name": tool_ids_by_parent_and_name,
            "root_agent_ids": root_agent_ids,
            "top_id": top_id,
            "top_entry_point_usage_example": top_node.get("entry_point_usage_example"),
        }


def _parse_nodespec_index_from_object(node_spec: NodeSpec) -> Dict[str, Any]:
    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    agent_name_to_ids: Dict[str, List[str]] = {}
    tool_name_to_ids: Dict[str, List[str]] = {}
    tool_ids_by_parent_and_name: Dict[Tuple[str, str], List[str]] = {}

    def walk(node: Any, parent_id: str) -> None:
        nid = str(getattr(node, "id", "") or "")
        name = str(getattr(node, "name", "") or "")
        ntype = str(getattr(getattr(node, "node_type", None), "type", "") or "")
        if not nid or not name:
            return

        inputs_obj = getattr(node, "inputs", None) or []
        input_names: List[str] = []
        for ip in inputs_obj:
            nm = str(getattr(ip, "name", "") or "").strip()
            if nm:
                input_names.append(nm)

        nodes_by_id[nid] = {
            "id": nid,
            "name": name,
            "type": ntype,
            "parent_id": parent_id,
            "inputs": input_names,
            "entry_point_usage_example": getattr(node, "entry_point_usage_example", None),
        }
        if ntype == "Agent":
            agent_name_to_ids.setdefault(name, []).append(nid)
        if ntype == "Tool":
            tool_name_to_ids.setdefault(name, []).append(nid)
            tool_ids_by_parent_and_name.setdefault((parent_id, name), []).append(nid)

        for ch in (getattr(node, "nodes", None) or []):
            walk(ch, nid)
        for ch in (getattr(node, "tool_list", None) or []):
            walk(ch, nid)

    walk(node_spec, "")

    top_id = str(getattr(node_spec, "id", "") or "")
    root_agent_ids: List[str] = []
    for nid, nd in nodes_by_id.items():
        if str(nd.get("type") or "") == "Agent" and (
            str(nd.get("parent_id") or "") == "" or str(nd.get("parent_id") or "") == top_id
        ):
            root_agent_ids.append(nid)

    return {
        "nodes_by_id": nodes_by_id,
        "agent_name_to_ids": agent_name_to_ids,
        "tool_name_to_ids": tool_name_to_ids,
        "tool_ids_by_parent_and_name": tool_ids_by_parent_and_name,
        "root_agent_ids": root_agent_ids,
        "top_id": top_id,
        "top_entry_point_usage_example": getattr(node_spec, "entry_point_usage_example", None),
    }


def _resolve_tool_id(
    tool_name: str,
    parent_agent_id: str,
    nodespec_index: Dict[str, Any],
) -> str:
    tname = str(tool_name or "").strip()
    pid = str(parent_agent_id or "").strip()
    if not tname:
        return ""
    exact = nodespec_index.get("tool_ids_by_parent_and_name", {}).get((pid, tname)) or []
    if exact:
        return str(exact[0])
    by_name = nodespec_index.get("tool_name_to_ids", {}).get(tname) or []
    if by_name:
        return str(by_name[0])
    return ""


def _resolve_agent_id(agent_name: str, nodespec_index: Dict[str, Any]) -> str:
    name = str(agent_name or "").strip()
    if not name:
        return ""
    cands = nodespec_index.get("agent_name_to_ids", {}).get(name) or []
    if cands:
        return str(cands[0])
    return ""


def _coerce_example_value(raw: Any, declared_type: str) -> Any:
    t = str(declared_type or "").strip().lower()
    if raw is None:
        return None
    s = str(raw).strip()
    if t in {"null", "none"}:
        return None
    if t in {"int", "integer"}:
        try:
            return int(s)
        except Exception:
            return raw
    if t in {"float", "number"}:
        try:
            return float(s)
        except Exception:
            return raw
    if t in {"bool", "boolean"}:
        if s.lower() in {"true", "1", "yes", "y"}:
            return True
        if s.lower() in {"false", "0", "no", "n"}:
            return False
        return raw
    return raw


def _load_query_from_query_file(parsed_path: Path) -> str:
    p = parsed_path.resolve()
    name = p.name
    cands: List[Path] = []
    if name.endswith(".llm_span_pipeline.json"):
        cands.append(p.with_name(name.replace(".llm_span_pipeline.json", ".prompt.txt")))
    if name.startswith("parsed_") and name.endswith(".json"):
        raw = name[len("parsed_") : -len(".json")]
        cands.append(p.with_name(raw + ".prompt.txt"))
    stem = p.stem
    if stem.startswith("parsed_"):
        stem = stem[len("parsed_") :]
    cands.append(p.with_name(stem + ".prompt.txt"))
    seen: set[str] = set()
    for c in cands:
        cs = str(c)
        if cs in seen:
            continue
        seen.add(cs)
        if c.exists():
            try:
                return c.read_text(encoding="utf-8").strip()
            except Exception:
                pass
    return ""


def _build_input_args(query: str, agent_id: str, nodespec_index: Dict[str, Any], parsed_path: Path) -> Dict[str, Any]:
    node = nodespec_index.get("nodes_by_id", {}).get(agent_id) or {}
    input_names = node.get("inputs") if isinstance(node.get("inputs"), list) else []
    out: Dict[str, Any] = {k: None for k in input_names if isinstance(k, str) and k}

    query_from_file = _load_query_from_query_file(parsed_path)
    effective_query = query_from_file or query

    usage = node.get("entry_point_usage_example")
    if usage is None:
        usage = nodespec_index.get("top_entry_point_usage_example")
    args = None
    if usage is not None:
        if isinstance(usage, dict):
            args = usage.get("arguments")
        else:
            args = getattr(usage, "arguments", None)
    if isinstance(args, list) and args:
        out = {}
        for a in args:
            if isinstance(a, dict):
                aname = str(a.get("name") or "").strip()
                declared_type = str(a.get("type") or "")
                example_val = a.get("example")
                is_task_input = bool(a.get("is_task_input", False))
            else:
                aname = str(getattr(a, "name", "") or "").strip()
                declared_type = str(getattr(a, "type", "") or "")
                example_val = getattr(a, "example", None)
                is_task_input = bool(getattr(a, "is_task_input", False))
            if not aname:
                continue
            key = aname[2:] if aname.startswith("--") else aname
            if is_task_input:
                out[key] = effective_query
            else:
                out[key] = _coerce_example_value(example_val, declared_type)

        # Safety fallback: if task input flag missing, still map common query args.
        if effective_query:
            for qk in ("task", "query", "input"):
                if qk in out and (out.get(qk) in (None, "", "<query_text>")):
                    out[qk] = effective_query

    if "input" in out:
        out["input"] = effective_query
    if "messages" in out:
        out["messages"] = [{"role": "user", "content": effective_query}] if effective_query else []
    if "agent" in out and out["agent"] is None:
        out["agent"] = {}
    if "config" in out and out["config"] is None:
        out["config"] = {}

    return out


def _append_unique_ordered(values: List[str], item: str) -> None:
    if item and item not in values:
        values.append(item)


def build_simple_flow(
    parsed_obj: Dict[str, Any],
    parsed_path: Path,
    nodespec_path: Optional[Path],
    node_spec: NodeSpec | None = None,
) -> Dict[str, Any]:
    events = parsed_obj.get("events") if isinstance(parsed_obj.get("events"), list) else []
    query = _resolve_user_query(parsed_obj, parsed_path)

    nodespec_index: Dict[str, Any] = {
        "nodes_by_id": {},
        "agent_name_to_id": {},
        "tool_name_to_candidates": {},
    }
    if node_spec is not None:
        nodespec_index = _parse_nodespec_index_from_object(node_spec)
    elif nodespec_path and nodespec_path.exists():
        nodespec_index = _parse_nodespec_index(nodespec_path)

    flow_events: List[Dict[str, Any]] = []
    invoked_tools: List[str] = []
    invoked_agents: List[str] = []

    pending_tool_by_call_id: Dict[str, Tuple[str, str]] = {}
    pending_tool_fifo_by_name: Dict[str, List[Tuple[str, str]]] = {}

    first_agent_id = ""
    last_agent_id = ""
    default_agent_id = ""
    roots = nodespec_index.get("root_agent_ids") if isinstance(nodespec_index.get("root_agent_ids"), list) else []
    if roots:
        default_agent_id = str(roots[0] or "")

    for e in events:
        if not isinstance(e, dict):
            continue
        et = str(e.get("type") or "").strip()
        name = str(e.get("name") or "").strip()
        seq = int(e.get("seq") or 0)
        tcid = str(e.get("tool_call_id") or "").strip()

        node_id = ""
        parent_agent_id = ""

        if et.lower() == "agent":
            node_id = _resolve_agent_id(name, nodespec_index)
            if (not node_id) and _is_generic_agent_runtime_name(name):
                node_id = last_agent_id or default_agent_id
            if node_id and not first_agent_id:
                first_agent_id = node_id
            if node_id:
                last_agent_id = node_id
        elif et.lower() == "system":
            # System prompts are typically attached to a concrete agent.
            node_id = _resolve_agent_id(name, nodespec_index)
            if (not node_id) and _is_generic_agent_runtime_name(name):
                node_id = last_agent_id or first_agent_id or default_agent_id

        tool_calls = e.get("tool_calls") if isinstance(e.get("tool_calls"), list) else []
        content = _truncate_content_text(e.get("content"))

        if et.lower() == "agent" and tool_calls:
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_name = str(tc.get("name") or "").strip()
                tc_id = str(tc.get("id") or "").strip()
                resolved_tool_id = _resolve_tool_id(tc_name, node_id, nodespec_index)
                parent_for_call = node_id
                if tc_id:
                    pending_tool_by_call_id[tc_id] = (resolved_tool_id, parent_for_call)
                pending_tool_fifo_by_name.setdefault(tc_name, []).append((resolved_tool_id, parent_for_call))

        if et.lower() == "tool":
            if tcid and tcid in pending_tool_by_call_id:
                node_id, parent_agent_id = pending_tool_by_call_id.pop(tcid)
            else:
                queue = pending_tool_fifo_by_name.get(name) or []
                if queue:
                    node_id, parent_agent_id = queue.pop(0)
                else:
                    node_id = _resolve_tool_id(name, "", nodespec_index)

        if et.lower() == "user":
            node_id = ""
        if et.lower() == "error" and not node_id:
            node_id = first_agent_id or default_agent_id

        out_name = name
        if _is_generic_agent_runtime_name(name):
            if et.lower() == "system":
                out_name = "system"
            elif et.lower() == "agent":
                resolved = str((nodespec_index.get("nodes_by_id", {}).get(node_id) or {}).get("name") or "").strip()
                out_name = resolved or "assistant"

        row: Dict[str, Any] = {
            "seq": seq,
            "node_id": node_id,
            "type": et,
            "name": out_name,
        }
        if tool_calls:
            normalized_tool_calls: List[Dict[str, Any]] = []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_name = str(tc.get("name") or "").strip()
                tc_row = dict(tc)
                # Trace-level call IDs are intentionally omitted in flow output.
                tc_row.pop("id", None)
                tc_row["tool_id"] = _resolve_tool_id(tc_name, node_id, nodespec_index)
                normalized_tool_calls.append(tc_row)
            row["tool_calls"] = normalized_tool_calls
        else:
            row["content"] = content

        flow_events.append(row)

        if et.lower() == "agent":
            _append_unique_ordered(invoked_agents, node_id)
            for tc in tool_calls:
                if isinstance(tc, dict):
                    tc_name = str(tc.get("name") or "").strip()
                    _append_unique_ordered(invoked_tools, _resolve_tool_id(tc_name, node_id, nodespec_index))
        elif et.lower() == "tool":
            _append_unique_ordered(invoked_tools, node_id)

    signature = _build_flow_signature(flow_events)
    chosen_agent_id = first_agent_id or default_agent_id
    input_args = _build_input_args(query=query, agent_id=chosen_agent_id, nodespec_index=nodespec_index, parsed_path=parsed_path)

    return {
        "flow_id": 0,
        "flow_signature_ordered": signature["flow_signature_ordered"],
        "flow_signature_key": signature["flow_signature_key"],
        "flow_signature_id": signature["flow_signature_id"],
        "flow_unique_components": signature["flow_unique_components"],
        "parsed_trace_file": str(parsed_path),
        "source_trace_file": str(parsed_obj.get("trace_file") or ""),
        "nodespec_file": str(nodespec_path) if nodespec_path else "",
        "input_args": input_args,
        "events": flow_events,
        "invoked_tools": invoked_tools,
        "invoked_agents": invoked_agents,
    }


def _default_output_path(parsed_trace_file: Path, parsed_obj: Dict[str, Any]) -> Path:
    source_trace = str(parsed_obj.get("trace_file") or "").strip()
    source_stem = Path(source_trace).stem if source_trace else parsed_trace_file.stem
    return parsed_trace_file.parent / f"flow_{source_stem}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed_trace", required=True, help="Path to parsed_trace_*.json")
    ap.add_argument("--out", default="", help="Optional output path. Default: sibling flow_<trace_stem>.json")
    ap.add_argument("--nodespec_file", default="", help="Optional nodespec/spec file path override.")
    args = ap.parse_args()

    parsed_path = Path(args.parsed_trace).resolve()
    if not parsed_path.exists():
        raise FileNotFoundError(f"Parsed trace not found: {parsed_path}")

    parsed_obj = json.loads(parsed_path.read_text(encoding="utf-8"))

    nodespec_path: Optional[Path]
    if str(args.nodespec_file).strip():
        nodespec_path = Path(args.nodespec_file).resolve()
    else:
        nodespec_path = _find_default_nodespec(parsed_path)

    flow_obj = build_simple_flow(parsed_obj=parsed_obj, parsed_path=parsed_path, nodespec_path=nodespec_path)

    out_path = Path(args.out).resolve() if str(args.out).strip() else _default_output_path(parsed_path, parsed_obj)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(flow_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[save_simple_flow] wrote {out_path}")


if __name__ == "__main__":
    main()
