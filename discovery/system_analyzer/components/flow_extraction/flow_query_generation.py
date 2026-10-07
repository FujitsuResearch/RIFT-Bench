from __future__ import annotations

import hashlib
import json
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Tuple, Optional

try:
    from .induce_runtime_mapping import induce_runtime_mapping
    from .prompts import (
        PROMPT_FLOW_QUERY_GENERATION,
        PROMPT_UNSEEN_FLOW_PROPOSAL,
    )
    from .trace_parsing import parse_trace_file, save_parsed_outputs
    from .model_client import call_model
    from . import trace_extraction
except ImportError:
    from .induce_runtime_mapping import induce_runtime_mapping
    from .prompts import (
        PROMPT_FLOW_QUERY_GENERATION,
        PROMPT_UNSEEN_FLOW_PROPOSAL,
    )
    from .trace_parsing import parse_trace_file, save_parsed_outputs
    from .model_client import call_model
    from . import trace_extraction


DEFAULT_KNOWN_QUERIES_PER_FLOW = 5
DEFAULT_UNSEEN_BATCH_SIZE = 5
DEFAULT_MAX_EXEC_STEPS = 1
DEFAULT_MAX_ATTEMPTS_PER_TARGET = 5
DEFAULT_NO_NEW_TARGET_ROUNDS_STOP = 4
DEFAULT_NO_NEW_FLOW_STEPS_STOP = 16
DEFAULT_DISCOVERY_MIN_STEPS = 16


def _json_loads_maybe(v: Any) -> Any:
    if isinstance(v, (dict, list)):
        return v
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _json_loose(text: str) -> Any:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    i = raw.find("{")
    j = raw.rfind("}")
    if i >= 0 and j > i:
        mid = raw[i : j + 1]
        try:
            return json.loads(mid)
        except Exception:
            return None
    return None


def _safe_name(s: Any, fallback: str = "unknown") -> str:
    t = str(s or "").strip()
    return t if t else fallback


def _norm_query_text(q: str) -> str:
    return " ".join(str(q or "").strip().lower().split())


def _safe_file_token(s: str, fallback: str = "call") -> str:
    t = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(s or "").strip()).strip("._")
    if not t:
        t = fallback
    # Keep filenames bounded.
    return t[:120]


def _llm_call_group(call_name: str) -> str:
    c = str(call_name or "").strip().lower()
    if c.startswith("unseen_flow_proposal"):
        return "unseen_flow_proposal"
    if c.startswith("target_query_"):
        return "target_query"
    if c.startswith("unseen_flow_queries_"):
        return "unseen_flow_queries"
    if c.startswith("known_flow_"):
        return "known_flow_queries"
    return "misc"


def _norm_event_name(event_type: str, name: str) -> str:
    et = (event_type or "").strip().lower()
    nm = (name or "").strip()
    if et == "user":
        return "user:user"
    if et == "system":
        return "system:system"
    if et == "error":
        return "error:error"
    return f"{et}:{nm or 'unknown'}"


def _extract_seed_query(trace_obj: Dict[str, Any]) -> str:
    spans = trace_obj.get("data", {}).get("spans")
    if not isinstance(spans, list):
        spans = trace_obj.get("spans")
    if not isinstance(spans, list):
        spans = []

    root = None
    for s in spans:
        if isinstance(s, dict) and s.get("parent_span_id") is None:
            root = s
            break

    candidates: List[str] = []

    if isinstance(root, dict):
        attrs = root.get("attributes") if isinstance(root.get("attributes"), dict) else {}
        sin = _json_loads_maybe(attrs.get("mlflow.spanInputs"))
        if isinstance(sin, dict):
            msg_list = sin.get("messages")
            if isinstance(msg_list, list):
                for m in msg_list:
                    if not isinstance(m, dict):
                        continue
                    role = str(m.get("role") or m.get("type") or "").lower()
                    if role in {"user", "human"}:
                        c = str(m.get("content") or "").strip()
                        if c:
                            candidates.append(c)
            c = str(sin.get("input") or "").strip()
            if c:
                candidates.append(c)

    info = trace_obj.get("info") if isinstance(trace_obj.get("info"), dict) else {}
    rp = str(info.get("request_preview") or "").strip()
    if rp and not rp.startswith("{"):
        candidates.append(rp)
    tip = _json_loads_maybe(info.get("trace_metadata", {}).get("mlflow.traceInputs") if isinstance(info.get("trace_metadata"), dict) else None)
    if isinstance(tip, dict):
        c = str(tip.get("input") or "").strip()
        if c:
            candidates.append(c)
        msgs = tip.get("messages")
        if isinstance(msgs, list):
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role") or m.get("type") or "").lower()
                if role in {"user", "human"}:
                    c = str(m.get("content") or "").strip()
                    if c:
                        candidates.append(c)

    for c in candidates:
        if c:
            return c
    return ""


def _load_seed_queries_from_validation_runs(outputs_dir: Path) -> Dict[str, str]:
    """
    Returns mapping: trace_file_name -> query_text
    using run_*.trace_path.txt + run_*.prompt.txt pairs.
    """
    out: Dict[str, str] = {}
    run_root = outputs_dir / "validation" / "execution_validation_runs"
    if not run_root.exists():
        run_root = outputs_dir / "execution_validation" / "execution_validation_runs"
    if not run_root.exists():
        return out
    for trace_path_file in sorted(run_root.glob("**/run_*.trace_path.txt")):
        prompt_file = trace_path_file.with_name(
            trace_path_file.name.replace(".trace_path.txt", ".prompt.txt")
        )
        if not prompt_file.exists():
            continue
        trace_path_text = trace_path_file.read_text(encoding="utf-8", errors="replace").strip()
        prompt_text = prompt_file.read_text(encoding="utf-8", errors="replace").strip()
        if not trace_path_text or not prompt_text:
            continue
        trace_name = Path(trace_path_text).name
        if not trace_name:
            continue
        # first write wins to keep deterministic behavior by sorted run order
        if trace_name not in out:
            out[trace_name] = prompt_text
    return out


def _load_validation_query_trace_links(outputs_dir: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    trace_to_query: Dict[str, str] = {}
    query_to_trace_path: Dict[str, str] = {}
    run_root = outputs_dir / "validation" / "execution_validation_runs"
    if not run_root.exists():
        run_root = outputs_dir / "execution_validation" / "execution_validation_runs"
    if not run_root.exists():
        return trace_to_query, query_to_trace_path
    for trace_path_file in sorted(run_root.glob("**/run_*.trace_path.txt")):
        prompt_file = trace_path_file.with_name(
            trace_path_file.name.replace(".trace_path.txt", ".prompt.txt")
        )
        if not prompt_file.exists():
            continue
        trace_path_text = trace_path_file.read_text(encoding="utf-8", errors="replace").strip()
        prompt_text = prompt_file.read_text(encoding="utf-8", errors="replace").strip()
        if not trace_path_text or not prompt_text:
            continue
        trace_name = Path(trace_path_text).name
        if trace_name and trace_name not in trace_to_query:
            trace_to_query[trace_name] = prompt_text
        qn = _norm_query_text(prompt_text)
        if qn and qn not in query_to_trace_path:
            query_to_trace_path[qn] = trace_path_text
    return trace_to_query, query_to_trace_path


def _write_flows_direct_list(
    *,
    out_dir: Path,
    known_queries_rows: List[Dict[str, Any]],
    recycled_from_unseen: List[Dict[str, Any]],
    generated_trace_rows: List[Dict[str, Any]],
) -> Path:
    known_sig_by_id: Dict[str, List[str]] = {}
    for row in known_queries_rows:
        if not isinstance(row, dict):
            continue
        fid = str(row.get("flow_id") or "").strip()
        sig = row.get("ordered_signature")
        if fid and isinstance(sig, list):
            known_sig_by_id[fid] = [str(x) for x in sig]

    lines: List[str] = []
    lines.append("=== KNOWN FLOWS (flow_id + signature + queries) ===")
    for i, row in enumerate(known_queries_rows, 1):
        if not isinstance(row, dict):
            continue
        fid = str(row.get("flow_id") or f"flow_{i}")
        sig = row.get("ordered_signature") if isinstance(row.get("ordered_signature"), list) else []
        queries = row.get("queries") if isinstance(row.get("queries"), list) else []
        lines.append(f"[{i}] flow_id={fid}  queries={len(queries)}")
        lines.append("  signature: " + (" -> ".join(str(x) for x in sig) if sig else "<none>"))
        for j, q in enumerate(queries, 1):
            lines.append(f"  {j}. {str(q)}")
        lines.append("")

    lines.append("=== NEWLY ADDED / RECYCLED (source flow name + landed_flow_id + signature + queries) ===")
    grouped_new: Dict[Tuple[str, str], List[str]] = {}
    for row in recycled_from_unseen:
        if not isinstance(row, dict):
            continue
        landed_id = str(row.get("landed_existing_flow_id") or "UNKNOWN")
        source_name = str(row.get("source_target_flow_name") or "")
        query = str(row.get("query") or "").strip()
        if not query:
            continue
        key = (landed_id, source_name)
        grouped_new.setdefault(key, []).append(query)

    for i, ((landed_id, source_name), queries) in enumerate(
        sorted(grouped_new.items(), key=lambda kv: (kv[0][0], kv[0][1])),
        1,
    ):
        landed_sig = known_sig_by_id.get(landed_id, [])
        lines.append(
            f"[{i}] source_target_flow_name={source_name} | landed_flow_id={landed_id} | queries={len(queries)}"
        )
        lines.append("  landed_signature: " + (" -> ".join(landed_sig) if landed_sig else "<unknown>"))
        for j, q in enumerate(queries, 1):
            lines.append(f"  {j}. {q}")
        lines.append("")

    lines.append("=== EXECUTED TRACE ROWS (trace_id + landed_flow_id + signatures + query) ===")
    for i, row in enumerate(generated_trace_rows, 1):
        if not isinstance(row, dict):
            continue
        query = str(row.get("query") or "").strip()
        if not query:
            continue
        trace_id = Path(str(row.get("trace_path") or "")).stem or f"row_{i}"
        landed_id = str(row.get("landed_flow_id") or "UNKNOWN")
        target_name = str(row.get("target_flow_name") or "")
        target_sig = row.get("target_flow_signature") if isinstance(row.get("target_flow_signature"), list) else []
        landed_sig = row.get("landed_signature") if isinstance(row.get("landed_signature"), list) else known_sig_by_id.get(landed_id, [])
        lines.append(f"[{i}] trace_id={trace_id} | target_flow_name={target_name} | landed_flow_id={landed_id}")
        lines.append("  target_signature: " + (" -> ".join(str(x) for x in target_sig) if target_sig else "<none>"))
        lines.append("  landed_signature: " + (" -> ".join(str(x) for x in landed_sig) if landed_sig else "<none>"))
        lines.append(f"  query: {query}")
        lines.append("")

    out_path = out_dir / "FLOWS_DIRECT_LIST.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _flow_signature(events: List[Dict[str, Any]]) -> Tuple[str, List[str], List[str]]:
    ordered: List[str] = []
    seen_set: set[str] = set()
    unique_components: List[str] = []

    def _append(tok: str) -> None:
        if not tok:
            return
        if not ordered or ordered[-1] != tok:
            ordered.append(tok)
        if tok not in seen_set:
            seen_set.add(tok)
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
            agent_name = _safe_name(e.get("name"), "unknown")
            if isinstance(tcs, list):
                for tc in tcs:
                    if not isinstance(tc, dict):
                        continue
                    tname = _safe_name(tc.get("name"), "unknown")
                    tc_names.append(tname)
            # Agent turn is exclusive: either tool-call turn OR message turn.
            if tc_names:
                # Order-insensitive tool-call group for a single agent turn.
                group = "+".join(sorted(tc_names, key=lambda x: x.lower()))
                _append(f"agent:{agent_name}.tool_call:{group}")
            else:
                _append(_norm_event_name("agent", agent_name))
            i += 1
            continue
        if et == "tool":
            # Adjacent tool responses are a single unordered group.
            names: List[str] = []
            while i < len(events):
                ee = events[i]
                if not isinstance(ee, dict) or str(ee.get("type") or "").strip().lower() != "tool":
                    break
                names.append(_safe_name(ee.get("name"), "unknown"))
                i += 1
            group = "+".join(sorted(names, key=lambda x: x.lower()))
            _append(f"tool:{group}")
            continue
        else:
            _append(_norm_event_name(et, _safe_name(e.get("name"), "unknown")))
            i += 1

    sig = "|".join(ordered)
    fid = hashlib.md5(sig.encode("utf-8")).hexdigest() if sig else "empty_flow"
    return fid, ordered, unique_components


def _sig_key(sig: List[str]) -> str:
    return "|".join([str(x).strip() for x in sig if str(x).strip()])


def _extract_error_content(events: List[Dict[str, Any]] | Any) -> str:
    if not isinstance(events, list):
        return ""
    for e in events:
        if not isinstance(e, dict):
            continue
        et = str(e.get("type") or "").strip().lower()
        if et != "error":
            continue
        for k in ("content", "error", "value", "message"):
            v = e.get(k)
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
    return ""


_SIG_TYPE_RE = re.compile(r"^(user|system|agent|tool|error)([:.])(.*)$", re.IGNORECASE)
_AGENT_TOOL_CALL_RE = re.compile(r"^agent:(?P<agent>.+?)\.tool_call:(?P<tools>.+)$", re.IGNORECASE)


def _normalize_signature_token(tok: Any) -> str:
    s = str(tok or "").strip()
    if not s:
        return ""
    low = s.lower()
    m_atc = _AGENT_TOOL_CALL_RE.match(s)
    if m_atc:
        agent = str(m_atc.group("agent") or "").strip() or "unknown"
        tools_body = str(m_atc.group("tools") or "").strip()
        parts = [p.strip() for p in tools_body.split("+") if p.strip()]
        parts = sorted(parts, key=lambda x: x.lower())
        return f"agent:{agent}.tool_call:{'+'.join(parts) if parts else 'unknown'}"
    if low.startswith("agent.tool_call.") and ":" not in s:
        body = s.split(".", 2)[-1].strip()
        parts = [p.strip() for p in body.split("+") if p.strip()]
        parts = sorted(parts, key=lambda x: x.lower())
        return "agent.tool_call:" + "+".join(parts)
    if low.startswith("agent.tool_call:"):
        body = s.split(":", 1)[-1].strip()
        parts = [p.strip() for p in body.split("+") if p.strip()]
        parts = sorted(parts, key=lambda x: x.lower())
        return "agent.tool_call:" + "+".join(parts)
    m = _SIG_TYPE_RE.match(s)
    if not m:
        return s
    et = m.group(1).lower()
    rest = (m.group(3) or "").strip()
    if et == "user":
        return "user:user"
    if et == "system":
        return "system:system"
    if et == "error":
        return "error:error"
    if et == "tool":
        parts = [p.strip() for p in rest.split("+") if p.strip()]
        parts = sorted(parts, key=lambda x: x.lower())
        return f"tool:{'+'.join(parts) if parts else 'unknown'}"
    return f"{et}:{rest or 'unknown'}"


def _is_valid_signature(sig: List[str]) -> bool:
    if not sig:
        return False
    pending_tool_results = 0
    prev_agent_turn_name = ""
    for i, tok in enumerate(sig):
        t = _normalize_signature_token(tok)
        if not t:
            return False
        # Disallow consecutive turns by the same agent.
        if t.startswith("agent:"):
            cur_agent = t.split(":", 1)[-1].split(".tool_call:", 1)[0].strip().lower()
            if prev_agent_turn_name and cur_agent and cur_agent == prev_agent_turn_name:
                return False
            prev_agent_turn_name = cur_agent
        else:
            prev_agent_turn_name = ""
        if t.startswith("agent.tool_call:") or ".tool_call:" in t:
            if i == 0:
                return False
            if t.startswith("agent.tool_call:"):
                body = t.split(":", 1)[-1].strip()
            else:
                body = t.split(".tool_call:", 1)[-1].strip()
            cnt = len([p for p in body.split("+") if p.strip()]) if body else 0
            pending_tool_results += max(1, cnt)
            continue
        if t.startswith("tool:"):
            body = t.split(":", 1)[-1].strip()
            cnt = len([p for p in body.split("+") if p.strip()]) if body else 0
            consume = max(1, cnt)
            if pending_tool_results < consume:
                return False
            pending_tool_results -= consume
            continue
        if pending_tool_results > 0:
            return False
    return pending_tool_results == 0


def _normalize_signature(sig: Any) -> List[str]:
    if not isinstance(sig, list):
        return []
    out: List[str] = []
    for raw in sig:
        t = _normalize_signature_token(raw)
        if not t:
            continue
        if out and out[-1] == t:
            continue
        # Cleanup pattern: agent turn immediately followed by same-agent tool_call.
        # Example:
        #   agent:agent -> agent:agent.tool_call:remember_tool
        # becomes:
        #   agent:agent.tool_call:remember_tool
        if out and t.startswith("agent:") and ".tool_call:" in t and out[-1].startswith("agent:") and ".tool_call:" not in out[-1]:
            prev_agent = out[-1].split(":", 1)[-1].strip().lower()
            cur_agent = t.split(":", 1)[-1].split(".tool_call:", 1)[0].strip().lower()
            if prev_agent and cur_agent and prev_agent == cur_agent:
                out[-1] = t
                continue
        out.append(t)
    return out


def _allowed_tool_names_from_structure(agentic_system_structure: Dict[str, Any]) -> set[str]:
    out: set[str] = set()
    if not isinstance(agentic_system_structure, dict):
        return out

    def _collect(node: Any) -> None:
        if not isinstance(node, dict):
            return
        ntype = str(node.get("type") or "").strip()
        name = str(node.get("name") or "").strip()
        if ntype == "Tool" and name:
            out.add(name)

    _collect(agentic_system_structure.get("top_level"))
    agn = agentic_system_structure.get("all_graph_nodes")
    if isinstance(agn, list):
        for n in agn:
            _collect(n)
    return out


def _tool_names_in_token(tok: str) -> List[str]:
    t = _normalize_signature_token(tok)
    if not t:
        return []
    if t.startswith("tool:"):
        body = t.split(":", 1)[-1].strip()
    elif t.startswith("agent.tool_call:"):
        body = t.split(":", 1)[-1].strip()
    elif t.startswith("agent:") and ".tool_call:" in t:
        body = t.split(".tool_call:", 1)[-1].strip()
    else:
        return []
    return [p.strip() for p in body.split("+") if p.strip()]


def _signature_uses_only_allowed_tools(sig: List[str], allowed_tools: set[str]) -> bool:
    if not allowed_tools:
        return True
    for tok in sig:
        names = _tool_names_in_token(tok)
        if not names:
            continue
        if any(n not in allowed_tools for n in names):
            return False
    return True


def _nodespec_summary_ids(node: Any) -> List[str]:
    ids = set()
    node_id = str(getattr(node, "id", "") or "").strip()
    if node_id:
        ids.add(node_id)
    duplicates = getattr(node, "duplicates", None)
    instances = getattr(duplicates, "instances", None) if duplicates is not None else None
    if isinstance(instances, list):
        for item in instances:
            if not isinstance(item, dict):
                continue
            iid = str(item.get("id") or "").strip()
            if iid:
                ids.add(iid)
    if not ids:
        ids.add(str(getattr(node, "name", "") or "").strip() or "unknown")
    return sorted(ids)


def _nodespec_hierarchy_summary_from_object(node_spec: Any) -> Dict[str, Any]:
    if node_spec is None:
        return {}

    def _block(node: Any) -> Dict[str, Any]:
        node_ids = _nodespec_summary_ids(node)
        children = []
        iter_children = getattr(node, "iter_children", None)
        for child in iter_children() if callable(iter_children) else []:
            child_ids = _nodespec_summary_ids(child)
            children.append(
                {
                    "name": str(getattr(child, "name", "") or "").strip() or "unknown",
                    "id": child_ids[0] if child_ids else "",
                }
            )
        return {
            "name": str(getattr(node, "name", "") or "").strip(),
            "type": str(getattr(getattr(node, "node_type", None), "type", "") or "").strip(),
            "description": str(getattr(node, "description", "") or "").strip(),
            "ids": node_ids,
            "children": children,
        }

    iter_descendants = getattr(node_spec, "iter_descendants", None)
    descendants = list(iter_descendants(include_self=True)) if callable(iter_descendants) else [node_spec]
    return {
        "top_level": _block(node_spec),
        "all_graph_nodes": [_block(node) for node in descendants[1:]],
    }


def _build_system_context(
    nodespec_file: Path | None,
    flow_catalog: Dict[str, Any],
    *,
    node_spec: Any = None,
) -> Dict[str, Any]:
    context: Dict[str, Any] = {"flow_count": len(flow_catalog.get("flows", []))}
    if node_spec is not None:
        try:
            hierarchy = _nodespec_hierarchy_summary_from_object(node_spec)
            if isinstance(hierarchy, dict) and hierarchy:
                context["hierarchical_structure"] = hierarchy
        except Exception as exc:
            context["hierarchical_structure_error"] = f"{exc.__class__.__name__}: {exc}"
    elif nodespec_file and nodespec_file.exists():
        context["nodespec_file"] = str(nodespec_file)
        try:
            from node_spec.structure_schema import NodeSpec

            hierarchy = NodeSpec.hierarchy_summary_from_nodespec_py(str(nodespec_file))
            if isinstance(hierarchy, dict) and hierarchy:
                context["hierarchical_structure"] = hierarchy
        except Exception as exc:
            context["hierarchical_structure_error"] = f"{exc.__class__.__name__}: {exc}"
    components: set[str] = set()
    for f in flow_catalog.get("flows", []):
        for tok in f.get("unique_components", []):
            components.add(str(tok))
    context["known_components"] = sorted(components)
    return context


def _llm_json(payload: Dict[str, Any], model: str) -> Dict[str, Any]:
    txt = call_model(payload, model=model)
    obj = _json_loose(txt)
    return obj if isinstance(obj, dict) else {}


def _llm_json_logged(
    *,
    payload: Dict[str, Any],
    model: str,
    raw_dir: Path,
    call_name: str,
    counter: List[int],
) -> Dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    counter[0] += 1
    idx = counter[0]
    token = _safe_file_token(call_name, fallback=f"call_{idx:04d}")
    group_dir = raw_dir / _llm_call_group(call_name)
    group_dir.mkdir(parents=True, exist_ok=True)
    payload_path = group_dir / f"{idx:04d}.{token}.payload.json"
    raw_path = group_dir / f"{idx:04d}.{token}.raw.txt"
    payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    txt = call_model(payload, model=model)
    raw_path.write_text(str(txt), encoding="utf-8")
    obj = _json_loose(txt)
    return obj if isinstance(obj, dict) else {}


def _generate_queries_for_flow(
    *,
    model: str,
    system_context: Dict[str, Any],
    requested_flow: Dict[str, Any] | List[str],
    n: int,
    example_queries: List[str] | None = None,
    previous_attempts: List[Dict[str, Any]] | None = None,
    call_name: str = "flow_query_generation",
    llm_raw_dir: Path | None = None,
    llm_counter: List[int] | None = None,
) -> List[Dict[str, Any]]:
    failed_attempts: List[Dict[str, Any]] = []
    for a in (previous_attempts or []):
        if not isinstance(a, dict):
            continue
        miss = str(a.get("miss_reason") or "").strip()
        has_error = bool(str(a.get("error") or "").strip())
        if miss or has_error:
            failed_attempts.append(a)

    payload = {
        "instruction": PROMPT_FLOW_QUERY_GENERATION,
        "agentic_system_structure": system_context.get("hierarchical_structure") if isinstance(system_context.get("hierarchical_structure"), dict) else system_context,
        "requested_flow": requested_flow,
        "example_queries": example_queries or [],
        "previous_attempts": failed_attempts,
        "required_count": n,
        "output_contract": {"queries": ["string"], "is_info_request": False},
    }
    if llm_raw_dir is not None and llm_counter is not None:
        obj = _llm_json_logged(
            payload=payload,
            model=model,
            raw_dir=llm_raw_dir,
            call_name=call_name,
            counter=llm_counter,
        )
    else:
        obj = _llm_json(payload, model=model)
    arr = obj.get("queries") if isinstance(obj.get("queries"), list) else []
    batch_info_flag = bool(obj.get("is_info_request")) if isinstance(obj, dict) else False
    out: List[Dict[str, Any]] = []
    seen = set()
    for q in arr:
        qq = ""
        is_info = batch_info_flag
        if isinstance(q, dict):
            qq = str(q.get("query") or "").strip()
            # Backward compatibility if model still returns per-item objects.
            is_info = bool(q.get("is_info_request"))
        else:
            qq = str(q or "").strip()
        if qq and qq not in seen:
            seen.add(qq)
            out.append({"query": qq, "is_info_request": is_info})
        if len(out) >= n:
            break
    seeds = example_queries or []
    while len(out) < n and seeds:
        out.append({"query": str(seeds[len(out) % len(seeds)]), "is_info_request": False})
    return out[:n]


def _count_attempts_toward_limit(attempts: List[Dict[str, Any]]) -> int:
    c = 0
    for a in attempts:
        if not isinstance(a, dict):
            continue
        if bool(a.get("is_info_request")):
            continue
        c += 1
    return c


def _propose_unseen_flows(
    *,
    model: str,
    system_context: Dict[str, Any],
    observed_flows: List[Dict[str, Any]],
    n_new_flows: int,
    rejected_examples: List[Dict[str, Any]] | None = None,
    llm_raw_dir: Path | None = None,
    llm_counter: List[int] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    agentic_system_structure = (
        system_context.get("hierarchical_structure")
        if isinstance(system_context.get("hierarchical_structure"), dict)
        else system_context
    )
    allowed_tools = _allowed_tool_names_from_structure(agentic_system_structure if isinstance(agentic_system_structure, dict) else {})
    payload = {
        "instruction": PROMPT_UNSEEN_FLOW_PROPOSAL,
        "agentic_system_structure": agentic_system_structure,
        "observed_flow_signatures": [f.get("ordered_signature", []) for f in observed_flows],
        "filtered_out_examples": (rejected_examples or [])[-40:],
        "required_count": n_new_flows,
        "output_contract": {
            "new_flows": [
                {
                    "target_flow_name": "string",
                    "ordered_signature": ["type:name"],
                    "rationale": "string",
                }
            ]
        },
    }
    if llm_raw_dir is not None and llm_counter is not None:
        obj = _llm_json_logged(
            payload=payload,
            model=model,
            raw_dir=llm_raw_dir,
            call_name="unseen_flow_proposal",
            counter=llm_counter,
        )
    else:
        obj = _llm_json(payload, model=model)
    rows = obj.get("new_flows") if isinstance(obj.get("new_flows"), list) else []
    out: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seen_sig = {
        "|".join(_normalize_signature(f.get("ordered_signature", [])))
        for f in observed_flows
        if _normalize_signature(f.get("ordered_signature", []))
    }
    reject_counts: Dict[str, int] = {}
    max_per_reason = 30

    def _maybe_reject(reason: str, row: Dict[str, Any] | None, sig_norm: List[str]) -> None:
        cnt = reject_counts.get(reason, 0)
        if cnt >= max_per_reason:
            return
        reject_counts[reason] = cnt + 1
        rr = row or {}
        rejected.append(
            {
                "target_flow_name": str(rr.get("target_flow_name") or f"rejected_{len(rejected)+1}"),
                "ordered_signature": sig_norm,
                "rationale": str(rr.get("rationale") or "").strip(),
                "reason": reason,
            }
        )

    for r in rows:
        if not isinstance(r, dict):
            _maybe_reject("invalid_row_type", None, [])
            continue
        sig_in = r.get("ordered_signature")
        if not isinstance(sig_in, list):
            # Backward-compat for earlier prompt schema wording.
            sig_in = r.get("events")
        sig_norm = _normalize_signature(sig_in)
        if not sig_norm:
            _maybe_reject("empty_or_unparseable_signature", r, [])
            continue
        # if not _is_valid_signature(sig_norm):
        #     _maybe_reject("invalid_signature_turn_semantics", r, sig_norm)
        #     continue
        if not _signature_uses_only_allowed_tools(sig_norm, allowed_tools):
            _maybe_reject("uses_unknown_or_disallowed_tool", r, sig_norm)
            continue
        sig_key = "|".join(sig_norm)
        if sig_key in seen_sig:
            _maybe_reject("duplicate_of_observed_or_already_accepted_signature", r, sig_norm)
            continue
        seen_sig.add(sig_key)
        out.append(
            {
                "target_flow_name": str(r.get("target_flow_name") or f"new_flow_{len(out)+1}"),
                "ordered_signature": sig_norm,
                "rationale": str(r.get("rationale") or "").strip(),
            }
        )
        if len(out) >= n_new_flows:
            break
    return out, rejected


def _find_nodespec(outputs_dir: Path) -> Path | None:
    m = re.search(r"use_case_(\d+)_outputs_\d+$", outputs_dir.name)
    if m:
        uc = m.group(1)
        spec_cand = Path("use_cases") / f"use_case_{uc}" / f"use_case_{uc}_spec.py"
        if spec_cand.exists():
            return spec_cand.resolve()
    spec_hits = sorted(outputs_dir.glob("*_spec.py"))
    if spec_hits:
        return spec_hits[0].resolve()
    for name in ("nodespec.py", "final_nodespec.py"):
        cand = outputs_dir / name
        if cand.exists():
            return cand
    return None


def _infer_use_case_number(outputs_dir: Path) -> int:
    checks = [outputs_dir.name, outputs_dir.as_posix()]
    for c in checks:
        m = re.search(r"use_case_(\d+)_outputs_", c)
        if m:
            return int(m.group(1))
    raise ValueError(f"Could not infer use case number from outputs_dir={outputs_dir}")


def _infer_zipfile_from_outputs_dir(outputs_dir: Path) -> Path | None:
    checks = [outputs_dir.name, outputs_dir.as_posix()]
    for c in checks:
        m = re.search(r"(.*)_outputs_\d+$", c)
        if not m:
            continue
        base = str(m.group(1) or "").strip().split("/")[-1]
        if not base:
            continue
        cand = Path("use_cases") / f"{base}.zip"
        if cand.exists():
            return cand.resolve()
    return None


def _safe_system_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "_", value.lower()).strip("_.-") or "system"


def _load_execution_command_example(outputs_dir: Path) -> Dict[str, Any] | None:
    p = (outputs_dir / "execution_command_example.json").resolve()
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _task_arg_names(exec_example: Dict[str, Any] | None) -> set[str]:
    if not isinstance(exec_example, dict):
        return {"--task", "--query", "--prompt"}
    raw = exec_example.get("task_arg_names")
    names: set[str] = set()
    if isinstance(raw, list):
        for x in raw:
            s = str(x or "").strip()
            if s:
                names.add(s)
    return names or {"--task", "--query", "--prompt"}


def _entrypoint_value(exec_example: Dict[str, Any] | None) -> str:
    if not isinstance(exec_example, dict):
        return ""
    ep = exec_example.get("entrypoint")
    if isinstance(ep, str):
        return ep.strip()
    if isinstance(ep, dict):
        return str(ep.get("value") or "").strip()
    return ""


def _command_template_from_exec_example(exec_example: Dict[str, Any] | None) -> str:
    if not isinstance(exec_example, dict):
        return trace_extraction.DEFAULT_COMMAND_TEMPLATE
    runner = str(exec_example.get("runner") or "python").strip() or "python"
    entrypoint = _entrypoint_value(exec_example)
    if not entrypoint:
        return trace_extraction.DEFAULT_COMMAND_TEMPLATE
    if not entrypoint.startswith("/"):
        entrypoint = f"/sandbox/{entrypoint.lstrip('./')}"
    task_names = _task_arg_names(exec_example)
    args = exec_example.get("args") if isinstance(exec_example.get("args"), list) else []
    task_arg = ""
    extra_parts: List[str] = []
    for item in args:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        if not task_arg and name in task_names:
            task_arg = name
            continue
        val = item.get("value")
        if val is None:
            extra_parts.append(shlex.quote(name))
        else:
            extra_parts.append(f"{shlex.quote(name)} {shlex.quote(str(val))}")
    if not task_arg:
        task_arg = "--task"
    parts = [shlex.quote(runner), shlex.quote(entrypoint)] + extra_parts + [shlex.quote(task_arg), "{query}"]
    return " ".join(p for p in parts if p)


@dataclass
class FlowQueryGenerationResult:
    out_dir: Path
    flow_catalog_file: Path
    known_queries_file: Path
    unseen_targets_file: Path
    unseen_queries_file: Path
    flow_counts_summary_file: Path


def _events_from_llm_span_pipeline_file(path: Path) -> List[Dict[str, Any]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []
    if not isinstance(obj, dict):
        return []
    for k in ("canonical_events", "events", "parsed_events"):
        ev = obj.get(k)
        if isinstance(ev, list):
            return [e for e in ev if isinstance(e, dict)]
    for outer in ("result", "data", "output"):
        box = obj.get(outer)
        if not isinstance(box, dict):
            continue
        for k in ("canonical_events", "events", "parsed_events"):
            ev = box.get(k)
            if isinstance(ev, list):
                return [e for e in ev if isinstance(e, dict)]
    return []


def _cleanup_flow_query_generation_outputs(out_dir: Path) -> None:
    for d in ("llm_raw", "generated_runs", "validation_parsed_traces"):
        shutil.rmtree(out_dir / d, ignore_errors=True)
    for name in (
        "flow_catalog.json",
        "known_flow_queries.json",
        "unseen_flow_targets.json",
        "unseen_flow_queries.json",
        "all_queries_manifest.json",
        "generated_trace_manifest.json",
        "recycled_queries_from_unseen.json",
        "flow_counts_summary.json",
        "FLOWS_DIRECT_LIST.txt",
    ):
        p = out_dir / name
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def _events_from_parsed_trace_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []
    ev = obj.get("events") if isinstance(obj, dict) else None
    if not isinstance(ev, list):
        return []
    return [e for e in ev if isinstance(e, dict)]


def _bootstrap_registry_from_existing_generated_manifest(
    *,
    out_dir: Path,
    flow_registry: Dict[str, Dict[str, Any]],
    sig_to_fid: Dict[str, str],
    query_to_flow: Dict[str, str],
    generated_trace_rows: List[Dict[str, Any]],
) -> None:
    manifest_path = out_dir / "generated_trace_manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return
    rows = manifest_obj.get("generated_trace_rows") if isinstance(manifest_obj, dict) else None
    if not isinstance(rows, list):
        return
    parsed_dir = out_dir / "generated_runs" / "parsed_traces"
    for row in rows:
        if not isinstance(row, dict):
            continue
        generated_trace_rows.append(dict(row))
        q = str(row.get("query") or "").strip()
        nq = _norm_query_text(q) if q else ""
        trace_path = str(row.get("trace_path") or "").strip()
        sig = row.get("landed_signature") if isinstance(row.get("landed_signature"), list) else []
        sig_list = [str(x) for x in sig] if isinstance(sig, list) else []
        if not sig_list and trace_path:
            parsed_trace = parsed_dir / f"parsed_{Path(trace_path).name}"
            ev = _events_from_parsed_trace_file(parsed_trace)
            if ev:
                _fid, recovered_sig, _uniq = _flow_signature(ev)
                sig_list = [str(x) for x in recovered_sig]
        sig_key = _sig_key(sig_list)
        if not sig_key:
            continue
        fid = sig_to_fid.get(sig_key, "")
        if not fid:
            fid = hashlib.md5(sig_key.encode("utf-8")).hexdigest()
            flow_registry[fid] = {
                "flow_id": fid,
                "ordered_signature": sig_list,
                "unique_components": list(dict.fromkeys(sig_list)),
                "trace_files": [],
                "queries": [],
            }
            sig_to_fid[sig_key] = fid
        fr = flow_registry.setdefault(
            fid,
            {
                "flow_id": fid,
                "ordered_signature": sig_list,
                "unique_components": list(dict.fromkeys(sig_list)),
                "trace_files": [],
                "queries": [],
            },
        )
        if trace_path:
            tname = Path(trace_path).name
            traces = fr.get("trace_files") if isinstance(fr.get("trace_files"), list) else []
            if tname and tname not in traces:
                traces.append(tname)
            fr["trace_files"] = traces
        if q:
            qlist = fr.get("queries") if isinstance(fr.get("queries"), list) else []
            if q not in qlist:
                qlist.append(q)
            fr["queries"] = qlist
        if nq and nq not in query_to_flow:
            query_to_flow[nq] = fid


def _dedupe_generated_trace_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = json.dumps(
            {
                "query": str(row.get("query") or "").strip(),
                "normalized_query": str(row.get("normalized_query") or "").strip(),
                "trace_path": str(row.get("trace_path") or "").strip(),
                "status": str(row.get("status") or "").strip(),
                "target_flow_name": str(row.get("target_flow_name") or "").strip(),
                "phase": str(row.get("phase") or "").strip(),
                "landed_flow_id": str(row.get("landed_flow_id") or "").strip(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def build_flow_counts_summary_from_existing(
    *,
    outputs_dir: Path | str,
    nodespec_file: Path | str | None = None,
    node_spec: Any = None,
    runtime_mapping_file: Path | str | None = None,
) -> Path:
    outputs_dir = Path(outputs_dir).resolve()
    out_dir = outputs_dir / "flow_extraction" / "flow_query_generation"
    out_dir.mkdir(parents=True, exist_ok=True)
    nodespec_path = Path(nodespec_file).resolve() if nodespec_file else (None if node_spec is not None else _find_nodespec(outputs_dir))
    runtime_mapping_path: Path | None = None
    if runtime_mapping_file:
        runtime_mapping_path = Path(runtime_mapping_file).resolve()
    else:
        cand_runtime_mapping = outputs_dir / "validation" / "runtime_mapping.json"
        if cand_runtime_mapping.exists():
            runtime_mapping_path = cand_runtime_mapping.resolve()

    baseline_by_key: Dict[str, List[str]] = {}
    baseline_queries_by_key: Dict[str, List[str]] = {}
    ev_runs_root = outputs_dir / "validation" / "execution_validation_runs"
    if not ev_runs_root.exists():
        ev_runs_root = outputs_dir / "execution_validation" / "execution_validation_runs"
    prompt_by_run: Dict[str, str] = {}
    for ptxt in sorted(ev_runs_root.glob("**/run_*.prompt.txt")):
        run_stem = ptxt.name.replace(".prompt.txt", "")
        prompt_by_run[run_stem] = ptxt.read_text(encoding="utf-8", errors="replace").strip()
    for f in sorted(ev_runs_root.glob("**/run_*.llm_span_pipeline.json")):
        run_stem = f.name.replace(".llm_span_pipeline.json", "")
        ev: List[Dict[str, Any]] = []
        trace_path_txt = f.with_name(run_stem + ".trace_path.txt")
        trace_path_abs: Path | None = None
        if trace_path_txt.exists():
            trace_path_raw = trace_path_txt.read_text(encoding="utf-8", errors="replace").strip()
            if trace_path_raw:
                source_trace_file = Path(trace_path_raw).name
                trace_candidate = Path(trace_path_raw)
                if trace_candidate.is_absolute() and trace_candidate.exists():
                    trace_path_abs = trace_candidate.resolve()
                else:
                    trace_roots = (outputs_dir / "validation" / "traces", ev_runs_root)
                    for trace_root in trace_roots:
                        matches = sorted(trace_root.glob(f"**/{source_trace_file}")) if trace_root.exists() else []
                        if matches:
                            trace_path_abs = matches[0].resolve()
                            break
        if trace_path_abs is not None and trace_path_abs.exists():
            try:
                parsed = parse_trace_file(
                    trace_file=trace_path_abs,
                    nodespec_file=nodespec_path,
                    node_spec=node_spec,
                    runtime_mapping_file=runtime_mapping_path,
                )
                parsed_events = parsed.get("events") if isinstance(parsed, dict) else None
                if isinstance(parsed_events, list):
                    ev = [e for e in parsed_events if isinstance(e, dict)]
            except Exception:
                ev = []
        if not ev:
            ev = _events_from_llm_span_pipeline_file(f)
        if not ev:
            continue
        fid, sig, _ = _flow_signature(ev)
        k = _sig_key(sig)
        if k and k not in baseline_by_key:
            baseline_by_key[k] = [str(x) for x in sig]
        if k:
            q = str(prompt_by_run.get(run_stem, "")).strip()
            if q:
                arr = baseline_queries_by_key.setdefault(k, [])
                if q not in arr:
                    arr.append(q)

    generated_by_key: Dict[str, List[str]] = {}
    generated_queries_by_key: Dict[str, List[str]] = {}
    gen_dir = out_dir / "generated_runs" / "parsed_traces"
    generated_query_by_trace_name: Dict[str, str] = {}
    gmf = out_dir / "generated_trace_manifest.json"
    if gmf.exists():
        try:
            gobj = json.loads(gmf.read_text(encoding="utf-8", errors="replace"))
            grows = gobj.get("generated_trace_rows") if isinstance(gobj, dict) else []
            if isinstance(grows, list):
                for r in grows:
                    if not isinstance(r, dict):
                        continue
                    tp = str(r.get("trace_path") or "").strip()
                    q = str(r.get("query") or "").strip()
                    if tp and q:
                        generated_query_by_trace_name[Path(tp).name] = q
        except Exception:
            pass
    for f in sorted(gen_dir.glob("parsed_trace_*.json")):
        if f.name.endswith(".debug.json"):
            continue
        try:
            obj = json.loads(f.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        ev = obj.get("events")
        if not isinstance(ev, list):
            continue
        fid, sig, _ = _flow_signature(ev)
        k = _sig_key(sig)
        if k and k not in generated_by_key:
            generated_by_key[k] = [str(x) for x in sig]
        if k:
            trace_name = f.name.replace("parsed_", "")
            q = str(generated_query_by_trace_name.get(trace_name, "")).strip()
            if q:
                arr = generated_queries_by_key.setdefault(k, [])
                if q not in arr:
                    arr.append(q)

    baseline_keys = set(baseline_by_key.keys())
    generated_keys = set(generated_by_key.keys())
    added_keys = generated_keys - baseline_keys
    total_keys = baseline_keys | generated_keys

    def _rows_from(keys: set[str], source: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        out_rows: List[Dict[str, Any]] = []
        for k in sorted(keys):
            sig = source.get(k) or []
            flow_id = hashlib.md5(k.encode("utf-8")).hexdigest() if k else "empty_flow"
            seen_q: set[str] = set()
            q_list: List[str] = []
            for q in (baseline_queries_by_key.get(k, []) + generated_queries_by_key.get(k, [])):
                qq = str(q).strip()
                if qq and qq not in seen_q:
                    seen_q.add(qq)
                    q_list.append(qq)
            out_rows.append(
                {
                    "flow_id": flow_id,
                    "signature_key": k,
                    "ordered_signature": sig,
                    "queries": q_list,
                    "queries_count": len(q_list),
                }
            )
        return out_rows

    summary = {
        "outputs_dir": str(outputs_dir),
        "baseline_source": str(ev_runs_root),
        "baseline_parser_nodespec_file": str(nodespec_path) if nodespec_path else "",
        "baseline_parser_runtime_mapping_file": str(runtime_mapping_path) if runtime_mapping_path else "",
        "generated_source": str(gen_dir),
        "counts": {
            "baseline_unique_flows": len(baseline_keys),
            "added_unique_flows": len(added_keys),
            "total_unique_flows": len(total_keys),
            "generated_unique_flows": len(generated_keys),
        },
        "signatures": {
            "baseline": _rows_from(baseline_keys, baseline_by_key),
            "generated": _rows_from(generated_keys, generated_by_key),
            "added": _rows_from(added_keys, generated_by_key),
            "total": _rows_from(total_keys, {**baseline_by_key, **generated_by_key}),
        },
    }
    report_file = out_dir / "flow_counts_summary.json"
    report_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report_file


def run_flow_query_generation(
    *,
    outputs_dir: Path | str,
    model: str,
    traces_dir: Path | str | None = None,
    nodespec_file: Path | str | None = None,
    node_spec: Any = None,
    runtime_mapping_file: Path | str | None = None,
    known_queries_per_flow: int = DEFAULT_KNOWN_QUERIES_PER_FLOW,
    execute_generated_queries: bool = True,
    query_timeout_sec: int = 180,
    max_exec_steps: int = DEFAULT_MAX_EXEC_STEPS,
    refresh_raw: bool = False,
    handler: Any = None,
) -> FlowQueryGenerationResult:
    outputs_dir = Path(outputs_dir).resolve()
    traces_dir = Path(traces_dir).resolve() if traces_dir else outputs_dir / "validation" / "traces"
    nodespec_file = Path(nodespec_file).resolve() if nodespec_file else (None if node_spec is not None else _find_nodespec(outputs_dir))
    if node_spec is None:
        if nodespec_file is None or not nodespec_file.is_file():
            raise FileNotFoundError(
                f"NodeSpec file not found; provide nodespec_file or place one under {outputs_dir}"
            )
        if nodespec_file.suffix.lower() == ".json":
            from node_spec.structure_schema import NodeSpec as NodeSpecModel

            node_spec = NodeSpecModel.from_json(path=str(nodespec_file))
        else:
            from .save_simple_flow import _load_nodespec_top

            node_spec = _load_nodespec_top(nodespec_file)

    if runtime_mapping_file:
        runtime_mapping_file = Path(runtime_mapping_file).resolve()
    else:
        runtime_mapping_file = outputs_dir / "validation" / "runtime_mapping.json"
        if not runtime_mapping_file.exists():
            mapping = induce_runtime_mapping(outputs_dir=outputs_dir, nodespec_file=nodespec_file, node_spec=node_spec)
            runtime_mapping_file.parent.mkdir(parents=True, exist_ok=True)
            runtime_mapping_file.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if node_spec.metadata is None:
        node_spec.metadata = {}
    node_spec.metadata["runtime_mapping_path"] = str(runtime_mapping_file)

    out_dir = outputs_dir / "flow_extraction" / "flow_query_generation"
    out_dir.mkdir(parents=True, exist_ok=True)
    if refresh_raw:
        _cleanup_flow_query_generation_outputs(out_dir)
    llm_raw_dir = out_dir / "llm_raw"
    validation_parsed_dir = out_dir / "validation_parsed_traces"
    validation_parsed_dir.mkdir(parents=True, exist_ok=True)
    llm_counter = [0]
    trace_to_query, query_to_trace_path = _load_validation_query_trace_links(outputs_dir)
    run_seed_queries = trace_to_query or _load_seed_queries_from_validation_runs(outputs_dir)

    flow_groups: Dict[str, Dict[str, Any]] = {}
    trace_files = sorted([p for p in traces_dir.glob("*.json") if p.is_file()])
    for tf in trace_files:
        parsed = parse_trace_file(
            tf,
            nodespec_file=nodespec_file,
            node_spec=node_spec,
            runtime_mapping_file=runtime_mapping_file,
        )
        save_parsed_outputs(
            parsed=parsed,
            parsed_out_file=validation_parsed_dir / f"parsed_{tf.name}",
            debug_out_file=validation_parsed_dir / f"parsed_{tf.stem}.debug.json",
        )
        trace_obj = json.loads(tf.read_text(encoding="utf-8"))
        seed_query = run_seed_queries.get(tf.name) or _extract_seed_query(trace_obj)
        events = parsed.get("events") if isinstance(parsed.get("events"), list) else []
        flow_id, ordered_sig, unique_components = _flow_signature(events)
        group = flow_groups.setdefault(
            flow_id,
            {
                "flow_id": flow_id,
                "ordered_signature": ordered_sig,
                "unique_components": unique_components,
                "trace_files": [],
                "seed_queries": [],
            },
        )
        group["trace_files"].append(tf.name)
        if seed_query and seed_query not in group["seed_queries"]:
            group["seed_queries"].append(seed_query)

    flows = sorted(flow_groups.values(), key=lambda x: (-len(x["trace_files"]), x["flow_id"]))
    print(
        f"[flow-query-gen] parsed validation traces={len(trace_files)} "
        f"observed flow patterns={len(flows)}"
    )
    flow_catalog = {
        "outputs_dir": str(outputs_dir),
        "traces_dir": str(traces_dir),
        "nodespec_file": str(nodespec_file) if nodespec_file else "",
        "runtime_mapping_file": str(runtime_mapping_file) if runtime_mapping_file else "",
        "trace_count": len(trace_files),
        "flow_count": len(flows),
        "flows": flows,
    }

    system_context = _build_system_context(nodespec_file, flow_catalog, node_spec=node_spec)
    # Canonical mutable flow registry.
    flow_registry: Dict[str, Dict[str, Any]] = {}
    sig_to_fid: Dict[str, str] = {}
    query_to_flow: Dict[str, str] = {}
    for f in flows:
        fid = str(f.get("flow_id"))
        sig = f.get("ordered_signature") if isinstance(f.get("ordered_signature"), list) else []
        sk = _sig_key(sig)
        flow_registry[fid] = {
            "flow_id": fid,
            "ordered_signature": sig,
            "unique_components": list(f.get("unique_components") or []),
            "trace_files": list(f.get("trace_files") or []),
            "queries": list(f.get("seed_queries") or []),
        }
        if sk:
            sig_to_fid[sk] = fid
        for q in (f.get("seed_queries") or []):
            nq = _norm_query_text(q)
            if nq and nq not in query_to_flow:
                query_to_flow[nq] = fid

    # Target->query->execute->parse loop (exact requested orchestration).
    unseen_targets: List[Dict[str, Any]] = []
    unseen_queries_rows: List[Dict[str, Any]] = []
    recycled_from_unseen: List[Dict[str, Any]] = []
    target_flow_attempt_report: List[Dict[str, Any]] = []
    generated_trace_rows: List[Dict[str, Any]] = []
    target_attempts: Dict[str, List[Dict[str, Any]]] = {}
    completed_target_keys: set[str] = set()
    attempted_target_keys: set[str] = set()
    stop_signal: Dict[str, Any] = {}
    unseen_generation_error = ""
    filtered_out_unseen_candidates: List[Dict[str, Any]] = []
    filtered_feedback_history: List[Dict[str, Any]] = []

    if not refresh_raw:
        _bootstrap_registry_from_existing_generated_manifest(
            out_dir=out_dir,
            flow_registry=flow_registry,
            sig_to_fid=sig_to_fid,
            query_to_flow=query_to_flow,
            generated_trace_rows=generated_trace_rows,
        )

    if execute_generated_queries:
        use_case_number = 0
        try:
            use_case_number = _infer_use_case_number(outputs_dir)
        except Exception:
            use_case_number = 0
        zipfile_path = _infer_zipfile_from_outputs_dir(outputs_dir)
        exec_example = _load_execution_command_example(outputs_dir)
        command_template = _command_template_from_exec_example(exec_example)
        generated_out_dir = out_dir / "generated_runs"
        steps = 0
        no_new_target_rounds = 0
        no_new_flow_steps = 0
        new_flow_discoveries = 0
        current_target: Dict[str, Any] | None = None
        pending_targets: List[Dict[str, Any]] = []

        while steps < max(1, int(max_exec_steps)):
            if current_target is None:
                # Drain already proposed targets before proposing additional unseen flows.
                if not pending_targets:
                    candidates, rejected_candidates = _propose_unseen_flows(
                        model=model,
                        system_context=system_context,
                        observed_flows=list(flow_registry.values()),
                        n_new_flows=DEFAULT_UNSEEN_BATCH_SIZE,
                        rejected_examples=filtered_feedback_history,
                        llm_raw_dir=llm_raw_dir,
                        llm_counter=llm_counter,
                    )
                    if rejected_candidates:
                        filtered_out_unseen_candidates.extend(rejected_candidates)
                        for rc in rejected_candidates:
                            filtered_feedback_history.append(
                                {
                                    "target_flow_name": rc.get("target_flow_name", ""),
                                    "ordered_signature": rc.get("ordered_signature", []),
                                    "reason": rc.get("reason", ""),
                                }
                            )
                        if len(filtered_feedback_history) > 80:
                            filtered_feedback_history = filtered_feedback_history[-80:]
                    for c in candidates:
                        key = _sig_key(c.get("ordered_signature") if isinstance(c.get("ordered_signature"), list) else [])
                        if key and key not in completed_target_keys and key not in attempted_target_keys:
                            pending_targets.append(c)
                            unseen_targets.append(c)
                            attempted_target_keys.add(key)
                    if not pending_targets:
                        if rejected_candidates:
                            no_new_target_rounds += 1
                            if no_new_target_rounds < int(DEFAULT_NO_NEW_TARGET_ROUNDS_STOP):
                                continue
                            stop_signal = {
                                "mode": "strict_loop",
                                "stopped_by": "no_new_or_unseen_targets_after_filtered_retries",
                                "steps": steps,
                                "new_flow_discoveries": new_flow_discoveries,
                                "proposal_rounds_without_accepted_targets": no_new_target_rounds,
                                "filtered_candidate_count": len(filtered_out_unseen_candidates),
                            }
                            break
                        stop_signal = {
                            "mode": "strict_loop",
                            "stopped_by": "no_new_or_unseen_targets",
                            "steps": steps,
                            "new_flow_discoveries": new_flow_discoveries,
                        }
                        break

                current_target = pending_targets.pop(0)
                no_new_target_rounds = 0

            tkey = _sig_key(current_target.get("ordered_signature") if isinstance(current_target.get("ordered_signature"), list) else [])
            attempts = target_attempts.setdefault(tkey, [])
            q_list = _generate_queries_for_flow(
                model=model,
                system_context=system_context,
                requested_flow=current_target,
                example_queries=[],
                previous_attempts=attempts,
                call_name=f"target_query_{str(current_target.get('target_flow_name') or 'target')}",
                n=1,
                llm_raw_dir=llm_raw_dir,
                llm_counter=llm_counter,
            )
            q_item = q_list[0] if q_list else {}
            query = str(q_item.get("query") or "").strip() if isinstance(q_item, dict) else ""
            is_info_request = bool(q_item.get("is_info_request")) if isinstance(q_item, dict) else False
            if not query:
                attempts.append({"error": "empty_query_generation", "is_info_request": False})
                if _count_attempts_toward_limit(attempts) >= int(DEFAULT_MAX_ATTEMPTS_PER_TARGET):
                    completed_target_keys.add(tkey)
                    current_target = None
                continue

            steps += 1
            nq = _norm_query_text(query)
            # Reuse existing known outcome if query already seen.
            known_fid = query_to_flow.get(nq, "")
            if known_fid:
                landed_sig: List[str] = []
                if known_fid in flow_registry:
                    s = flow_registry[known_fid].get("ordered_signature")
                    if isinstance(s, list):
                        landed_sig = [str(x) for x in s]
                landed_key = _sig_key(landed_sig)
                success_known = bool(landed_key and landed_key == tkey)
                recycled_from_unseen.append(
                    {
                        "query": query,
                        "is_info_request": is_info_request,
                        "source_target_flow_name": current_target.get("target_flow_name"),
                        "landed_existing_flow_id": known_fid,
                        "mode": "known_prompt_reuse",
                    }
                )
                attempts.append(
                    {
                        "query": query,
                        "landed_flow_id": known_fid,
                        "landed_signature": landed_sig,
                        "mode": "reuse",
                        "miss_reason": "reused_query_already_lands_known_flow",
                        "is_info_request": is_info_request,
                    }
                )
                target_flow_attempt_report.append(
                    {
                        "query": query,
                        "target_flow_name": str(current_target.get("target_flow_name") or ""),
                        "target_flow_signature": current_target.get("ordered_signature")
                        if isinstance(current_target.get("ordered_signature"), list)
                        else [],
                        "resulted_flow_id": known_fid,
                        "resulted_flow_signature": landed_sig,
                        "successful": success_known,
                        "mode": "reuse",
                    }
                )
                no_new_flow_steps += 1
                if success_known:
                    completed_target_keys.add(tkey)
                    current_target = None
                    continue
                if _count_attempts_toward_limit(attempts) >= int(DEFAULT_MAX_ATTEMPTS_PER_TARGET):
                    completed_target_keys.add(tkey)
                    current_target = None
                continue

            # Execute system.
            try:
                zipfile_path = handler.path_to_zipfile

                with tempfile.TemporaryDirectory() as traces_dir:
                    trace_path = trace_extraction._run_trace_extraction(
                        zipfile_path=str(zipfile_path),
                        task=query,
                        handler=handler,
                        command_template=command_template,
                        out_dir=str(traces_dir),
                        system_name=_safe_system_name(f"temp_system_flow_{outputs_dir.name}"),
                        query_timeout_sec=int(query_timeout_sec),
                    )

                    # Parse resulting trace -> extract actual flow.
                    parsed = parse_trace_file(
                        trace_path,
                        nodespec_file=nodespec_file,
                        node_spec=node_spec,
                        runtime_mapping_file=runtime_mapping_file,
                    )
                    parsed_dir = generated_out_dir / "parsed_traces"
                    trace_name = Path(trace_path).name
                    parsed_name = f"parsed_{trace_name}"
                    if parsed_name.endswith(".json"):
                        parsed_debug_name = parsed_name[:-5] + ".debug.json"
                    else:
                        parsed_debug_name = parsed_name + ".debug.json"
                    save_parsed_outputs(
                        parsed=parsed,
                        parsed_out_file=parsed_dir / parsed_name,
                        debug_out_file=parsed_dir / parsed_debug_name,
                    )
                    evs = parsed.get("events") if isinstance(parsed.get("events"), list) else []
                    actual_fid, actual_sig, actual_unique = _flow_signature(evs)
                    actual_key = _sig_key(actual_sig)
                    error_content = _extract_error_content(evs)

                    if actual_key in sig_to_fid:
                        landed_fid = sig_to_fid[actual_key]
                        flow_registry[landed_fid].setdefault("queries", []).append(query)
                        recycled_from_unseen.append(
                            {
                                "query": query,
                                "is_info_request": is_info_request,
                                "source_target_flow_name": current_target.get("target_flow_name"),
                                "landed_existing_flow_id": landed_fid,
                                "mode": "executed_existing_flow",
                            }
                        )
                        no_new_flow_steps += 1
                    else:
                        landed_fid = actual_fid
                        sig_to_fid[actual_key] = landed_fid
                        flow_registry[landed_fid] = {
                            "flow_id": landed_fid,
                            "ordered_signature": actual_sig,
                            "unique_components": actual_unique,
                            "trace_files": [Path(trace_path).name],
                            "queries": [query],
                        }
                        new_flow_discoveries += 1
                        no_new_flow_steps = 0
                    query_to_flow[nq] = landed_fid
                    generated_trace_rows.append(
                        {
                            "query": query,
                            "normalized_query": nq,
                            "target_flow_name": current_target.get("target_flow_name"),
                            "target_flow_signature": current_target.get("ordered_signature")
                            if isinstance(current_target.get("ordered_signature"), list)
                            else [],
                            "status": "generated",
                            "trace_path": str(trace_path),
                            "landed_flow_id": landed_fid,
                            "landed_signature": actual_sig,
                            "successful": bool(actual_key == tkey),
                            "is_info_request": is_info_request,
                            "error_content": error_content,
                        }
                    )
                    target_flow_attempt_report.append(
                        {
                            "query": query,
                            "target_flow_name": str(current_target.get("target_flow_name") or ""),
                            "target_flow_signature": current_target.get("ordered_signature")
                            if isinstance(current_target.get("ordered_signature"), list)
                            else [],
                            "resulted_flow_id": landed_fid,
                            "resulted_flow_signature": actual_sig,
                            "successful": bool(actual_key == tkey),
                            "mode": "generated",
                            "error_content": error_content,
                        }
                    )
                    attempt_row = {
                        "query": query,
                        "is_info_request": is_info_request,
                        "trace_path": str(trace_path),
                        "landed_flow_id": landed_fid,
                        "landed_signature": actual_sig,
                        "miss_reason": (
                            ""
                            if landed_fid == sig_to_fid.get(tkey, "")
                            else "landed_other_flow_after_execution"
                        ),
                    }
                    if error_content:
                        attempt_row["error_content"] = error_content
                    attempts.append(attempt_row)
                    if actual_key == tkey:
                        completed_target_keys.add(tkey)
                        current_target = None
                        continue
                    if _count_attempts_toward_limit(attempts) >= int(DEFAULT_MAX_ATTEMPTS_PER_TARGET):
                        # move on after fixed number of tested attempts per target
                        completed_target_keys.add(tkey)
                        current_target = None

            except Exception as exc:
                generated_trace_rows.append(
                    {
                        "query": query,
                        "normalized_query": nq,
                        "target_flow_name": current_target.get("target_flow_name"),
                        "status": "error",
                        "trace_path": "",
                        "error": f"{exc.__class__.__name__}: {exc}",
                        "is_info_request": is_info_request,
                    }
                )
                target_flow_attempt_report.append(
                    {
                        "query": query,
                        "target_flow_name": str(current_target.get("target_flow_name") or ""),
                        "target_flow_signature": current_target.get("ordered_signature")
                        if isinstance(current_target.get("ordered_signature"), list)
                        else [],
                        "resulted_flow_id": "",
                        "resulted_flow_signature": [],
                        "successful": False,
                        "mode": "execution_error",
                        "error": f"{exc.__class__.__name__}: {exc}",
                    }
                )
                attempts.append({"query": query, "error": f"{exc.__class__.__name__}: {exc}", "is_info_request": is_info_request})
                attempts[-1]["miss_reason"] = "execution_error"
                no_new_flow_steps += 1
                if _count_attempts_toward_limit(attempts) >= int(DEFAULT_MAX_ATTEMPTS_PER_TARGET):
                    completed_target_keys.add(tkey)
                    current_target = None
                continue

            
        if not stop_signal:
            stop_signal = {
                "mode": "strict_loop",
                "stopped_by": "max_exec_steps",
                "steps": steps,
                "max_exec_steps": int(max_exec_steps),
                "no_new_target_rounds": no_new_target_rounds,
                "no_new_flow_steps": no_new_flow_steps,
                "new_flow_discoveries": new_flow_discoveries,
            }
    else:
        stop_signal = {
            "mode": "strict_loop",
            "stopped_by": "skip_trace_execution",
            "steps": 0,
        }

    # Phase 2: ensure per-flow verified query coverage up to known_queries_per_flow.
    known_queries_rows: List[Dict[str, Any]] = []
    known_fill_report: List[Dict[str, Any]] = []
    for fid, fr in sorted(flow_registry.items(), key=lambda kv: kv[0]):
        uniq: List[str] = []
        seenq: set[str] = set()
        for q in fr.get("queries", []):
            nq = _norm_query_text(q)
            if nq and nq not in seenq:
                seenq.add(nq)
                uniq.append(q)
        fr["queries"] = uniq
        target_n = int(known_queries_per_flow)
        flow_sig = fr.get("ordered_signature") if isinstance(fr.get("ordered_signature"), list) else []
        target_key = _sig_key(flow_sig)
        rounds_without_k_change = 0
        round_idx = 0
        while execute_generated_queries and len(uniq) < target_n:
            round_idx += 1
            missing_before = target_n - len(uniq)
            add = _generate_queries_for_flow(
                model=model,
                system_context=system_context,
                requested_flow=flow_sig,
                n=missing_before,
                example_queries=uniq,
                previous_attempts=[],
                call_name=f"known_flow_fill_{fid}_r{round_idx}",
                llm_raw_dir=llm_raw_dir,
                llm_counter=llm_counter,
            )
            for item in add:
                q = str(item.get("query") or "").strip() if isinstance(item, dict) else ""
                if not q:
                    continue
                nq = _norm_query_text(q)
                if not nq or nq in seenq:
                    continue
                known_landed_fid = query_to_flow.get(nq, "")
                if known_landed_fid:
                    landed_sig: List[str] = []
                    if known_landed_fid in flow_registry:
                        s = flow_registry[known_landed_fid].get("ordered_signature")
                        if isinstance(s, list):
                            landed_sig = [str(x) for x in s]
                    if _sig_key(landed_sig) == target_key or str(known_landed_fid) == str(fid):
                        seenq.add(nq)
                        uniq.append(q)
                    continue

                try:
                    if handler is not None:
                        trace_path = trace_extraction._run_trace_extraction(
                            zipfile_path=str(handler.path_to_zipfile),
                            task=q,
                            handler=handler,
                            command_template=command_template,
                            out_dir=str(generated_out_dir),
                            system_name=_safe_system_name(f"temp_system_flow_{outputs_dir.name}"),
                            query_timeout_sec=int(query_timeout_sec),
                        )
                    elif use_case_number > 0:
                        trace_path = trace_extraction.extract_trace_path_from_use_case(
                            use_case_number=use_case_number,
                            task=q,
                            handler=handler,
                            out_dir=str(generated_out_dir),
                            query_timeout_sec=int(query_timeout_sec),
                        )
                    else:
                        if zipfile_path is None:
                            raise ValueError(
                                f"Could not infer zip file from outputs_dir={outputs_dir}"
                            )
                        trace_path = trace_extraction._run_trace_extraction(
                            zipfile_path=str(zipfile_path),
                            task=q,
                            handler=handler,
                            command_template=command_template,
                            system_name=_safe_system_name(f"temp_system_flow_{outputs_dir.name}"),
                            out_dir=str(generated_out_dir),
                            query_timeout_sec=int(query_timeout_sec),
                        )
                except Exception as exc:
                    generated_trace_rows.append(
                        {
                            "query": q,
                            "normalized_query": nq,
                            "target_flow_name": f"known_flow_{fid}",
                            "status": "error",
                            "trace_path": "",
                            "error": f"{exc.__class__.__name__}: {exc}",
                            "phase": "known_flow_fill",
                        }
                    )
                    continue

                parsed = parse_trace_file(
                    trace_path,
                    nodespec_file=nodespec_file,
                    node_spec=node_spec,
                    runtime_mapping_file=runtime_mapping_file,
                )
                parsed_dir = generated_out_dir / "parsed_traces"
                trace_name = Path(trace_path).name
                parsed_name = f"parsed_{trace_name}"
                if parsed_name.endswith(".json"):
                    parsed_debug_name = parsed_name[:-5] + ".debug.json"
                else:
                    parsed_debug_name = parsed_name + ".debug.json"
                save_parsed_outputs(
                    parsed=parsed,
                    parsed_out_file=parsed_dir / parsed_name,
                    debug_out_file=parsed_dir / parsed_debug_name,
                )
                evs = parsed.get("events") if isinstance(parsed.get("events"), list) else []
                actual_fid, actual_sig, actual_unique = _flow_signature(evs)
                actual_key = _sig_key(actual_sig)
                error_content = _extract_error_content(evs)

                if actual_key in sig_to_fid:
                    landed_fid = sig_to_fid[actual_key]
                    flow_registry[landed_fid].setdefault("queries", []).append(q)
                else:
                    landed_fid = actual_fid
                    sig_to_fid[actual_key] = landed_fid
                    flow_registry[landed_fid] = {
                        "flow_id": landed_fid,
                        "ordered_signature": actual_sig,
                        "unique_components": actual_unique,
                        "trace_files": [Path(trace_path).name],
                        "queries": [q],
                    }
                query_to_flow[nq] = landed_fid
                success_fill = bool(actual_key == target_key)
                generated_trace_rows.append(
                    {
                        "query": q,
                        "normalized_query": nq,
                        "target_flow_name": f"known_flow_{fid}",
                        "target_flow_signature": flow_sig,
                        "status": "generated",
                        "trace_path": trace_path,
                        "landed_flow_id": landed_fid,
                        "landed_signature": actual_sig,
                        "successful": success_fill,
                        "phase": "known_flow_fill",
                        "error_content": error_content,
                    }
                )
                if success_fill:
                    seenq.add(nq)
                    uniq.append(q)

            missing_after = target_n - len(uniq)
            known_fill_report.append(
                {
                    "flow_id": fid,
                    "round": round_idx,
                    "missing_before": missing_before,
                    "missing_after": missing_after,
                }
            )
            if missing_after <= 0:
                break
            if missing_after == missing_before:
                rounds_without_k_change += 1
            else:
                rounds_without_k_change = 0
            if rounds_without_k_change >= 3:
                break

        fr["queries"] = uniq
        known_queries_rows.append(
            {
                "flow_id": fid,
                "ordered_signature": flow_sig,
                "queries": uniq[: int(known_queries_per_flow)],
                "counts": {"final_count": min(len(uniq), int(known_queries_per_flow))},
            }
        )

    flow_catalog_file = out_dir / "flow_catalog.json"
    known_queries_file = out_dir / "known_flow_queries.json"
    unseen_targets_file = out_dir / "unseen_flow_targets.json"
    unseen_queries_file = out_dir / "unseen_flow_queries.json"
    all_queries_manifest_file = out_dir / "all_queries_manifest.json"
    generated_trace_manifest_file = out_dir / "generated_trace_manifest.json"
    flow_counts_summary_file = out_dir / "flow_counts_summary.json"

    # IMPORTANT: flow_catalog must reflect the FINAL observed registry after query execution.
    # The initial `flow_catalog` above is built before unseen-target execution.
    final_flows = sorted(
        flow_registry.values(),
        key=lambda x: (-len(x.get("trace_files") or []), str(x.get("flow_id") or "")),
    )
    final_trace_files = {
        str(tf)
        for f in final_flows
        for tf in (f.get("trace_files") if isinstance(f.get("trace_files"), list) else [])
        if str(tf).strip()
    }
    final_flow_catalog = {
        "outputs_dir": str(outputs_dir),
        "traces_dir": str(traces_dir),
        "nodespec_file": str(nodespec_file) if nodespec_file else "",
        "runtime_mapping_file": str(runtime_mapping_file) if runtime_mapping_file else "",
        "trace_count": len(final_trace_files),
        "flow_count": len(final_flows),
        "flows": final_flows,
    }
    flow_catalog_file.write_text(json.dumps(final_flow_catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    known_queries_file.write_text(json.dumps({"known_flow_queries": known_queries_rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    unseen_targets_file.write_text(
        json.dumps(
            {
                "unseen_flow_targets": unseen_targets,
                "filtered_out_flow_candidates": filtered_out_unseen_candidates,
                "stop_signal": stop_signal,
                "error": unseen_generation_error,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    unseen_queries_file.write_text(json.dumps({"unseen_flow_queries": unseen_queries_rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "recycled_queries_from_unseen.json").write_text(
        json.dumps({"recycled_queries_from_unseen": recycled_from_unseen}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    all_queries: List[Dict[str, Any]] = []
    for row in known_queries_rows:
        fid = str(row.get("flow_id"))
        for q in row.get("queries") or []:
            nq = _norm_query_text(q)
            all_queries.append(
                {
                    "source": "known_flow",
                    "flow_id": fid,
                    "query": q,
                    "normalized_query": nq,
                    "existing_trace_path": query_to_trace_path.get(nq, ""),
                }
            )
    for row in unseen_queries_rows:
        tname = str(row.get("target_flow_name") or "")
        for q in row.get("queries") or []:
            nq = _norm_query_text(q)
            all_queries.append(
                {
                    "source": "unseen_target",
                    "target_flow_name": tname,
                    "query": q,
                    "normalized_query": nq,
                    "existing_trace_path": query_to_trace_path.get(nq, ""),
                }
            )
    all_queries_manifest_file.write_text(json.dumps({"all_queries": all_queries}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Add existing known links into generated manifest for completeness.
    for row in all_queries:
        q = str(row.get("query") or "").strip()
        nq = str(row.get("normalized_query") or "")
        existing_trace_path = str(row.get("existing_trace_path") or "").strip()
        if q and existing_trace_path:
            generated_trace_rows.append(
                {
                    **row,
                    "status": "existing",
                    "trace_path": existing_trace_path,
                }
            )
    generated_trace_rows = _dedupe_generated_trace_rows(generated_trace_rows)
    generated_trace_manifest_file.write_text(
        json.dumps(
            {
                "execute_generated_queries": bool(execute_generated_queries),
                "query_timeout_sec": int(query_timeout_sec),
                "refresh_raw": bool(refresh_raw),
                "generated_trace_rows": generated_trace_rows,
                "target_flow_attempt_report": target_flow_attempt_report,
                "known_flow_fill_report": known_fill_report,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    generated_attempts = sum(
        1 for row in generated_trace_rows
        if str(row.get("status") or "").lower() in {"generated", "error"}
    )
    generated_errors = sum(
        1 for row in generated_trace_rows
        if str(row.get("status") or "").lower() == "error"
    )
    print(
        f"[flow-query-gen] generated-query sandbox attempts={generated_attempts} "
        f"errors={generated_errors}"
    )
    _write_flows_direct_list(
        out_dir=out_dir,
        known_queries_rows=known_queries_rows,
        recycled_from_unseen=recycled_from_unseen,
        generated_trace_rows=generated_trace_rows,
    )
    flow_counts_summary_file = build_flow_counts_summary_from_existing(
        outputs_dir=outputs_dir,
        nodespec_file=nodespec_file,
        node_spec=node_spec,
        runtime_mapping_file=runtime_mapping_file,
    )

    # Complete Flow Extraction by enriching the in-memory NodeSpec from both
    # validation traces and any generated-query traces. The run_si entry point
    # supplies raw NodeSpec JSON, so write the resulting schema object back to
    # that JSON file instead of routing through Python-source-only helpers.
    from .add_tool_io_pairs_from_traces import add_tool_io_pairs_from_out_dir
    from .add_top_level_flows import add_top_level_flows_from_out_dir

    tool_io_updates = []
    json_nodespec_file = bool(nodespec_file and nodespec_file.suffix.lower() == ".json")
    for parsed_dir in (
        validation_parsed_dir,
        out_dir / "generated_runs" / "parsed_traces",
    ):
        if parsed_dir.is_dir():
            helper_args = (
                {"node_spec": node_spec}
                if json_nodespec_file or nodespec_file is None
                else {"nodes_file": str(nodespec_file)}
            )
            tool_io_updates.append(
                add_tool_io_pairs_from_out_dir(
                    out_dir=outputs_dir,
                    traces_dir=str(parsed_dir),
                    max_pairs_per_tool=20,
                    dry_run=False,
                    **helper_args,
                )
            )
    tools_updated = sum(int(item.get("tools_updated") or 0) for item in tool_io_updates)
    print(f"[flow-query-gen] tool I/O enrichment updated tools={tools_updated}")

    flow_helper_args = (
        {"node_spec": node_spec}
        if json_nodespec_file or nodespec_file is None
        else {"nodes_file": str(nodespec_file)}
    )
    flow_update = add_top_level_flows_from_out_dir(
        out_dir=outputs_dir,
        exclude_inventory_queries=True,
        write_flow_trace_files=True,
        **flow_helper_args,
    )
    print(
        "[flow-query-gen] NodeSpec flow extraction "
        f"flows_written={flow_update.get('flows_written')} "
        f"unique_flow_ids={flow_update.get('unique_flow_ids')} "
        f"skipped_inventory_queries={flow_update.get('skipped_inventory_queries')}"
    )
    if json_nodespec_file and nodespec_file is not None:
        node_spec.to_json(path=str(nodespec_file))
        print(f"[flow-query-gen] wrote enriched NodeSpec: {nodespec_file}")

    return FlowQueryGenerationResult(
        out_dir=out_dir,
        flow_catalog_file=flow_catalog_file,
        known_queries_file=known_queries_file,
        unseen_targets_file=unseen_targets_file,
        unseen_queries_file=unseen_queries_file,
        flow_counts_summary_file=flow_counts_summary_file,
    )
