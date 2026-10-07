from __future__ import annotations

import difflib
from collections import deque
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Deque, Dict, List, Set, Tuple

from node_spec.structure_schema import FlowSpec
from trace_parser.nodespec_loader import load_nodes_by_var

from trace_parser.parser_core import (
    GENERIC_MESSAGE_LIST_PATHS,
    TOP_DEFAULT_CONTENT_KEYS,
    TOP_DEFAULT_NAME_KEYS,
    TOP_DEFAULT_ROLE_KEYS,
    TOP_DEFAULT_TIMESTAMP_KEYS,
    TOP_DEFAULT_TOOL_CALL_KEYS,
    TOP_DEFAULT_TOOL_RESULT_KEYS,
    Schema,
    parse_trace_with_schema,
)


def build_runtime_schema() -> Schema:
    return Schema(
        use_case="runtime_trace_parsing",
        trace_count=1,
        message_list_paths=list(GENERIC_MESSAGE_LIST_PATHS),
        role_keys=list(TOP_DEFAULT_ROLE_KEYS),
        content_keys=list(TOP_DEFAULT_CONTENT_KEYS),
        name_keys=list(TOP_DEFAULT_NAME_KEYS),
        tool_calls_keys=list(TOP_DEFAULT_TOOL_CALL_KEYS),
        tool_result_keys=list(TOP_DEFAULT_TOOL_RESULT_KEYS),
        timestamp_keys=list(TOP_DEFAULT_TIMESTAMP_KEYS),
    )


def _norm_name(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def _normalized_debug_name(s: str) -> str:
    return str(s or "")


def _name_keys(name: str) -> List[str]:
    raw = str(name or "").strip()
    if not raw:
        return []
    keys: List[str] = []
    seen: Set[str] = set()

    def _add(v: str) -> None:
        n = _norm_name(v)
        if n and n not in seen:
            seen.add(n)
            keys.append(n)

    _add(raw)
    for token in raw.replace("/", ".").replace(":", ".").split("."):
        t = token.strip()
        if t:
            _add(t)
    return keys


def _name_tokens(name: str) -> List[str]:
    raw = str(name or "").strip().lower()
    if not raw:
        return []
    cur: List[str] = []
    out: List[str] = []
    for ch in raw:
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                tok = "".join(cur)
                if tok:
                    out.append(tok)
                cur = []
    if cur:
        tok = "".join(cur)
        if tok:
            out.append(tok)
    return out


def _node_type_name(node: Dict[str, Any]) -> str:
    nt = node.get("node_type")
    if isinstance(nt, dict):
        return str(nt.get("type") or "")
    return str(nt or "")


def _load_python_module(file_path: Path):
    file_path = Path(file_path).resolve()
    parent = str(file_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    # Prefer colocated NodeSpec_schema.py next to the spec file.
    local_schema = file_path.parent / "NodeSpec_schema.py"
    if local_schema.exists():
        schema_name = f"flow_nodespec_schema_{hashlib.md5(str(local_schema).encode('utf-8')).hexdigest()}"
        schema_spec = importlib.util.spec_from_file_location(schema_name, str(local_schema))
        if schema_spec is not None and schema_spec.loader is not None:
            schema_mod = importlib.util.module_from_spec(schema_spec)
            sys.modules[schema_name] = schema_mod
            sys.modules["NodeSpec_schema"] = schema_mod
            schema_spec.loader.exec_module(schema_mod)

    mod_name = f"flow_nodespec_{hashlib.md5(str(file_path).encode('utf-8')).hexdigest()}"
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_from_nodespec_utils(path: Path) -> Tuple[List[str], List[str], str]:
    module = _load_python_module(path)
    roots: List[Any] = []
    for attr in ("ALL_NODES", "MAIN_GRAPH_NODES"):
        arr = getattr(module, attr, None)
        if isinstance(arr, list):
            for item in arr:
                if hasattr(item, "list_agents") and hasattr(item, "list_tools"):
                    roots.append(item)
    if not roots:
        raise RuntimeError("no NodeSpec roots with list_agents/list_tools found")

    # Aggregate across all discovered roots to avoid missing tool inventories
    # when a single chosen root has partial/empty helper output.
    agents_set: Set[str] = set()
    tools_set: Set[str] = set()
    for r in roots:
        agent_rows = r.list_agents() if callable(getattr(r, "list_agents", None)) else []
        tool_rows = r.list_tools() if callable(getattr(r, "list_tools", None)) else []
        for x in agent_rows:
            if isinstance(x, dict):
                nm = str(x.get("name") or "").strip()
                if nm:
                    agents_set.add(nm)
        for x in tool_rows:
            if isinstance(x, dict):
                nm = str(x.get("name") or "").strip()
                if nm:
                    tools_set.add(nm)
    agents = sorted(agents_set)
    tools = sorted(tools_set)
    # Some generated NodeSpec schema helpers can return empty lists even when
    # ALL_NODES is populated; force fallback parsing in that case.
    if not agents and not tools:
        raise RuntimeError("nodespec utility methods returned empty agent/tool inventories")
    return agents, tools, "nodespec_utility_methods"


def load_expected_names_from_nodespec(nodespec_file: Path | str) -> Dict[str, List[str]]:
    path = Path(nodespec_file).resolve()
    tool_schemas: Dict[str, Dict[str, Set[str]]] = {}

    def _collect_tool_schemas_from_nodes(nodes_by_var: Dict[str, Dict[str, Any]], var_order: List[str]) -> None:
        for var in var_order:
            node = nodes_by_var.get(var) or {}
            if _node_type_name(node) != "Tool":
                continue
            nm = str(node.get("name") or "").strip()
            if not nm:
                continue
            allowed: Set[str] = set()
            required: Set[str] = set()
            inputs = node.get("inputs")
            if isinstance(inputs, list):
                for ip in inputs:
                    if isinstance(ip, dict):
                        in_name = str(ip.get("name") or "").strip().lower()
                        req = bool(ip.get("required", False))
                    else:
                        in_name = str(getattr(ip, "name", "") or "").strip().lower()
                        req = bool(getattr(ip, "required", False))
                    if not in_name:
                        continue
                    allowed.add(in_name)
                    if req:
                        required.add(in_name)
            tool_schemas[nm] = {"allowed": allowed, "required": required}

    try:
        agents, tools, source = _extract_from_nodespec_utils(path)
        var_order, nodes_by_var = load_nodes_by_var(path)
        _collect_tool_schemas_from_nodes(nodes_by_var, var_order)
        return {"agents": agents, "tools": tools, "tool_schemas": tool_schemas, "source": source}
    except Exception:
        var_order, nodes_by_var = load_nodes_by_var(path)
        agents: Set[str] = set()
        tools: Set[str] = set()
        _collect_tool_schemas_from_nodes(nodes_by_var, var_order)
        for var in var_order:
            node = nodes_by_var.get(var) or {}
            name = str(node.get("name") or "").strip()
            if not name:
                continue
            t = _node_type_name(node)
            if t == "Agent":
                agents.add(name)
            elif t == "Tool":
                tools.add(name)
        return {"agents": sorted(agents), "tools": sorted(tools), "tool_schemas": tool_schemas, "source": "fallback_load_nodes_by_var"}


def _arg_keys_from_tool_call(tc: Dict[str, Any]) -> Set[str]:
    args = tc.get("arguments")
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                args = parsed
        except Exception:
            args = None
    if not isinstance(args, dict):
        return set()
    out: Set[str] = set()
    for k, v in args.items():
        kk = str(k).strip().lower()
        if not kk or kk == "value" or v is None:
            continue
        out.add(kk)
    return out


def _candidate_tools_by_args(
    arg_keys: Set[str],
    expected_tools: List[str],
    tool_schemas: Dict[str, Dict[str, Set[str]]],
) -> List[str]:
    if not expected_tools:
        return []
    if not arg_keys:
        return list(expected_tools)
    cands: List[str] = []
    for tname in expected_tools:
        schema = tool_schemas.get(tname) or {}
        allowed = set(schema.get("allowed") or set())
        required = set(schema.get("required") or set())
        if not allowed:
            continue
        if not required.issubset(arg_keys):
            continue
        if not arg_keys.issubset(allowed):
            continue
        cands.append(tname)
    return cands


def _build_norm_map(expected_names: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for name in expected_names:
        for k in _name_keys(name):
            out.setdefault(k, []).append(name)
    return out


def _resolve_expected_name(
    observed_name: str,
    expected_names: List[str],
    norm_map: Dict[str, List[str]],
    *,
    component_type: str,
) -> Tuple[str, str]:
    observed = str(observed_name or "").strip()
    if not observed:
        if len(expected_names) == 1:
            return expected_names[0], "empty_to_only_expected"
        return observed, "empty"
    obs_keys = _name_keys(observed)
    # 1) exact normalized candidate hit
    for k in obs_keys:
        cands = sorted(set(norm_map.get(k) or []))
        if len(cands) == 1:
            return cands[0], "exact_norm"
    # 1b) deterministic best-effort mapping for generic runtime labels
    # Closed-list behavior only; no free-form aliasing.
    obs_norm = _norm_name(observed)
    if component_type == "agent" and obs_norm in {
        "assistant",
        "ai",
        "model",
        "crewkickoff",
        "teamquery",
        "litellmcompletion",
    }:
        # If there is a single expected agent, bind generic runtime labels to it.
        if len(expected_names) == 1:
            return expected_names[0], "generic_runtime_to_only_expected_agent"
        return observed, "generic_runtime_ambiguous"
    if component_type == "tool" and obs_norm in {"tool", "function", "call"}:
        # Only safe when there is exactly one expected tool.
        if len(expected_names) == 1:
            return expected_names[0], "generic_runtime_to_only_tool"
    # 1c) deterministic token containment for composite runtime names.
    # Example: compute_factorial -> factorial, when unambiguous.
    if component_type == "tool":
        obs_tokens = set(_name_tokens(observed))
        if obs_tokens:
            token_matches: List[str] = []
            for exp in expected_names:
                exp_tokens = set(_name_tokens(exp))
                if not exp_tokens:
                    continue
                # expected tool token set must be fully present in observed name
                # and expected should be specific enough to avoid over-matching.
                if exp_tokens.issubset(obs_tokens) and sum(len(t) for t in exp_tokens) >= 5:
                    token_matches.append(exp)
            token_matches = sorted(set(token_matches))
            if len(token_matches) == 1:
                return token_matches[0], "token_subset_unambiguous"
    # 2) fuzzy fallback (forced best-match when expected inventory exists)
    if not obs_norm or not expected_names:
        return observed, "unresolved"
    scored: List[Tuple[float, str]] = []
    for exp in expected_names:
        exp_norm = _norm_name(exp)
        if not exp_norm:
            continue
        ratio = difflib.SequenceMatcher(None, obs_norm, exp_norm).ratio()
        scored.append((ratio, exp))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return observed, "unresolved"
    best_ratio, best_name = scored[0]
    second_ratio = scored[1][0] if len(scored) > 1 else 0.0
    if best_ratio >= 0.90 and (best_ratio - second_ratio) >= 0.06:
        return best_name, "fuzzy_high_conf"
    # Forced canonicalization policy:
    # when expected names exist, always map to the best lexical candidate
    # instead of leaving unresolved runtime labels.
    return best_name, "fuzzy_forced_best"


def _apply_deterministic_name_mapping(
    parsed: Dict[str, Any],
    expected_agents: List[str],
    expected_tools: List[str],
    tool_schemas: Dict[str, Dict[str, Set[str]]] | None = None,
) -> Dict[str, Any]:
    events = parsed.get("events") if isinstance(parsed.get("events"), list) else []
    agent_norm_map = _build_norm_map(expected_agents)
    tool_norm_map = _build_norm_map(expected_tools)
    schemas = tool_schemas or {}
    actions: List[Dict[str, Any]] = []
    pending_by_id: Dict[str, str] = {}
    pending_fifo: Deque[str] = deque()

    for e in events:
        if not isinstance(e, dict):
            continue
        et = str(e.get("type") or "").strip().lower()
        observed_name = str(e.get("name") or "").strip()
        if et in {"agent", "system"}:
            mapped, reason = _resolve_expected_name(
                observed_name,
                expected_agents,
                agent_norm_map,
                component_type="agent",
            )
            if mapped != observed_name:
                e["name"] = mapped
            actions.append(
                {
                    "seq": e.get("seq"),
                    "event_type": et,
                    "observed_name": _normalized_debug_name(observed_name),
                    "mapped_name": e.get("name"),
                    "reason": reason,
                }
            )
            tool_calls = e.get("tool_calls")
            if et == "agent" and isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    tc_obs = str(tc.get("name") or "").strip()
                    tc_args = _arg_keys_from_tool_call(tc)
                    arg_candidates = _candidate_tools_by_args(tc_args, expected_tools, schemas)
                    candidate_pool = arg_candidates if arg_candidates else expected_tools
                    tc_mapped, tc_reason = _resolve_expected_name(
                        tc_obs,
                        candidate_pool,
                        _build_norm_map(candidate_pool),
                        component_type="tool",
                    )
                    if tc_mapped != tc_obs:
                        tc["name"] = tc_mapped
                    tcid = str(tc.get("id") or tc.get("tool_id") or "").strip()
                    if tc_mapped:
                        if tcid:
                            pending_by_id[tcid] = tc_mapped
                        pending_fifo.append(tc_mapped)
                    actions.append(
                        {
                            "seq": e.get("seq"),
                            "event_type": "agent.tool_call",
                            "observed_name": _normalized_debug_name(tc_obs),
                            "mapped_name": tc.get("name"),
                            "reason": f"{tc_reason};args_candidates={len(arg_candidates)}",
                        }
                    )
        elif et == "tool":
            tcid = str(e.get("tool_call_id") or "").strip()
            mapped = ""
            reason = ""
            if tcid and tcid in pending_by_id:
                mapped = str(pending_by_id.pop(tcid) or "")
                reason = "paired_by_tool_call_id"
                try:
                    pending_fifo.remove(mapped)
                except Exception:
                    pass
            else:
                mapped, reason = _resolve_expected_name(
                    observed_name,
                    expected_tools,
                    tool_norm_map,
                    component_type="tool",
                )
                if (not mapped or _is_generic_observed_name(observed_name)) and pending_fifo:
                    mapped = str(pending_fifo.popleft() or mapped)
                    reason = "paired_by_fifo_tool_call"
            if mapped != observed_name:
                e["name"] = mapped
            actions.append(
                {
                    "seq": e.get("seq"),
                    "event_type": "tool",
                    "observed_name": _normalized_debug_name(observed_name),
                    "mapped_name": e.get("name"),
                    "reason": reason,
                }
            )

    dbg = parsed.get("debug") if isinstance(parsed.get("debug"), dict) else {}
    dbg["expected_component_names"] = {"agents": expected_agents, "tools": expected_tools}
    dbg["deterministic_name_mapping"] = actions
    parsed["debug"] = dbg
    return parsed


def _enforce_expected_name_inventory(
    parsed: Dict[str, Any],
    expected_agents: List[str],
    expected_tools: List[str],
    tool_schemas: Dict[str, Dict[str, Set[str]]] | None = None,
) -> Dict[str, Any]:
    """
    Hard guard: force all event/tool-call names onto expected inventories.
    This runs after other mapping stages and before writing parsed outputs.
    """
    events = parsed.get("events") if isinstance(parsed.get("events"), list) else []
    if not events:
        return parsed
    agent_norm_map = _build_norm_map(expected_agents)
    tool_norm_map = _build_norm_map(expected_tools)
    schemas = tool_schemas or {}
    actions: List[Dict[str, Any]] = []
    pending_by_id: Dict[str, str] = {}
    pending_fifo: Deque[str] = deque()

    for e in events:
        if not isinstance(e, dict):
            continue
        et = str(e.get("type") or "").strip().lower()
        observed_name = str(e.get("name") or "").strip()
        if et in {"agent", "system"} and expected_agents:
            mapped, reason = _resolve_expected_name(
                observed_name,
                expected_agents,
                agent_norm_map,
                component_type="agent",
            )
            if mapped:
                e["name"] = mapped
            actions.append(
                {
                    "seq": e.get("seq"),
                    "event_type": et,
                    "observed_name": _normalized_debug_name(observed_name),
                    "mapped_name": e.get("name"),
                    "reason": f"enforce_expected_inventory:{reason}",
                }
            )
            tool_calls = e.get("tool_calls")
            if isinstance(tool_calls, list) and expected_tools:
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    tc_obs = str(tc.get("name") or "").strip()
                    tc_args = _arg_keys_from_tool_call(tc)
                    arg_candidates = _candidate_tools_by_args(tc_args, expected_tools, schemas)
                    candidate_pool = arg_candidates if arg_candidates else expected_tools
                    tc_mapped, tc_reason = _resolve_expected_name(
                        tc_obs,
                        candidate_pool,
                        _build_norm_map(candidate_pool),
                        component_type="tool",
                    )
                    if tc_mapped:
                        tc["name"] = tc_mapped
                        tcid = str(tc.get("id") or tc.get("tool_id") or "").strip()
                        if tcid:
                            pending_by_id[tcid] = tc_mapped
                        pending_fifo.append(tc_mapped)
                    actions.append(
                        {
                            "seq": e.get("seq"),
                            "event_type": "agent.tool_call",
                            "observed_name": _normalized_debug_name(tc_obs),
                            "mapped_name": tc.get("name"),
                            "reason": f"enforce_expected_inventory:{tc_reason};args_candidates={len(arg_candidates)}",
                        }
                    )
        elif et == "tool" and expected_tools:
            tcid = str(e.get("tool_call_id") or "").strip()
            mapped = ""
            reason = ""
            if tcid and tcid in pending_by_id:
                mapped = str(pending_by_id.pop(tcid) or "")
                reason = "enforce_expected_inventory:paired_by_tool_call_id"
                try:
                    pending_fifo.remove(mapped)
                except Exception:
                    pass
            else:
                mapped, reason = _resolve_expected_name(
                    observed_name,
                    expected_tools,
                    tool_norm_map,
                    component_type="tool",
                )
                if (not mapped or _is_generic_observed_name(observed_name)) and pending_fifo:
                    mapped = str(pending_fifo.popleft() or mapped)
                    reason = "enforce_expected_inventory:paired_by_fifo_tool_call"
            if mapped:
                e["name"] = mapped
            actions.append(
                {
                    "seq": e.get("seq"),
                    "event_type": "tool",
                    "observed_name": _normalized_debug_name(observed_name),
                    "mapped_name": e.get("name"),
                    "reason": f"enforce_expected_inventory:{reason}",
                }
            )

    dbg = parsed.get("debug") if isinstance(parsed.get("debug"), dict) else {}
    dbg["enforce_expected_inventory"] = actions
    parsed["debug"] = dbg
    return parsed


def _load_runtime_mapping(runtime_mapping_file: Path | str) -> Dict[str, Any]:
    p = Path(runtime_mapping_file).resolve()
    return json.loads(p.read_text(encoding="utf-8"))


def _apply_runtime_alias_mapping(parsed: Dict[str, Any], runtime_mapping: Dict[str, Any]) -> Dict[str, Any]:
    events = parsed.get("events") if isinstance(parsed.get("events"), list) else []
    reg = runtime_mapping.get("alias_registry") if isinstance(runtime_mapping.get("alias_registry"), dict) else {}
    agent_map = reg.get("agent") if isinstance(reg.get("agent"), dict) else {}
    tool_map = reg.get("tool") if isinstance(reg.get("tool"), dict) else {}
    expected_agents = runtime_mapping.get("expected_agent_names")
    expected_agent_norm = {_norm_name(x): str(x) for x in (expected_agents or []) if str(x).strip()}
    actions: List[Dict[str, Any]] = []

    for e in events:
        if not isinstance(e, dict):
            continue
        et = str(e.get("type") or "").strip().lower()
        observed_name = str(e.get("name") or "").strip()
        if et in {"agent", "system"}:
            key = _norm_name(observed_name)
            if key in agent_map:
                mapped = str(agent_map.get(key) or observed_name)
                observed_is_generic = (not observed_name) or (key in {"agent", "assistant", "ai", "model"})
                mapped_norm = _norm_name(mapped)
                mapped_is_expected = mapped_norm in expected_agent_norm if expected_agent_norm else True
                if observed_is_generic and mapped_is_expected and mapped != observed_name:
                    e["name"] = mapped
                actions.append(
                    {
                        "seq": e.get("seq"),
                        "event_type": et,
                        "observed_name": _normalized_debug_name(observed_name),
                        "mapped_name": e.get("name"),
                        "reason": "runtime_alias_registry",
                    }
                )
            tool_calls = e.get("tool_calls")
            if et == "agent" and isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    tc_obs = str(tc.get("name") or "").strip()
                    tkey = _norm_name(tc_obs)
                    if tkey in tool_map:
                        tc_mapped = str(tool_map.get(tkey) or tc_obs)
                        if tc_mapped != tc_obs:
                            tc["name"] = tc_mapped
                        actions.append(
                            {
                                "seq": e.get("seq"),
                                "event_type": "agent.tool_call",
                                "observed_name": _normalized_debug_name(tc_obs),
                                "mapped_name": tc.get("name"),
                                "reason": "runtime_alias_registry",
                            }
                        )
        elif et == "tool":
            key = _norm_name(observed_name)
            if key in tool_map:
                mapped = str(tool_map.get(key) or observed_name)
                if mapped != observed_name:
                    e["name"] = mapped
                actions.append(
                    {
                        "seq": e.get("seq"),
                        "event_type": "tool",
                        "observed_name": _normalized_debug_name(observed_name),
                        "mapped_name": e.get("name"),
                        "reason": "runtime_alias_registry",
                    }
                )

    dbg = parsed.get("debug") if isinstance(parsed.get("debug"), dict) else {}
    dbg["runtime_alias_mapping"] = actions
    parsed["debug"] = dbg
    return parsed


def _map_flow_target_to_expected_tool(target: str, expected_tools: List[str]) -> str:
    tgt = str(target or "").strip()
    if not tgt or not expected_tools:
        return tgt
    tgt_norm = _norm_name(tgt)
    if not tgt_norm:
        return tgt

    # 1) exact normalized match
    for exp in expected_tools:
        if _norm_name(exp) == tgt_norm:
            return str(exp)

    # 2) common namespacing variant: functions.<target>
    pref = f"functions.{tgt}"
    pref_norm = _norm_name(pref)
    for exp in expected_tools:
        if _norm_name(exp) == pref_norm:
            return str(exp)

    # 3) unambiguous suffix token match (e.g., ...get_teams)
    tail = tgt.split(".")[-1].split("_")[-2:]
    tail_str = "".join(tail)
    if tail_str:
        cands = [exp for exp in expected_tools if tail_str in _norm_name(exp)]
        if len(cands) == 1:
            return str(cands[0])

    return tgt


def _apply_flow_link_call_target_mapping(
    parsed: Dict[str, Any],
    expected_tools: List[str] | None = None,
) -> Dict[str, Any]:
    events = parsed.get("events") if isinstance(parsed.get("events"), list) else []
    dbg = parsed.get("debug") if isinstance(parsed.get("debug"), dict) else {}
    flow_links = dbg.get("flow_links") if isinstance(dbg.get("flow_links"), list) else []

    call_id_to_target: Dict[str, str] = {}
    for link in flow_links:
        if not isinstance(link, dict):
            continue
        if str(link.get("kind") or "").strip().lower() != "tool":
            continue
        fid = str(link.get("flow_id") or "").strip()
        target = str(link.get("target") or "").strip()
        if not fid.startswith("tool:") or not target:
            continue
        cid = fid.split("tool:", 1)[-1].strip()
        if cid:
            call_id_to_target[cid] = target

    if not call_id_to_target:
        return parsed

    actions: List[Dict[str, Any]] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        et = str(e.get("type") or "").strip().lower()
        seq = e.get("seq")

        if et == "agent":
            tool_calls = e.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    cid = str(tc.get("id") or "").strip()
                    if not cid:
                        continue
                    target = call_id_to_target.get(cid)
                    if not target:
                        continue
                    mapped_target = _map_flow_target_to_expected_tool(target, expected_tools or [])
                    observed = str(tc.get("name") or "").strip()
                    if observed != mapped_target:
                        tc["name"] = mapped_target
                    actions.append(
                        {
                            "seq": seq,
                            "event_type": "agent.tool_call",
                            "call_id": cid,
                            "observed_name": _normalized_debug_name(observed),
                            "mapped_name": tc.get("name"),
                            "reason": "flow_links_call_target",
                        }
                    )
        elif et == "tool":
            cid = str(e.get("tool_call_id") or "").strip()
            if not cid:
                continue
            target = call_id_to_target.get(cid)
            if not target:
                continue
            mapped_target = _map_flow_target_to_expected_tool(target, expected_tools or [])
            observed = str(e.get("name") or "").strip()
            if observed != mapped_target:
                e["name"] = mapped_target
            actions.append(
                {
                    "seq": seq,
                    "event_type": "tool",
                    "call_id": cid,
                    "observed_name": _normalized_debug_name(observed),
                    "mapped_name": e.get("name"),
                    "reason": "flow_links_call_target",
                }
            )

    dbg = parsed.get("debug") if isinstance(parsed.get("debug"), dict) else {}
    dbg["flow_link_call_target_mapping"] = actions
    parsed["debug"] = dbg
    return parsed


def _parse_attr_value(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    s = v.strip()
    if not s:
        return v
    try:
        return json.loads(s)
    except Exception:
        s2 = s.strip('"')
        if s2 != s:
            try:
                return json.loads(s2)
            except Exception:
                return v
        return v


def _is_wrapper_like_name(name: str) -> bool:
    n = _norm_name(name)
    if not n:
        return True
    wrapper_tokens = (
        "litellm",
        "completion",
        "completions",
        "chatmodel",
        "openai",
        "azureopenai",
        "llm",
        "gpt",
        "model",
        "chain",
        "crewkickoff",
        "kickoff",
        "taskexecutesync",
    )
    return any(tok in n for tok in wrapper_tokens)


def _extract_span_identity_hints(trace_obj: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    data = trace_obj.get("data") if isinstance(trace_obj.get("data"), dict) else trace_obj
    spans = data.get("spans") if isinstance(data, dict) else None
    if not isinstance(spans, list):
        return {}

    out: Dict[str, Dict[str, str]] = {}
    parent_by_span: Dict[str, str] = {}
    for span in spans:
        if not isinstance(span, dict):
            continue
        sid = str(span.get("span_id") or "").strip()
        if not sid:
            continue
        parent_by_span[sid] = str(span.get("parent_span_id") or "").strip()

        attrs = span.get("attributes") if isinstance(span.get("attributes"), dict) else {}
        span_type = str(_parse_attr_value(attrs.get("mlflow.spanType")) or "").strip().upper()
        span_name = str(span.get("name") or "").strip()
        role_attr = str(_parse_attr_value(attrs.get("role")) or "").strip()
        agent_role_attr = str(_parse_attr_value(attrs.get("agent_role")) or "").strip()
        agent_attr = str(_parse_attr_value(attrs.get("agent")) or "").strip()

        span_inputs = _parse_attr_value(attrs.get("mlflow.spanInputs"))
        if not isinstance(span_inputs, dict):
            span_inputs = {}
        role_hint = str(span_inputs.get("agent_role") or span_inputs.get("role") or "").strip()
        tool_hint = str(span_inputs.get("tool_name") or span_inputs.get("name") or "").strip()

        hint: Dict[str, str] = {}
        if span_type == "AGENT":
            candidate = role_attr or role_hint or span_name
            if candidate and not _is_wrapper_like_name(candidate):
                hint["agent_name"] = candidate
        elif span_type == "TOOL":
            t = tool_hint or span_name
            if t and not _is_wrapper_like_name(t):
                hint["tool_name"] = t
            a = agent_role_attr or role_hint or role_attr
            if a and not _is_wrapper_like_name(a):
                hint["agent_name"] = a
        else:
            a = agent_attr or agent_role_attr or role_attr or role_hint
            if a and not _is_wrapper_like_name(a):
                hint["agent_name"] = a
            if tool_hint and not _is_wrapper_like_name(tool_hint):
                hint["tool_name"] = tool_hint

        if hint:
            out[sid] = hint

    # Propagate hints down to descendant spans (e.g., LLM transport spans under AGENT spans).
    def _walk_up(span_id: str) -> List[str]:
        chain: List[str] = []
        cur = str(span_id or "")
        seen: Set[str] = set()
        while cur and cur not in seen:
            chain.append(cur)
            seen.add(cur)
            cur = parent_by_span.get(cur) or ""
        return chain

    for sid in list(parent_by_span.keys()):
        chain = _walk_up(sid)
        merged: Dict[str, str] = dict(out.get(sid) or {})
        for anc in chain[1:]:
            ah = out.get(anc) or {}
            if not merged.get("agent_name") and str(ah.get("agent_name") or "").strip():
                merged["agent_name"] = str(ah.get("agent_name") or "").strip()
            if not merged.get("tool_name") and str(ah.get("tool_name") or "").strip():
                merged["tool_name"] = str(ah.get("tool_name") or "").strip()
            if merged.get("agent_name") and merged.get("tool_name"):
                break
        if merged:
            out[sid] = merged
    return out


def _is_generic_observed_name(observed_name: str) -> bool:
    n = _norm_name(observed_name)
    if not n:
        return True
    return n in {
        "assistant",
        "ai",
        "model",
        "litellmcompletion",
        "completions",
        "llm",
        "tool",
        "function",
        "call",
        "gpt4",
        "gpt4o",
    }


def _alias_path_priority(path: str) -> int:
    p = str(path or "").lower()
    if "spanoutputs" in p:
        return 0
    if "spaninputs" in p:
        return 1
    return 2


def _event_time_distance(event_time_ns: int, span_times: Dict[str, Tuple[int, int]], span_id: str) -> int:
    t = span_times.get(str(span_id) or "")
    if not t:
        return 10**30
    st, en = t
    if event_time_ns <= 0:
        return min(abs(st), abs(en))
    return min(abs(st - event_time_ns), abs(en - event_time_ns))


def _apply_span_identity_hints(
    parsed: Dict[str, Any],
    span_hints: Dict[str, Dict[str, str]],
    span_times: Dict[str, Tuple[int, int]],
) -> Dict[str, Any]:
    events = parsed.get("events") if isinstance(parsed.get("events"), list) else []
    actions: List[Dict[str, Any]] = []

    for e in events:
        if not isinstance(e, dict):
            continue

        et = str(e.get("type") or "").strip().lower()
        observed_name = str(e.get("name") or "").strip()
        if not _is_generic_observed_name(observed_name):
            continue

        event_time_ns = int(e.get("time_ns") or 0)
        candidate_rows: List[Tuple[str, str]] = []
        sid = str(e.get("span_id") or "").strip()
        if sid:
            candidate_rows.append((sid, str(e.get("path") or "")))
        aliases = e.get("aliases")
        if isinstance(aliases, list):
            for a in aliases:
                if not isinstance(a, dict):
                    continue
                aid = str(a.get("span_id") or "").strip()
                if aid:
                    candidate_rows.append((aid, str(a.get("path") or "")))

        # Prefer outputs first, then inputs, then other paths; within each,
        # use nearest time proximity to the event timestamp.
        dedup: List[Tuple[str, str]] = []
        seen: Set[str] = set()
        for csid, cpath in candidate_rows:
            if csid in seen:
                continue
            seen.add(csid)
            dedup.append((csid, cpath))
        dedup.sort(
            key=lambda row: (
                _alias_path_priority(row[1]),
                _event_time_distance(event_time_ns, span_times, row[0]),
            )
        )

        mapped_name = ""
        chosen_span_id = ""
        chosen_path = ""
        for csid, cpath in dedup:
            h = span_hints.get(csid)
            if not (isinstance(h, dict) and h):
                continue
            if et in {"agent", "system"}:
                cand = str(h.get("agent_name") or "").strip()
            elif et == "tool":
                cand = str(h.get("tool_name") or "").strip()
            else:
                cand = ""
            # Stop at the first non-empty, non-generic candidate.
            if cand and not _is_generic_observed_name(cand):
                mapped_name = cand
                chosen_span_id = csid
                chosen_path = cpath
                break

        if not mapped_name:
            continue
        e["name"] = mapped_name
        actions.append(
            {
                "seq": e.get("seq"),
                "event_type": et,
                "span_id": sid,
                "resolved_from_span_id": chosen_span_id,
                "resolved_from_path": chosen_path,
                "observed_name": _normalized_debug_name(observed_name),
                "mapped_name": mapped_name,
                "reason": "span_identity_hint",
            }
        )

    dbg = parsed.get("debug") if isinstance(parsed.get("debug"), dict) else {}
    dbg["span_identity_hint_mapping"] = actions
    parsed["debug"] = dbg
    return parsed


def _to_flowspec(parsed: Dict[str, Any]) -> FlowSpec:
    return _to_flowspec_with_ids(parsed, None)


def _to_flowspec_with_ids(parsed: Dict[str, Any], nodespec_file: Path | str | None) -> FlowSpec:
    events_raw = parsed.get("events") if isinstance(parsed.get("events"), list) else []
    events: List[Dict[str, Any]] = []

    GENERIC_AGENT_RUNTIME_NAMES_NORM = {
        "assistant",
        "agent",
        "ai",
        "model",
        "crewkickoff",
        "teamquery",
        "litellmcompletion",
    }

    agent_name_to_id: Dict[str, str] = {}
    id_to_name: Dict[str, str] = {}
    tool_name_to_ids: Dict[str, List[str]] = {}
    tool_inputs_by_id: Dict[str, Set[str]] = {}
    tool_required_by_id: Dict[str, Set[str]] = {}
    all_agent_ids: List[str] = []
    if nodespec_file:
        try:
            var_order, nodes_by_var = load_nodes_by_var(Path(nodespec_file).resolve())
            def _walk(node: Dict[str, Any], fallback_id: str = "") -> None:
                nm = str(node.get("name") or "").strip()
                # Mirror flow_extraction behavior: when explicit id is absent in
                # generated NodeSpec files, fall back to variable name.
                nid = str(node.get("id") or "").strip() or str(fallback_id or "").strip()
                if not nm or not nid:
                    return
                nkey = _norm_name(nm)
                nt = _node_type_name(node)
                if nt == "Agent" and nkey and nkey not in agent_name_to_id:
                    agent_name_to_id[nkey] = nid
                    all_agent_ids.append(nid)
                elif nt == "Tool" and nkey:
                    tool_name_to_ids.setdefault(nkey, []).append(nid)
                    allowed: Set[str] = set()
                    required: Set[str] = set()
                    ips = node.get("inputs") if isinstance(node.get("inputs"), list) else []
                    for ip in ips:
                        if isinstance(ip, dict):
                            in_name = str(ip.get("name") or "").strip().lower()
                            req = bool(ip.get("required", False))
                        else:
                            in_name = str(getattr(ip, "name", "") or "").strip().lower()
                            req = bool(getattr(ip, "required", False))
                        if not in_name:
                            continue
                        allowed.add(in_name)
                        if req:
                            required.add(in_name)
                    tool_inputs_by_id[nid] = allowed
                    tool_required_by_id[nid] = required
                if nt in {"Agent", "Tool"}:
                    id_to_name[nid] = nm
                for ch in node.get("nodes") if isinstance(node.get("nodes"), list) else []:
                    if isinstance(ch, dict):
                        _walk(ch, "")
                for ch in node.get("tool_list") if isinstance(node.get("tool_list"), list) else []:
                    if isinstance(ch, dict):
                        _walk(ch, "")

            for var in var_order:
                node = nodes_by_var.get(var) or {}
                if isinstance(node, dict):
                    _walk(node, str(var or ""))
        except Exception:
            pass

    default_agent_id = all_agent_ids[0] if all_agent_ids else ""

    def _resolve_agent_id(name: str) -> str:
        nn = _norm_name(name)
        if nn in agent_name_to_id:
            return agent_name_to_id[nn]
        if (not nn) or (nn in GENERIC_AGENT_RUNTIME_NAMES_NORM):
            return default_agent_id
        return ""

    def _tool_arg_keys(tc_like: Dict[str, Any]) -> Set[str]:
        args = tc_like.get("arguments")
        if isinstance(args, str):
            try:
                parsed_args = json.loads(args)
                if isinstance(parsed_args, dict):
                    args = parsed_args
            except Exception:
                args = None
        if not isinstance(args, dict):
            return set()
        out: Set[str] = set()
        for k, v in args.items():
            kk = str(k or "").strip().lower()
            if not kk or kk == "value" or v is None:
                continue
            out.add(kk)
        return out

    def _resolve_tool_id(name: str, tc_like: Dict[str, Any] | None = None) -> str:
        ids = tool_name_to_ids.get(_norm_name(name), [])
        if not ids:
            return ""
        if len(ids) == 1:
            return ids[0]
        if not tc_like:
            return ids[0]
        arg_keys = _tool_arg_keys(tc_like)
        if not arg_keys:
            return ids[0]
        # Prefer required ⊆ args ⊆ allowed.
        candidates: List[str] = []
        for tid in ids:
            allowed = tool_inputs_by_id.get(tid, set())
            required = tool_required_by_id.get(tid, set())
            if allowed and required.issubset(arg_keys) and arg_keys.issubset(allowed):
                candidates.append(tid)
        return candidates[0] if candidates else ids[0]

    invoked_tools: List[str] = []
    invoked_agents: List[str] = []
    pending_tool_by_call_id: Dict[str, Tuple[str, str]] = {}
    pending_tool_fifo_by_name: Dict[str, Deque[Tuple[str, str]]] = {}
    first_agent_id = ""
    last_agent_id = ""

    for ev in events_raw:
        if not isinstance(ev, dict):
            continue
        row = dict(ev)
        et = str(row.get("type") or "").strip().lower()
        nm = str(row.get("name") or "").strip()
        node_id = ""

        if et in {"agent", "system"}:
            node_id = _resolve_agent_id(nm)
            if node_id and not first_agent_id:
                first_agent_id = node_id
            if node_id:
                last_agent_id = node_id
            tcs = row.get("tool_calls") if isinstance(row.get("tool_calls"), list) else []
            out_tcs: List[Dict[str, Any]] = []
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                tc_row = dict(tc)
                tc_name = str(tc_row.get("name") or "").strip()
                tcid = str(tc_row.get("id") or tc_row.get("tool_id") or "").strip()
                tool_id = _resolve_tool_id(tc_name, tc_row)
                if tool_id:
                    tc_row["tool_id"] = tool_id
                    if tcid:
                        pending_tool_by_call_id[tcid] = (tool_id, node_id)
                    q = pending_tool_fifo_by_name.setdefault(tc_name, deque())
                    q.append((tool_id, node_id))
                out_tcs.append(tc_row)
            row["tool_calls"] = out_tcs
            if node_id:
                invoked_agents.append(node_id)
        elif et == "tool":
            tcid = str(row.get("tool_call_id") or "").strip()
            parent_agent_id = ""
            if tcid and tcid in pending_tool_by_call_id:
                node_id, parent_agent_id = pending_tool_by_call_id.pop(tcid)
            else:
                q = pending_tool_fifo_by_name.get(nm)
                if q:
                    node_id, parent_agent_id = q.popleft()
            if not node_id:
                node_id = _resolve_tool_id(nm, row)
            if parent_agent_id:
                last_agent_id = parent_agent_id
            if node_id:
                invoked_tools.append(node_id)
        elif et == "error":
            node_id = last_agent_id or first_agent_id or default_agent_id

        row["node_id"] = node_id
        resolved_name = str(id_to_name.get(node_id) or "").strip()
        if et in {"agent", "tool"} and resolved_name:
            row["name"] = resolved_name
        events.append(row)

    return FlowSpec(
        flow_id=0,
        flow_index=0,
        flow_signature_ordered=list(parsed.get("flow_signature_ordered") or []),
        flow_signature_key=str(parsed.get("flow_signature_key") or ""),
        flow_signature_id=str(parsed.get("flow_signature_id") or ""),
        flow_unique_components=list(parsed.get("flow_unique_components") or []),
        parsed_trace_file=str(parsed.get("parsed_trace_file") or ""),
        source_trace_file=str(parsed.get("source_trace_file") or parsed.get("trace_file") or ""),
        input_args=dict(parsed.get("input_args") or {}),
        events=events,
        invoked_tools=list(dict.fromkeys(invoked_tools)),
        invoked_agents=list(dict.fromkeys(invoked_agents)),
    )


def parse_trace_file(
    trace_file: Path | str,
    nodespec_file: Path | str | None = None,
    runtime_mapping_file: Path | str | None = None,
) -> FlowSpec:
    trace_path = Path(trace_file).resolve()
    schema = build_runtime_schema()
    parsed = parse_trace_with_schema(trace_path, schema)
    raw_trace_obj = json.loads(trace_path.read_text(encoding="utf-8"))
    span_hints = _extract_span_identity_hints(raw_trace_obj)
    span_times: Dict[str, Tuple[int, int]] = {}
    spans = raw_trace_obj.get("spans")
    if isinstance(spans, list):
        for s in spans:
            if not isinstance(s, dict):
                continue
            sid = str(s.get("span_id") or "").strip()
            if not sid:
                continue
            st = int(s.get("start_time_unix_nano") or 0)
            en = int(s.get("end_time_unix_nano") or 0)
            span_times[sid] = (st, en)
    if span_hints:
        parsed = _apply_span_identity_hints(parsed=parsed, span_hints=span_hints, span_times=span_times)
    if runtime_mapping_file:
        runtime_mapping = _load_runtime_mapping(runtime_mapping_file)
        parsed = _apply_runtime_alias_mapping(parsed=parsed, runtime_mapping=runtime_mapping)
    if nodespec_file:
        expected = load_expected_names_from_nodespec(nodespec_file)
        parsed = _apply_deterministic_name_mapping(
            parsed=parsed,
            expected_agents=expected.get("agents") or [],
            expected_tools=expected.get("tools") or [],
            tool_schemas=expected.get("tool_schemas") if isinstance(expected.get("tool_schemas"), dict) else {},
        )
        parsed = _enforce_expected_name_inventory(
            parsed=parsed,
            expected_agents=expected.get("agents") or [],
            expected_tools=expected.get("tools") or [],
            tool_schemas=expected.get("tool_schemas") if isinstance(expected.get("tool_schemas"), dict) else {},
        )
        dbg = parsed.get("debug") if isinstance(parsed.get("debug"), dict) else {}
        dbg["expected_component_source"] = str(expected.get("source") or "")
        parsed["debug"] = dbg
    return _to_flowspec_with_ids(parsed, nodespec_file)


def save_parsed_outputs(
    *,
    parsed: FlowSpec | Dict[str, Any],
    parsed_out_file: Path | str,
    debug_out_file: Path | str,
) -> None:
    parsed_path = Path(parsed_out_file).resolve()
    debug_path = Path(debug_out_file).resolve()
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    parsed_obj = parsed.model_dump(mode="json", exclude_none=False) if isinstance(parsed, FlowSpec) else parsed
    parsed_path.write_text(json.dumps(parsed_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    debug_obj = {
        "trace_file": parsed_obj.get("source_trace_file"),
        "schema_use_case": "",
        "debug": {},
    }
    debug_path.write_text(json.dumps(debug_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_flow_output(
    *,
    flow: FlowSpec | Dict[str, Any],
    flow_out_file: Path | str,
) -> None:
    out_path = Path(flow_out_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flow_obj = flow.model_dump(mode="json", exclude_none=False) if isinstance(flow, FlowSpec) else flow
    out_path.write_text(json.dumps(flow_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_single_trace(
    trace_file: Path | str,
    *,
    nodespec_file: Path | str | None = None,
    runtime_mapping_file: Path | str | None = None,
    parsed_out_file: Path | str | None = None,
    debug_out_file: Path | str | None = None,
) -> FlowSpec | Path:
    parsed = parse_trace_file(
        trace_file=trace_file,
        nodespec_file=nodespec_file,
        runtime_mapping_file=runtime_mapping_file,
    )
    if parsed_out_file is None:
        return parsed
    out_path = Path(parsed_out_file).resolve()
    save_flow_output(flow=parsed, flow_out_file=out_path)
    if debug_out_file:
        dbg_path = Path(debug_out_file).resolve()
        dbg_path.parent.mkdir(parents=True, exist_ok=True)
        dbg_obj = {
            "trace_file": parsed.source_trace_file,
            "schema_use_case": "",
            "debug": {},
        }
        dbg_path.write_text(json.dumps(dbg_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path
