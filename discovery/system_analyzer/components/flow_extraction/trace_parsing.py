from __future__ import annotations

import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
    from .global_utils import load_nodes_by_var
    from .parser_core import (
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
except ImportError:
    from global_utils import load_nodes_by_var
    from flow_extraction.parser_core import (
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

    # choose richest root deterministically
    best = max(
        roots,
        key=lambda r: (
            len(r.list_agents() if callable(getattr(r, "list_agents", None)) else []),
            len(r.list_tools() if callable(getattr(r, "list_tools", None)) else []),
        ),
    )
    agent_rows = best.list_agents() if callable(getattr(best, "list_agents", None)) else []
    tool_rows = best.list_tools() if callable(getattr(best, "list_tools", None)) else []
    agents = sorted({str(x.get("name") or "").strip() for x in agent_rows if isinstance(x, dict) and str(x.get("name") or "").strip()})
    tools = sorted({str(x.get("name") or "").strip() for x in tool_rows if isinstance(x, dict) and str(x.get("name") or "").strip()})
    return agents, tools, "nodespec_utility_methods"


def _extract_expected_names_from_object(node_spec: Any) -> Dict[str, List[str]]:
    agents = node_spec.list_agents()
    tools = node_spec.list_tools()
    tool_list = [str(tool.get("name") or "").strip() for tool in tools]
    agent_list = [str(agent.get("name") or "").strip() for agent in agents]
    return {"agents": sorted(agent_list), "tools": sorted(tool_list), "source": "nodespec_object"}


def load_expected_names_from_nodespec(
    nodespec_file: Path | str | None = None,
    *,
    node_spec: Any = None,
) -> Dict[str, List[str]]:
    if node_spec is not None:
        return _extract_expected_names_from_object(node_spec)
    if nodespec_file is None:
        raise ValueError("Either `node_spec` or `nodespec_file` must be provided")
    path = Path(nodespec_file).resolve()
    try:
        agents, tools, source = _extract_from_nodespec_utils(path)
        return {"agents": agents, "tools": tools, "source": source}
    except Exception:
        var_order, nodes_by_var = load_nodes_by_var(path)
        agents: Set[str] = set()
        tools: Set[str] = set()
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
        return {"agents": sorted(agents), "tools": sorted(tools), "source": "fallback_load_nodes_by_var"}


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
    # 2) strict fuzzy fallback
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
    return observed, "unresolved"


def _apply_deterministic_name_mapping(parsed: Dict[str, Any], expected_agents: List[str], expected_tools: List[str]) -> Dict[str, Any]:
    events = parsed.get("events") if isinstance(parsed.get("events"), list) else []
    agent_norm_map = _build_norm_map(expected_agents)
    tool_norm_map = _build_norm_map(expected_tools)
    actions: List[Dict[str, Any]] = []

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
                    tc_mapped, tc_reason = _resolve_expected_name(
                        tc_obs,
                        expected_tools,
                        tool_norm_map,
                        component_type="tool",
                    )
                    if tc_mapped != tc_obs:
                        tc["name"] = tc_mapped
                    actions.append(
                        {
                            "seq": e.get("seq"),
                            "event_type": "agent.tool_call",
                            "observed_name": _normalized_debug_name(tc_obs),
                            "mapped_name": tc.get("name"),
                            "reason": tc_reason,
                        }
                    )
        elif et == "tool":
            mapped, reason = _resolve_expected_name(
                observed_name,
                expected_tools,
                tool_norm_map,
                component_type="tool",
            )
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


def _load_runtime_mapping(runtime_mapping_file: Path | str) -> Dict[str, Any]:
    p = Path(runtime_mapping_file).resolve()
    return json.loads(p.read_text(encoding="utf-8"))


def _apply_runtime_alias_mapping(parsed: Dict[str, Any], runtime_mapping: Dict[str, Any]) -> Dict[str, Any]:
    events = parsed.get("events") if isinstance(parsed.get("events"), list) else []
    reg = runtime_mapping.get("alias_registry") if isinstance(runtime_mapping.get("alias_registry"), dict) else {}
    agent_map = reg.get("agent") if isinstance(reg.get("agent"), dict) else {}
    tool_map = reg.get("tool") if isinstance(reg.get("tool"), dict) else {}
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
                if mapped != observed_name:
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


def parse_trace_file(
    trace_file: Path | str,
    nodespec_file: Path | str | None = None,
    node_spec: Any = None,
    runtime_mapping_file: Path | str | None = None,
) -> Dict[str, Any]:
    trace_path = Path(trace_file).resolve()
    schema = build_runtime_schema()
    parsed = parse_trace_with_schema(trace_path, schema)
    if runtime_mapping_file:
        runtime_mapping = _load_runtime_mapping(runtime_mapping_file)
        parsed = _apply_runtime_alias_mapping(parsed=parsed, runtime_mapping=runtime_mapping)
    if node_spec is not None or nodespec_file:
        expected = load_expected_names_from_nodespec(nodespec_file, node_spec=node_spec)
        parsed = _apply_deterministic_name_mapping(
            parsed=parsed,
            expected_agents=expected.get("agents") or [],
            expected_tools=expected.get("tools") or [],
        )
        dbg = parsed.get("debug") if isinstance(parsed.get("debug"), dict) else {}
        dbg["expected_component_source"] = str(expected.get("source") or "")
        parsed["debug"] = dbg
    return parsed


def save_parsed_outputs(
    *,
    parsed: Dict[str, Any],
    parsed_out_file: Path | str,
    debug_out_file: Path | str,
) -> None:
    parsed_path = Path(parsed_out_file).resolve()
    debug_path = Path(debug_out_file).resolve()
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    parsed_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    debug_obj = {
        "trace_file": parsed.get("trace_file"),
        "schema_use_case": parsed.get("schema_use_case"),
        "debug": parsed.get("debug") if isinstance(parsed.get("debug"), dict) else {},
    }
    debug_path.write_text(json.dumps(debug_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
