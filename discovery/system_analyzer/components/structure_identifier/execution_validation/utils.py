from __future__ import annotations

import json
import os
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

try:
    from . import trace_extraction
    from .parser_core import (
        GENERIC_AGENT_RUNTIME_NAMES_NORM,
        GENERIC_MESSAGE_LIST_PATHS,
        TOP_DEFAULT_CONTENT_KEYS,
        TOP_DEFAULT_NAME_KEYS,
        TOP_DEFAULT_ROLE_KEYS,
        TOP_DEFAULT_TIMESTAMP_KEYS,
        TOP_DEFAULT_TOOL_CALL_KEYS,
        TOP_DEFAULT_TOOL_RESULT_KEYS,
        GENERIC_TOOL_RUNTIME_NAMES_NORM,
        Schema,
        parse_trace_with_schema,
    )
    from .prompts import (
        build_inventory_parse_payload,
        build_inventory_name_mapping_payload,
        build_name_mapping_payload,
        build_trace_alias_resolution_payload,
    )
    from ..global_utils import (
        add_child_ref,
        build_parent_child_maps,
        call_with_cache_text,
        next_var,
        node_type_name,
        norm,
        parse_json_loose,
        load_nodes_by_var,
        write_nodes_output,
    )
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    CLEAN_CODE_ROOT = Path(__file__).resolve().parents[1]
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    if str(CLEAN_CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CLEAN_CODE_ROOT))
    import trace_extraction
    from parser_core import (
        GENERIC_AGENT_RUNTIME_NAMES_NORM,
        GENERIC_MESSAGE_LIST_PATHS,
        TOP_DEFAULT_CONTENT_KEYS,
        TOP_DEFAULT_NAME_KEYS,
        TOP_DEFAULT_ROLE_KEYS,
        TOP_DEFAULT_TIMESTAMP_KEYS,
        TOP_DEFAULT_TOOL_CALL_KEYS,
        TOP_DEFAULT_TOOL_RESULT_KEYS,
        GENERIC_TOOL_RUNTIME_NAMES_NORM,
        Schema,
        parse_trace_with_schema,
    )
    from prompts import (
        build_inventory_parse_payload,
        build_inventory_name_mapping_payload,
        build_name_mapping_payload,
        build_trace_alias_resolution_payload,
    )
    from global_utils import (
        add_child_ref,
        build_parent_child_maps,
        call_with_cache_text,
        next_var,
        node_type_name,
        norm,
        parse_json_loose,
        load_nodes_by_var,
        write_nodes_output,
    )


SERVER_TYPES = {"Local_MCP_server", "External_MCP_server"}


@dataclass(frozen=True)
class ToolOwnerPair:
    """One expected tool together with the component that owns it."""

    owner_name: str
    owner_kind: str  # agent|server
    tool_name: str

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.owner_kind, self.owner_name, self.tool_name)


@dataclass
class ExpectedComponents:
    """The expected agents, servers, and tool ownership relations from the NodeSpec."""

    agents: Set[str]
    servers: Set[str]
    tools_by_owner: List[ToolOwnerPair]
    server_to_ancestor_agents: Dict[str, Set[str]]
    server_to_ancestor_servers: Dict[str, Set[str]]


@dataclass
class ObservedComponents:
    """The components and tool activity observed in one parsed runtime trace."""

    actor_names: Set[str]
    agents: Set[str]
    tools_by_owner: Set[Tuple[str, str]]  # (owner, tool)
    tool_names_any: Set[str]
    tool_io_pairs: List[Dict[str, Any]]
    inventory_tools: List[Dict[str, Any]]


def merge_observed_components(
    *,
    observed: ObservedComponents,
    observed_agents_all: Set[str],
    observed_actor_names_all: Set[str],
    observed_tools_by_owner_all: Set[Tuple[str, str]],
    observed_tool_names_all: Set[str],
    observed_tool_io_all: List[Dict[str, Any]],
    observed_inventory_tools_all: List[Dict[str, Any]],
) -> None:
    """Merge one attempt's observed component summary into the run-level aggregate collections."""

    observed_agents_all.update(observed.agents)
    observed_actor_names_all.update(observed.actor_names)
    observed_tools_by_owner_all.update(observed.tools_by_owner)
    observed_tool_names_all.update(observed.tool_names_any)
    observed_tool_io_all.extend(observed.tool_io_pairs)
    observed_inventory_tools_all.extend(observed.inventory_tools)


def observed_snapshot(observed: ObservedComponents) -> Dict[str, Any]:
    # Return the processed attempt payload used by the caller and the final report.
    return {
        "agents": sorted(observed.agents),
        "actor_names": sorted(observed.actor_names),
        "tools_by_owner": [{"owner": o, "tool": t} for (o, t) in sorted(observed.tools_by_owner)],
        "tool_names_any": sorted(observed.tool_names_any),
    }


def _norm_name(s: str) -> str:
    """Normalize a name for loose matching by lowercasing and keeping only alphanumerics."""
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def _is_generic_runtime_name(name: str, component_type: str) -> bool:
    """Return whether a runtime name is too generic to trust as a component identity."""
    n = _norm_name(name)
    if not n:
        return True
    if component_type == "agent":
        return n in GENERIC_AGENT_RUNTIME_NAMES_NORM
    if component_type == "tool":
        return n in GENERIC_TOOL_RUNTIME_NAMES_NORM
    return False


def _norm_name_candidates(name: str) -> List[str]:
    """
    Deterministic lookup keys for names that may carry dotted wrappers/prefixes.
    Example: functions.code_interpreter -> ["functionscodeinterpreter", "codeinterpreter"].
    """
    raw = str(name or "").strip()
    if not raw:
        return []
    keys: List[str] = []
    seen: Set[str] = set()

    def _add(v: str) -> None:
        nv = _norm_name(v)
        if nv and nv not in seen:
            seen.add(nv)
            keys.append(nv)

    _add(raw)
    if "." in raw:
        _add(raw.split(".", 1)[-1])   # remove first dotted prefix
        _add(raw.rsplit(".", 1)[-1])  # last segment
    return keys


def _deterministic_map_name(
    *,
    observed_name: str,
    expected_by_norm: Dict[str, str],
    alias_map: Dict[str, str],
) -> Tuple[str, str]:
    """
    Returns (mapped_name, reason) or ("", "") if no deterministic match.
    """
    for key in _norm_name_candidates(observed_name):
        aliased = alias_map.get(key, "")
        if aliased and _norm_name(aliased) in expected_by_norm:
            return expected_by_norm[_norm_name(aliased)], "deterministic_global_alias_map"
    for key in _norm_name_candidates(observed_name):
        if key in expected_by_norm:
            return expected_by_norm[key], "deterministic_exact_norm_match"
    return "", ""


def _truncate_text(value: Any, max_chars: int) -> str:
    """Convert a value to text and cap it to at most `max_chars` characters."""
    s = str(value or "")
    if len(s) <= max_chars:
        return s
    return s[:max_chars]


def _cap_tool_calls(tool_calls: Any, max_items: int = 20) -> List[Dict[str, Any]]:
    """Return at most `max_items` normalized tool-call dicts from a parsed tool-call list."""
    if not isinstance(tool_calls, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in tool_calls[:max_items]:
        if not isinstance(item, dict):
            continue
        out.append(item)
    return out


def _event_excerpt(ev: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact event summary used as evidence in mapping prompts and logs."""
    if not isinstance(ev, dict):
        return {}
    return {
        "seq": ev.get("seq"),
        "type": ev.get("type"),
        "name": ev.get("name"),
        "content_excerpt": str(ev.get("content") or "")[:320],
        "tool_calls_count": len(ev.get("tool_calls") or []) if isinstance(ev.get("tool_calls"), list) else 0,
        "span_id": ev.get("span_id"),
        "path": ev.get("path"),
    }


def map_agent_tool_events_to_expected(
    *,
    events: List[Dict[str, Any]],
    expected_agent_names: List[str],
    expected_tool_names: List[str],
    model: str,
    refresh_raw: bool,
    raw_dir: Path,
    use_llm: bool = True,
    global_agent_alias_by_norm: Dict[str, str] | None = None,
    global_tool_alias_by_norm: Dict[str, str] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Map parsed agent/tool event names onto expected component names and record mapping decisions."""

    # Prepare mutable outputs plus lightweight caller context used in LLM mapping prompts.
    mapped_events: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []
    events_by_seq = {ev.get("seq"): ev for ev in events if isinstance(ev, dict)}
    owner_by_seq: Dict[Any, str] = {}
    last_actor = ""
    for ev in events:
        if not isinstance(ev, dict):
            continue
        et = str(ev.get("type") or "").strip().lower()
        name = str(ev.get("name") or "").strip()
        owner_by_seq[ev.get("seq")] = last_actor
        if et in {"agent", "system"} and name:
            last_actor = name

    # Build deterministic lookup tables and reuse any previously learned alias mappings.
    expected_agent_by_norm = {_norm_name(x): x for x in expected_agent_names if x}
    expected_tool_by_norm = {_norm_name(x): x for x in expected_tool_names if x}
    global_agent_alias_by_norm = global_agent_alias_by_norm or {}
    global_tool_alias_by_norm = global_tool_alias_by_norm or {}

    map_dir = raw_dir / "name_mapping_expected"
    map_dir.mkdir(parents=True, exist_ok=True)

    # Process each parsed event and only remap agent/tool events.
    for idx, ev in enumerate(events):
        e = dict(ev)
        et = str(e.get("type") or "").strip().lower()
        if et not in {"agent", "tool"}:
            mapped_events.append(e)
            continue

        observed_name = str(e.get("name") or "").strip()
        expected_names = expected_agent_names if et == "agent" else expected_tool_names
        expected_by_norm = expected_agent_by_norm if et == "agent" else expected_tool_by_norm

        observed_is_generic = _is_generic_runtime_name(observed_name, et)
        alias_map = global_agent_alias_by_norm if et == "agent" else global_tool_alias_by_norm
        canonical_name, deterministic_reason = _deterministic_map_name(
            observed_name=observed_name,
            expected_by_norm=expected_by_norm,
            alias_map=alias_map,
        )
        # First try deterministic exact/alias-based mapping before asking the model.
        if canonical_name and not _is_generic_runtime_name(canonical_name, et):
            det_alias_scope = (
                "generic_runtime"
                if deterministic_reason == "deterministic_exact_norm_match" and observed_is_generic
                else "specific"
            )
            e["name"] = canonical_name
            # Record the deterministic mapping decision and keep the renamed event.
            actions.append(
                {
                    "event_index": idx,
                    "event_seq": e.get("seq"),
                    "event_type": et,
                    "observed_name": observed_name,
                    "decision": "map_to_expected",
                    "mapped_name": canonical_name,
                    "reason": deterministic_reason,
                    "alias_scope": det_alias_scope,
                }
            )
            mapped_events.append(e)
            continue

        # If deterministic mapping fails, optionally ask the model to decide whether this is a known component.
        if use_llm:

            # Build local context around this event so the model can judge the intended component identity.
            seq = e.get("seq")
            prev_ev = events_by_seq.get((int(seq) - 1) if isinstance(seq, int) else None, {})
            next_ev = events_by_seq.get((int(seq) + 1) if isinstance(seq, int) else None, {})
            payload = build_name_mapping_payload(
                component_type=et,
                observed_name=observed_name,
                event_evidence={
                    "type": e.get("type"),
                    "content": _truncate_text(e.get("content"), 5000),
                    "tool_calls": _cap_tool_calls(e.get("tool_calls")),
                    "span_id": e.get("span_id"),
                    "path": e.get("path"),
                    "aliases": e.get("aliases"),
                    "caller_name": owner_by_seq.get(e.get("seq"), ""),
                    "previous_event": _event_excerpt(prev_ev),
                    "next_event": _event_excerpt(next_ev),
                },
                expected_names=expected_names,
            )
            # Run the name-mapping prompt and parse the model decision.
            # Run the cached parse prompt once before deciding whether a stricter retry is needed.
            # Run the inventory-specific mapping prompt for this one extracted tool row.
            text, _source = call_with_cache_text(
                payload,
                payload_path=map_dir / f"event_{idx:04d}.payload.json",
                raw_path=map_dir / f"event_{idx:04d}.raw.txt",
                model=model,
                refresh_raw=bool(refresh_raw),
                retriever=None,
            )
            obj = parse_json_loose(text)
            # Read the model decision and treat malformed output as a failed mapping.
            decision = str(obj.get("decision") or "").strip() if isinstance(obj, dict) else ""
            mapped_name = str(obj.get("mapped_name") or "").strip() if isinstance(obj, dict) else ""
            reason = str(obj.get("reason") or "").strip() if isinstance(obj, dict) else "invalid_mapper_output"
            alias_scope = "specific"

            # Only accept the model result when it resolves to one allowed expected name.
            if decision == "map_to_expected" and _norm_name(mapped_name) in expected_by_norm:
                e["name"] = expected_by_norm[_norm_name(mapped_name)]
            else:
                decision = "new_component"
                mapped_name = ""
                alias_scope = ""
        else:
            # Without LLM mapping, leave unresolved names as new-component candidates.
            decision = "new_component"
            mapped_name = ""
            reason = "report_only_no_llm_mapping"
            alias_scope = ""

        # Persist the final mapping decision for downstream alias handling and reporting.
        actions.append(
            {
                "event_index": idx,
                "event_seq": e.get("seq"),
                "event_type": et,
                "observed_name": observed_name,
                "decision": decision,
                "mapped_name": mapped_name,
                "reason": reason,
                "alias_scope": alias_scope,
            }
        )
        mapped_events.append(e)

    # Return both the remapped event stream and the per-event mapping decisions.
    return mapped_events, actions


def resolve_single_name_per_trace(
    *,
    trace_id: str,
    events: List[Dict[str, Any]],
    mapping_actions: List[Dict[str, Any]],
    model: str,
    refresh_raw: bool,
    raw_dir: Path,
    use_llm: bool = True,
) -> List[Dict[str, Any]]:
    """Choose one final trace-visible alias for each expected agent/tool within a single trace."""

    # Collect all observed aliases that were mapped to each expected component.
    by_key: Dict[Tuple[str, str], Set[str]] = {}
    seen_expected_keys: Set[Tuple[str, str]] = set()
    for action in mapping_actions:
        if not isinstance(action, dict):
            continue
        if str(action.get("decision") or "").strip() != "map_to_expected":
            continue

        event_type = str(action.get("event_type") or "").strip().lower()
        expected_name = str(action.get("mapped_name") or "").strip()
        observed_name = str(action.get("observed_name") or "").strip()
        alias_scope = str(action.get("alias_scope") or "").strip().lower()

        if event_type not in {"agent", "tool"} or not expected_name:
            continue

        seen_expected_keys.add((event_type, expected_name))

        if alias_scope != "specific":
            continue
        if not observed_name:
            continue

        by_key.setdefault((event_type, expected_name), set()).add(observed_name)

    # Ensure the canonical expected name is always available as a fallback alias candidate.
    for key in seen_expected_keys:
        by_key.setdefault(key, set()).add(key[1])

    # Build lightweight event lookup and resolve one chosen alias per expected component.
    by_seq = {ev.get("seq"): ev for ev in events if isinstance(ev, dict)}
    out: List[Dict[str, Any]] = []
    for (et, expected_name), aliases in sorted(by_key.items(), key=lambda x: (x[0][0], x[0][1])):
        alias_list = sorted([a for a in aliases if a])
        if not alias_list:
            continue
        if len(alias_list) == 1:
            out.append(
                {
                    "trace_id": trace_id,
                    "component_type": et,
                    "expected_name": expected_name,
                    "chosen_trace_name": alias_list[0],
                    "aliases": alias_list,
                    "reason": "single_alias",
                    "source": "deterministic",
                }
            )
            continue

        # Gather a small amount of local trace evidence for each competing alias.
        evidence: List[Dict[str, Any]] = []
        for alias in alias_list:
            count = 0
            for a in mapping_actions:
                if not isinstance(a, dict):
                    continue
                if str(a.get("event_type") or "").strip().lower() != et:
                    continue
                if str(a.get("mapped_name") or "").strip() != expected_name:
                    continue
                if str(a.get("observed_name") or "").strip() != alias:
                    continue
                seq = a.get("event_seq")
                ev = by_seq.get(seq, {})
                evidence.append(
                    {
                        "alias": alias,
                        "seq": seq,
                        "type": ev.get("type"),
                        "content_excerpt": str(ev.get("content") or "")[:320],
                        "tool_calls_count": len(ev.get("tool_calls") or []),
                    }
                )
                count += 1
                if count >= 2:
                    break

        # Default deterministically, then optionally ask the model to pick the best alias.
        chosen = alias_list[0]
        reason = "deterministic_first_sorted_alias"
        source = "deterministic"
        if use_llm:
            payload = build_trace_alias_resolution_payload(
                component_type=et,
                expected_name=expected_name,
                alias_candidates=alias_list,
                evidence=evidence,
            )
            text, _src = call_with_cache_text(
                payload,
                payload_path=raw_dir / f"{trace_id}.{et}.{_norm_name(expected_name)}.payload.json",
                raw_path=raw_dir / f"{trace_id}.{et}.{_norm_name(expected_name)}.raw.txt",
                model=model,
                refresh_raw=bool(refresh_raw),
                retriever=None,
            )
            obj = parse_json_loose(text)
            candidate = str(obj.get("chosen_name") or "").strip() if isinstance(obj, dict) else ""
            if candidate in alias_list:
                chosen = candidate
                reason = str(obj.get("reason") or "").strip() if isinstance(obj, dict) else "llm_selected"
                source = "llm"

        out.append(
            {
                "trace_id": trace_id,
                "component_type": et,
                "expected_name": expected_name,
                "chosen_trace_name": chosen,
                "aliases": alias_list,
                "reason": reason,
                "source": source,
            }
        )
    # Return one chosen trace-facing name row per expected component seen in this trace.
    return out


def extract_tool_calls(ev: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the event's tool calls as a clean list of dict payloads."""
    raw = ev.get("tool_calls")
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for it in raw:
        if isinstance(it, dict):
            out.append(it)
    return out


def observe_components(events: List[Dict[str, Any]]) -> ObservedComponents:
    """Scan mapped events and derive the observed agents, tools, and tool IO pairs for one trace."""
    # Accumulate the component-level view that execution validation uses downstream.
    actor_names: Set[str] = set()
    agents: Set[str] = set()
    tools_by_owner: Set[Tuple[str, str]] = set()
    tool_names_any: Set[str] = set()
    tool_io_pairs: List[Dict[str, Any]] = []

    # Track the latest active agent plus unresolved tool calls waiting for tool-result events.
    last_agent_name = ""
    pending_tool_calls_by_id: Dict[str, Dict[str, Any]] = {}
    pending_tool_calls_by_name: Dict[str, List[Dict[str, Any]]] = {}

    # Walk the event stream once to infer agent presence, tool ownership, and completed tool IO pairs.
    for ev in events:
        et = str(ev.get("type") or "").strip().lower()
        name = str(ev.get("name") or "").strip()
        if name:
            actor_names.add(name)

        if et == "agent" and name:
            last_agent_name = name
            agents.add(name)

        # Agent-side tool calls open pending records but do not emit IO pairs yet.
        calls = extract_tool_calls(ev)
        if calls:
            owner = name if (et == "agent" and name) else last_agent_name
            for c in calls:
                tname = str(c.get("name") or "").strip()
                if not tname:
                    continue
                tool_names_any.add(tname)
                if owner:
                    tools_by_owner.add((owner, tname))
                pending_row = {
                    "owner": owner,
                    "tool": tname,
                    "input": c.get("arguments") if isinstance(c.get("arguments"), dict) else c,
                    "event_seq": ev.get("seq"),
                }
                tcid = str(c.get("id") or c.get("tool_call_id") or c.get("call_id") or "").strip()
                if tcid:
                    pending_tool_calls_by_id[tcid] = pending_row
                pending_tool_calls_by_name.setdefault(norm(tname), []).append(pending_row)

        # Tool-result events close one pending call and only then emit a final IO pair.
        if et == "tool" and name:
            tool_names_any.add(name)
            pending_row = None
            tcid = str(ev.get("tool_call_id") or "").strip()
            if tcid:
                pending_row = pending_tool_calls_by_id.pop(tcid, None)
                if pending_row is not None:
                    pending_name = norm(str(pending_row.get("tool") or ""))
                    bucket = pending_tool_calls_by_name.get(pending_name, [])
                    if pending_row in bucket:
                        bucket.remove(pending_row)
                    if not bucket:
                        pending_tool_calls_by_name.pop(pending_name, None)
            if pending_row is None:
                bucket = pending_tool_calls_by_name.get(norm(name), [])
                if bucket:
                    pending_row = bucket.pop(0)
                    if not bucket:
                        pending_tool_calls_by_name.pop(norm(name), None)
                    for pending_id, row in list(pending_tool_calls_by_id.items()):
                        if row is pending_row:
                            pending_tool_calls_by_id.pop(pending_id, None)
                            break
            if pending_row:
                owner = str(pending_row.get("owner") or "")
                if owner:
                    tools_by_owner.add((owner, name))
                tool_io_pairs.append(
                    {
                        "owner": owner,
                        "tool": str(pending_row.get("tool") or name),
                        "input": pending_row.get("input"),
                        "output": ev.get("content"),
                        "event_seq": pending_row.get("event_seq"),
                        "tool_event_seq": ev.get("seq"),
                    }
                )

    # Deduplicate repeated tool IO rows before returning the observed summary.
    seen = set()
    dedup_pairs: List[Dict[str, Any]] = []
    for row in tool_io_pairs:
        key = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
        if key in seen:
            continue
        seen.add(key)
        dedup_pairs.append(row)

    # Return the normalized observed component snapshot for this trace.
    return ObservedComponents(
        actor_names=actor_names,
        agents=agents,
        tools_by_owner=tools_by_owner,
        tool_names_any=tool_names_any,
        tool_io_pairs=dedup_pairs,
        inventory_tools=[],
    )


def map_resolve_observe_events(
    *,
    events: List[Dict[str, Any]],
    trace_id: str,
    expected_agent_names: List[str],
    expected_tool_names: List[str],
    model: str,
    refresh_raw: bool,
    run_dir: Path,
    run_no: int,
    use_llm_mapping: bool,
    use_llm_single_name: bool,
    global_agent_alias_by_norm: Dict[str, str] | None = None,
    global_tool_alias_by_norm: Dict[str, str] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], ObservedComponents]:
    """Map parsed event names, resolve per-trace aliases, and build the observed component view."""

    # First map raw parsed names into the expected validation naming space.
    mapped_events, mapping_actions = map_agent_tool_events_to_expected(
        events=[e for e in events if isinstance(e, dict)],
        expected_agent_names=expected_agent_names,
        expected_tool_names=expected_tool_names,
        model=model,
        refresh_raw=refresh_raw,
        raw_dir=run_dir / "name_mapping_expected" / f"run_{run_no:03d}",
        use_llm=use_llm_mapping,
        global_agent_alias_by_norm=global_agent_alias_by_norm,
        global_tool_alias_by_norm=global_tool_alias_by_norm,
    )

    # Then choose one canonical trace-visible alias per expected component for reporting/debugging.
    single_name_rows = resolve_single_name_per_trace(
        trace_id=trace_id,
        events=mapped_events,
        mapping_actions=mapping_actions,
        model=model,
        refresh_raw=refresh_raw,
        raw_dir=run_dir / "name_resolution_per_trace" / f"run_{run_no:03d}",
        use_llm=use_llm_single_name,
    )
    # Finally derive the observed agents/tools/tool-IO summary from the mapped events.
    observed = observe_components(mapped_events)
    return mapped_events, mapping_actions, single_name_rows, observed


def _maybe_json_loads(text: str) -> Any:
    """Best-effort JSON parse that also handles fenced blocks and embedded objects."""
    try:
        return json.loads(text)
    except Exception:
        pass
    s = str(text or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
        try:
            return json.loads(s)
        except Exception:
            return None
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except Exception:
            return None
    return None


def _parse_json_dict_deterministic(text: str) -> Dict[str, Any] | None:
    """Best-effort deterministic JSON dict parse without new LLM calls."""
    s = str(text or "").strip()
    if not s:
        return None

    # 1) Strict parse.
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            obj.setdefault("__source_path__", str(c))
            return obj
    except Exception:
        pass

    # 2) Existing loose parser.
    try:
        obj = parse_json_loose(s)
        if isinstance(obj, dict):
            obj.setdefault("__source_path__", str(c))
            return obj
    except Exception:
        pass

    # 3) Existing fenced/substring parser.
    obj = _maybe_json_loads(s)
    if isinstance(obj, dict):
        return obj

    # 4) Try decoding from any object-start position and keep first dict.
    dec = json.JSONDecoder()
    start = s.find("{")
    while start >= 0:
        try:
            obj, _end = dec.raw_decode(s[start:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        start = s.find("{", start + 1)

    return None


def extract_inventory_tools_from_events(
    *,
    events: List[Dict[str, Any]],
    allowed_parents: List[Dict[str, Any]] | None = None,
    model: str = "",
    refresh_raw: bool = False,
    raw_dir: Path | None = None,
    use_llm: bool = True,
    trace_file: Path | None = None,
    allow_unknown_owners: bool = False,
) -> List[Dict[str, Any]]:
    """Extract tool inventory rows from agent/system responses in one trace.

    The function first tries deterministic parsing from JSON-shaped response content.
    It then optionally falls back to an LLM prompt for responses that may describe
    tools in free text. Returned rows are deduplicated by owner/tool.
    """
    out: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()

    # Normalize the allowed owners so parsed inventory rows can be attached only
    # to known agents/servers unless unknown owners are explicitly allowed.
    allowed_owner_by_norm: Dict[str, Tuple[str, str]] = {}
    for row in allowed_parents or []:
        if not isinstance(row, dict):
            continue
        nm = str(row.get("name") or "").strip()
        ok = (
            str(row.get("parent_kind") or row.get("owner_kind") or "").strip().lower()
            or "agent_or_server"
        )
        if nm:
            allowed_owner_by_norm[norm(nm)] = (nm, ok)

    # Collect richer content from the raw trace when possible, because event-level
    # content may be truncated compared with the original span payload.
    content_candidates: List[Tuple[int, str]] = []
    full_content_by_key: Dict[Tuple[str, str], str] = {}
    if isinstance(trace_file, Path) and trace_file.exists():
        try:
            def _jloads_maybe(v: Any) -> Any:
                """Parse JSON-looking strings, otherwise return the original value."""
                if isinstance(v, str):
                    txt = v.strip()
                    if txt and txt[0] in "{[":
                        try:
                            return json.loads(txt)
                        except Exception:
                            return v
                return v

            def _resolve_path(root: Any, path: str) -> Any:
                """Resolve one dotted/bracket path inside a nested dict/list payload."""
                cur = root
                for raw_tok in [t for t in str(path or "").split(".") if t]:
                    tok = raw_tok.replace("[*]", "[0]")
                    bracket_pos = tok.find("[")
                    key = tok[:bracket_pos] if bracket_pos >= 0 else tok
                    if key:
                        if not isinstance(cur, dict):
                            return None
                        cur = cur.get(key)
                    rest = tok[len(key):]
                    while rest.startswith("["):
                        end = rest.find("]")
                        if end < 0:
                            return None
                        idx_txt = rest[1:end].strip()
                        try:
                            idx = int(idx_txt)
                        except Exception:
                            return None
                        if not isinstance(cur, list) or idx < 0 or idx >= len(cur):
                            return None
                        cur = cur[idx]
                        rest = rest[end + 1 :]
                return cur

            trace_obj = json.loads(trace_file.read_text(encoding="utf-8"))
            data = trace_obj.get("data") if isinstance(trace_obj.get("data"), dict) else {}
            spans = data.get("spans") if isinstance(data.get("spans"), list) else trace_obj.get("spans")
            span_by_id: Dict[str, Dict[str, Any]] = {}
            for s in spans or []:
                if not isinstance(s, dict):
                    continue
                sid = str(s.get("span_id") or "").strip()
                if not sid:
                    continue
                attrs = s.get("attributes") if isinstance(s.get("attributes"), dict) else {}
                span_root = {
                    "spanInputs": _jloads_maybe(attrs.get("mlflow.spanInputs")),
                    "spanOutputs": _jloads_maybe(attrs.get("mlflow.spanOutputs")),
                    "metadata": _jloads_maybe(attrs.get("metadata")),
                    "attributes": attrs,
                }
                span_by_id[sid] = span_root

            # Keep the longest recovered content per (span_id, path) pair.
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                sid = str(ev.get("span_id") or "").strip()
                pth = str(ev.get("path") or "").strip()
                if not sid or not pth:
                    continue
                span_root = span_by_id.get(sid)
                if not isinstance(span_root, dict):
                    continue
                node = _resolve_path(span_root, pth)
                if node is None:
                    continue
                content_val = node.get("content") if isinstance(node, dict) and "content" in node else node
                if isinstance(content_val, (dict, list)):
                    try:
                        content_text = json.dumps(content_val, ensure_ascii=False)
                    except Exception:
                        content_text = str(content_val)
                else:
                    content_text = str(content_val or "")
                if not content_text:
                    continue
                key = (sid, pth)
                prev = full_content_by_key.get(key, "")
                if len(content_text) > len(prev):
                    full_content_by_key[key] = content_text
        except Exception:
            full_content_by_key = {}

    # First pass: deterministic extraction from JSON-shaped response content.
    for ev in events:
        if not isinstance(ev, dict):
            continue
        et = str(ev.get("type") or "").strip().lower()
        if et not in {"agent", "system"}:
            continue
        sid = str(ev.get("span_id") or "").strip()
        pth = str(ev.get("path") or "").strip()
        content = ""
        if sid and pth:
            content = str(full_content_by_key.get((sid, pth), "")).strip()
        if not content:
            content = str(ev.get("content") or "").strip()
        if not content:
            continue
        seq = int(ev.get("seq")) if isinstance(ev.get("seq"), int) else -1
        content_candidates.append((seq, content))
        obj = _maybe_json_loads(content)
        if not isinstance(obj, dict):
            continue
        tools = obj.get("tools")
        if not isinstance(tools, list):
            continue

        # Validate and normalize the owner carried by the payload itself.
        owner_from_payload = str(obj.get("parent") or obj.get("owner") or "").strip()
        row_owner = owner_from_payload
        row_owner_kind = "agent_or_server"
        if row_owner and norm(row_owner) in allowed_owner_by_norm:
            row_owner, row_owner_kind = allowed_owner_by_norm[norm(row_owner)]
        elif not (allow_unknown_owners and row_owner):
            continue

        # Emit one inventory row per tool, skipping placeholders and duplicates.
        for t in tools:
            if not isinstance(t, dict):
                continue
            tool_name_raw = str(t.get("name") or "").strip()
            if not tool_name_raw:
                continue
            if "<tool_name>" in tool_name_raw:
                continue
            if "<owner_name>" in row_owner:
                continue
            mapped_tool = tool_name_raw
            key = (norm(row_owner), norm(mapped_tool))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "owner": row_owner,
                    "owner_kind": row_owner_kind,
                    "tool": mapped_tool,
                    "description": str(t.get("description") or "").strip(),
                    "inputs": t.get("inputs") if isinstance(t.get("inputs"), list) else [],
                    "outputs": t.get("outputs") if isinstance(t.get("outputs"), list) else [],
                    "side_effects": t.get("side_effects"),
                    "constraints": t.get("constraints"),
                    "event_seq": ev.get("seq"),
                    "source": "inventory_content_json",
                }
            )

    # Second pass: optionally ask the model to structure free-text inventory responses.
    if use_llm and model and raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        for idx, (seq, content) in enumerate(content_candidates):

            # Ask the model to convert one response text into the grouped inventory schema.
            payload = build_inventory_parse_payload(
                allowed_parents=allowed_parents or [],
                response_text=content,
            )

            # Run the cached parse prompt once before deciding whether a stricter retry is needed.
            # Run the inventory-specific mapping prompt for this one extracted tool row.
            text, _source = call_with_cache_text(
                payload,
                payload_path=raw_dir / f"event_{idx:04d}.payload.json",
                raw_path=raw_dir / f"event_{idx:04d}.raw.txt",
                model=model,
                refresh_raw=bool(refresh_raw),
                retriever=None,
            )
            obj = _parse_json_dict_deterministic(text)
            needs_retry = not isinstance(obj, dict)
            if isinstance(obj, dict):
                grouped_probe = obj.get("tools_by_parent")
                if not isinstance(grouped_probe, list):
                    grouped_probe = obj.get("tools_by_owner")

                # Empty grouped output is treated as unusable and gets one strict retry.
                usable_groups = 0
                if isinstance(grouped_probe, list):
                    for grp in grouped_probe:
                        if not isinstance(grp, dict):
                            continue
                        tools_probe = grp.get("tools")
                        if isinstance(tools_probe, list) and any(isinstance(t, dict) and str(t.get("name") or "").strip() for t in tools_probe):
                            usable_groups += 1
                if usable_groups == 0:
                    needs_retry = True

            if needs_retry:

                # Retry once with a stricter prompt when the first response is malformed or empty.
                retry_payload = dict(payload)
                retry_payload["prompt"] = (
                    str(payload.get("prompt") or "")
                    + "\n\nRETRY REQUIREMENTS (STRICT):\n"
                    + "- Return valid strict JSON only.\n"
                    + "- Do not include markdown/code fences.\n"
                    + "- Do not truncate; close all brackets/braces.\n"
                    + "- Keep output exactly in the required output schema.\n"
                    + "- Include tools_by_parent entries with non-empty tools lists when tools are present in the response text.\n"
                    + "- Assign every extracted tool to exactly one allowed parent."
                )
                retry_text, _retry_source = call_with_cache_text(
                    retry_payload,
                    payload_path=raw_dir / f"event_{idx:04d}.retry1.payload.json",
                    raw_path=raw_dir / f"event_{idx:04d}.retry1.raw.txt",
                    model=model,
                    refresh_raw=True,
                    retriever=None,
                )
                obj = _parse_json_dict_deterministic(retry_text)
            if not isinstance(obj, dict):
                continue

            # Consume the grouped owner->tools shape and add any new rows.
            grouped = obj.get("tools_by_parent")
            if not isinstance(grouped, list):
                grouped = obj.get("tools_by_owner")
            if isinstance(grouped, list):
                for grp in grouped:
                    if not isinstance(grp, dict):
                        continue
                    grp_owner_raw = str(grp.get("parent") or grp.get("owner") or "").strip()
                    grp_owner_kind = (
                        str(grp.get("parent_kind") or grp.get("owner_kind") or "").strip().lower()
                        or "agent_or_server"
                    )
                    grp_owner = grp_owner_raw

                    # Normalize the returned owner to an allowed canonical parent when possible.
                    if norm(grp_owner_raw) in allowed_owner_by_norm:
                        grp_owner, grp_owner_kind = allowed_owner_by_norm[norm(grp_owner_raw)]
                    elif grp_owner_raw and not allow_unknown_owners:
                        continue
                    if not grp_owner:
                        continue

                    # Ignore groups that do not contain a usable tools list.
                    tools = grp.get("tools")
                    if not isinstance(tools, list):
                        continue
                    for t in tools:

                        # Convert each extracted tool into the standard inventory row shape.
                        if not isinstance(t, dict):
                            continue
                        tool_name = str(t.get("name") or "").strip()
                        if not tool_name:
                            continue
                        key = (norm(grp_owner), norm(tool_name))
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append(
                            {
                                "owner": grp_owner,
                                "owner_kind": grp_owner_kind,
                                "tool": tool_name,
                                "description": str(t.get("description") or "").strip(),
                                "inputs": t.get("inputs") if isinstance(t.get("inputs"), list) else [],
                                "outputs": t.get("outputs") if isinstance(t.get("outputs"), list) else [],
                                "side_effects": t.get("side_effects"),
                                "constraints": t.get("constraints"),
                                "event_seq": seq if seq >= 0 else None,
                                "source": "inventory_content_llm_parse",
                            }
                        )

            # Unassigned and legacy single-owner outputs are intentionally ignored here.
            unassigned = obj.get("unassigned_tools")
            if isinstance(unassigned, list):
                pass
            tools = obj.get("tools")
            if isinstance(tools, list):
                continue

    return out


def _dedupe_inventory_keep_lowest_server(
    *,
    rows: List[Dict[str, Any]],
    server_to_ancestor_servers: Dict[str, Set[str]],
) -> List[Dict[str, Any]]:
    """Drop ancestor-server inventory duplicates when a deeper server exposes the same tool."""
    if not rows:
        return []

    # Compare duplicates within each normalized tool name bucket.
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        t = str(r.get("tool") or "").strip()
        if not t:
            continue
        grouped.setdefault(norm(t), []).append(r)

    out: List[Dict[str, Any]] = []
    for _tnorm, bucket in grouped.items():

        # Only server-owned rows participate in the ancestor-vs-descendant dedupe rule.
        server_rows = [r for r in bucket if str(r.get("owner_kind") or "").strip().lower() == "server"]
        drop_server_norms: Set[str] = set()
        server_owner_norms = {norm(str(r.get("owner") or "")) for r in server_rows if str(r.get("owner") or "").strip()}

        # Mark ancestor servers for removal when a descendant server reports the same tool.
        for sn in server_owner_norms:
            ancestors = set()
            for sname, s_anc in (server_to_ancestor_servers or {}).items():
                if norm(sname) != sn:
                    continue
                ancestors = {norm(a) for a in (s_anc or set())}
                break
            for anc in ancestors:
                if anc in server_owner_norms:
                    drop_server_norms.add(anc)

        # Keep all non-server rows, plus only the lowest matching server rows.
        for r in bucket:
            ok = str(r.get("owner_kind") or "").strip().lower()
            on = norm(str(r.get("owner") or ""))
            if ok == "server" and on in drop_server_norms:
                continue
            out.append(r)
    return out


def map_inventory_tools_to_expected(
    *,
    rows: List[Dict[str, Any]],
    expected_tool_names: List[str],
    expected_tools_by_owner: List["ToolOwnerPair"],
    mapping_actions: List[Dict[str, Any]] | None = None,
    model: str,
    refresh_raw: bool,
    raw_dir: Path,
    use_llm: bool = True,
    global_tool_alias_by_norm: Dict[str, str] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Map extracted inventory tool names onto expected tool names and record mapping decisions."""

    mapped_rows: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []

    # Build deterministic lookup tables for expected tool names and their known owners.
    expected_by_norm = {_norm_name(x): x for x in expected_tool_names if str(x or "").strip()}
    owners_by_tool_norm: Dict[str, List[Tuple[str, str]]] = {}
    for p in expected_tools_by_owner or []:
        if not isinstance(p, ToolOwnerPair):
            continue
        tn = _norm_name(str(p.tool_name or ""))
        on = str(p.owner_name or "").strip()
        ok = str(p.owner_kind or "").strip().lower()
        if tn and on and ok in {"agent", "server"}:
            owners_by_tool_norm.setdefault(tn, []).append((on, ok))

    # Reuse any globally learned aliases and seed expected-name alias sets from prior mapping actions.
    alias_map = global_tool_alias_by_norm or {}
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Keep per-expected-name alias sets for the LLM payload, plus a reverse lookup
    # from any normalized alias back to the canonical expected tool name.
    expected_name_aliases: Dict[str, Set[str]] = {x: set() for x in expected_tool_names if str(x or "").strip()}
    alias_to_canonical_norm: Dict[str, str] = {}
    for canonical in expected_name_aliases.keys():
        alias_to_canonical_norm[_norm_name(canonical)] = canonical

    # Fold in previously learned specific tool aliases from earlier runtime mapping decisions.
    for a in mapping_actions or []:
        if not isinstance(a, dict):
            continue
        if str(a.get("event_type") or "").strip().lower() != "tool":
            continue
        if str(a.get("decision") or "").strip() != "map_to_expected":
            continue
        if str(a.get("alias_scope") or "").strip().lower() != "specific":
            continue
        observed_alias = str(a.get("observed_name") or "").strip()
        canonical = str(a.get("mapped_name") or "").strip()

        # Ignore malformed entries and aliases that do not point to known expected tools.
        if not observed_alias or not canonical or canonical not in expected_name_aliases:
            continue
        if _norm_name(observed_alias) != _norm_name(canonical):
            expected_name_aliases[canonical].add(observed_alias)
        alias_to_canonical_norm[_norm_name(observed_alias)] = canonical

    # Process each extracted inventory row independently.
    for idx, row in enumerate(rows or []):
        if not isinstance(row, dict):
            continue
        rr = dict(row)

        # Work on a copy so the input rows stay unchanged.
        observed_name = str(rr.get("tool") or "").strip()
        if not observed_name:
            mapped_rows.append(rr)
            continue

        # Try deterministic mapping first, then fall back to the inventory-specific LLM mapper.
        decision = "new_component"
        mapped_name = ""
        reason = "no_match"
        alias_scope = "specific"

        mapped_name, deterministic_reason = _deterministic_map_name(
            observed_name=observed_name,
            expected_by_norm=expected_by_norm,
            alias_map=alias_map,
        )

        # A deterministic hit immediately resolves this inventory tool to an expected tool.
        if mapped_name:
            decision = "map_to_expected"
            reason = deterministic_reason
            alias_scope = "specific"
        elif use_llm and model:

            # Otherwise ask the model to choose among expected tools plus their known aliases.
            expected_names_payload = {
                canonical: sorted(aliases)
                for canonical, aliases in expected_name_aliases.items()
            }
            payload = build_inventory_name_mapping_payload(
                observed_name=observed_name,
                event_evidence={
                    "type": "inventory_record",
                    "owner": rr.get("owner"),
                    "owner_kind": rr.get("owner_kind"),
                    "description": _truncate_text(rr.get("description"), 2000),
                    "inputs": rr.get("inputs") if isinstance(rr.get("inputs"), list) else [],
                    "outputs": rr.get("outputs") if isinstance(rr.get("outputs"), list) else [],
                },
                expected_names=expected_names_payload,
            )
            # Run the inventory-specific mapping prompt for this one extracted tool row.
            text, _source = call_with_cache_text(
                payload,
                payload_path=raw_dir / f"row_{idx:04d}.payload.json",
                raw_path=raw_dir / f"row_{idx:04d}.raw.txt",
                model=model,
                refresh_raw=bool(refresh_raw),
                retriever=None,
            )
            obj = parse_json_loose(text)

            # Read the model decision and treat malformed output as a failed mapping.
            decision = str(obj.get("decision") or "").strip() if isinstance(obj, dict) else ""
            mapped_name = str(obj.get("mapped_name") or "").strip() if isinstance(obj, dict) else ""
            reason = str(obj.get("reason") or "").strip() if isinstance(obj, dict) else "invalid_mapper_output"

            # Resolve the returned name back to a canonical expected tool name.
            mapped_norm = _norm_name(mapped_name)
            if decision == "map_to_expected":
                if mapped_norm in expected_by_norm:
                    mapped_name = expected_by_norm[mapped_norm]
                elif mapped_norm in alias_to_canonical_norm:
                    mapped_name = alias_to_canonical_norm[mapped_norm]
                else:
                    decision = "new_component"
                    mapped_name = ""

            # Any invalid or non-canonical mapping is downgraded back to a new-component signal.
            if not (decision == "map_to_expected" and _norm_name(mapped_name) in expected_by_norm):
                decision = "new_component"
                mapped_name = ""
                alias_scope = "specific"

        if decision == "map_to_expected" and mapped_name:

            # Rewrite the row to the canonical expected tool name.
            rr["tool"] = expected_by_norm[_norm_name(mapped_name)]

            # If the expected tool has known owners, rebind this row to a canonical owner.
            tool_owners = owners_by_tool_norm.get(_norm_name(rr["tool"]), [])
            if tool_owners:
                servers = [x for x in tool_owners if x[1] == "server"]
                agents = [x for x in tool_owners if x[1] == "agent"]
                chosen_owner, chosen_kind = (servers[0] if servers else (agents[0] if agents else tool_owners[0]))
                rr["owner"] = chosen_owner
                rr["owner_kind"] = chosen_kind

            # Persist any newly learned specific aliases for later rows and later runs.
            if alias_scope == "specific":
                for nk in _norm_name_candidates(observed_name):
                    alias_map[nk] = rr["tool"]
                if rr["tool"] in expected_name_aliases and _norm_name(observed_name) != _norm_name(rr["tool"]):
                    expected_name_aliases[rr["tool"]].add(observed_name)
                alias_to_canonical_norm[_norm_name(observed_name)] = rr["tool"]

        actions.append(
            {
                "row_index": idx,
                "owner": rr.get("owner"),
                "observed_tool": observed_name,
                "decision": decision,
                "mapped_name": rr.get("tool") if decision == "map_to_expected" else "",
                "reason": reason,
                "alias_scope": alias_scope,
            }
        )
        mapped_rows.append(rr)
    return mapped_rows, actions


def _build_local_alias_maps_from_mapping_actions(
    mapping_actions: List[Dict[str, Any]],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build per-trace normalized alias maps for agents and tools from successful mapping actions."""

    agent_alias_by_norm: Dict[str, str] = {}
    tool_alias_by_norm: Dict[str, str] = {}

    for a in mapping_actions:
        if not isinstance(a, dict):
            continue
        if str(a.get("decision") or "").strip() != "map_to_expected":
            continue
        et = str(a.get("event_type") or "").strip().lower()
        alias_scope = str(a.get("alias_scope") or "").strip().lower()
        observed = str(a.get("observed_name") or "").strip()
        mapped = str(a.get("mapped_name") or "").strip()
        name_keys = _norm_name_candidates(observed)
        if not name_keys or not mapped:
            continue
        if et == "agent":
            if alias_scope != "specific":
                continue
            for nk in name_keys:
                agent_alias_by_norm.setdefault(nk, mapped)
        elif et == "tool":
            # Only specific aliases are valid for tool normalization as well.
            if alias_scope != "specific":
                continue
            for nk in name_keys:
                tool_alias_by_norm.setdefault(nk, mapped)
    return agent_alias_by_norm, tool_alias_by_norm


def _canonicalize_tool_name(name: str, expected_tool_by_norm: Dict[str, str], tool_alias_by_norm: Dict[str, str]) -> str:
    """Resolve one observed tool name to its canonical expected tool name when possible."""
    raw = str(name or "").strip()
    if not raw:
        return raw
    n = _norm_name(raw)
    if n in tool_alias_by_norm:
        return tool_alias_by_norm[n]
    if n in expected_tool_by_norm:
        return expected_tool_by_norm[n]
    if "." in raw:
        suffix = raw.rsplit(".", 1)[-1].strip()
        ns = _norm_name(suffix)
        if ns in tool_alias_by_norm:
            return tool_alias_by_norm[ns]
        if ns in expected_tool_by_norm:
            return expected_tool_by_norm[ns]
    return raw


def _canonicalize_agent_name(name: str, expected_agent_by_norm: Dict[str, str], agent_alias_by_norm: Dict[str, str]) -> str:
    """Resolve one observed agent name to its canonical expected agent name when possible."""
    raw = str(name or "").strip()
    if not raw:
        return raw
    n = _norm_name(raw)
    if n in agent_alias_by_norm:
        return agent_alias_by_norm[n]
    if n in expected_agent_by_norm:
        return expected_agent_by_norm[n]
    return raw


def _normalize_observed_for_validation(
    *,
    observed: ObservedComponents,
    mapping_actions: List[Dict[str, Any]],
    expected_agent_names: List[str],
    expected_tool_names: List[str],
) -> None:
    """Canonicalize observed agent/tool names in-place before validation checks run."""

    # Build deterministic lookup tables and reuse any previously learned alias mappings.
    expected_agent_by_norm = {_norm_name(x): x for x in expected_agent_names if x}
    expected_tool_by_norm = {_norm_name(x): x for x in expected_tool_names if x}
    agent_alias_by_norm, tool_alias_by_norm = _build_local_alias_maps_from_mapping_actions(mapping_actions)

    # Normalize the observed agent sets and keep actor_names aligned with the canonical agent names.
    canon_agents: Set[str] = set()
    canon_actor_names: Set[str] = set(observed.actor_names)
    for a in observed.agents:
        ca = _canonicalize_agent_name(a, expected_agent_by_norm, agent_alias_by_norm)
        if ca:
            canon_agents.add(ca)
            canon_actor_names.add(ca)
    observed.agents = canon_agents
    observed.actor_names = canon_actor_names

    # Normalize owner/tool pairs so later validation compares canonical names only.
    canon_pairs: Set[Tuple[str, str]] = set()
    for o, t in observed.tools_by_owner:
        co = _canonicalize_agent_name(o, expected_agent_by_norm, agent_alias_by_norm)
        ct = _canonicalize_tool_name(t, expected_tool_by_norm, tool_alias_by_norm)
        if co and ct:
            canon_pairs.add((co, ct))
    observed.tools_by_owner = canon_pairs

    # Normalize the flat set of observed tool names as well.
    canon_tool_names: Set[str] = set()
    for t in observed.tool_names_any:
        ct = _canonicalize_tool_name(t, expected_tool_by_norm, tool_alias_by_norm)
        if ct:
            canon_tool_names.add(ct)
    observed.tool_names_any = canon_tool_names

    # Normalize the richer tool IO rows while preserving their non-name payload fields.
    norm_io: List[Dict[str, Any]] = []
    for row in observed.tool_io_pairs:
        if not isinstance(row, dict):
            continue
        rr = dict(row)
        rr["owner"] = _canonicalize_agent_name(str(rr.get("owner") or ""), expected_agent_by_norm, agent_alias_by_norm)
        rr["tool"] = _canonicalize_tool_name(str(rr.get("tool") or ""), expected_tool_by_norm, tool_alias_by_norm)
        norm_io.append(rr)
    observed.tool_io_pairs = norm_io

    # Normalize extracted inventory rows in the same way.
    if isinstance(observed.inventory_tools, list):
        norm_inventory: List[Dict[str, Any]] = []
        for row in observed.inventory_tools:
            if not isinstance(row, dict):
                continue
            rr = dict(row)
            rr["owner"] = _canonicalize_agent_name(
                str(rr.get("owner") or ""),
                expected_agent_by_norm,
                agent_alias_by_norm,
            ) or str(rr.get("owner") or "")
            rr["tool"] = _canonicalize_tool_name(
                str(rr.get("tool") or ""),
                expected_tool_by_norm,
                tool_alias_by_norm,
            )
            norm_inventory.append(rr)
        observed.inventory_tools = norm_inventory


def contains_name(target: str, names: Iterable[str]) -> bool:
    """Return whether the target name appears in the iterable after normalization."""
    nt = norm(target)
    if not nt:
        return False
    for n in names:
        if norm(str(n)) == nt:
            return True
    return False


def contains_pair(owner: str, tool: str, observed_pairs: Set[Tuple[str, str]]) -> bool:
    """Return whether the normalized owner/tool pair appears in the observed pair set."""
    no = norm(owner)
    nt = norm(tool)
    for o, t in observed_pairs:
        if norm(o) == no and norm(t) == nt:
            return True
    return False


def update_validated_from_observed(
    *,
    expected_agents: List[str],
    expected_tools_by_agent: List["ToolOwnerPair"],
    observed: ObservedComponents,
    validated: Set[Tuple[Any, ...]],
) -> List[Tuple[Any, ...]]:
    """Mark expected agents and tools as validated when they appear in the observed trace view."""
    def agent_key(a: str) -> Tuple[str, str]:
        return ("agent", a)

    def tool_key(p: ToolOwnerPair) -> Tuple[str, str, str]:
        return ("tool", p.owner_name, p.tool_name)

    newly_validated: List[Tuple[Any, ...]] = []

    # Mark expected agents as validated when they appear either as explicit agents
    # or more generally among the observed actor names.
    for a in expected_agents:
        if contains_name(a, observed.agents) or contains_name(a, observed.actor_names):
            k = agent_key(a)
            if k not in validated:
                validated.add(k)
                newly_validated.append(k)

    # Mark expected owner/tool pairs as validated when the normalized observed
    # tool ownership set contains the same pair.
    for p in expected_tools_by_agent:
        if contains_pair(p.owner_name, p.tool_name, observed.tools_by_owner):
            k = tool_key(p)
            if k not in validated:
                validated.add(k)
                newly_validated.append(k)
    return newly_validated


def process_attempt_events(
    *,
    events: List[Dict[str, Any]],
    trace_id: str,
    expected_agent_names: List[str],
    expected_tool_names: List[str],
    expected_tools_by_agent: List["ToolOwnerPair"],
    expected_tools_by_owner_all: List["ToolOwnerPair"],
    expected_server_to_ancestor_agents: Dict[str, Set[str]],
    expected_server_to_ancestor_servers: Dict[str, Set[str]],
    model: str,
    refresh_raw: bool,
    run_dir: Path,
    run_no: int,
    use_llm_mapping: bool,
    use_llm_single_name: bool,
    use_llm_inventory_parse: bool,
    parse_inventory: bool,
    trace_file: Path | None = None,
    allow_unknown_inventory_owners: bool = False,
    allowed_parents: List[Dict[str, Any]] | None = None,
    global_agent_alias_by_norm: Dict[str, str] | None = None,
    global_tool_alias_by_norm: Dict[str, str] | None = None,
    validated: Set[Tuple[Any, ...]],
) -> Dict[str, Any]:
    """Map one attempt's parsed events into observed components, inventory rows, and validation hits."""

    # First resolve names and gather the raw observed component view from the mapped trace.
    mapped_events, mapping_actions, single_name_rows, observed = map_resolve_observe_events(
        events=events,
        trace_id=trace_id,
        expected_agent_names=expected_agent_names,
        expected_tool_names=expected_tool_names,
        model=model,
        refresh_raw=refresh_raw,
        run_dir=run_dir,
        run_no=run_no,
        use_llm_mapping=use_llm_mapping,
        use_llm_single_name=use_llm_single_name,
        global_agent_alias_by_norm=global_agent_alias_by_norm,
        global_tool_alias_by_norm=global_tool_alias_by_norm,
    )

    # Optionally extract tool inventory rows from the trace content for the inventory pass.
    inventory_tools: List[Dict[str, Any]] = []
    if parse_inventory:
        allowed_parents_rows = (
            [row for row in (allowed_parents or []) if isinstance(row, dict)]
            if allowed_parents
            else (
                [{"name": a, "parent_kind": "agent"} for a in sorted(set(expected_agent_names))]
                + [
                    {"name": s, "parent_kind": "server"}
                    for s in sorted(set(expected_server_to_ancestor_agents.keys()))
                ]
            )
        )
        inventory_tools = extract_inventory_tools_from_events(
            events=mapped_events,
            allowed_parents=allowed_parents_rows,
            model=model,
            refresh_raw=bool(refresh_raw),
            raw_dir=run_dir / "inventory_parse" / f"run_{run_no:03d}",
            use_llm=bool(use_llm_inventory_parse),
            trace_file=trace_file,
            allow_unknown_owners=bool(allow_unknown_inventory_owners),
        )
        inventory_tools, _inventory_map_actions = map_inventory_tools_to_expected(
            rows=inventory_tools,
            expected_tool_names=sorted(set(expected_tool_names)),
            expected_tools_by_owner=expected_tools_by_owner_all,
            mapping_actions=mapping_actions,
            model=model,
            refresh_raw=bool(refresh_raw),
            raw_dir=run_dir / "inventory_name_mapping" / f"run_{run_no:03d}",
            use_llm=bool(use_llm_mapping),
            global_tool_alias_by_norm=global_tool_alias_by_norm,
        )
        inventory_tools = _dedupe_inventory_keep_lowest_server(
            rows=inventory_tools,
            server_to_ancestor_servers=expected_server_to_ancestor_servers,
        )

    # Fold extracted inventory rows back into the observed tool/component sets.
    if inventory_tools:
        for row in inventory_tools:
            owner = str(row.get("owner") or "").strip()
            tool = str(row.get("tool") or "").strip()
            if not owner or not tool:
                continue
            observed.tools_by_owner.add((owner, tool))
            observed.tool_names_any.add(tool)
            observed.tool_io_pairs.append(
                {
                    "owner": owner,
                    "tool": tool,
                    "input": row.get("inputs") if isinstance(row.get("inputs"), list) else [],
                    "output": row.get("outputs") if isinstance(row.get("outputs"), list) else [],
                    "description": row.get("description"),
                    "side_effects": row.get("side_effects"),
                    "constraints": row.get("constraints"),
                    "event_seq": row.get("event_seq"),
                    "source": row.get("source") or "inventory_content_json",
                    "owner_kind": row.get("owner_kind"),
                }
            )
        observed.inventory_tools = inventory_tools
    else:
        observed.inventory_tools = []

    # Normalize observed names into the expected naming space before validation checks.
    _normalize_observed_for_validation(
        observed=observed,
        mapping_actions=mapping_actions,
        expected_agent_names=expected_agent_names,
        expected_tool_names=expected_tool_names,
    )

    # Mark any expected agents/tools that were observed in this attempt as validated.
    newly_validated = update_validated_from_observed(
        expected_agents=expected_agent_names,
        expected_tools_by_agent=expected_tools_by_agent,
        observed=observed,
        validated=validated,
    )

    return {
        "mapped_events": mapped_events,
        "mapping_actions": mapping_actions,
        "single_name_rows": single_name_rows,
        "observed": observed,
        "inventory_tools": observed.inventory_tools,
        "newly_validated": newly_validated,
    }


def merge_alias_maps_from_actions(
    *,
    mapping_actions: List[Dict[str, Any]],
    global_agent_alias_by_norm: Dict[str, str],
    global_tool_alias_by_norm: Dict[str, str],
) -> None:
    """Merge newly learned specific aliases from one run into the global alias caches."""

    # Global cache: persist only specific aliases to avoid generic-runtime drift.
    agent_alias_by_norm: Dict[str, str] = {}
    tool_alias_by_norm: Dict[str, str] = {}
    for a in mapping_actions:
        if not isinstance(a, dict):
            continue
        if str(a.get("decision") or "").strip() != "map_to_expected":
            continue
        if str(a.get("alias_scope") or "").strip().lower() != "specific":
            continue
        et = str(a.get("event_type") or "").strip().lower()
        observed = str(a.get("observed_name") or "").strip()
        mapped = str(a.get("mapped_name") or "").strip()
        name_keys = _norm_name_candidates(observed)
        if not name_keys or not mapped:
            continue

        # Collect aliases separately for agents and tools before merging them globally.
        if et == "agent":
            for nk in name_keys:
                agent_alias_by_norm.setdefault(nk, mapped)
        elif et == "tool":
            for nk in name_keys:
                tool_alias_by_norm.setdefault(nk, mapped)

    # Extend the global caches without overwriting aliases learned earlier.
    for k, v in agent_alias_by_norm.items():
        if k and v:
            global_agent_alias_by_norm.setdefault(k, v)
    for k, v in tool_alias_by_norm.items():
        if k and v:
            global_tool_alias_by_norm.setdefault(k, v)


def load_execution_command_example(args: Any) -> Dict[str, Any] | None:
    """Load the first available execution-command example JSON from the known config locations."""

    candidates: List[Path] = []
    p = str(getattr(args, "execution_command_example_json", "") or "").strip()
    if p:
        candidates.append(Path(p).resolve())
    entry_path = str(getattr(args, "entry", "") or "").strip()
    if entry_path:
        candidates.append((Path(entry_path).resolve().parent / "execution_command_example.json").resolve())
    # Return the first readable JSON object candidate.
    for c in candidates:
        if not c.exists():
            continue
        try:
            obj = json.loads(c.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(obj, dict):
            obj.setdefault("__source_path__", str(c))
            return obj
    return None


def _task_arg_names(exec_example: Dict[str, Any] | None) -> Set[str]:
    """Return the argument names that may carry the user query in the execution command."""
    if not isinstance(exec_example, dict):
        return {"--task", "--query", "--prompt"}
    raw = exec_example.get("task_arg_names")
    names: Set[str] = set()
    if isinstance(raw, list):
        for x in raw:
            txt = str(x or "").strip()
            if txt:
                names.add(txt)
    return names or {"--task", "--query", "--prompt"}


def _entrypoint_value(exec_example: Dict[str, Any] | None) -> str:
    """Extract the configured execution entrypoint path from the execution example payload."""
    if not isinstance(exec_example, dict):
        return ""
    ep = exec_example.get("entrypoint")
    if isinstance(ep, str):
        return ep.strip()
    if isinstance(ep, dict):
        return str(ep.get("value") or "").strip()
    return ""


def command_template_from_exec_example(exec_example: Dict[str, Any] | None) -> str:
    """Build a command template used to execute one query against the target system."""
    if not isinstance(exec_example, dict):
        return trace_extraction.DEFAULT_COMMAND_TEMPLATE

    runner = str(exec_example.get("runner") or "python").strip() or "python"
    entrypoint = _entrypoint_value(exec_example)
    if not entrypoint:
        return trace_extraction.DEFAULT_COMMAND_TEMPLATE

    # Resolve relative entrypoints from the execution-example file location when possible.
    if not Path(entrypoint).is_absolute():
        source_path = str(exec_example.get("__source_path__") or "").strip()
        if source_path:
            entrypoint = str((Path(source_path).resolve().parent / entrypoint).resolve())

    # Separate the query-bearing argument from all other fixed command arguments.
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

    # Emit the final shell template with a `{query}` placeholder for later substitution.
    parts = [shlex.quote(runner), shlex.quote(entrypoint)] + extra_parts + [shlex.quote(task_arg), "{query}"]
    return " ".join(p for p in parts if p)


def parse_trace_to_canonical(
    *,
    trace_file: Path,
    out_file: Path,
    raw_dir: Path,
    model: str,
    refresh_raw: bool,
    expected_agent_names: List[str] | None = None,
    expected_tool_names: List[str] | None = None,
) -> Dict[str, Any]:
    """Parse one raw trace file, normalize its events, and write canonical/debug artifacts."""
    # Configure the output caps used to keep canonical traces compact.
    max_content_chars = 10000
    max_tool_calls = 50
    # Build the parser schema used for execution-validation traces.
    schema = Schema(
        use_case="validation_runtime",
        trace_count=1,
        message_list_paths=list(GENERIC_MESSAGE_LIST_PATHS),
        role_keys=list(TOP_DEFAULT_ROLE_KEYS),
        content_keys=list(TOP_DEFAULT_CONTENT_KEYS),
        name_keys=list(TOP_DEFAULT_NAME_KEYS),
        tool_calls_keys=list(TOP_DEFAULT_TOOL_CALL_KEYS),
        tool_result_keys=list(TOP_DEFAULT_TOOL_RESULT_KEYS),
        timestamp_keys=list(TOP_DEFAULT_TIMESTAMP_KEYS),
    )
    # Parse the raw trace into the shared event structure from the parser core.
    parsed = parse_trace_with_schema(trace_file, schema)

    # Rebuild the parsed events into the compact canonical format used downstream.
    events_in = parsed.get("events") if isinstance(parsed.get("events"), list) else []
    canonical_events: List[Dict[str, Any]] = []
    for idx, e in enumerate(events_in, start=1):
        if not isinstance(e, dict):
            continue
        canonical_events.append(
            {
                "seq": int(e.get("seq") or idx),
                "name": str(e.get("name") or ""),
                "type": str(e.get("type") or ""),
                "content": _truncate_text(e.get("content"), max_content_chars),
                "tool_calls": _cap_tool_calls(e.get("tool_calls"), max_items=max_tool_calls),
                "span_id": str(e.get("span_id") or ""),
                "path": str(e.get("path") or ""),
                "time_ns": int(e.get("time_ns") or 0),
                "aliases": e.get("aliases") if isinstance(e.get("aliases"), list) else [],
            }
        )

    # Prepare the persisted canonical output and the parallel debug artifact.
    final = {
        "trace_file": trace_file.name,
        "model": "parser_core",
        "events": canonical_events,
    }
    debug = {
        "trace_file": trace_file.name,
        "model": "parser_core",
        "debug": parsed.get("debug") if isinstance(parsed.get("debug"), dict) else {},
    }
    # Write both artifacts to disk and return the canonical payload.
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_file.with_suffix(".debug.json").write_text(json.dumps(debug, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return final


def find_tool_io_pairs_for_validated(
    observed_pairs: List[Dict[str, Any]],
    validated_tools: Set[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    """Keep only runtime tool IO examples that belong to validated owner/tool pairs."""
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in observed_pairs:

        # Keep ToolIOPair examples runtime-only.
        # Inventory rows encode schema and should stay in inputs/outputs fields,
        # not become example pairs.
        source = str(row.get("source") or "").strip().lower()
        if source.startswith("inventory_content_"):
            continue
        owner = str(row.get("owner") or "")
        tool = str(row.get("tool") or "")
        if not owner or not tool:
            continue

        # Retain only examples whose normalized owner/tool pair is already validated.
        matched = False
        for vo, vt in validated_tools:
            if norm(vo) == norm(owner) and norm(vt) == norm(tool):
                matched = True
                break
        if not matched:
            continue

        # Deduplicate repeated examples before returning them in the report.
        key = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _find_owner_var_for_apply(nodes_by_var: Dict[str, Dict[str, Any]], owner_name: str) -> str:
    """Return the node var for one owner name when it already exists in the graph."""
    wanted = norm(owner_name)
    if not wanted:
        return ""
    for var, node in nodes_by_var.items():
        if norm(str(node.get("name") or "")) == wanted and node_type_name(node) in {"System", "Agent", "Local_MCP_server", "External_MCP_server"}:
            return var
    return ""


def apply_validation_additions(
    *,
    nodes_py: Path,
    out_path: Path,
    add_candidates: List[Dict[str, Any]],
    external_mcp_tool_nodes: List[Dict[str, Any]],
    report_out: Path | None = None,
) -> Dict[str, Any]:
    """Apply validated agents/tools to the NodeSpec graph and write the updated graph plus apply report."""
    # Load the current graph and prepare bookkeeping for new vars and emitted changes.
    var_order, nodes_by_var = load_nodes_by_var(nodes_py)
    used_vars = set(var_order)
    root_var = next((var for var in var_order if node_type_name(nodes_by_var.get(var, {})) == "System"), var_order[0] if var_order else "")
    changes: Dict[str, Any] = {
        "added_agents": [],
        "added_tools": [],
        "added_external_mcp_tools": [],
        "skipped": [],
        "output_nodes_file": str(out_path),
    }

    def _tool_node(name: str, description: str, inputs: List[Dict[str, Any]], outputs: List[Dict[str, Any]], source: str) -> Dict[str, Any]:
        """Build a new Tool node payload in the NodeSpec graph format."""
        return {
            "name": name,
            "is_graph": False,
            "node_type": {"type": "Tool", "other_description": None},
            "description": description or "Added from execution validation.",
            "code_execution": False,
            "inputs": inputs,
            "outputs": outputs,
            "tool_example_pairs": [],
            "external_connections": [],
            "required_keys": {"enabled": False, "keys": None},
            "duplicates": {"exists": False, "instances": None},
            "tool_list": [],
            "nodes": [],
            "internal_edges": [],
            "metadata": {"created_by": "execution_validation", "source": source},
        }

    def _agent_node(name: str, description: str) -> Dict[str, Any]:
        """Build a new Agent node payload in the NodeSpec graph format."""
        return {
            "name": name,
            "is_graph": False,
            "node_type": {"type": "Agent", "other_description": None},
            "description": description or "Added from execution validation.",
            "code_execution": False,
            "inputs": [],
            "outputs": [],
            "tool_example_pairs": [],
            "external_connections": [],
            "required_keys": {"enabled": False, "keys": None},
            "duplicates": {"exists": False, "instances": None},
            "tool_list": [],
            "nodes": [],
            "internal_edges": [],
            "metadata": {"created_by": "execution_validation", "source": "validated_new_agent"},
        }

    # Add regular validated agent/tool candidates into the graph.
    for cand in add_candidates:
        if not isinstance(cand, dict):
            continue
        ctype = str(cand.get("component_type") or "").strip().lower()
        name = str(cand.get("name") or "").strip()
        owner = str(cand.get("owner") or "").strip()
        description = str(cand.get("description") or "").strip()
        if ctype not in {"agent", "tool"} or not name:
            changes["skipped"].append({"candidate": cand, "reason": "invalid_candidate"})
            continue

        # Skip candidates whose same-type node name already exists in the graph.
        if any(norm(str(node.get("name") or "")) == norm(name) and node_type_name(node) == ("Agent" if ctype == "agent" else "Tool") for node in nodes_by_var.values()):
            changes["skipped"].append({"candidate": cand, "reason": "already_exists"})
            continue
        new_var = next_var(name, used_vars)
        if ctype == "agent":
            # New agents attach under the System root.
            nodes_by_var[new_var] = _agent_node(name, description)
            var_order.append(new_var)
            if root_var and root_var in nodes_by_var:
                add_child_ref(nodes_by_var[root_var], new_var, field="nodes")
            changes["added_agents"].append({"name": name, "var": new_var, "parent_var": root_var})
            continue

        # New tools attach under an existing owner, using tool_list for MCP servers.
        owner_var = _find_owner_var_for_apply(nodes_by_var, owner)
        if not owner_var:
            changes["skipped"].append({"candidate": cand, "reason": "owner_not_found"})
            continue
        owner_type = node_type_name(nodes_by_var.get(owner_var, {}))
        attach_field = "tool_list" if owner_type in {"Local_MCP_server", "External_MCP_server"} else "nodes"
        nodes_by_var[new_var] = _tool_node(name, description, [], [], "validated_new_tool")
        var_order.append(new_var)
        add_child_ref(nodes_by_var[owner_var], new_var, field=attach_field)
        changes["added_tools"].append({"name": name, "var": new_var, "owner": owner, "owner_var": owner_var, "attach_field": attach_field})

    # Add external MCP tool nodes, preserving extracted IO metadata when available.
    for row in external_mcp_tool_nodes:
        if not isinstance(row, dict):
            continue
        owner = str(row.get("owner_server") or "").strip()
        name = str(row.get("name") or "").strip()
        if not owner or not name:
            changes["skipped"].append({"candidate": row, "reason": "invalid_external_mcp_tool"})
            continue
        owner_var = _find_owner_var_for_apply(nodes_by_var, owner)
        if not owner_var:
            changes["skipped"].append({"candidate": row, "reason": "external_owner_not_found"})
            continue

        # Avoid adding the same external tool twice under the same owner server.
        if any(norm(str(nodes_by_var.get(child_var, {}).get("name") or "")) == norm(name) and node_type_name(nodes_by_var.get(child_var, {})) == "Tool" for child_var in (nodes_by_var.get(owner_var, {}).get("tool_list") or []) if isinstance(child_var, str)):
            changes["skipped"].append({"candidate": row, "reason": "external_tool_already_exists"})
            continue
        new_var = next_var(name, used_vars)
        nodes_by_var[new_var] = _tool_node(
            name,
            str(row.get("description") or "").strip(),
            row.get("inputs") if isinstance(row.get("inputs"), list) else [],
            row.get("outputs") if isinstance(row.get("outputs"), list) else [],
            "validated_external_mcp_tool",
        )
        var_order.append(new_var)
        add_child_ref(nodes_by_var[owner_var], new_var, field="tool_list")
        changes["added_external_mcp_tools"].append({"name": name, "var": new_var, "owner_server": owner, "owner_var": owner_var})

    # Write the updated graph file and, optionally, a standalone JSON apply report.
    write_nodes_output(out_path, var_order, nodes_by_var, changes, "EXECUTION_VALIDATION_APPLY_REPORT")
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(changes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return changes


def short_event_excerpt(events: List[Dict[str, Any]], max_items: int = 6) -> List[Dict[str, Any]]:
    """Return a compact event slice for reporting."""
    out: List[Dict[str, Any]] = []
    for ev in events[:max_items]:
        out.append(
            {
                "seq": ev.get("seq"),
                "name": ev.get("name"),
                "type": ev.get("type"),
                "content": str(ev.get("content") or "")[:280],
                "tool_calls_count": len(ev.get("tool_calls") or []),
            }
        )
    # Return one chosen trace-facing name row per expected component seen in this trace.
    return out


def inventory_context(rows: List[Dict[str, Any]], max_items: int = 12) -> str:
    """Build a short inventory context string for flexible activation prompts."""
    parts: List[str] = []
    for row in rows[:max_items]:
        if not isinstance(row, dict):
            continue
        owner = str(row.get("owner") or "").strip()
        tool = str(row.get("tool") or "").strip()
        desc = str(row.get("description") or "").strip()
        if not tool:
            continue
        label = f"{owner}.{tool}" if owner else tool
        parts.append(f"{label}: {desc}" if desc else label)
    return "; ".join(parts)


def tool_row_from_expected(pair: ToolOwnerPair) -> Dict[str, Any]:
    """Convert one expected tool-owner pair into a mutable candidate row."""
    # Return the processed attempt payload used by the caller and the final report.
    return {
        "owner": pair.owner_name,
        "owner_kind": pair.owner_kind,
        "tool": pair.tool_name,
        "description": "",
        "from_nodespec": True,
    }


def is_generic_agent_name(name: str) -> bool:
    """Return whether a name is too generic to treat as a discovered agent."""
    return norm(name) in {norm(x) for x in GENERIC_AGENT_RUNTIME_NAMES_NORM}


def merge_tool_candidates(expected_rows: List[Dict[str, Any]], inventory_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge expected and inventory tool rows by normalized owner/tool key."""
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # Walk both sources together and collapse rows that describe the same owner/tool pair.
    for row in expected_rows + inventory_rows:
        if not isinstance(row, dict):
            continue
        owner = str(row.get("owner") or "").strip()
        tool = str(row.get("tool") or "").strip()
        if not owner or not tool:
            continue
        key = (norm(owner), norm(tool))
        current = merged.get(key)

        # First row for a key initializes the merged record.
        if current is None:
            merged[key] = {
                "owner": owner,
                "owner_kind": str(row.get("owner_kind") or "agent_or_server").strip().lower() or "agent_or_server",
                "tool": tool,
                "description": str(row.get("description") or "").strip(),
                "from_nodespec": bool(row.get("from_nodespec")),
            }
            continue

        # Later rows fill missing metadata and preserve whether any source came from the NodeSpec.
        if not current.get("description") and str(row.get("description") or "").strip():
            current["description"] = str(row.get("description") or "").strip()
        current["from_nodespec"] = bool(current.get("from_nodespec")) or bool(row.get("from_nodespec"))
        if current.get("owner_kind") == "agent_or_server" and str(row.get("owner_kind") or "").strip():
            current["owner_kind"] = str(row.get("owner_kind") or "").strip().lower()

    # Return a stable owner/tool ordering for downstream prompt construction.
    return sorted(merged.values(), key=lambda row: (str(row.get("owner") or ""), str(row.get("tool") or "")))


def record_attempt_key(kind: str, owner: str, name: str) -> Tuple[str, str, str]:
    """Return the normalized attempt-key tuple used for reporting."""
    return (kind, owner, name)


def execute_validation_attempt(
    *,
    run_counter: int,
    out_dir: Path,
    run_dir: Path,
    args: Any,
    phase: str,
    prompt_variant: int,
    query_text: str,
    command_template: str,
    trace_timeout_sec: int,
    target_component: Dict[str, Any],
    parse_inventory: bool,
    allow_unknown_inventory_owners: bool,
    expected_agent_names: List[str],
    expected_tool_names: List[str],
    expected_tools_by_owner_all: List[ToolOwnerPair],
    expected_server_to_ancestor_agents: Dict[str, Set[str]],
    expected_server_to_ancestor_servers: Dict[str, Set[str]],
    validated: Set[Tuple[Any, ...]],
    global_agent_alias_by_norm: Dict[str, str],
    global_tool_alias_by_norm: Dict[str, str],
    observed_agents_all: Set[str],
    observed_actor_names_all: Set[str],
    observed_tools_by_owner_all: Set[Tuple[str, str]],
    observed_tool_names_all: Set[str],
    observed_tool_io_all: List[Dict[str, Any]],
    observed_inventory_tools_all: List[Dict[str, Any]],
    rounds: List[Dict[str, Any]],
    failed_attempts: List[Dict[str, Any]],
) -> Tuple[int, Dict[str, Any] | None]:
    """Run one validation prompt, parse its trace, and merge its observations into accumulators."""

    # Allocate this attempt's identifier and output artifact paths.
    run_counter += 1
    prompt_file = run_dir / f"run_{run_counter:03d}.prompt.txt"
    trace_ref_file = run_dir / f"run_{run_counter:03d}.trace_path.txt"
    canon_out = run_dir / f"run_{run_counter:03d}.llm_span_pipeline.json"

    # Find prior runs in this directory whose saved prompt matches the current query.
    matched_prompt_files: List[Path] = []
    print(f"DEBUG:execute_validation_attempt:start run={run_counter} phase={phase!r} prompt_variant={prompt_variant} query={query_text!r}")
    for existing_prompt_file in sorted(run_dir.glob("run_*.prompt.txt")):
        if not existing_prompt_file.is_file():
            continue
        try:
            if existing_prompt_file.read_text(encoding="utf-8").strip() == query_text.strip():
                matched_prompt_files.append(existing_prompt_file)
        except Exception:
            continue

    # Reuse the latest matching prompt file when one already exists; otherwise use the new run slot.
    selected_prompt_file = matched_prompt_files[-1] if matched_prompt_files else prompt_file
    selected_trace_ref_file = selected_prompt_file.with_name(
        selected_prompt_file.name.replace(".prompt.txt", ".trace_path.txt")
    )
    print(f"DEBUG:execute_validation_attempt:matched_prompts count={len(matched_prompt_files)} selected_prompt={str(selected_prompt_file)!r} selected_trace_ref={str(selected_trace_ref_file)!r}")
    try:

        # No matching prompt exists yet, so persist this prompt and generate a fresh trace.
        if not matched_prompt_files:
            prompt_file.write_text(query_text + "\n", encoding="utf-8")
            selected_prompt_file = prompt_file
            selected_trace_ref_file = trace_ref_file
            if not query_text.strip():
                raise ValueError("empty query")
            print("DEBUG:execute_validation_attempt:generating_new_trace")
            print("DEBUG:execute_validation_attempt:regenerating_missing_trace_ref")
            trace_path = Path(
                trace_extraction._run_trace_extraction(
                    task=query_text,
                    command_template=command_template,
                    out_dir=str(run_dir),
                    query_timeout_sec=int(trace_timeout_sec or 180),
                )
            )
            selected_trace_ref_file.write_text(str(trace_path) + "\n", encoding="utf-8")
            print(f"DEBUG:execute_validation_attempt:wrote_trace_ref trace_path={str(trace_path)!r}")

        # A matching prompt exists, but its trace reference is missing, so regenerate just the trace ref.
        elif not selected_trace_ref_file.exists():
            if not query_text.strip():
                raise ValueError("empty query")
            trace_path = Path(
                trace_extraction._run_trace_extraction(
                    task=query_text,
                    command_template=command_template,
                    out_dir=str(run_dir),
                    query_timeout_sec=int(trace_timeout_sec or 180),
                )
            )
            selected_trace_ref_file.write_text(str(trace_path) + "\n", encoding="utf-8")
        else:
            trace_path = Path(selected_trace_ref_file.read_text(encoding="utf-8").strip()).resolve()
            print(f"DEBUG:execute_validation_attempt:reusing_trace_ref trace_path={str(trace_path)!r}")

        # Store per-trace raw parsing artifacts under a trace-specific directory.
        raw_dir = out_dir / "llm_span_events_raw_optimal" / trace_path.stem

        # Convert the raw trace into the canonical event format used downstream.
        parsed = parse_trace_to_canonical(
            trace_file=trace_path,
            out_file=canon_out,
            raw_dir=raw_dir,
            model=args.model,
            refresh_raw=bool(args.refresh_raw),
            expected_agent_names=expected_agent_names,
            expected_tool_names=expected_tool_names,
        )

        # Resolve names, extract inventory/tool evidence, and update validated components.
        processed = process_attempt_events(
            events=parsed.get("events") if isinstance(parsed.get("events"), list) else [],
            trace_id=trace_path.stem,
            expected_agent_names=expected_agent_names,
            expected_tool_names=expected_tool_names,
            expected_tools_by_agent=expected_tools_by_owner_all,
            expected_tools_by_owner_all=expected_tools_by_owner_all,
            expected_server_to_ancestor_agents=expected_server_to_ancestor_agents,
            expected_server_to_ancestor_servers=expected_server_to_ancestor_servers,
            model=args.model,
            refresh_raw=bool(args.refresh_raw),
            run_dir=run_dir,
            run_no=run_counter,
            use_llm_mapping=True,
            use_llm_single_name=False,
            use_llm_inventory_parse=True,
            parse_inventory=parse_inventory,
            trace_file=trace_path,
            allowed_parents=[],
            allow_unknown_inventory_owners=allow_unknown_inventory_owners,
            global_agent_alias_by_norm=global_agent_alias_by_norm,
            global_tool_alias_by_norm=global_tool_alias_by_norm,
            validated=validated,
        )

    except Exception as exc:

        # Record failures without aborting the whole validation run.
        failed_attempts.append(
            {
                "run": run_counter,
                "phase": phase,
                "prompt_variant": prompt_variant,
                "target_component": target_component,
                "query_text": query_text,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        rounds.append(
            {
                "run": run_counter,
                "phase": phase,
                "prompt_variant": prompt_variant,
                "status": "failed",
                "target_component": target_component,
                "query_text": query_text,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return run_counter, None

    # Merge successful attempt outputs back into the run-level shared state.
    merge_alias_maps_from_actions(
        mapping_actions=processed["mapping_actions"],
        global_agent_alias_by_norm=global_agent_alias_by_norm,
        global_tool_alias_by_norm=global_tool_alias_by_norm,
    )
    merge_observed_components(
        observed=processed["observed"],
        observed_agents_all=observed_agents_all,
        observed_actor_names_all=observed_actor_names_all,
        observed_tools_by_owner_all=observed_tools_by_owner_all,
        observed_tool_names_all=observed_tool_names_all,
        observed_tool_io_all=observed_tool_io_all,
        observed_inventory_tools_all=observed_inventory_tools_all,
    )

    # Persist the canonicalized trace with the post-processed event view.
    parsed_out = {
        **({} if not isinstance(parsed, dict) else parsed),
        "events": processed["mapped_events"],
        "trace_file": trace_path.name,
        "model": "parser_core",
    }
    canon_out.write_text(json.dumps(parsed_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rounds.append(
        {
            "run": run_counter,
            "phase": phase,
            "prompt_variant": prompt_variant,
            "status": "ok",
            "target_component": target_component,
            "query_text": query_text,
            "trace_file": str(trace_path),
            "canonical_file": str(canon_out),
            "canonical_excerpt": short_event_excerpt(processed["mapped_events"]),
            "name_mapping_actions": processed["mapping_actions"],
            "newly_validated": [list(x) for x in processed["newly_validated"]],
            "observed_snapshot": observed_snapshot(processed["observed"]),
            "inventory_tools_extracted": processed.get("inventory_tools") if isinstance(processed.get("inventory_tools"), list) else [],
        }
    )
    return run_counter, processed


def load_expected_components(nodes_py: Path) -> ExpectedComponents:
    """Load the expected agents, servers, tool ownership, and server ancestry from a NodeSpec file."""
    var_order, nodes_by_var = load_nodes_by_var(nodes_py)
    if not var_order:
        raise RuntimeError(f"No nodes found in {nodes_py}")

    # First collect the named agents and servers that appear in the graph.
    agents: Set[str] = set()
    servers: Set[str] = set()
    for var in var_order:
        node = nodes_by_var.get(var, {})
        name = str(node.get("name") or "").strip()
        ntype = node_type_name(node)
        if not name:
            continue
        if ntype == "Agent":
            agents.add(name)
        elif ntype in SERVER_TYPES:
            servers.add(name)

    # Build parent/child navigation maps so ownership and ancestry can be derived from the graph.
    children_by_parent, parents_by_child = build_parent_child_maps(var_order, nodes_by_var)

    def descendant_vars(root_var: str) -> Set[str]:
        """Return all descendant vars reachable below one parent var."""
        seen_local: Set[str] = set()
        stack: List[str] = list(children_by_parent.get(root_var, []))
        while stack:
            cur = stack.pop()
            if cur in seen_local:
                continue
            seen_local.add(cur)
            stack.extend(children_by_parent.get(cur, []))
        return seen_local

    # For each agent/server owner, collect every descendant Tool node as an expected tool relation.
    tools_by_owner: List[ToolOwnerPair] = []
    seen: Set[Tuple[str, str, str]] = set()
    for pvar in var_order:
        pnode = nodes_by_var.get(pvar, {})
        ptype = node_type_name(pnode)
        pname = str(pnode.get("name") or "").strip()
        if not pname:
            continue

        owner_kind = ""
        if ptype == "Agent":
            owner_kind = "agent"
        elif ptype in SERVER_TYPES:
            owner_kind = "server"
        else:
            continue

        for cvar in sorted(descendant_vars(pvar)):
            cnode = nodes_by_var.get(cvar, {})
            ctype = node_type_name(cnode)
            if ctype != "Tool":
                continue
            tname = str(cnode.get("name") or "").strip()
            if not tname:
                continue
            row = ToolOwnerPair(owner_name=pname, owner_kind=owner_kind, tool_name=tname)
            if row.key in seen:
                continue
            seen.add(row.key)
            tools_by_owner.append(row)

    # Build a direct lookup from server name to graph var for the ancestry pass.
    var_by_name: Dict[str, str] = {}
    for v in var_order:
        n = nodes_by_var.get(v, {})
        nm = str(n.get("name") or "").strip()
        if nm:
            var_by_name[nm] = v

    def ancestor_vars(root_var: str) -> Set[str]:
        """Return all ancestor vars reachable above one child var."""
        seen_local: Set[str] = set()
        stack: List[str] = list(parents_by_child.get(root_var, []))
        while stack:
            cur = stack.pop()
            if cur in seen_local:
                continue
            seen_local.add(cur)
            stack.extend(parents_by_child.get(cur, []))
        return seen_local

    # For each server, collect the ancestor agents and ancestor servers above it in the graph.
    server_to_ancestor_agents: Dict[str, Set[str]] = {}
    server_to_ancestor_servers: Dict[str, Set[str]] = {}
    for sname in sorted(servers):
        svar = var_by_name.get(sname, "")
        if not svar:
            server_to_ancestor_agents[sname] = set()
            server_to_ancestor_servers[sname] = set()
            continue
        agent_names: Set[str] = set()
        server_names: Set[str] = set()
        for avar in ancestor_vars(svar):
            anode = nodes_by_var.get(avar, {})
            atype = node_type_name(anode)
            aname = str(anode.get("name") or "").strip()
            if atype == "Agent":
                if aname:
                    agent_names.add(aname)
            elif atype in SERVER_TYPES:
                if aname:
                    server_names.add(aname)
        server_to_ancestor_agents[sname] = agent_names
        server_to_ancestor_servers[sname] = server_names

    return ExpectedComponents(
        agents=agents,
        servers=servers,
        tools_by_owner=tools_by_owner,
        server_to_ancestor_agents=server_to_ancestor_agents,
        server_to_ancestor_servers=server_to_ancestor_servers,
    )
