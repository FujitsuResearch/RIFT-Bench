from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROLE_HINT_KEYS = {"role", "source", "type"}
CONTENT_HINT_KEYS = {"content", "text", "message", "raw"}
TOOL_HINT_KEYS = {"tool_calls", "function_call", "tool_call", "tool_call_id", "arguments", "args"}
MESSAGE_NAME_KEYS = {"name", "source"}
INDUCE_TOOL_CALL_KEYS = {"tool_calls", "function_call", "tool_call", "arguments", "args"}
INDUCE_TOOL_RESULT_KEYS = {"tool_call_id", "status", "is_error", "artifact", "result", "results"}
INDUCE_TIMESTAMP_KEYS = {"created_at", "timestamp", "time", "time_unix_nano"}
FILTER_PATH_HINTS = ("messages", "choices")
GENERIC_MESSAGE_LIST_PATHS = (
    "spanOutputs.messages",
    "spanOutputs.update.messages",
    "spanInputs.messages",
    "spanInputs.state.messages",
    "spanOutputs.tasks_output[*].messages",
)
TOP_DEFAULT_ROLE_KEYS = ("role", "source", "type")
TOP_DEFAULT_CONTENT_KEYS = ("content", "text", "message", "raw")
TOP_DEFAULT_NAME_KEYS = ("name", "source")
TOP_DEFAULT_TOOL_CALL_KEYS = ("tool_calls", "function_call", "tool_call", "arguments", "args")
TOP_DEFAULT_TOOL_RESULT_KEYS = ("tool_call_id", "status", "is_error", "artifact", "result")
TOP_DEFAULT_TIMESTAMP_KEYS = ("created_at", "timestamp", "time_unix_nano")

USER_ROLEISH = {"user", "human", "usermessage"}
TEXTMESSAGE_ROLEISH = {"textmessage"}
SYSTEM_ROLEISH = {"system", "systemmessage"}
TOOL_ROLEISH = {"tool", "toolmessage", "toolcallexecutionevent"}
AGENT_ROLEISH = {"ai", "assistant", "aimessage", "assistantmessage", "toolcallrequestevent", "toolcallsummarymessage"}
USER_SYSTEM_TYPES = {"user", "system"}
USER_SOURCEISH = set(USER_ROLEISH)
TOP_LEVEL_TOOL_RESULT_ID_KEYS = ("tool_call_id",)
TOOL_EXEC_PAYLOAD_ID_KEYS = ("call_id", "tool_call_id", "id")
TOOL_EXEC_PAYLOAD_OUTPUT_KEYS = ("content", "result", "output")
GENERIC_AGENT_RUNTIME_NAMES_NORM = {"assistant", "agent", "teamquery", "crewkickoff", "ai", "model", "litellmcompletion"}
GENERIC_TOOL_RUNTIME_NAMES_NORM = {"tool", "function", "call"}
USER_AGENT_ECHO_DEDUP_MAX_GAP_NS = 5_000_000_000
USER_STRONG_PATH_TOKENS = ("spanInputs.messages", "spanInputs.state.messages")
USER_EXCLUDED_PATH_TOKENS = ("spanOutputs.messages", "spanOutputs.update.messages", "spanOutputs.tasks_output")
USER_MAX_CONFIDENT_LIST_INDEX = 1
USER_MAX_CONFIDENT_SPAN_INDEX = 1
ENABLE_USER_FALLBACK_PROMOTION = False

ROUTING_OBJECT_KEYS = {"routing"}
TERMINAL_ROUTING_TARGETS = {"finish", "__end__"}
ROUTING_DIRECT_KEYS = ("goto", "next")
RELAY_HINT_TOKENS = (
    "spanOutputs.messages",
    "spanOutputs.update.messages",
    "spanInputs.messages",
    "spanInputs.state.messages",
)
ROUTING_PATH_PRIORITY_KEYS = ("spanOutputs.goto", "spanOutputs.update.next", "spanOutputs.next")
WRAPPER_TERMINAL_NAME_TOKENS = ("runnablesequence", "jsonoutputparser", "langgraph")

WRAPPER_NAME_TOKENS = (
    "langgraph",
    "runnablesequence",
    "jsonoutputparser",
    "azurechatopenai",
    "chatopenai",
    "custommiddleware",
    "before_agent",
    "after_agent",
    "before_model",
    "after_model",
    "call_model",
    "model",
    "tools",
    "should_continue",
)
WRAPPER_NAME_EXACT = {"agent"}

EXCEPTION_EVENT_NAME = "exception"
RELAY_NAMELESS_DEDUP_MAX_GAP_NS = 5_000_000_000
ROUTING_CLUSTER_GAP_NS = 2_000_000_000
ROUTING_TERMINAL_BRIDGE_MAX_GAP_NS = 3_000_000_000
WRAPPER_TERMINAL_FOLLOWUP_MAX_GAP_NS = 3_000_000_000
SPLIT_TOOL_BATCH_BY_CALL_ID = True
ROUTING_RANK_PREFIX_PENALTIES = {"jsonoutputparser": -6, "runnablesequence": -5}
ROUTING_RANK_NAME_BONUS_TOKENS = ("supervisor", "_node")


def _json_loads_maybe(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    s = v.strip()
    if not s:
        return v
    if s.startswith('"') and s.endswith('"'):
        try:
            s = json.loads(s)
            if not isinstance(s, str):
                return s
        except Exception:
            pass
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except Exception:
            return v
    return v


def _parse_attr_value(v: Any) -> Any:
    parsed = _json_loads_maybe(v)
    if parsed is v and isinstance(v, str):
        s = v.strip().strip('"')
        return _json_loads_maybe(s)
    return parsed


def _parse_attr_scalar(v: Any) -> Any:
    if isinstance(v, dict):
        for key in ("stringValue", "boolValue", "doubleValue", "intValue"):
            if key in v:
                return v.get(key)
    return _parse_attr_value(v)


def _clean_hint_text(v: Any) -> str:
    s = str(v or "").strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()
    return s


def _extract_span_identity_hints(trace: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    spans = _extract_spans(trace)
    if not spans:
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
        span_type = _clean_hint_text(_parse_attr_scalar(attrs.get("mlflow.spanType"))).upper()
        span_name = _clean_hint_text(span.get("name"))
        agent_attr = _clean_hint_text(_parse_attr_scalar(attrs.get("agent")))
        role_attr = _clean_hint_text(_parse_attr_scalar(attrs.get("role")))
        agent_role_attr = _clean_hint_text(_parse_attr_scalar(attrs.get("agent_role")))
        span_inputs = _parse_attr_scalar(attrs.get("mlflow.spanInputs"))
        if not isinstance(span_inputs, dict):
            span_inputs = {}
        role_hint = _clean_hint_text(span_inputs.get("agent_role") or span_inputs.get("role") or "")
        tool_hint = _clean_hint_text(span_inputs.get("tool_name") or span_inputs.get("name") or "")

        hint: Dict[str, str] = {}
        if span_type == "AGENT":
            candidate = role_attr or role_hint or span_name
            if candidate and not _is_wrapper_name(candidate):
                hint["agent_name"] = candidate
        elif span_type == "TOOL":
            t = tool_hint or span_name
            if t and not _is_wrapper_name(t):
                hint["tool_name"] = t
            a = agent_role_attr or role_hint or role_attr
            if a and not _is_wrapper_name(a):
                hint["agent_name"] = a
        else:
            a = agent_attr or agent_role_attr or role_attr or role_hint
            if a and not _is_wrapper_name(a):
                hint["agent_name"] = a
            if tool_hint and not _is_wrapper_name(tool_hint):
                hint["tool_name"] = tool_hint
        if hint:
            out[sid] = hint

    def _walk_up(span_id: str) -> List[str]:
        chain: List[str] = []
        cur = str(span_id or "")
        seen: set[str] = set()
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
                merged["agent_name"] = _clean_hint_text(ah.get("agent_name"))
            if not merged.get("tool_name") and str(ah.get("tool_name") or "").strip():
                merged["tool_name"] = _clean_hint_text(ah.get("tool_name"))
            if merged.get("agent_name") and merged.get("tool_name"):
                break
        if merged:
            out[sid] = merged
    return out


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


def _apply_span_identity_hints(events: List[Dict[str, Any]], trace: Dict[str, Any]) -> None:
    if not events:
        return
    hints = _extract_span_identity_hints(trace)
    if not hints:
        return

    span_times: Dict[str, Tuple[int, int]] = {}
    for s in _extract_spans(trace):
        if not isinstance(s, dict):
            continue
        sid = str(s.get("span_id") or "").strip()
        if not sid:
            continue
        span_times[sid] = (int(s.get("start_time_unix_nano") or 0), int(s.get("end_time_unix_nano") or 0))

    for e in events:
        if not isinstance(e, dict):
            continue
        et = str(e.get("type") or "").strip().lower()
        observed_name = str(e.get("name") or "").strip()
        if not _is_generic_runtime_name(observed_name, "agent" if et in {"agent", "system"} else "tool" if et == "tool" else ""):
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
        dedup: List[Tuple[str, str]] = []
        seen: set[str] = set()
        for csid, cpath in candidate_rows:
            if csid in seen:
                continue
            seen.add(csid)
            dedup.append((csid, cpath))
        dedup.sort(key=lambda row: (_alias_path_priority(row[1]), _event_time_distance(event_time_ns, span_times, row[0])))
        hint: Dict[str, str] | None = None
        for csid, _ in dedup:
            h = hints.get(csid)
            if isinstance(h, dict) and h:
                hint = h
                break
        if not isinstance(hint, dict):
            continue
        mapped = ""
        if et in {"agent", "system"}:
            mapped = _clean_hint_text(hint.get("agent_name"))
        elif et == "tool":
            mapped = _clean_hint_text(hint.get("tool_name"))
        mapped = _normalize_event_name(mapped)
        if mapped and not _is_generic_runtime_name(mapped, "agent" if et in {"agent", "system"} else "tool") and not _is_wrapper_name(mapped):
            e["name"] = mapped


def _extract_spans(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = trace.get("data") if isinstance(trace.get("data"), dict) else trace
    spans = data.get("spans") if isinstance(data, dict) else None
    out = [s for s in (spans or []) if isinstance(s, dict)]
    out.sort(key=lambda s: int(s.get("start_time_unix_nano") or 0))
    return out


def _span_roots(span: Dict[str, Any]) -> Dict[str, Any]:
    attrs = span.get("attributes") if isinstance(span.get("attributes"), dict) else {}
    return {
        "spanInputs": _parse_attr_value(attrs.get("mlflow.spanInputs")),
        "spanOutputs": _parse_attr_value(attrs.get("mlflow.spanOutputs")),
        "metadata": _parse_attr_value(attrs.get("metadata")),
        "attributes": attrs,
        "events": span.get("events"),
    }


def _walk(obj: Any, path: str = "") -> Iterable[Tuple[str, Any]]:
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            ktxt = str(k)
            child = f"{path}.{ktxt}" if path else ktxt
            yield from _walk(v, child)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            child = f"{path}[{i}]"
            yield from _walk(v, child)


def _canonicalize_path(path: str) -> str:
    return re.sub(r"\[\d+\]", "[*]", path)


def _split_path(path: str) -> List[str]:
    return [p for p in path.split(".") if p]


def _resolve_token(container: Any, token: str) -> List[Any]:
    m = re.match(r"^([^\[]*)(.*)$", token)
    if not m:
        return []
    key = m.group(1) or ""
    suffix = m.group(2) or ""

    bases: List[Any]
    if key:
        if not isinstance(container, dict):
            return []
        bases = [container.get(key)]
    else:
        bases = [container]

    for idx_txt in re.findall(r"\[([^\]]+)\]", suffix):
        next_bases: List[Any] = []
        for base in bases:
            if not isinstance(base, list):
                continue
            if idx_txt == "*":
                next_bases.extend(base)
            else:
                try:
                    idx = int(idx_txt)
                except ValueError:
                    continue
                if 0 <= idx < len(base):
                    next_bases.append(base[idx])
        bases = next_bases
        if not bases:
            break
    return [b for b in bases if b is not None]


def _resolve_path_all(root: Dict[str, Any], path: str) -> List[Any]:
    curs: List[Any] = [root]
    for tok in _split_path(path):
        next_curs: List[Any] = []
        for c in curs:
            next_curs.extend(_resolve_token(c, tok))
        curs = next_curs
        if not curs:
            return []
    return curs


def _looks_message_dict(d: Dict[str, Any]) -> bool:
    keys = {str(k).lower() for k in d.keys()}
    has_roleish = any(k in keys for k in ROLE_HINT_KEYS)
    has_contentish = any(k in keys for k in CONTENT_HINT_KEYS)
    has_toolish = any(k in keys for k in TOOL_HINT_KEYS)
    return (has_roleish and has_contentish) or (has_contentish and has_toolish) or (has_roleish and has_toolish)


def _score_message_list(lst: List[Any]) -> int:
    if not lst:
        return 0
    score = 0
    for item in lst:
        if not isinstance(item, dict):
            continue
        if _looks_message_dict(item):
            score += 3
        keys = {str(k).lower() for k in item.keys()}
        if "tool_calls" in keys or "function_call" in keys:
            score += 2
        if "content" in keys:
            score += 1
    return score


@dataclass
class Schema:
    use_case: str
    trace_count: int
    message_list_paths: List[str]
    role_keys: List[str]
    content_keys: List[str]
    name_keys: List[str]
    tool_calls_keys: List[str]
    tool_result_keys: List[str]
    timestamp_keys: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "use_case": self.use_case,
            "trace_count": self.trace_count,
            "message_list_paths": self.message_list_paths,
            "field_keys": {
                "role": self.role_keys,
                "content": self.content_keys,
                "name": self.name_keys,
                "tool_calls": self.tool_calls_keys,
                "tool_result": self.tool_result_keys,
                "timestamp": self.timestamp_keys,
            },
        }


def induce_schema(use_case: str, trace_files: List[Path]) -> Schema:
    path_scores: Dict[str, int] = {}
    role_keys: Dict[str, int] = {}
    content_keys: Dict[str, int] = {}
    name_keys: Dict[str, int] = {}
    tool_call_keys: Dict[str, int] = {}
    tool_result_keys: Dict[str, int] = {}
    ts_keys: Dict[str, int] = {}

    for tf in trace_files:
        trace = json.loads(tf.read_text(encoding="utf-8"))
        spans = _extract_spans(trace)
        for span in spans:
            roots = _span_roots(span)
            for root_name, root_obj in roots.items():
                for path, node in _walk(root_obj, root_name):
                    cpath = _canonicalize_path(path)
                    if isinstance(node, list) and node:
                        score = _score_message_list(node)
                        if score > 0:
                            path_scores[cpath] = path_scores.get(cpath, 0) + score
                            for it in node:
                                if not isinstance(it, dict):
                                    continue
                                for k in it.keys():
                                    kl = str(k).lower()
                                    if kl in ROLE_HINT_KEYS:
                                        role_keys[kl] = role_keys.get(kl, 0) + 1
                                    if kl in CONTENT_HINT_KEYS:
                                        content_keys[kl] = content_keys.get(kl, 0) + 1
                                    if kl in MESSAGE_NAME_KEYS:
                                        name_keys[kl] = name_keys.get(kl, 0) + 1
                                    if kl in INDUCE_TOOL_CALL_KEYS:
                                        tool_call_keys[kl] = tool_call_keys.get(kl, 0) + 1
                                    if kl in INDUCE_TOOL_RESULT_KEYS:
                                        tool_result_keys[kl] = tool_result_keys.get(kl, 0) + 1
                                    if kl in INDUCE_TIMESTAMP_KEYS:
                                        ts_keys[kl] = ts_keys.get(kl, 0) + 1

    sorted_paths = [k for k, _ in sorted(path_scores.items(), key=lambda kv: (-kv[1], kv[0]))]
    filtered_paths = [p for p in sorted_paths if any(h in p for h in FILTER_PATH_HINTS)]
    if not filtered_paths:
        filtered_paths = sorted_paths

    def top_keys(counter: Dict[str, int], defaults: List[str]) -> List[str]:
        ks = [k for k, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))]
        for d in defaults:
            if d not in ks:
                ks.append(d)
        return ks[:8]

    return Schema(
        use_case=use_case,
        trace_count=len(trace_files),
        message_list_paths=list(GENERIC_MESSAGE_LIST_PATHS),
        role_keys=top_keys(role_keys, list(TOP_DEFAULT_ROLE_KEYS)),
        content_keys=top_keys(content_keys, list(TOP_DEFAULT_CONTENT_KEYS)),
        name_keys=top_keys(name_keys, list(TOP_DEFAULT_NAME_KEYS)),
        tool_calls_keys=top_keys(tool_call_keys, list(TOP_DEFAULT_TOOL_CALL_KEYS)),
        tool_result_keys=top_keys(tool_result_keys, list(TOP_DEFAULT_TOOL_RESULT_KEYS)),
        timestamp_keys=top_keys(ts_keys, list(TOP_DEFAULT_TIMESTAMP_KEYS)),
    )


def _stringify(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True)
    return str(v)


def _event_content_from_message(message: Dict[str, Any], ev_type: str, tool_calls: List[Dict[str, Any]]) -> str:
    """
    Canonical content policy:
    - Agent events that are pure tool-call envelopes should not duplicate payload in `content`.
    - Tool events that carry execution payload arrays should keep only user-facing tool output text.
    """
    raw = message.get("content") if "content" in message else message.get("raw", "")
    obj = _parse_attr_value(raw)

    if ev_type == "agent" and tool_calls:
        # Common shape: content is a list of call specs (id/name/arguments).
        if isinstance(obj, list) and obj and all(isinstance(it, dict) for it in obj):
            is_call_envelope = all(
                (
                    ("arguments" in it or ("function" in it and isinstance(it.get("function"), dict)))
                    and ("name" in it or ("function" in it and isinstance(it.get("function"), dict) and it["function"].get("name")))
                )
                for it in obj
            )
            if is_call_envelope:
                return ""

    if ev_type == "tool":
        # Common shape: list of {call_id, content, ...}. Keep only textual tool outputs.
        if isinstance(obj, list) and obj and all(isinstance(it, dict) for it in obj):
            outs: List[str] = []
            for it in obj:
                c = it.get("content")
                if c is None:
                    continue
                cs = str(c).strip()
                if cs:
                    outs.append(cs)
            if outs:
                return "\n\n".join(outs)
        # Also handle singular payload dicts.
        if isinstance(obj, dict):
            # Search-like tools often return {"results": [...]} where results is the real answer payload.
            if isinstance(obj.get("results"), list):
                return json.dumps(obj.get("results"), ensure_ascii=False)
            for k in ("content", "result", "output"):
                if k in obj and obj.get(k) is not None:
                    cs = str(obj.get(k)).strip()
                    if cs:
                        return cs

    return _stringify(raw)


def _strip_stale_tool_calls(ev_type: str, content: str, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Some frameworks echo previous tool calls alongside the final textual response:
    - agent event
    - non-empty content
    - tool_calls present but all have empty arguments
    These refs are not fresh call intents and should not block dedupe.
    """
    if ev_type != "agent" or not tool_calls:
        return tool_calls
    if not str(content or "").strip():
        return tool_calls
    all_empty_args = True
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        args = tc.get("arguments")
        if isinstance(args, dict) and args:
            all_empty_args = False
            break
    return [] if all_empty_args else tool_calls


def _split_tool_batch_payload(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Split tool execution batches into one payload per call_id/tool_call_id when available.
    Input shape handled:
      content: [ {call_id|tool_call_id, name?, content|result|output}, ... ]
    """
    if not SPLIT_TOOL_BATCH_BY_CALL_ID:
        return []
    content_obj = _parse_attr_value(message.get("content"))
    if not (isinstance(content_obj, list) and content_obj):
        return []
    if not all(isinstance(it, dict) for it in content_obj):
        return []

    out: List[Dict[str, Any]] = []
    for it in content_obj:
        call_id = str(it.get("call_id") or it.get("tool_call_id") or "").strip()
        if not call_id:
            continue
        text = ""
        for k in ("content", "result", "output"):
            if it.get(k) is not None and str(it.get(k)).strip():
                text = str(it.get(k)).strip()
                break
        if not text and isinstance(it.get("results"), list):
            text = json.dumps(it.get("results"), ensure_ascii=False)
        if not text:
            continue
        out.append(
            {
                "tool_call_id": call_id,
                "name": str(it.get("name") or "").strip(),
                "content": text,
            }
        )
    return out


def _canonical_content(content: str) -> str:
    s = str(content or "").strip()
    if not s:
        return ""
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            obj = json.loads(s)
            return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            pass
    return re.sub(r"\s+", " ", s)


def _parse_iso_to_ns(v: Any) -> Optional[int]:
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return int(dt.timestamp() * 1_000_000_000)
    except Exception:
        return None


def _event_nano_to_ns(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        n = int(v)
    except Exception:
        return None
    # Some traces store event nanos with microsecond precision scale.
    if n < 10_000_000_000_000_000:
        return n * 1000
    return n


def _routing_payload(content: str) -> Optional[str]:
    s = str(content or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    if set(obj.keys()) != ROUTING_OBJECT_KEYS:
        return None
    routing = obj.get("routing")
    if not isinstance(routing, list):
        return None
    return json.dumps({"routing": routing}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _routing_targets_from_payload(payload: Optional[str]) -> List[str]:
    if not payload:
        return []
    try:
        obj = json.loads(payload)
    except Exception:
        return []
    if not isinstance(obj, dict):
        return []
    routing = obj.get("routing")
    if not isinstance(routing, list):
        return []
    out: List[str] = []
    for it in routing:
        s = _stringify(it).strip()
        if s:
            out.append(s)
    return out


def _first_nonterminal_target(content: str) -> str:
    rp = _routing_payload(content)
    targets = _routing_targets_from_payload(rp)
    for t in targets:
        tl = str(t).strip().lower()
        if tl and tl not in TERMINAL_ROUTING_TARGETS:
            return str(t).strip()
    return ""


def _append_alias(event: Dict[str, Any], span_id: Any, path: Any, max_aliases: int = 12) -> None:
    sid = str(span_id or "")
    p = str(path or "")
    if not sid and not p:
        return
    aliases = event.setdefault("_aliases", [])
    key = json.dumps({"span_id": sid, "path": p}, ensure_ascii=False, sort_keys=True)
    if key in {json.dumps(a, ensure_ascii=False, sort_keys=True) for a in aliases if isinstance(a, dict)}:
        return
    if len(aliases) < max_aliases:
        aliases.append({"span_id": sid, "path": p})
    else:
        event["_alias_overflow"] = int(event.get("_alias_overflow") or 0) + 1


def _merge_aliases(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for a in src.get("_aliases") or []:
        if not isinstance(a, dict):
            continue
        _append_alias(dst, a.get("span_id"), a.get("path"))


def _routing_targets(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        out: List[str] = []
        for it in v:
            s = _stringify(it).strip()
            if s:
                out.append(s)
        return out
    s = _stringify(v).strip()
    return [s] if s else []


def _normalize_tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    tc = message.get("tool_calls")
    if isinstance(tc, list):
        for it in tc:
            if not isinstance(it, dict):
                continue
            name = it.get("name")
            if not name and isinstance(it.get("function"), dict):
                name = it["function"].get("name")
            args = it.get("args")
            if args is None and isinstance(it.get("function"), dict):
                args = _parse_attr_value(it["function"].get("arguments"))
            args = _parse_attr_value(args)
            if not isinstance(args, dict):
                args = {}
            out.append({"name": str(name or ""), "arguments": args, "id": str(it.get("id") or "")})

    fc = message.get("function_call")
    if isinstance(fc, dict):
        args = _parse_attr_value(fc.get("arguments"))
        if not isinstance(args, dict):
            args = {}
        out.append({"name": str(fc.get("name") or ""), "arguments": args, "id": str(message.get("id") or "")})

    if not out and message.get("type") == "ToolCallRequestEvent" and isinstance(message.get("content"), list):
        for it in message.get("content"):
            if not isinstance(it, dict):
                continue
            args = _parse_attr_value(it.get("arguments"))
            if not isinstance(args, dict):
                args = {}
            out.append({"name": str(it.get("name") or ""), "arguments": args, "id": str(it.get("id") or "")})

    # Fallback: some traces serialize tool calls as JSON text in `content`.
    # Example:
    #   content='[{"arguments":"{\\"ticker\\":\\"AAPL\\"}","id":"call_x","name":"get_stock_price"}]'
    if not out:
        content_obj = _parse_attr_value(message.get("content"))
        if isinstance(content_obj, list):
            for it in content_obj:
                if not isinstance(it, dict):
                    continue
                name = str(it.get("name") or "").strip()
                if not name:
                    continue
                args = _parse_attr_value(it.get("arguments"))
                if not isinstance(args, dict):
                    args = _parse_attr_value(it.get("args"))
                if not isinstance(args, dict):
                    args = {}
                out.append(
                    {
                        "name": name,
                        "arguments": args,
                        "id": str(it.get("id") or it.get("tool_call_id") or it.get("call_id") or ""),
                    }
                )

    dedup: Dict[str, Dict[str, Any]] = {}
    for item in out:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        dedup[key] = item
    return list(dedup.values())


def _infer_type(message: Dict[str, Any]) -> str:
    role_l = str(message.get("role") or "").strip().lower()
    source_l = str(message.get("source") or "").strip().lower()
    type_l = str(message.get("type") or "").strip().lower()
    name_l = str(message.get("name") or "").strip().lower()
    explicit_user_origin = (source_l in USER_SOURCEISH) or (name_l in USER_SOURCEISH)
    ambiguous_origin = not source_l and not name_l

    # Prefer explicit message type over source/role, because many frameworks
    # put producer-agent names in source even for tool execution events.
    if type_l in USER_ROLEISH:
        return "user" if (explicit_user_origin or ambiguous_origin) else "agent"
    if type_l in TEXTMESSAGE_ROLEISH:
        return "user" if source_l in USER_SOURCEISH else "agent"
    if type_l in SYSTEM_ROLEISH:
        return "system"
    if type_l in TOOL_ROLEISH:
        return "tool"
    if type_l in AGENT_ROLEISH:
        return "agent"

    if role_l in USER_ROLEISH:
        return "user" if (explicit_user_origin or ambiguous_origin) else "agent"
    if role_l in TEXTMESSAGE_ROLEISH:
        return "user" if source_l in USER_SOURCEISH else "agent"
    if role_l in SYSTEM_ROLEISH:
        return "system"
    if role_l in TOOL_ROLEISH:
        return "tool"
    if role_l in AGENT_ROLEISH:
        return "agent"

    # Tool execution payload pattern fallback:
    # content is often a list of dicts with call_id/tool_call_id + content.
    content_obj = _parse_attr_value(message.get("content"))
    if isinstance(content_obj, list) and content_obj and all(isinstance(it, dict) for it in content_obj):
        has_tool_exec_shape = any(
            (
                any(str(it.get(k) or "").strip() for k in TOOL_EXEC_PAYLOAD_ID_KEYS)
            )
            and any(k in it for k in TOOL_EXEC_PAYLOAD_OUTPUT_KEYS)
            for it in content_obj
            if isinstance(it, dict)
        )
        if has_tool_exec_shape:
            return "tool"

    if any(message.get(k) is not None for k in TOP_LEVEL_TOOL_RESULT_ID_KEYS):
        return "tool"
    if _normalize_tool_calls(message):
        return "agent"
    if source_l and source_l not in USER_SOURCEISH:
        return "agent"
    return "agent"


def _infer_name(message: Dict[str, Any], ev_type: str) -> str:
    if ev_type == "user":
        return "user"
    return _normalize_event_name(message.get("name") or message.get("source"))


def _normalize_event_name(raw: Any) -> str:
    s = str(raw or "").strip()
    # Normalize framework instance suffixes like "agent_1", "RunnableSequence_5".
    # Keep only the semantic base name.
    return re.sub(r"_(\d+)$", "", s)


def _norm_name(s: Any) -> str:
    return "".join(ch for ch in str(s or "").lower() if ch.isalnum())


def _is_generic_runtime_name(name: str, ev_type: str) -> bool:
    n = _norm_name(name)
    if not n:
        return True
    if ev_type == "agent":
        return n in GENERIC_AGENT_RUNTIME_NAMES_NORM
    if ev_type == "tool":
        return n in GENERIC_TOOL_RUNTIME_NAMES_NORM
    return False


def _tool_call_ids_from_message(message: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    tcid = str(message.get("tool_call_id") or "").strip()
    if tcid:
        out.append(tcid)
    content_obj = _parse_attr_value(message.get("content"))
    if isinstance(content_obj, list):
        for it in content_obj:
            if not isinstance(it, dict):
                continue
            for k in TOOL_EXEC_PAYLOAD_ID_KEYS:
                cid = str(it.get(k) or "").strip()
                if cid:
                    out.append(cid)
    dedup: List[str] = []
    seen = set()
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        dedup.append(x)
    return dedup


def _is_confident_user_event(e: Dict[str, Any]) -> bool:
    if str(e.get("type") or "") != "user":
        return False
    if e.get("tool_calls") or str(e.get("tool_call_id") or ""):
        return False
    content = str(e.get("_canonical_content") or "")
    if not content:
        return False
    if _routing_payload(content) is not None:
        return False
    paths: List[str] = [str(e.get("path") or "")]
    for a in (e.get("_aliases") or []):
        if isinstance(a, dict):
            paths.append(str(a.get("path") or ""))
    for a in (e.get("aliases") or []):
        if isinstance(a, dict):
            paths.append(str(a.get("path") or ""))
    if not any(any(tok in p for tok in USER_STRONG_PATH_TOKENS) for p in paths):
        return False
    return True


def _resolve_tool_name_from_message(
    message: Dict[str, Any],
    fallback_name: str,
    pending_tool_name_by_call_id: Dict[str, str],
) -> str:
    # 1) direct top-level id linkage
    tcid = str(message.get("tool_call_id") or "").strip()
    if tcid and tcid in pending_tool_name_by_call_id:
        return _normalize_event_name(pending_tool_name_by_call_id[tcid])

    # 2) content list id linkage
    content_obj = _parse_attr_value(message.get("content"))
    if isinstance(content_obj, list):
        for it in content_obj:
            if not isinstance(it, dict):
                continue
            for k in TOOL_EXEC_PAYLOAD_ID_KEYS:
                cid = str(it.get(k) or "").strip()
                if cid and cid in pending_tool_name_by_call_id:
                    return _normalize_event_name(pending_tool_name_by_call_id[cid])
            explicit_name = str(it.get("name") or "").strip()
            if explicit_name:
                return _normalize_event_name(explicit_name)

    return _normalize_event_name(fallback_name)


def _span_type_name(span: Dict[str, Any]) -> str:
    attrs = span.get("attributes") if isinstance(span.get("attributes"), dict) else {}
    st = _parse_attr_value(attrs.get("mlflow.spanType"))
    return str(st or "").strip().strip('"').upper()


def _fallback_tool_content_from_outputs(outputs: Any) -> str:
    if not isinstance(outputs, dict):
        return ""
    for key in ("result", "result_as_answer", "output", "value", "text", "content", "message"):
        v = outputs.get(key)
        if v is None:
            continue
        if isinstance(v, str):
            s = v.strip()
            if s:
                return s
            continue
        try:
            return json.dumps(v, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(v)
    return ""


def _has_equivalent_tool_event_for_span(
    raw_events: List[Dict[str, Any]],
    *,
    span_id: str,
    tool_name: str,
    content: str,
) -> bool:
    tn = _norm_name(tool_name)
    cc = _canonical_content(content)
    if not tn or not cc:
        return False
    for e in raw_events:
        if not isinstance(e, dict):
            continue
        if str(e.get("type") or "") != "tool":
            continue
        if str(e.get("span_id") or "") != str(span_id or ""):
            continue
        if _norm_name(str(e.get("name") or "")) != tn:
            continue
        if _canonical_content(str(e.get("content") or "")) != cc:
            continue
        return True
    return False


def _is_wrapper_name(name: str) -> bool:
    n = str(name or "").strip().lower()
    if not n:
        return True
    if any(tok in n for tok in WRAPPER_NAME_TOKENS):
        return True
    if n in WRAPPER_NAME_EXACT:
        return True
    return False


def _event_time_ns(
    message: Dict[str, Any],
    start_ns: int,
    end_ns: int,
    ev_type: str,
    container_path: str,
) -> int:
    created = _parse_iso_to_ns(message.get("created_at"))
    if created is not None:
        return created
    # For transcript-like events extracted from span outputs, anchor to span end.
    # This avoids pulling later output-list items (for example control/user relays)
    # ahead of tool results due to start-time bias.
    if container_path.startswith("spanOutputs"):
        return end_ns or start_ns
    if ev_type in USER_SYSTEM_TYPES:
        return start_ns
    if ev_type == "tool":
        return end_ns or start_ns
    if _normalize_tool_calls(message):
        return start_ns
    return end_ns or start_ns


def parse_trace_with_schema(trace_file: Path, schema: Schema) -> Dict[str, Any]:
    trace = json.loads(trace_file.read_text(encoding="utf-8"))
    spans = _extract_spans(trace)
    parent_by_span: Dict[str, str] = {}
    span_name_by_span: Dict[str, str] = {}
    for s in spans:
        sid = str(s.get("span_id") or "")
        if not sid:
            continue
        parent_by_span[sid] = str(s.get("parent_span_id") or "")
        span_name_by_span[sid] = _normalize_event_name(s.get("name"))

    def _nearest_semantic_span_name(span_id: str) -> str:
        cur = str(span_id or "")
        seen: set[str] = set()
        depth = 0
        while cur and cur not in seen and depth < 32:
            seen.add(cur)
            nm = str(span_name_by_span.get(cur) or "").strip()
            if nm and not _is_wrapper_name(nm):
                return nm
            cur = str(parent_by_span.get(cur) or "")
            depth += 1
        return ""

    raw_events: List[Dict[str, Any]] = []
    pending_tool_name_by_call_id: Dict[str, str] = {}
    idx = 0
    for span_idx, span in enumerate(spans):
        span_id = str(span.get("span_id") or "")
        start_ns = int(span.get("start_time_unix_nano") or 0)
        end_ns = int(span.get("end_time_unix_nano") or 0)
        roots = _span_roots(span)
        span_type = _span_type_name(span)
        span_name = _normalize_event_name(span.get("name"))
        span_emitted_tool_event = False

        for p in schema.message_list_paths:
            resolved_lists = [x for x in _resolve_path_all(roots, p) if isinstance(x, list)]
            for lst in resolved_lists:
                for j, item in enumerate(lst):
                    if not isinstance(item, dict):
                        continue
                    ev_type = _infer_type(item)
                    if ev_type == "tool":
                        split_parts = _split_tool_batch_payload(item)
                        if split_parts:
                            fallback_tool_name = _resolve_tool_name_from_message(
                                item,
                                _infer_name(item, ev_type),
                                pending_tool_name_by_call_id,
                            )
                            for part in sorted(split_parts, key=lambda x: _norm_name(x.get("tool_call_id"))):
                                part_name = _normalize_event_name(
                                    part.get("name")
                                    or pending_tool_name_by_call_id.get(str(part.get("tool_call_id") or ""), "")
                                    or fallback_tool_name
                                )
                                raw_events.append(
                                    {
                                        "event_id": f"E{idx:06d}",
                                        "type": "tool",
                                        "name": part_name,
                                        "content": str(part.get("content") or ""),
                                        "tool_calls": [],
                                        "span_id": span_id,
                                        "span_index": span_idx,
                                        "start_ns": start_ns,
                                        "end_ns": end_ns,
                                        "time_ns": _event_time_ns(item, start_ns, end_ns, "tool", p),
                                        "path": f"{p}[{j}]",
                                        "container_path": p,
                                        "list_index": j,
                                        "tool_call_id": str(part.get("tool_call_id") or ""),
                                    }
                                )
                                idx += 1
                                span_emitted_tool_event = True
                            # Batch handled; skip default tool-event path.
                            continue
                    tool_calls = _normalize_tool_calls(item)
                    content = _event_content_from_message(item, ev_type, tool_calls)
                    tool_calls = _strip_stale_tool_calls(ev_type, content, tool_calls)
                    inferred_name = _infer_name(item, ev_type)
                    if ev_type == "tool":
                        inferred_name = _resolve_tool_name_from_message(item, inferred_name, pending_tool_name_by_call_id)
                    if not content.strip() and not tool_calls and ev_type != "user":
                        # skip empty noise unless user marker
                        continue
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        tcid = str(tc.get("id") or "").strip()
                        tname = str(tc.get("name") or "").strip()
                        if tcid and tname:
                            pending_tool_name_by_call_id[tcid] = _normalize_event_name(tname)
                    for cid in _tool_call_ids_from_message(item):
                        if cid and inferred_name:
                            if cid not in pending_tool_name_by_call_id and not _is_generic_runtime_name(inferred_name, "tool"):
                                pending_tool_name_by_call_id[cid] = inferred_name
                    raw_events.append(
                        {
                            "event_id": f"E{idx:06d}",
                            "type": ev_type,
                            "name": inferred_name,
                            "content": content,
                            "tool_calls": tool_calls,
                            "span_id": span_id,
                            "span_index": span_idx,
                            "start_ns": start_ns,
                            "end_ns": end_ns,
                            "time_ns": _event_time_ns(item, start_ns, end_ns, ev_type, p),
                            "path": f"{p}[{j}]",
                            "container_path": p,
                            "list_index": j,
                            "tool_call_id": str(item.get("tool_call_id") or ""),
                        }
                    )
                    idx += 1
                    if ev_type == "tool":
                        span_emitted_tool_event = True

        # Additional single-message extractor for LLM choice-style spans.
        outputs = roots.get("spanOutputs")
        if isinstance(outputs, dict) and isinstance(outputs.get("choices"), list) and outputs["choices"]:
            first = outputs["choices"][0] if isinstance(outputs["choices"][0], dict) else {}
            msg = first.get("message")
            if isinstance(msg, dict):
                ev_type = _infer_type(msg)
                tool_calls = _normalize_tool_calls(msg)
                content = _event_content_from_message(msg, ev_type, tool_calls)
                tool_calls = _strip_stale_tool_calls(ev_type, content, tool_calls)
                inferred_name = _infer_name(msg, ev_type)
                if ev_type == "tool":
                    inferred_name = _resolve_tool_name_from_message(msg, inferred_name, pending_tool_name_by_call_id)
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    tcid = str(tc.get("id") or "").strip()
                    tname = str(tc.get("name") or "").strip()
                    if tcid and tname:
                        pending_tool_name_by_call_id[tcid] = _normalize_event_name(tname)
                raw_events.append(
                    {
                        "event_id": f"E{idx:06d}",
                        "type": ev_type,
                        "name": inferred_name,
                        "content": content,
                        "tool_calls": tool_calls,
                        "span_id": span_id,
                        "span_index": span_idx,
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                        "time_ns": _event_time_ns(
                            msg,
                            start_ns,
                            end_ns,
                            ev_type,
                            "spanOutputs.choices[0].message",
                        ),
                        "path": "spanOutputs.choices[0].message",
                        "container_path": "spanOutputs.choices[0].message",
                        "list_index": 0,
                        "tool_call_id": str(msg.get("tool_call_id") or ""),
                    }
                )
                idx += 1
                if ev_type == "tool":
                    span_emitted_tool_event = True

        # Capture routing/control transitions as agent events (not a separate type).
        if isinstance(outputs, dict):
            routing_candidates: List[Tuple[str, List[str]]] = []
            for k in ROUTING_DIRECT_KEYS:
                targets = _routing_targets(outputs.get(k))
                if targets:
                    routing_candidates.append((f"spanOutputs.{k}", targets))
            upd = outputs.get("update")
            if isinstance(upd, dict):
                targets = _routing_targets(upd.get("next"))
                if targets:
                    routing_candidates.append(("spanOutputs.update.next", targets))

            for pth, targets in routing_candidates:
                raw_events.append(
                    {
                        "event_id": f"E{idx:06d}",
                        "type": "agent",
                        "name": "",
                        "content": json.dumps({"routing": targets}, ensure_ascii=False, sort_keys=True),
                        "tool_calls": [],
                        "span_id": span_id,
                        "span_index": span_idx,
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                        "time_ns": end_ns or start_ns,
                        "path": pth,
                        "container_path": pth,
                        "list_index": 0,
                        "tool_call_id": "",
                    }
                )
                idx += 1

        # Additive fallback:
        # If this span is explicitly TOOL-typed but no tool event was extracted
        # from message-shaped payloads, emit one generic tool-result event.
        if span_type == "TOOL" and not span_emitted_tool_event:
            tool_name = span_name
            if not tool_name or _is_generic_runtime_name(tool_name, "tool"):
                tool_name = _nearest_semantic_span_name(span_id)
            if not tool_name or _is_generic_runtime_name(tool_name, "tool"):
                tool_name = "tool"
            fb_content = _fallback_tool_content_from_outputs(outputs)
            if not fb_content:
                # keep behavior conservative: only emit when there is any output payload
                # to avoid generating empty synthetic noise.
                if isinstance(outputs, dict) and outputs:
                    try:
                        fb_content = json.dumps(outputs, ensure_ascii=False, sort_keys=True)
                    except Exception:
                        fb_content = str(outputs)
            if fb_content:
                if _has_equivalent_tool_event_for_span(
                    raw_events,
                    span_id=span_id,
                    tool_name=tool_name,
                    content=fb_content,
                ):
                    continue
                raw_events.append(
                    {
                        "event_id": f"E{idx:06d}",
                        "type": "tool",
                        "name": _normalize_event_name(tool_name),
                        "content": fb_content,
                        "tool_calls": [],
                        "span_id": span_id,
                        "span_index": span_idx,
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                        "time_ns": end_ns or start_ns,
                        "path": "spanOutputs",
                        "container_path": "spanOutputs",
                        "list_index": 0,
                        "tool_call_id": "",
                    }
                )
                idx += 1

        # Capture explicit exceptions from span events.
        span_events = span.get("events")
        if isinstance(span_events, list):
            for k, sev in enumerate(span_events):
                if not isinstance(sev, dict):
                    continue
                if str(sev.get("name") or "").strip().lower() != EXCEPTION_EVENT_NAME:
                    continue
                attrs = sev.get("attributes") if isinstance(sev.get("attributes"), dict) else {}
                err_msg = str(attrs.get("exception.message") or "").strip()
                err_type = str(attrs.get("exception.type") or "").strip()
                if not err_msg and not err_type:
                    continue
                ev_time = _event_nano_to_ns(sev.get("time_unix_nano")) or end_ns or start_ns
                raw_events.append(
                    {
                        "event_id": f"E{idx:06d}",
                        "type": "error",
                        "name": err_type,
                        "content": err_msg,
                        "tool_calls": [],
                        "span_id": span_id,
                        "span_index": span_idx,
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                        "time_ns": ev_time,
                        "path": f"events[{k}]",
                        "container_path": "events",
                        "list_index": k,
                        "tool_call_id": "",
                        "error_stacktrace": str(attrs.get("exception.stacktrace") or ""),
                    }
                )
                idx += 1

    # Deterministic dedupe.
    dedup: Dict[str, Dict[str, Any]] = {}
    for e in raw_events:
        tc_norm = json.dumps(sorted(e.get("tool_calls") or [], key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False)), sort_keys=True, ensure_ascii=False)
        canon_content = _canonical_content(e.get("content") or "")
        # tool_call_id is only identity-bearing for tool result events.
        # For non-tool events it often appears as relay metadata and blocks safe dedupe.
        dedupe_tool_call_id = str(e.get("tool_call_id") or "") if str(e.get("type") or "") == "tool" else ""
        # Do not globally dedupe routing events across turns; repeated identical
        # routes from the same supervisor at different times are semantically distinct.
        routing_payload = _routing_payload(str(e.get("content") or ""))
        key_parts = [
            e.get("type") or "",
            e.get("name") or "",
            canon_content,
            tc_norm,
            dedupe_tool_call_id,
        ]
        if routing_payload is not None and str(e.get("type") or "") == "agent":
            key_parts.append(e.get("event_id") or "")
        key = "||".join(key_parts)
        if key in dedup:
            old = dedup[key]
            old["time_ns"] = min(int(old.get("time_ns") or 0), int(e.get("time_ns") or 0))
            old_paths = old.setdefault("_paths", [old.get("path", "")])
            if e["path"] not in old_paths:
                if len(old_paths) < 8:
                    old_paths.append(e["path"])
                else:
                    old["_path_overflow"] = int(old.get("_path_overflow") or 0) + 1
            _append_alias(old, e.get("span_id"), e.get("path"))
            continue
        ne = dict(e)
        ne["_paths"] = [e["path"]]
        ne["_path_overflow"] = 0
        ne["_canonical_content"] = canon_content
        ne["_aliases"] = []
        ne["_alias_overflow"] = 0
        _append_alias(ne, e.get("span_id"), e.get("path"))
        dedup[key] = ne

    events = list(dedup.values())

    # Drop echoed relay duplicates: same content/type/name appears first in outputs and later in inputs.
    # Keep the outputs-origin event and drop the inputs-origin echo when both are plain text events.
    # Exception: for user events, prefer inputs-origin and drop outputs-origin copy.
    drop_idx: set[int] = set()
    for i, a in enumerate(events):
        if i in drop_idx:
            continue
        if a.get("tool_calls") or str(a.get("tool_call_id") or ""):
            continue
        a_type = str(a.get("type") or "")
        a_name = str(a.get("name") or "")
        a_content = str(a.get("_canonical_content") or "")
        if not a_content:
            continue
        a_is_output = "spanOutputs" in str(a.get("path") or "")
        for j, b in enumerate(events):
            if i == j or j in drop_idx:
                continue
            if b.get("tool_calls") or str(b.get("tool_call_id") or ""):
                continue
            if str(b.get("type") or "") != a_type:
                continue
            if str(b.get("name") or "") != a_name:
                continue
            if str(b.get("_canonical_content") or "") != a_content:
                continue
            b_is_input = "spanInputs" in str(b.get("path") or "")
            if not (a_is_output and b_is_input):
                continue
            if int(b.get("time_ns") or 0) < int(a.get("time_ns") or 0):
                continue
            if a_type == "user":
                # user must come from inputs; keep b, drop a
                keep = b
                drop = a
                keep_paths = keep.setdefault("_paths", [keep.get("path", "")])
                for pth in drop.get("_paths") or [drop.get("path", "")]:
                    if pth not in keep_paths:
                        if len(keep_paths) < 8:
                            keep_paths.append(pth)
                        else:
                            keep["_path_overflow"] = int(keep.get("_path_overflow") or 0) + 1
                _merge_aliases(keep, drop)
                drop_idx.add(i)
                break
            drop_idx.add(j)

    if drop_idx:
        events = [e for i, e in enumerate(events) if i not in drop_idx]

    # Second dedupe pass (extra-safe):
    # collapse nearby named-vs-unnamed relay duplicates for agent text payloads.
    # Rules:
    # - same type=agent + same canonical content
    # - one has name and the other has empty name
    # - no tool_calls / no tool_call_id
    # - close timestamps
    # - provenance suggests relay-style message paths
    relay_hint_tokens = RELAY_HINT_TOKENS
    drop_idx_2: set[int] = set()
    for i, a in enumerate(events):
        if i in drop_idx_2:
            continue
        if str(a.get("type") or "") != "agent":
            continue
        if a.get("tool_calls") or str(a.get("tool_call_id") or ""):
            continue
        a_content = str(a.get("_canonical_content") or "")
        if not a_content:
            continue
        a_name = str(a.get("name") or "").strip()
        a_path = str(a.get("path") or "")
        if not any(tok in a_path for tok in relay_hint_tokens):
            continue

        for j, b in enumerate(events):
            if i == j or j in drop_idx_2:
                continue
            if str(b.get("type") or "") != "agent":
                continue
            if b.get("tool_calls") or str(b.get("tool_call_id") or ""):
                continue
            if str(b.get("_canonical_content") or "") != a_content:
                continue
            t_a = int(a.get("time_ns") or 0)
            t_b = int(b.get("time_ns") or 0)
            if abs(t_a - t_b) > RELAY_NAMELESS_DEDUP_MAX_GAP_NS:
                continue
            b_path = str(b.get("path") or "")
            if not any(tok in b_path for tok in relay_hint_tokens):
                continue

            b_name = str(b.get("name") or "").strip()
            if bool(a_name) == bool(b_name):
                continue

            # Preserve repeated short acknowledgements; only dedupe substantial relay text.
            if len(a_content) < 40:
                continue

            keep_i = i if a_name else j
            drop_j = j if a_name else i
            if drop_j in drop_idx_2:
                continue
            # Prefer dropping unnamed relay that is not later than the named copy.
            if str(events[drop_j].get("name") or "").strip():
                continue
            if int(events[keep_i].get("time_ns") or 0) < int(events[drop_j].get("time_ns") or 0):
                continue

            keep = events[keep_i]
            drop = events[drop_j]
            keep_paths = keep.setdefault("_paths", [keep.get("path", "")])
            drop_paths = drop.get("_paths") or [drop.get("path", "")]
            for pth in drop_paths:
                if pth not in keep_paths:
                    if len(keep_paths) < 8:
                        keep_paths.append(pth)
                    else:
                        keep["_path_overflow"] = int(keep.get("_path_overflow") or 0) + 1
            _merge_aliases(keep, drop)
            drop_idx_2.add(drop_j)

    if drop_idx_2:
        events = [e for i, e in enumerate(events) if i not in drop_idx_2]

    # User echo pruning:
    # drop user events that are near-duplicate echoes of agent content.
    drop_user_echo_idx: set[int] = set()
    for i, a in enumerate(events):
        if str(a.get("type") or "") != "user":
            continue
        if _is_confident_user_event(a):
            continue
        if a.get("tool_calls") or str(a.get("tool_call_id") or ""):
            continue
        a_content = str(a.get("_canonical_content") or "")
        if len(a_content) < 32:
            continue
        t_a = int(a.get("time_ns") or 0)
        for j, b in enumerate(events):
            if i == j:
                continue
            if str(b.get("type") or "") != "agent":
                continue
            if b.get("tool_calls") or str(b.get("tool_call_id") or ""):
                continue
            if str(b.get("_canonical_content") or "") != a_content:
                continue
            t_b = int(b.get("time_ns") or 0)
            if t_a < t_b:
                continue
            if t_a - t_b > USER_AGENT_ECHO_DEDUP_MAX_GAP_NS:
                continue
            drop_user_echo_idx.add(i)
            keep = events[j]
            drop = events[i]
            keep_paths = keep.setdefault("_paths", [keep.get("path", "")])
            for pth in drop.get("_paths") or [drop.get("path", "")]:
                if pth not in keep_paths:
                    if len(keep_paths) < 8:
                        keep_paths.append(pth)
                    else:
                        keep["_path_overflow"] = int(keep.get("_path_overflow") or 0) + 1
            _merge_aliases(keep, drop)
            break
    if drop_user_echo_idx:
        events = [e for i, e in enumerate(events) if i not in drop_user_echo_idx]

    # Tool duplicate reconciliation:
    # Some traces expose the same tool output via both message extraction and
    # TOOL-span fallback. Collapse strict duplicates (same name+content) while
    # preserving the richer event where available.
    tool_seen: Dict[Tuple[str, str], int] = {}
    tool_drop: set[int] = set()
    for i, e in enumerate(events):
        if str(e.get("type") or "") != "tool":
            continue
        nm = _norm_name(e.get("name"))
        cc = str(e.get("_canonical_content") or "")
        if not nm or not cc:
            continue
        k = (nm, cc)
        if k not in tool_seen:
            tool_seen[k] = i
            continue
        j = tool_seen[k]
        keep = events[j]
        cur = e
        keep_tcid = str(keep.get("tool_call_id") or "").strip()
        cur_tcid = str(cur.get("tool_call_id") or "").strip()
        # Prefer event with explicit tool_call_id, then earlier event.
        if (not keep_tcid and cur_tcid):
            tool_drop.add(j)
            tool_seen[k] = i
        else:
            tool_drop.add(i)
    if tool_drop:
        events = [e for i, e in enumerate(events) if i not in tool_drop]

    # User confidence reconciliation:
    # Keep user labels only when strong path/position evidence indicates original
    # user task input. Otherwise demote to agent to avoid user/agent echo noise.
    for e in events:
        if str(e.get("type") or "") != "user":
            continue
        paths: List[str] = [str(e.get("path") or "")]
        for a in (e.get("_aliases") or []):
            if isinstance(a, dict):
                paths.append(str(a.get("path") or ""))
        for a in (e.get("aliases") or []):
            if isinstance(a, dict):
                paths.append(str(a.get("path") or ""))
        if not any("spanInputs" in p for p in paths):
            e["type"] = "agent"
            if str(e.get("name") or "").strip().lower() == "user":
                e["name"] = ""
            continue
        if _is_confident_user_event(e):
            continue
        e["type"] = "agent"
        if str(e.get("name") or "").strip().lower() == "user":
            e["name"] = ""

    # Optional safeguard: preserve one primary user task event when no confident
    # user survives. Disabled by default to avoid false-positive user labels.
    if ENABLE_USER_FALLBACK_PROMOTION and not any(str(e.get("type") or "") == "user" for e in events):
        candidates: List[Dict[str, Any]] = []
        for e in events:
            if e.get("tool_calls") or str(e.get("tool_call_id") or ""):
                continue
            content = str(e.get("_canonical_content") or "")
            if not content or _routing_payload(content) is not None:
                continue
            path = str(e.get("path") or "")
            if any(tok in path for tok in USER_EXCLUDED_PATH_TOKENS):
                continue
            if not any(tok in path for tok in USER_STRONG_PATH_TOKENS):
                continue
            if int(e.get("list_index") or 0) > USER_MAX_CONFIDENT_LIST_INDEX:
                continue
            candidates.append(e)
        if candidates:
            chosen = min(
                candidates,
                key=lambda e: (
                    int(e.get("time_ns") or 0),
                    int(e.get("span_index") or 0),
                    int(e.get("list_index") or 0),
                ),
            )
            chosen["type"] = "user"
            chosen["name"] = "user"

    # Enforce a single user task event per trace:
    # keep the earliest confident user and demote any additional user labels.
    user_events = [e for e in events if str(e.get("type") or "") == "user"]
    if len(user_events) > 1:
        keep = min(
            user_events,
            key=lambda e: (
                int(e.get("time_ns") or 0),
                int(e.get("span_index") or 0),
                int(e.get("list_index") or 0),
            ),
        )
        for e in user_events:
            if e is keep:
                continue
            e["type"] = "agent"
            if str(e.get("name") or "").strip().lower() == "user":
                e["name"] = ""


    # Third dedupe pass: collapse routing relay bursts (same routing target repeated
    # by parser/wrapper/supervisor spans within a short time window).
    keep_routing_idx: set[int] = set(range(len(events)))
    routing_items: List[Tuple[int, Dict[str, Any], str]] = []
    for idx_e, ev in enumerate(events):
        rp = _routing_payload(str(ev.get("content") or ""))
        if (
            str(ev.get("type") or "") == "agent"
            and not ev.get("tool_calls")
            and not str(ev.get("tool_call_id") or "")
            and rp is not None
        ):
            routing_items.append((idx_e, ev, rp))

    routing_items.sort(key=lambda x: (int(x[1].get("time_ns") or 0), x[0]))

    k = 0
    while k < len(routing_items):
        base_idx, base_ev, base_payload = routing_items[k]
        cluster = [base_idx]
        last_t = int(base_ev.get("time_ns") or 0)
        m = k + 1
        while m < len(routing_items):
            idx_m, ev_m, payload_m = routing_items[m]
            t_m = int(ev_m.get("time_ns") or 0)
            if payload_m != base_payload:
                break
            if t_m - last_t > ROUTING_CLUSTER_GAP_NS:
                break
            cluster.append(idx_m)
            last_t = t_m
            m += 1

        if len(cluster) == 1:
            k = m
            continue

        def _routing_rank(e: Dict[str, Any]) -> Tuple[int, int, int]:
            p = str(e.get("path") or "")
            n = str(e.get("name") or "").lower()
            path_score = 0
            if ROUTING_PATH_PRIORITY_KEYS[0] in p:
                path_score += 40
            if ROUTING_PATH_PRIORITY_KEYS[1] in p:
                path_score += 30
            if ROUTING_PATH_PRIORITY_KEYS[2] in p:
                path_score += 10
            for prefix, penalty in ROUTING_RANK_PREFIX_PENALTIES.items():
                if n.startswith(prefix):
                    path_score += penalty
            if any(tok in n for tok in ROUTING_RANK_NAME_BONUS_TOKENS):
                path_score += 4
            return (
                path_score,
                1 if str(e.get("name") or "").strip() else 0,
                -int(e.get("time_ns") or 0),
            )

        keep = max(cluster, key=lambda idx_: _routing_rank(events[idx_]))

        keep_e = events[keep]
        keep_paths = keep_e.setdefault("_paths", [keep_e.get("path", "")])
        for idx_ in cluster:
            if idx_ == keep:
                continue
            if idx_ in keep_routing_idx:
                keep_routing_idx.remove(idx_)
            drop_paths = events[idx_].get("_paths") or [events[idx_].get("path", "")]
            for pth in drop_paths:
                if pth not in keep_paths:
                    if len(keep_paths) < 8:
                        keep_paths.append(pth)
                    else:
                        keep_e["_path_overflow"] = int(keep_e.get("_path_overflow") or 0) + 1
            keep_e["time_ns"] = min(int(keep_e.get("time_ns") or 0), int(events[idx_].get("time_ns") or 0))
            _merge_aliases(keep_e, events[idx_])

        k = m

    events = [e for idx_, e in enumerate(events) if idx_ in keep_routing_idx]

    # Fourth dedupe pass: prune internal terminal relay routing events when they are
    # transient bridge signals (FINISH/__end__) between two concrete handoffs.
    terminal_targets = TERMINAL_ROUTING_TARGETS
    events.sort(
        key=lambda e: (
            int(e.get("time_ns") or 0),
            int(e.get("span_index") or 0),
            str(e.get("container_path") or ""),
            int(e.get("list_index") or 0),
            e.get("event_id") or "",
        )
    )
    drop_terminal_idx: set[int] = set()
    for i, e in enumerate(events):
        rp = _routing_payload(str(e.get("content") or ""))
        if rp is None:
            continue
        targets = [t.lower() for t in _routing_targets_from_payload(rp)]
        if not targets or not all(t in terminal_targets for t in targets):
            continue

        prev_idx = -1
        next_idx = -1
        for j in range(i - 1, -1, -1):
            if _routing_payload(str(events[j].get("content") or "")) is not None:
                prev_idx = j
                break
        for j in range(i + 1, len(events)):
            if _routing_payload(str(events[j].get("content") or "")) is not None:
                next_idx = j
                break
        if prev_idx < 0 or next_idx < 0:
            continue

        prev_targets = [t.lower() for t in _routing_targets_from_payload(_routing_payload(str(events[prev_idx].get("content") or "")))]
        next_targets = [t.lower() for t in _routing_targets_from_payload(_routing_payload(str(events[next_idx].get("content") or "")))]
        if not prev_targets or not next_targets:
            continue
        if all(t in terminal_targets for t in next_targets):
            continue
        if int(events[next_idx].get("time_ns") or 0) - int(events[prev_idx].get("time_ns") or 0) > ROUTING_TERMINAL_BRIDGE_MAX_GAP_NS:
            continue
        drop_terminal_idx.add(i)

    if drop_terminal_idx:
        events = [e for i, e in enumerate(events) if i not in drop_terminal_idx]

    # Fifth pass: wrapper-terminal routing noise pruning.
    # Drop FINISH/__end__ routing emitted by wrapper/framework spans when a concrete
    # non-terminal routing handoff follows shortly after.
    wrapper_name_tokens = WRAPPER_TERMINAL_NAME_TOKENS
    events.sort(
        key=lambda e: (
            int(e.get("time_ns") or 0),
            int(e.get("span_index") or 0),
            str(e.get("container_path") or ""),
            int(e.get("list_index") or 0),
            e.get("event_id") or "",
        )
    )
    drop_wrapper_terminal_idx: set[int] = set()
    for i, e in enumerate(events):
        rp = _routing_payload(str(e.get("content") or ""))
        if rp is None:
            continue
        targets = [t.lower() for t in _routing_targets_from_payload(rp)]
        if not targets or not all(t in terminal_targets for t in targets):
            continue
        name_l = str(e.get("name") or "").strip().lower()
        if not any(tok in name_l for tok in wrapper_name_tokens):
            continue
        next_idx = -1
        for j in range(i + 1, len(events)):
            if _routing_payload(str(events[j].get("content") or "")) is not None:
                next_idx = j
                break
        if next_idx < 0:
            continue
        next_targets = [
            t.lower()
            for t in _routing_targets_from_payload(_routing_payload(str(events[next_idx].get("content") or "")))
        ]
        if not next_targets or all(t in terminal_targets for t in next_targets):
            continue
        if int(events[next_idx].get("time_ns") or 0) - int(e.get("time_ns") or 0) > WRAPPER_TERMINAL_FOLLOWUP_MAX_GAP_NS:
            continue
        drop_wrapper_terminal_idx.add(i)

    if drop_wrapper_terminal_idx:
        events = [e for i, e in enumerate(events) if i not in drop_wrapper_terminal_idx]
    for e in events:
        aliases = [a for a in (e.get("_aliases") or []) if isinstance(a, dict)]
        if not aliases:
            aliases = [{"span_id": str(e.get("span_id") or ""), "path": str(e.get("path") or "")}]
        primary = aliases[0]
        if str(e.get("type") or "") == "user":
            for cand in aliases:
                if "spanInputs" in str(cand.get("path") or ""):
                    primary = cand
                    break
        e["span_id"] = str(primary.get("span_id") or e.get("span_id") or "")
        e["path"] = str(primary.get("path") or e.get("path") or "")
        extra = aliases[1:]
        if int(e.get("_alias_overflow") or 0) > 0:
            extra = extra + [{"span_id": "", "path": f"...(+{int(e.get('_alias_overflow') or 0)} more)"}]
        e["aliases"] = extra

    def _type_rank(e: Dict[str, Any]) -> int:
        t = str(e.get("type") or "")
        has_tc = bool(e.get("tool_calls"))
        if t == "user":
            return 0
        if t == "system":
            return 1
        if t == "agent" and has_tc:
            return 2
        if t == "tool":
            return 3
        if t == "agent":
            return 4
        return 9

    events.sort(
        key=lambda e: (
            int(e.get("time_ns") or 0),
            int(e.get("span_index") or 0),
            str(e.get("container_path") or ""),
            int(e.get("list_index") or 0),
            _type_rank(e),
            e.get("event_id") or "",
        )
    )

    # If a system event is immediately followed by an agent event, inherit
    # the adjacent agent name for easier downstream attribution.
    for i in range(len(events) - 1):
        cur = events[i]
        nxt = events[i + 1]
        if str(cur.get("type") or "") != "system":
            continue
        if str(nxt.get("type") or "") != "agent":
            continue
        if str(cur.get("name") or "").strip():
            continue
        next_name = str(nxt.get("name") or "").strip()
        if next_name:
            cur["name"] = next_name

    # Late name backfill for empty agent names, after dedupe is finalized.
    # Flow extraction should not use span names for agent identity.
    for e in events:
        if str(e.get("type") or "") != "agent":
            continue
        existing = str(e.get("name") or "").strip()
        # Preserve explicit message-level names only.
        if existing:
            continue
        # Prefer nearest non-wrapper semantic span owner for unresolved agents.
        span_id = str(e.get("span_id") or "").strip()
        cand = _normalize_event_name(_nearest_semantic_span_name(span_id)) if span_id else ""
        if cand and not _is_wrapper_name(cand):
            e["name"] = cand
            continue
        e["name"] = ""

    # Final consolidation after late name backfill.
    # Some events become mergeable only after names are filled.
    final_dedup: Dict[str, Dict[str, Any]] = {}
    for e in events:
        tc_norm = json.dumps(
            sorted(e.get("tool_calls") or [], key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False)),
            sort_keys=True,
            ensure_ascii=False,
        )
        canon_content = _canonical_content(e.get("content") or "")
        dedupe_tool_call_id = str(e.get("tool_call_id") or "") if str(e.get("type") or "") == "tool" else ""
        routing_payload = _routing_payload(str(e.get("content") or ""))
        key = "||".join(
            [
                str(e.get("type") or ""),
                str(e.get("name") or ""),
                canon_content,
                tc_norm,
                dedupe_tool_call_id,
                str(e.get("event_id") or "") if (routing_payload is not None and str(e.get("type") or "") == "agent") else "",
            ]
        )
        if key not in final_dedup:
            final_dedup[key] = dict(e)
            continue
        cur = final_dedup[key]
        if int(e.get("time_ns") or 0) < int(cur.get("time_ns") or 0):
            cur["time_ns"] = e.get("time_ns")
        _merge_aliases(cur, e)
    events = list(final_dedup.values())

    events.sort(
        key=lambda e: (
            int(e.get("time_ns") or 0),
            int(e.get("span_index") or 0),
            str(e.get("container_path") or ""),
            int(e.get("list_index") or 0),
            _type_rank(e),
            e.get("event_id") or "",
        )
    )

    _apply_span_identity_hints(events, trace)

    for i, e in enumerate(events, start=1):
        e["seq"] = i

    # Deterministic flow links (debug):
    # - tool flows: agent(tool_call) -> tool result (by id or tool name fallback)
    # - routing flows: routing event -> next agent message with matching target name
    flow_links: List[Dict[str, Any]] = []
    pending_tool_by_id: Dict[str, Tuple[str, int]] = {}
    pending_tool_by_name: Dict[str, List[Tuple[str, int]]] = {}
    pending_route_by_target: Dict[str, List[Tuple[str, int]]] = {}

    for e in events:
        seq = int(e.get("seq") or 0)
        et = str(e.get("type") or "")
        name = str(e.get("name") or "").strip()
        content = str(e.get("content") or "")
        tool_calls = e.get("tool_calls") or []

        # Open tool-call flows.
        if et == "agent" and isinstance(tool_calls, list) and tool_calls:
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_name = str(tc.get("name") or "").strip()
                tc_id = str(tc.get("id") or "").strip()
                if not tc_name and not tc_id:
                    continue
                fid = f"tool:{tc_id}" if tc_id else f"tool:{tc_name}@{seq}"
                opener = (fid, seq)
                if tc_id:
                    pending_tool_by_id[tc_id] = opener
                pending_tool_by_name.setdefault(tc_name, []).append(opener)

        # Close tool-call flows on tool output.
        if et == "tool":
            tcid = str(e.get("tool_call_id") or "").strip()
            fid = ""
            open_seq = 0
            if tcid and tcid in pending_tool_by_id:
                fid, open_seq = pending_tool_by_id.pop(tcid)
            else:
                cands = pending_tool_by_name.get(name) or []
                if cands:
                    fid, open_seq = cands.pop(0)
            if fid:
                flow_links.append(
                    {
                        "flow_id": fid,
                        "kind": "tool",
                        "target": name,
                        "open_seq": open_seq,
                        "close_seq": seq,
                    }
                )

        # Open routing flow.
        rt = _first_nonterminal_target(content) if et == "agent" and not tool_calls else ""
        if rt:
            rf = f"route:{seq}:{rt}"
            pending_route_by_target.setdefault(rt, []).append((rf, seq))

        # Close routing flow when target agent speaks (non-routing text event).
        is_routing_event = _routing_payload(content) is not None
        if et == "agent" and name and not tool_calls and content and not is_routing_event:
            cands = pending_route_by_target.get(name) or []
            if cands:
                rf, open_seq = cands.pop(0)
                flow_links.append(
                    {
                        "flow_id": rf,
                        "kind": "routing",
                        "target": name,
                        "open_seq": open_seq,
                        "close_seq": seq,
                    }
                )

    return {
        "trace_file": trace_file.name,
        "schema_use_case": schema.use_case,
        "events": [
            {
                "seq": e["seq"],
                "type": e["type"],
                "name": e["name"],
                "content": e["content"],
                "tool_calls": e["tool_calls"],
                "tool_call_id": e.get("tool_call_id") or "",
                "span_id": e["span_id"],
                "time_ns": e["time_ns"],
                "path": e["path"],
                "aliases": e.get("aliases") or [],
            }
            for e in events
        ],
        "debug": {
            "num_events_raw": len(raw_events),
            "num_events_deduped": len(events),
            "flow_links": flow_links,
        },
    }


def _pick_one_trace(trace_files: List[Path]) -> Path:
    return sorted(trace_files)[0]


def run(base_dir: Path, out_dir: Path, use_cases: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"use_cases": {}}

    for uc in use_cases:
        traces_dir = base_dir / f"use_case_{uc}_outputs" / "validation" / "traces"
        trace_files = sorted(traces_dir.glob("*.json"))
        if not trace_files:
            summary["use_cases"][uc] = {"status": "no_traces_found", "traces_dir": str(traces_dir)}
            continue

        schema = induce_schema(uc, trace_files)
        uc_dir = out_dir / f"use_case_{uc}"
        uc_dir.mkdir(parents=True, exist_ok=True)

        schema_path = uc_dir / "induced_schema.json"
        schema_path.write_text(json.dumps(schema.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        sample_trace = _pick_one_trace(trace_files)
        parsed = parse_trace_with_schema(sample_trace, schema)
        parsed_path = uc_dir / f"parsed_{sample_trace.stem}.json"
        parsed_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        summary["use_cases"][uc] = {
            "status": "ok",
            "num_traces_for_induction": len(trace_files),
            "schema_path": str(schema_path),
            "sample_trace": str(sample_trace),
            "parsed_output_path": str(parsed_path),
            "num_events_raw": parsed["debug"]["num_events_raw"],
            "num_events_deduped": parsed["debug"]["num_events_deduped"],
            "top_message_paths": schema.message_list_paths[:5],
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[parser_core] wrote {summary_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="modular_imp")
    ap.add_argument("--out_dir", default="modular_imp/flow_extraction/schema_induction_outputs")
    ap.add_argument("--use_cases", default="1,2,3,4,5,6")
    args = ap.parse_args()

    use_cases = [u.strip() for u in args.use_cases.split(",") if u.strip()]
    run(Path(args.base_dir).resolve(), Path(args.out_dir).resolve(), use_cases)


if __name__ == "__main__":
    main()
