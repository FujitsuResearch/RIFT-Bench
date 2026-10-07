#!/usr/bin/env python3
"""Child creation component: root-down child discovery and materialization."""

import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
    from ..model_client import call_model
    from ..rag.context_request import (
        _append_rag_event,
        extract_context_request_needs,
        merge_aggregated_context,
        retrieve_context_for_needs,
        run_task_with_rag,
        try_parse_context_request,
    )
    from .prompts import (
        CHILD_DISCOVERY_DELTA_PROMPT,
        CHILD_MATERIALIZATION_PROMPT,
        CHILD_RUNTIME_JUDGE_PROMPT,
        DUPLICATE_CANDIDATE_PROMPT,
        DUPLICATE_DECISION_PROMPT,
    )
    from ..global_utils import (
        add_child_ref,
        cache_payload_matches,
        parse_json_loose,
        next_var,
        node_type_name,
        read_snippet,
        safe_name,
        to_int_or_none,
        unique_json_items,
    )
except ImportError:
    from model_client import call_model
    from rag.context_request import (
        _append_rag_event,
        extract_context_request_needs,
        merge_aggregated_context,
        retrieve_context_for_needs,
        run_task_with_rag,
        try_parse_context_request,
    )
    from child_creation.prompts import (
        CHILD_DISCOVERY_DELTA_PROMPT,
        CHILD_MATERIALIZATION_PROMPT,
        CHILD_RUNTIME_JUDGE_PROMPT,
        DUPLICATE_CANDIDATE_PROMPT,
        DUPLICATE_DECISION_PROMPT,
    )
    from global_utils import (
        add_child_ref,
        cache_payload_matches,
        parse_json_loose,
        next_var,
        node_type_name,
        read_snippet,
        safe_name,
        to_int_or_none,
        unique_json_items,
    )


def _normalize_ref_line_and_snippet(ref: Dict[str, Any]) -> Dict[str, Any] | None:
    """Normalize a raw code reference into a strict file/line/snippet-backed reference."""
    if not isinstance(ref, dict):
        return None
    file_path = str(ref.get("file") or "").strip()
    if not file_path:
        return None

    line_val = ref.get("line")
    if line_val is None:
        ls = to_int_or_none(ref.get("line_start"))
        le = to_int_or_none(ref.get("line_end"))
        if ls is not None and le is not None:
            line_val = (ls, le)
        elif ls is not None:
            line_val = ls
        elif le is not None:
            line_val = le
    if isinstance(line_val, list) and len(line_val) == 2:
        a = to_int_or_none(line_val[0])
        b = to_int_or_none(line_val[1])
        if a is not None and b is not None:
            line_val = (a, b)
    elif isinstance(line_val, tuple) and len(line_val) == 2:
        a = to_int_or_none(line_val[0])
        b = to_int_or_none(line_val[1])
        if a is not None and b is not None:
            line_val = (a, b)
    elif not isinstance(line_val, int):
        iv = to_int_or_none(line_val)
        line_val = iv if iv is not None else None

    # Strict mode: snippet must be fetched from real code using file+line.
    snippet = ""
    if line_val is not None:
        snippet = read_snippet(file_path, line_val)
    if not line_val or not snippet:
        return None

    kind = str(ref.get("kind") or "other").strip() or "other"
    out = {
        "kind": kind,
        "file": file_path,
        "line": line_val,
        "snippet": snippet,
    }
    if kind == "other":
        out["other_kind_description"] = str(ref.get("other_kind_description") or "attachment").strip() or "attachment"
    return out


def _dedupe_code_refs(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate code references by file and line while preserving first-seen order."""
    seen: Set[Tuple[str, str]] = set()
    out: List[Dict[str, Any]] = []
    for r in items:
        if not isinstance(r, dict):
            continue
        file_path = str(r.get("file") or "")
        line = r.get("line")
        line_key = json.dumps(line, ensure_ascii=False, sort_keys=True, default=str)
        key = (file_path, line_key)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _normalize_code_refs(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize raw code references into strict validated references and dedupe them."""
    normed: List[Dict[str, Any]] = []
    for r in items:
        nr = _normalize_ref_line_and_snippet(r)
        if nr is not None:
            normed.append(nr)
    return _dedupe_code_refs(unique_json_items(normed))


def _code_refs_add(base: List[Dict[str, Any]], adds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge additional code references into an existing list and re-normalize the result."""
    all_items = [x for x in [*base, *adds] if isinstance(x, dict)]
    return _normalize_code_refs(all_items)


def _normalize_child_usage_refs(adds: Any) -> List[Dict[str, Any]]:
    """Normalize raw child reference candidates into valid `usage` code references only."""
    if not isinstance(adds, list):
        return []
    out: List[Dict[str, Any]] = []
    for raw in adds:
        if not isinstance(raw, dict):
            continue
        ref = dict(raw)
        ref["kind"] = "usage"
        ref["other_kind_description"] = None
        nr = _normalize_ref_line_and_snippet(ref)
        if not isinstance(nr, dict):
            continue
        nr["kind"] = "usage"
        nr["other_kind_description"] = None
        out.append(nr)
    return out


def _mirror_child_usage_to_parent_attachment(child_usage_refs: List[Dict[str, Any]], parent_var: str, child_name: str) -> List[Dict[str, Any]]:
    """Convert child-side usage references into parent-side attachment references."""
    out: List[Dict[str, Any]] = []
    for raw in child_usage_refs:
        if not isinstance(raw, dict):
            continue
        ref = dict(raw)
        ref["kind"] = "other"
        ref["other_kind_description"] = "attachment"
        # Keep line/file/snippet so parent refs remain code-backed after normalization.
        base_snippet = str(ref.get("snippet") or "").strip()
        if base_snippet:
            ref["snippet"] = f"Attachment evidence for parent {parent_var} -> child {child_name}: {base_snippet}"
        out.append(ref)
    return out


def _discovery_children_refs_invalid(parsed: Dict[str, Any]) -> bool:
    """Return whether discovery output is missing valid child attachment code references."""
    if not isinstance(parsed, dict):
        return False
    children_add = parsed.get("children_add")
    if not isinstance(children_add, list):
        return False
    for child in children_add:
        if not isinstance(child, dict):
            continue
        raw_refs = child.get("code_references_add_child")
        if not isinstance(raw_refs, list):
            raw_refs = child.get("code_references_add")
        if not isinstance(raw_refs, list):
            return True
        norm = _normalize_child_usage_refs(raw_refs)
        if len(norm) == 0:
            return True
    return False


def _materialized_child_refs_invalid(parsed: Dict[str, Any]) -> bool:
    """Return whether materialization output is missing valid child code references."""
    if not isinstance(parsed, dict):
        return False
    child_node = parsed.get("child_node")
    if not isinstance(child_node, dict):
        return False
    refs = child_node.get("code_references")
    if not isinstance(refs, list):
        return False
    # If refs are present, they must be resolvable to real code snippets.
    if len(refs) == 0:
        return False
    norm = _normalize_code_refs([x for x in refs if isinstance(x, dict)])
    return len(norm) == 0


def _coerce_node_type_dict(type_name: str, other_desc: str | None = None) -> Dict[str, Any]:
    allowed = {
        "LLM",
        "Tool",
        "Database",
        "System",
        "Local_MCP_server",
        "External_MCP_server",
        "Agent",
        "Deterministic_controller",
        "other",
    }
    t = str(type_name or "").strip()
    if t not in allowed:
        t = "other"
    return {"__call__": "NodeType", "type": t, "other_description": other_desc if t == "other" else None}


def _normalize_child_node(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a materialized child node into the stage's expected in-memory node shape."""
    name = str(raw.get("name") or "").strip() or "unnamed_component"
    nt = raw.get("node_type") if isinstance(raw.get("node_type"), dict) else {}
    nt_name = str(nt.get("type") or raw.get("node_type_name") or "other")
    nt_other = nt.get("other_description") if isinstance(nt.get("other_description"), str) else str(raw.get("other_node_type_description") or "") or None
    out = {
        "name": name,
        "node_type": _coerce_node_type_dict(nt_name, nt_other),
        "description": str(raw.get("description") or ""),
        "code_references": _normalize_code_refs([x for x in (raw.get("code_references") if isinstance(raw.get("code_references"), list) else []) if isinstance(x, dict)]),
        "inputs": [x for x in (raw.get("inputs") if isinstance(raw.get("inputs"), list) else []) if isinstance(x, dict)],
        "outputs": [x for x in (raw.get("outputs") if isinstance(raw.get("outputs"), list) else []) if isinstance(x, dict)],
        "metadata": dict(raw.get("metadata")) if isinstance(raw.get("metadata"), dict) else {},
        "tool_list": [],
        "nodes": [],
        "internal_edges": [],
    }
    return out


def _child_field_for_parent(parent_node: Dict[str, Any], child_node: Dict[str, Any]) -> str:
    """Choose whether a child should attach under `nodes` or `tool_list` for this parent."""
    ptype = node_type_name(parent_node).lower()
    ctype = node_type_name(child_node)
    if "mcp_server" in ptype and ctype == "Tool":
        return "tool_list"
    return "nodes"


def _build_child_proposals_for_parent(
    *,
    parent_var: str,
    parent_node: Dict[str, Any],
    guidance_summary: str,
    retriever: Any,
    model: str,
    raw_parent_dir: Path,
    refresh_raw: bool,
    phase_max_rounds: int,
    no_change_patience: int,
    use_first_round_context_request_hint: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Iteratively discover child proposals for one parent, including retrieval-backed retry rounds."""

    proposals: List[Dict[str, Any]] = []
    seen_sig: Set[Tuple[str, str, str]] = set()
    no_change = 0
    rounds = max(1, int(phase_max_rounds))
    retrieval_calls = 0
    retrieved_evidence_context: str | None = None
    retrieval_history: List[Dict[str, Any]] = []
    seen_query_keys: Set[Tuple[str, str, str, int]] = set()
    for r in range(1, rounds + 1):

        # Build the current round payload from the parent state and accumulated retrieval context.
        discovery_prompt = CHILD_DISCOVERY_DELTA_PROMPT
        if r == 1 and bool(use_first_round_context_request_hint):
            discovery_prompt = (
                discovery_prompt
                + "\n\nFIRST CONTEXT ROUND: if any attachment evidence is missing, return context_request now and do not finalize in this round."
            )
        payload = {
            "prompt": discovery_prompt,
            "parent_var": parent_var,
            "parent_node": parent_node,
            "guidance_summary": guidance_summary,
            "discovered_children_so_far": proposals,
            "retrieved_evidence_context": retrieved_evidence_context,
            "retrieval_history": retrieval_history,
        }
        payload_path = raw_parent_dir / f"discover_round_{r}.payload.json"
        raw_path = raw_parent_dir / f"discover_round_{r}.txt"
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


        # Reuse cached output when the exact same payload was already run.
        if raw_path.exists() and not refresh_raw and cache_payload_matches(payload_path, payload):
            text = raw_path.read_text(encoding="utf-8", errors="replace")
            parsed = parse_json_loose(text)
            source = "cache"

        else:
            rag_log_path = raw_path.parent / f"{raw_path.name}.rag.log"
            _append_rag_event(
                "round_start",
                {
                    "round": r,
                    "discovered_children_so_far": len(proposals),
                    "has_retrieved_evidence_context": bool(retrieved_evidence_context),
                    "retrieval_history_len": len(retrieval_history),
                },
                debug_log_path=rag_log_path,
            )
            text = call_model(payload, model)
            raw_path.write_text(text, encoding="utf-8")
            parsed = parse_json_loose(text)
            source = "model"


        # Retry once when discovered child references cannot be resolved to valid code-backed evidence.
        if _discovery_children_refs_invalid(parsed):
            # Retry once with the same payload when references cannot be resolved to real code snippets.
            rag_log_path = raw_path.parent / f"{raw_path.name}.rag.log"
            _append_rag_event(
                "retry_same_payload_invalid_code_references",
                {"round": r, "reason": "discovery_children_refs_invalid"},
                debug_log_path=rag_log_path,
            )
            text_retry = call_model(payload, model)
            raw_path.write_text(text_retry, encoding="utf-8")
            parsed_retry = parse_json_loose(text_retry)
            if isinstance(parsed_retry, dict):
                parsed = parsed_retry
                source = "model_retry"


        # Normalize newly proposed children and attach their mirrored parent evidence.
        children_add = parsed.get("children_add") if isinstance(parsed.get("children_add"), list) else []
        added = 0
        for raw in children_add:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            ntype = str(raw.get("node_type") or "").strip()
            desc = str(raw.get("description") or "").strip()
            if not name or not ntype:
                continue
            sig = (name.lower(), ntype, desc.lower())
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            child_adds = _normalize_child_usage_refs(raw.get("code_references_add_child"))
            parent_adds = _mirror_child_usage_to_parent_attachment(child_adds, parent_var=parent_var, child_name=name)
            proposal = {
                "name": name,
                "node_type": ntype,
                "other_node_type_description": str(raw.get("other_node_type_description") or "") or None,
                "description": desc,
                "code_references_add_parent": parent_adds,
                "code_references_add_child": child_adds,
            }
            proposals.append(proposal)
            parent_node["code_references"] = _code_refs_add(
                parent_node.get("code_references") if isinstance(parent_node.get("code_references"), list) else [],
                parent_adds,
            )
            added += 1


        # Process any context request returned by the model for the next round.
        is_complete = bool(parsed.get("is_complete")) if isinstance(parsed.get("is_complete"), bool) else False

        req = try_parse_context_request(parsed)
        parsed_needs = extract_context_request_needs(req if isinstance(req, dict) else None)

        ctx, hist = retrieve_context_for_needs(
            parsed_needs=parsed_needs,
            retriever=retriever,
            seen_need_keys=seen_query_keys,
            key_prefix="context_request_need",
        )

        if hist:
            retrieval_calls += len(hist)
            existing_ctx = str(retrieved_evidence_context or "").strip()

            if ctx:
                retrieved_evidence_context = merge_aggregated_context(existing_ctx, ctx)

            retrieval_history = [*retrieval_history, *hist]
            rag_log_path = raw_path.parent / f"{raw_path.name}.rag.log"
            _append_rag_event(
                "context_request_retrieved",
                {
                    "round": r,
                    "needs": [str(x.get("query") or "") for x in parsed_needs if isinstance(x, dict)],
                    "retrieval_batch": hist,
                },
                debug_log_path=rag_log_path,
            )

        # Decide whether to stop or continue based on new proposals, retrieval activity, and completion state.
        # Do not count as "no change" when new retrieval was requested/executed in this round;
        # otherwise the loop can stop before the model gets a chance to use newly retrieved context.
        if added == 0 and not hist:
            no_change += 1
        else:
            no_change = 0
        if is_complete or no_change >= max(1, int(no_change_patience)):
            return proposals, {
                "rounds": r,
                "source": source,
                "retrieval_rounds_last_call": 1 if hist else 0,
                "retrieval_calls_last_call": len(retrieval_history),
                "is_complete": is_complete,
                "completion_reason": str(parsed.get("completion_reason") or ""),
                "no_change_stop": no_change >= max(1, int(no_change_patience)),
            }

    return proposals, {
        "rounds": rounds,
        "source": "model",
        "retrieval_rounds_last_call": rounds,
        "retrieval_calls_last_call": retrieval_calls,
        "is_complete": False,
        "completion_reason": "max_rounds",
        "no_change_stop": False,
    }


def _materialize_child(
    *,
    parent_var: str,
    parent_node: Dict[str, Any],
    proposal: Dict[str, Any],
    nodespec_schema_catalog: str,
    guidance_summary: str,
    retriever: Any,
    model: str,
    raw_child_dir: Path,
    refresh_raw: bool,
    rag_max_rounds: int,
) -> Tuple[Dict[str, Any] | None, Dict[str, Any]]:
    """Materialize one discovered child proposal into a normalized child node with preserved attachment evidence."""

    # Build the materialization payload from the parent, proposal, schema, and guidance inputs.
    payload = {
        "prompt": CHILD_MATERIALIZATION_PROMPT,
        "parent_var": parent_var,
        "parent_node": parent_node,
        "child_proposal": proposal,
        "nodespec_schema_catalog": nodespec_schema_catalog,
        "guidance_summary": guidance_summary,
    }
    payload_path = raw_child_dir / "materialize.payload.json"
    raw_path = raw_child_dir / "materialize.txt"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Reuse cached materialization output when the payload matches a previous run.
    if raw_path.exists() and not refresh_raw and cache_payload_matches(payload_path, payload):
        parsed = parse_json_loose(raw_path.read_text(encoding="utf-8", errors="replace"))
        source = "cache"
        meta = {"rounds": 0, "retrieval_calls": 0}

    # Perfomr the LLM call when the payload does not match a previous run.
    else:
        rag_log_path = raw_path.parent / f"{raw_path.name}.rag.log"
        text = run_task_with_rag(payload, model, retriever, max_rounds=rag_max_rounds, debug_log_path=rag_log_path)
        raw_path.write_text(text, encoding="utf-8")
        parsed = parse_json_loose(text)
        source = "model"
        meta = {"rounds": None, "retrieval_calls": None}

    # Retry once when the materialized child has unusable code references.
    if _materialized_child_refs_invalid(parsed):

        # Retry once with same payload when child_node references are invalid/non-resolvable.
        rag_log_path = raw_path.parent / f"{raw_path.name}.rag.log"
        _append_rag_event(
            "retry_same_payload_invalid_code_references",
            {"reason": "materialized_child_refs_invalid"},
            debug_log_path=rag_log_path,
        )
        text_retry = run_task_with_rag(payload, model, retriever, max_rounds=rag_max_rounds, debug_log_path=rag_log_path)
        raw_path.write_text(text_retry, encoding="utf-8")
        parsed_retry = parse_json_loose(text_retry)
        if isinstance(parsed_retry, dict):
            parsed = parsed_retry
            source = "model_retry"

    # Normalize the final child node shape and merge discovery-stage child attachment references back in.
    raw_node = parsed.get("child_node") if isinstance(parsed.get("child_node"), dict) else {}
    if not raw_node:
        raw_node = {
            "name": proposal.get("name"),
            "node_type": {"type": proposal.get("node_type"), "other_description": proposal.get("other_node_type_description")},
            "description": proposal.get("description"),
            "code_references": proposal.get("code_references_add_child") if isinstance(proposal.get("code_references_add_child"), list) else [],
        }
    node = _normalize_child_node(raw_node)
    node["code_references"] = _code_refs_add(
        node.get("code_references") if isinstance(node.get("code_references"), list) else [],
        proposal.get("code_references_add_child") if isinstance(proposal.get("code_references_add_child"), list) else [],
    )
    return node, {
        "source": source,
        "retrieval_rounds": meta.get("rounds"),
        "retrieval_calls": meta.get("retrieval_calls"),
        "excluded": False,
        "exclude_reason": "",
    }


def _judge_child_runtime_inclusion(
    *,
    parent_var: str,
    parent_node: Dict[str, Any],
    child_var: str,
    child_node: Dict[str, Any],
    guidance_summary: str,
    model: str,
    raw_child_dir: Path,
    refresh_raw: bool,
) -> Dict[str, Any]:
    """Judge whether a materialized child candidate should be included under the given parent."""

    # Build the runtime-judge payload from the parent, candidate child, and guidance context.
    payload = {
        "prompt": CHILD_RUNTIME_JUDGE_PROMPT,
        "parent_var": parent_var,
        "parent_node": parent_node,
        "candidate_child_var": child_var,
        "candidate_child_node": child_node,
        "guidance_summary": guidance_summary,
    }
    payload_path = raw_child_dir / "judge.payload.json"
    raw_path = raw_child_dir / "judge.txt"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Reuse cached judge output when the payload matches a previous run.
    if raw_path.exists() and not refresh_raw and cache_payload_matches(payload_path, payload):
        parsed = parse_json_loose(raw_path.read_text(encoding="utf-8", errors="replace"))
        source = "cache"
    else:
        text = call_model(payload, model)
        raw_path.write_text(text, encoding="utf-8")
        parsed = parse_json_loose(text)
        source = "model"

    # Normalize the judge result into the standard include/reason/source shape.
    include_child = bool(parsed.get("include_child")) if isinstance(parsed, dict) and isinstance(parsed.get("include_child"), bool) else True
    reason = str(parsed.get("reason") or "").strip() if isinstance(parsed, dict) else ""
    return {"include_child": include_child, "reason": reason, "source": source}


def _existing_nodes_catalog(nodes_by_var: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for var, node in nodes_by_var.items():
        if not isinstance(node, dict):
            continue
        out.append(
            {
                "var": var,
                "name": str(node.get("name") or ""),
                "node_type": node_type_name(node),
                "description": str(node.get("description") or ""),
            }
        )
    return out


def _dedupe_candidate_vars(values: Any, nodes_by_var: Dict[str, Dict[str, Any]]) -> List[str]:
    """Normalize and dedupe candidate var names, keeping only vars that exist in the graph."""
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for v in values:
        s = str(v or "").strip()
        if not s or s in seen:
            continue
        if s not in nodes_by_var:
            continue
        seen.add(s)
        out.append(s)
    return out


def _merge_duplicate_reused_metadata(
    node: Dict[str, Any],
    *,
    parent_var: str,
    proposal_name: str,
    reason: str,
) -> None:
    """Record metadata explaining that an existing node was reused for a duplicate child proposal."""
    md = node.get("metadata")
    if not isinstance(md, dict):
        md = {}
    events = md.get("duplicate_reused_events")
    if not isinstance(events, list):
        events = []
    events = unique_json_items(
        [
            *[x for x in events if isinstance(x, dict)],
            {
                "parent_var": parent_var,
                "proposal_name": proposal_name,
                "reason": reason,
            },
        ]
    )
    md["duplicate_reused_events"] = events
    node["metadata"] = md


def _resolve_duplicate_child_global(
    *,
    parent_var: str,
    parent_node: Dict[str, Any],
    candidate_child_node: Dict[str, Any],
    guidance_summary: str,
    nodes_by_var: Dict[str, Dict[str, Any]],
    model: str,
    retriever: Any,
    raw_child_dir: Path,
    refresh_raw: bool,
) -> Dict[str, Any]:
    """Resolve whether a candidate child should reuse an existing graph node instead of creating a new one."""

    # Return early when there are no existing graph nodes to compare against.
    if not nodes_by_var:
        return {
            "matched_var": None,
            "reason": "",
            "candidate_meta": {"source": "none"},
            "decision_meta": [],
            "plausible_matches": [],
        }

    # Ask the model for a short list of plausible duplicate candidates from the existing graph.
    c_payload = {
        "prompt": DUPLICATE_CANDIDATE_PROMPT,
        "parent_var": parent_var,
        "parent_node": parent_node,
        "candidate_child_node": candidate_child_node,
        "existing_nodes_catalog": [x for x in _existing_nodes_catalog(nodes_by_var) if str(x.get("var") or "") != parent_var],
        "guidance_summary": guidance_summary,
    }
    c_payload_path = raw_child_dir / "dedupe_candidates.payload.json"
    c_raw_path = raw_child_dir / "dedupe_candidates.txt"
    c_payload_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        c_payload_text = json.dumps(c_payload, indent=2, ensure_ascii=False)
    except Exception as exc:
        bad_path = _debug_find_non_jsonable(c_payload) or "unknown"
        raise
    c_payload_path.write_text(c_payload_text, encoding="utf-8")

    # Reuse cached duplicate-candidate output when the payload matches a previous run.
    if c_raw_path.exists() and not refresh_raw and cache_payload_matches(c_payload_path, c_payload):
        c_obj = parse_json_loose(c_raw_path.read_text(encoding="utf-8", errors="replace"))
        c_source = "cache"
    else:
        c_log_path = c_raw_path.parent / f"{c_raw_path.name}.rag.log"
        c_text = run_task_with_rag(c_payload, model, retriever, max_rounds=8, debug_log_path=c_log_path)
        c_raw_path.write_text(c_text, encoding="utf-8")
        c_obj = parse_json_loose(c_text)
        c_source = "model"

    # Normalize the plausible duplicate shortlist and stop early if nothing viable was found.
    plausible = _dedupe_candidate_vars(c_obj.get("plausible_matches") if isinstance(c_obj, dict) else [], nodes_by_var)
    if not plausible:
        return {
            "matched_var": None,
            "reason": "",
            "candidate_meta": {"source": c_source},
            "decision_meta": [],
            "plausible_matches": [],
        }

    # Compare the candidate child against each plausible existing node until one is confirmed as the same component.
    decision_meta: List[Dict[str, Any]] = []
    for cand_var in plausible:
        existing_node = nodes_by_var.get(cand_var)
        if not isinstance(existing_node, dict):
            continue
        d_payload = {
            "prompt": DUPLICATE_DECISION_PROMPT,
            "parent_var": parent_var,
            "parent_node": parent_node,
            "candidate_child_node": candidate_child_node,
            "existing_var": cand_var,
            "existing_node": existing_node,
            "guidance_summary": guidance_summary,
        }
        safe_var = safe_name(cand_var, max_len=48)
        d_payload_path = raw_child_dir / f"dedupe_decision_{safe_var}.payload.json"
        d_raw_path = raw_child_dir / f"dedupe_decision_{safe_var}.txt"
        d_payload_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            d_payload_text = json.dumps(d_payload, indent=2, ensure_ascii=False)
        except Exception as exc:
            bad_path = _debug_find_non_jsonable(d_payload) or "unknown"

            raise
        d_payload_path.write_text(d_payload_text, encoding="utf-8")

        # Reuse cached duplicate-decision output when the per-candidate payload already matches.
        if d_raw_path.exists() and not refresh_raw and cache_payload_matches(d_payload_path, d_payload):
            d_obj = parse_json_loose(d_raw_path.read_text(encoding="utf-8", errors="replace"))
            d_source = "cache"
        else:
            d_log_path = d_raw_path.parent / f"{d_raw_path.name}.rag.log"
            d_text = run_task_with_rag(d_payload, model, retriever, max_rounds=8, debug_log_path=d_log_path)
            d_raw_path.write_text(d_text, encoding="utf-8")
            d_obj = parse_json_loose(d_text)
            d_source = "model"
        is_same = bool(d_obj.get("is_same_component")) if isinstance(d_obj, dict) and isinstance(d_obj.get("is_same_component"), bool) else False
        reason = str(d_obj.get("reason") or "").strip() if isinstance(d_obj, dict) else ""
        decision_meta.append({"candidate_var": cand_var, "is_same_component": is_same, "reason": reason, "source": d_source})

        # Return immediately once one existing node is judged to be the same runtime component.
        if is_same and cand_var != parent_var:
            return {
                "matched_var": cand_var,
                "reason": reason,
                "candidate_meta": {"source": c_source},
                "decision_meta": decision_meta,
                "plausible_matches": plausible,
            }

    # Return the evaluated shortlist and per-candidate decisions when no duplicate match is confirmed.
    return {
        "matched_var": None,
        "reason": "",
        "candidate_meta": {"source": c_source},
        "decision_meta": decision_meta,
        "plausible_matches": plausible,
    }


def process_parent_child_expansion(
    *,
    parent_var: str,
    nodes_by_var: Dict[str, Dict[str, Any]],
    var_order: List[str],
    guidance_summary: str,
    nodespec_schema_catalog: str,
    retriever: Any,
    model: str,
    raw_root_dir: Path,
    refresh_raw: bool,
    rag_max_rounds: int,
    phase_a_max_rounds: int = 8,
    phase_a_no_change_patience: int = 2,
    use_first_round_context_request_hint_discovery: bool = False,
) -> Dict[str, Any]:
    """Run the find-materialize-judge-dedup pipeline once for a single parent node."""
    if parent_var not in nodes_by_var:
        raise KeyError(f"Unknown parent var: {parent_var}")


    # Resolve the parent node and its per-parent working directory for this expansion pass.
    parent_node = nodes_by_var[parent_var]
    parent_raw_dir = raw_root_dir / "pass_dynamic" / safe_name(parent_var, max_len=96)
    parent_raw_dir.mkdir(parents=True, exist_ok=True)


    # Discover candidate children for this parent, including retrieval-backed follow-up rounds.
    proposals, discover_meta = _build_child_proposals_for_parent(
        parent_var=parent_var,
        parent_node=parent_node,
        guidance_summary=guidance_summary,
        retriever=retriever,
        model=model,
        raw_parent_dir=parent_raw_dir,
        refresh_raw=refresh_raw,
        phase_max_rounds=phase_a_max_rounds,
        no_change_patience=phase_a_no_change_patience,
        use_first_round_context_request_hint=use_first_round_context_request_hint_discovery,
    )

    created_here: List[str] = []
    excluded_here: List[Dict[str, Any]] = []
    reused_here: List[Dict[str, Any]] = []
    attached_child_vars: List[str] = []
    created_nodes = 0


    # Materialize, judge, dedupe, and attach each discovered child proposal.
    for idx, proposal in enumerate(proposals, start=1):

        # Materialize the proposal into a concrete child node candidate.
        child_raw_dir = parent_raw_dir / f"child_{idx:03d}_{safe_name(str(proposal.get('name') or 'child'))}"
        child_raw_dir.mkdir(parents=True, exist_ok=True)
        child_node, mat_meta = _materialize_child(
            parent_var=parent_var,
            parent_node=parent_node,
            proposal=proposal,
            nodespec_schema_catalog=nodespec_schema_catalog,
            guidance_summary=guidance_summary,
            retriever=retriever,
            model=model,
            raw_child_dir=child_raw_dir,
            refresh_raw=refresh_raw,
            rag_max_rounds=rag_max_rounds,
        )
        if child_node is None:
            excluded_here.append(
                {
                    "proposal_name": str(proposal.get("name") or ""),
                    "reason": str(mat_meta.get("exclude_reason") or "excluded"),
                }
            )
            continue


        # Run the runtime-inclusion judge before attaching the child to the graph.
        tentative_child_var = next_var(str(child_node.get("name") or "child"), set(nodes_by_var.keys()))
        judge = _judge_child_runtime_inclusion(
            parent_var=parent_var,
            parent_node=parent_node,
            child_var=tentative_child_var,
            child_node=child_node,
            guidance_summary=guidance_summary,
            model=model,
            raw_child_dir=child_raw_dir,
            refresh_raw=refresh_raw,
        )
        if not bool(judge.get("include_child")):
            excluded_here.append(
                {
                    "proposal_name": str(proposal.get("name") or ""),
                    "reason": str(judge.get("reason") or "excluded_by_runtime_judge"),
                    "excluded_by": "runtime_judge",
                }
            )
            continue

        # Resolve against the existing graph so duplicate children are reused instead of recreated.
        dup = _resolve_duplicate_child_global(
            parent_var=parent_var,
            parent_node=parent_node,
            candidate_child_node=child_node,
            guidance_summary=guidance_summary,
            nodes_by_var=nodes_by_var,
            model=model,
            retriever=retriever,
            raw_child_dir=child_raw_dir,
            refresh_raw=refresh_raw,
        )
        matched_var = str(dup.get("matched_var") or "").strip()
        if matched_var and matched_var in nodes_by_var:
            existing_node = nodes_by_var[matched_var]
            existing_node["code_references"] = _code_refs_add(
                existing_node.get("code_references") if isinstance(existing_node.get("code_references"), list) else [],
                child_node.get("code_references") if isinstance(child_node.get("code_references"), list) else [],
            )
            _merge_duplicate_reused_metadata(
                existing_node,
                parent_var=parent_var,
                proposal_name=str(proposal.get("name") or ""),
                reason=str(dup.get("reason") or ""),
            )
            attach_field = _child_field_for_parent(parent_node, existing_node)
            add_child_ref(parent_node, matched_var, field=attach_field)
            attached_child_vars.append(matched_var)
            reused_here.append(
                {
                    "parent_var": parent_var,
                    "proposal_name": str(proposal.get("name") or ""),
                    "matched_var": matched_var,
                    "reason": str(dup.get("reason") or ""),
                }
            )
            continue

        # Add a brand-new child node to the graph and attach it under the current parent.
        child_var = next_var(str(child_node.get("name") or "child"), set(nodes_by_var.keys()))
        nodes_by_var[child_var] = child_node
        var_order.append(child_var)
        created_nodes += 1
        created_here.append(child_var)
        attached_child_vars.append(child_var)
        attach_field = _child_field_for_parent(parent_node, child_node)
        add_child_ref(parent_node, child_var, field=attach_field)

    return {
        "parent_var": parent_var,
        "phase_a": discover_meta,
        "created_children_vars": created_here,
        "reused_existing_children": reused_here,
        "excluded_children": excluded_here,
        "attached_child_vars": attached_child_vars,
        "created_nodes": created_nodes,
    }


def expand_children_from_parent(
    *,
    start_parent_var: str,
    nodes_by_var: Dict[str, Dict[str, Any]],
    var_order: List[str],
    guidance_summary: str,
    nodespec_schema_catalog: str,
    retriever: Any,
    model: str,
    raw_root_dir: Path,
    refresh_raw: bool,
    rag_max_rounds: int,
    phase_a_max_rounds: int = 8,
    phase_a_no_change_patience: int = 2,
    use_first_round_context_request_hint_discovery: bool = False,
) -> Dict[str, Any]:
    """Expand children recursively from a single starting parent over the current graph state."""
    q: deque[str] = deque([start_parent_var])
    processed: Set[str] = set()
    parent_runs: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    created_nodes = 0

    while q:
        parent_var = q.popleft()
        if parent_var in processed or parent_var not in nodes_by_var:
            continue
        processed.add(parent_var)
        try:
            run = process_parent_child_expansion(
                parent_var=parent_var,
                nodes_by_var=nodes_by_var,
                var_order=var_order,
                guidance_summary=guidance_summary,
                nodespec_schema_catalog=nodespec_schema_catalog,
                retriever=retriever,
                model=model,
                raw_root_dir=raw_root_dir,
                refresh_raw=refresh_raw,
                rag_max_rounds=rag_max_rounds,
                phase_a_max_rounds=phase_a_max_rounds,
                phase_a_no_change_patience=phase_a_no_change_patience,
                use_first_round_context_request_hint_discovery=use_first_round_context_request_hint_discovery,
            )
            parent_runs.append(run)
            created_nodes += int(run.get("created_nodes") or 0)
            for child_var in run.get("attached_child_vars") if isinstance(run.get("attached_child_vars"), list) else []:
                if isinstance(child_var, str) and child_var in nodes_by_var:
                    q.append(child_var)
        except Exception as exc:
            errors.append({"parent_var": parent_var, "error": str(exc)})

    return {
        "seed_vars": [start_parent_var],
        "parents_processed": len(processed),
        "created_nodes": created_nodes,
        "parent_runs": parent_runs,
        "errors": errors,
    }
