#!/usr/bin/env python3
"""Connectivity pass reusable helpers."""

import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
    from ..connectivity_pass.prompts import CONNECTIVITY_EDGE_PROMPT_BASE
    from ..global_utils import (
        as_var,
        build_parent_child_maps,
        cache_payload_matches,
        parse_json_loose
    )
    from ..rag.context_request import _append_rag_event, run_task_with_rag
except ImportError:
    from connectivity_pass.prompts import CONNECTIVITY_EDGE_PROMPT_BASE
    from global_utils import (
        as_var,
        build_parent_child_maps,
        cache_payload_matches,
        parse_json_loose
    )
    from rag.context_request import _append_rag_event, run_task_with_rag



def _call_connectivity_once(
    *,
    payload: Dict[str, Any],
    payload_path: Path,
    raw_path: Path,
    model: str,
    retriever: Any,
    refresh_raw: bool,
    rag_max_rounds: int,
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """Run one RAG-backed connectivity call and return its parsed final output plus call metadata."""
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if raw_path.exists() and not refresh_raw and cache_payload_matches(payload_path, payload):
        return parse_json_loose(raw_path.read_text(encoding="utf-8", errors="replace")), "cache", {"rounds": 0, "retrievals": 0}

    rag_log_path = raw_path.parent / f"{raw_path.name}.rag.log"
    _append_rag_event(
        "connectivity_iteration_call_start",
        {"payload_keys": sorted(payload.keys())},
        debug_log_path=rag_log_path,
    )
    text = run_task_with_rag(payload, model, retriever, max_rounds=rag_max_rounds, debug_log_path=rag_log_path)
    raw_path.write_text(text, encoding="utf-8")
    parsed = parse_json_loose(text)
    return parsed, "model", {"rounds": None, "retrievals": None}


def _edges_list(parent_node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the parent node's normalized internal edge list, keeping only dict edge objects."""
    vals = parent_node.get("internal_edges")
    if not isinstance(vals, list):
        vals = []
    out = [x for x in vals if isinstance(x, dict)]
    parent_node["internal_edges"] = out
    return out


def _add_edge(parent_node: Dict[str, Any], frm: str, to: str, description: str | None = None) -> bool:
    """Add one internal edge if the same `from_`/`to` pair does not already exist."""
    if not frm or not to:
        return False
    edges = _edges_list(parent_node)
    key = (frm, to)
    for e in edges:
        if (str(e.get("from_") or ""), str(e.get("to") or "")) == key:
            return False
    edges.append(
        {
            "__call__": "Edge",
            "from_": frm,
            "to": to,
            "condition": None,
            "description": description or None,
        }
    )
    parent_node["internal_edges"] = edges
    return True


def _remove_edge(parent_node: Dict[str, Any], frm: str, to: str) -> bool:
    """Remove one internal edge by its `from_`/`to` pair if present."""
    edges = _edges_list(parent_node)
    kept: List[Dict[str, Any]] = []
    removed = False
    for e in edges:
        if (str(e.get("from_") or ""), str(e.get("to") or "")) == (frm, to):
            removed = True
            continue
        kept.append(e)
    if removed:
        parent_node["internal_edges"] = kept
    return removed


def _edge_child_vars(parent_node: Dict[str, Any]) -> List[str]:
    """Return only `nodes` children, which are the children allowed to participate in internal edges."""
    vals = parent_node.get("nodes") if isinstance(parent_node.get("nodes"), list) else []
    out: List[str] = []
    for item in vals:
        v = as_var(item)
        if v:
            out.append(v)
    seen: Set[str] = set()
    uniq: List[str] = []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def _connectivity_analysis(parent_node: Dict[str, Any], child_vars: List[str] | None = None) -> Dict[str, Any]:
    """Compute the shared connectivity state for one parent graph."""
    children = list(child_vars) if isinstance(child_vars, list) else _edge_child_vars(parent_node)
    if not children:
        return {
            "children": [],
            "touched": set(),
            "start_children": [],
            "end_children": [],
            "unreachable": [],
            "dead_end": [],
        }

    edges = _edges_list(parent_node)
    adj: Dict[str, Set[str]] = {}
    touched: Set[str] = set()
    start_children: List[str] = []
    end_children: List[str] = []

    for e in edges:
        f = str(e.get("from_") or "")
        t = str(e.get("to") or "")
        if not f or not t:
            continue
        if f in children:
            touched.add(f)
        if t in children:
            touched.add(t)
        if f == "START" and t in children and t not in start_children:
            start_children.append(t)
        if t == "END" and f in children and f not in end_children:
            end_children.append(f)
        adj.setdefault(f, set()).add(t)
    reachable: Set[str] = set()
    q: deque[str] = deque(["START"])
    while q:
        cur = q.popleft()
        if cur in reachable:
            continue
        reachable.add(cur)
        for nxt in adj.get(cur, set()):
            if nxt not in reachable:
                q.append(nxt)
    unreachable = [c for c in children if c not in reachable]

    can_end: Set[str] = set()
    rev: Dict[str, Set[str]] = {}
    for frm, tos in adj.items():
        for to in tos:
            rev.setdefault(to, set()).add(frm)
    q2: deque[str] = deque(["END"])
    while q2:
        cur = q2.popleft()
        if cur in can_end:
            continue
        can_end.add(cur)
        for prev in rev.get(cur, set()):
            if prev not in can_end:
                q2.append(prev)
    dead_end = [c for c in children if c not in can_end]
    return {
        "children": children,
        "touched": touched,
        "start_children": start_children,
        "end_children": end_children,
        "unreachable": unreachable,
        "dead_end": dead_end,
    }


def _detach_unlinked_nodes_children(parent_node: Dict[str, Any]) -> List[str]:
    """Remove `nodes` children with no incident internal edge and return their vars."""
    nodes_children = sorted(_edge_child_vars(parent_node))
    parent_node["nodes"] = nodes_children
    if not nodes_children:
        return []

    child_set = set(nodes_children)
    touched: Set[str] = set()
    for e in _edges_list(parent_node):
        if not isinstance(e, dict):
            continue
        frm = str(e.get("from_") or "").strip()
        to = str(e.get("to") or "").strip()
        if frm in child_set:
            touched.add(frm)
        if to in child_set:
            touched.add(to)

    detached = sorted([v for v in nodes_children if v not in touched])
    if detached:
        keep = sorted([v for v in nodes_children if v in touched])
        parent_node["nodes"] = keep
    return detached


def _orphan_cascade_delete(
    *,
    initial_vars: List[str],
    nodes_by_var: Dict[str, Dict[str, Any]],
    var_order: List[str],
    root_var: str,
) -> Dict[str, Any]:
    """Delete detached orphan nodes and cascade the deletion to children that become parentless."""

    pending: deque[str] = deque([v for v in initial_vars if isinstance(v, str)])
    queued: Set[str] = set(v for v in pending)
    deleted: List[str] = []

    while pending:
        cur = pending.popleft()
        queued.discard(cur)
        if cur == root_var or cur not in nodes_by_var:
            continue

        _, parents_by_child = build_parent_child_maps(var_order, nodes_by_var)
        if parents_by_child.get(cur):
            continue

        node = nodes_by_var.get(cur, {})
        child_refs: List[str] = []
        for fld in ("nodes", "tool_list"):
            vals = node.get(fld) if isinstance(node.get(fld), list) else []
            for item in vals:
                cv = as_var(item)
                if cv and cv in nodes_by_var and cv != cur:
                    child_refs.append(cv)

        nodes_by_var.pop(cur, None)
        if cur in var_order:
            var_order.remove(cur)
        deleted.append(cur)

        for cv in child_refs:
            if cv in nodes_by_var and cv not in queued:
                pending.append(cv)
                queued.add(cv)

    return {
        "deleted_vars": deleted,
        "deleted_count": len(deleted),
    }


def _child_nodes_for_prompt(
    parent_node: Dict[str, Any],
    nodes_by_var: Dict[str, Dict[str, Any]],
    *,
    include_virtual_anchors: bool = False,
) -> List[Dict[str, Any]]:
    """Build the prompt-ready child node list, optionally with START/END anchors."""
    out: List[Dict[str, Any]] = []
    if include_virtual_anchors:
        out.extend(
            [
                {
                    "var": "START",
                    "node": {
                        "name": "START",
                        "node_type": {"type": "virtual_anchor"},
                        "description": "Virtual graph entry anchor. Must connect to at least one child.",
                    },
                },
                {
                    "var": "END",
                    "node": {
                        "name": "END",
                        "node_type": {"type": "virtual_anchor"},
                        "description": "Virtual graph exit anchor. At least one child must connect to END.",
                    },
                },
            ]
        )
    for cv in _edge_child_vars(parent_node):
        node = nodes_by_var.get(cv)
        if isinstance(node, dict):
            out.append({"var": cv, "node": node})
    return out


def _allowed_endpoints(parent_node: Dict[str, Any]) -> Set[str]:
    """Return the valid edge endpoint names for this parent graph."""
    return {"START", "END", *_edge_child_vars(parent_node)}


def _prune_edges_to_allowed_endpoints(parent_node: Dict[str, Any]) -> int:
    """Drop internal edges whose endpoints are outside this parent's allowed connectivity set."""
    allowed = _allowed_endpoints(parent_node)
    kept: List[Dict[str, Any]] = []
    removed = 0
    for edge in _edges_list(parent_node):
        frm = str(edge.get("from_") or "").strip()
        to = str(edge.get("to") or "").strip()
        if frm not in allowed or to not in allowed:
            removed += 1
            continue
        kept.append(edge)
    parent_node["internal_edges"] = kept
    return removed


def _apply_edge_mutations(
    parent_node: Dict[str, Any],
    *,
    edges_remove: List[Dict[str, Any]],
    edges_add: List[Dict[str, Any]],
    allowed_endpoints: Set[str],
) -> Dict[str, Any]:
    """Apply edge removals/additions and report accepted versus rejected changes."""

    removed = 0
    added = 0
    rejected_edges: List[Dict[str, Any]] = []

    for edge in edges_remove:
        if not isinstance(edge, dict):
            continue
        frm = str(edge.get("from_") or "").strip()
        to = str(edge.get("to") or "").strip()
        if frm and to and _remove_edge(parent_node, frm, to):
            removed += 1

    for edge in edges_add:
        if not isinstance(edge, dict):
            continue
        frm = str(edge.get("from_") or "").strip()
        to = str(edge.get("to") or "").strip()
        description = edge.get("description")
        if not frm or not to:
            rejected_edges.append({
                "edge": edge,
                "reason": "missing_endpoint",
            })
            continue
        if frm not in allowed_endpoints or to not in allowed_endpoints:
            rejected_edges.append({
                "edge": edge,
                "reason": "invalid_endpoint",
            })
            continue
        if _add_edge(parent_node, frm, to, description if isinstance(description, str) else None):
            added += 1

    return {
        "edges_added": added,
        "edges_removed": removed,
        "rejected_edges": rejected_edges,
    }


def _connectivity_issue_report(parent_node: Dict[str, Any], child_vars: List[str] | None = None) -> Dict[str, Any]:
    """Compute the current connectivity issues for one parent graph."""
    analysis = _connectivity_analysis(parent_node, child_vars=child_vars)
    children = list(analysis["children"])
    touched = set(analysis["touched"])
    start_children = list(analysis["start_children"])
    end_children = list(analysis["end_children"])
    unreachable = list(analysis["unreachable"])
    dead_end = list(analysis["dead_end"])
    isolated = sorted([c for c in children if c not in touched])

    issues: List[Dict[str, Any]] = []
    seen_issue_keys: Set[Tuple[str, str]] = set()

    def _add_issue(kind: str, child: str = "", details: str = "") -> None:
        key = (kind, child)
        if key in seen_issue_keys:
            return
        seen_issue_keys.add(key)
        issues.append({"kind": kind, "child": child or None, "details": details})

    for child in isolated:
        _add_issue("isolated_child", child, "child has no incident internal edge")
    for child in unreachable:
        _add_issue("unreachable_child", child, "child is not reachable from START")
    for child in dead_end:
        _add_issue("dead_end_child", child, "child cannot reach END")

    if children and len(start_children) == 0:
        _add_issue("missing_start_entry", details="no START edge reaches any child")
    elif len(start_children) > 1:
        _add_issue("multiple_start_entries", details="more than one child has a START entry edge")

    if children and len(end_children) == 0:
        _add_issue("missing_end_exit", details="no child reaches END")
    elif len(end_children) > 1:
        _add_issue("multiple_end_exits", details="more than one child has an END exit edge")

    return {
        "child_vars": children,
        "isolated_children": isolated,
        "unreachable_children": unreachable,
        "dead_end_children": dead_end,
        "start_entry_children": start_children,
        "end_exit_children": end_children,
        "issues": issues,
        "has_issues": bool(issues),
    }


def _repair_loop_mode_instruction(iteration_index: int, max_iterations: int) -> str:
    """Return the outer-loop instruction text to inject at the top of the connectivity prompt."""
    if iteration_index <= 1:
        return "Be only evidence-based, use only explicit code evidence for edge changes."
    if iteration_index >= max_iterations:
        return "You must resolve ALL the remaining listed connectivity issues in this call. If relations between children are not explicit, prefer a hub structure. If an LLM child exists, prefer that LLM as the hub. Use sequential or other non-hub relations only when explicit evidence supports them."
    return "If relations between children are not explicit, prefer a hub structure. If an LLM child exists, prefer that LLM as the hub. Use sequential or other non-hub relations only when explicit evidence supports them."


def process_parent_connectivity(
    *,
    parent_var: str,
    parent_node: Dict[str, Any],
    nodes_by_var: Dict[str, Dict[str, Any]],
    var_order: List[str],
    root_var: str,
    guidance_summary: str,
    retriever: Any,
    model: str,
    raw_parent_dir: Path,
    refresh_raw: bool,
    rag_max_rounds: int,
    max_iterations: int,
) -> Dict[str, Any]:
    """Run connectivity repair and post-cleanup for one parent node and return its results."""
    if isinstance(parent_node.get("nodes"), list) and parent_node.get("nodes"):
        conn_meta = repair_parent_connectivity_iterative(
            parent_var=parent_var,
            parent_node=parent_node,
            guidance_summary=guidance_summary,
            retriever=retriever,
            model=model,
            raw_parent_dir=raw_parent_dir,
            refresh_raw=refresh_raw,
            rag_max_rounds=rag_max_rounds,
            nodes_by_var=nodes_by_var,
            max_iterations=max_iterations,
        )
    else:
        conn_meta = {
            "iterations": [],
            "edges_added": 0,
            "edges_removed": 0,
            "final_issues": {"issues": [], "has_issues": False},
            "unresolved": False,
            "rejected_edges": [],
        }

    detached_nodes = _detach_unlinked_nodes_children(parent_node)
    orphan_cleanup = _orphan_cascade_delete(
        initial_vars=detached_nodes,
        nodes_by_var=nodes_by_var,
        var_order=var_order,
        root_var=root_var,
    )
    return {
        "connectivity": conn_meta,
        "detached_nodes": detached_nodes,
        "orphan_cleanup": orphan_cleanup,
        "child_vars_to_queue": [
            child_var for child_var in _child_vars(parent_node) if child_var in nodes_by_var
        ],
    }


def repair_parent_connectivity_iterative(
    *,
    parent_var: str,
    parent_node: Dict[str, Any],
    guidance_summary: str,
    retriever: Any,
    model: str,
    raw_parent_dir: Path,
    refresh_raw: bool,
    rag_max_rounds: int,
    nodes_by_var: Dict[str, Dict[str, Any]],
    max_iterations: int,
) -> Dict[str, Any]:
    """Iteratively detect and repair one parent node's internal connectivity until issues are resolved or iterations are exhausted."""

    pruned_edges = _prune_edges_to_allowed_endpoints(parent_node)
    children = _edge_child_vars(parent_node)
    if not children:
        return {
            "iterations": [],
            "edges_added": 0,
            "edges_removed": pruned_edges,
            "final_issues": {"has_issues": False, "issues": []},
            "unresolved": False,
            "rejected_edges": [],
        }

    if len(children) == 1:
        only = children[0]
        parent_node["internal_edges"] = []
        _add_edge(parent_node, "START", only, "single_child_entry")
        _add_edge(parent_node, only, "END", "single_child_exit")
        return {
            "iterations": [],
            "edges_added": 2,
            "edges_removed": pruned_edges,
            "final_issues": {"has_issues": False, "issues": []},
            "unresolved": False,
            "rejected_edges": [],
        }

    total_added = 0
    total_removed = pruned_edges
    rejected_edges: List[Dict[str, Any]] = []
    iteration_runs: List[Dict[str, Any]] = []
    iterations = max(1, int(max_iterations))

    for iteration_index in range(1, iterations + 1):
        issues_before = _connectivity_issue_report(parent_node, child_vars=children)
        if not bool(issues_before.get("has_issues")):
            break

        loop_mode_instruction = _repair_loop_mode_instruction(iteration_index, iterations)
        prompt_text = CONNECTIVITY_EDGE_PROMPT_BASE.replace("<<OUTER_LOOP_INSTRUCTION>>", loop_mode_instruction)
        payload = {
            "prompt": prompt_text,
            "parent_var": parent_var,
            "parent_node": parent_node,
            "child_vars": children,
            "child_nodes": _child_nodes_for_prompt(parent_node, nodes_by_var, include_virtual_anchors=True),
            "current_edges": _edges_list(parent_node),
            "connectivity_issues": issues_before,
            "guidance_summary": guidance_summary,
            "retrieved_evidence_context": None,
        }
        obj, source, meta = _call_connectivity_once(
            payload=payload,
            payload_path=raw_parent_dir / f"connect_iter_{iteration_index:02d}.payload.json",
            raw_path=raw_parent_dir / f"connect_iter_{iteration_index:02d}.txt",
            model=model,
            retriever=retriever,
            refresh_raw=refresh_raw,
            rag_max_rounds=rag_max_rounds,
        )
        mut = _apply_edge_mutations(
            parent_node,
            edges_remove=obj.get("edges_remove") if isinstance(obj.get("edges_remove"), list) else [],
            edges_add=obj.get("edges_add") if isinstance(obj.get("edges_add"), list) else [],
            allowed_endpoints=_allowed_endpoints(parent_node),
        )
        total_added += int(mut.get("edges_added") or 0)
        total_removed += int(mut.get("edges_removed") or 0)
        rejected_edges.extend(mut.get("rejected_edges") if isinstance(mut.get("rejected_edges"), list) else [])

        issues_after = _connectivity_issue_report(parent_node, child_vars=children)
        iteration_runs.append({
            "iteration": iteration_index,
            "mode_instruction": loop_mode_instruction,
            "source": source,
            "retrieval_rounds": int(meta.get("rounds") or 0),
            "retrieval_calls": int(meta.get("retrievals") or 0),
            "issues_before": issues_before.get("issues") if isinstance(issues_before.get("issues"), list) else [],
            "issues_after": issues_after.get("issues") if isinstance(issues_after.get("issues"), list) else [],
            "edges_added": int(mut.get("edges_added") or 0),
            "edges_removed": int(mut.get("edges_removed") or 0),
            "rejected_edges": mut.get("rejected_edges") if isinstance(mut.get("rejected_edges"), list) else [],
        })

        if not bool(issues_after.get("has_issues")):
            break

    final_issues = _connectivity_issue_report(parent_node, child_vars=children)
    return {
        "iterations": iteration_runs,
        "edges_added": total_added,
        "edges_removed": total_removed,
        "final_issues": final_issues,
        "unresolved": bool(final_issues.get("has_issues")),
        "rejected_edges": rejected_edges,
    }


__all__ = [
    "repair_parent_connectivity_iterative",
    "process_parent_connectivity",
    "_detach_unlinked_nodes_children",
    "_orphan_cascade_delete"
]
