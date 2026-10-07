#!/usr/bin/env python3
"""Helpers for node_refinement component."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Set

try:
    from ..global_utils import (
        build_parent_child_maps,
        detach_var_from_parents,
        extract_var_name,
        node_type_name,
    )
    from ..graph_correctness.utils import resolve_orphan_node
except ImportError:
    from global_utils import (
        build_parent_child_maps,
        detach_var_from_parents,
        extract_var_name,
        node_type_name,
    )
    from graph_correctness.utils import resolve_orphan_node

KNOWN_NODE_TYPES: Set[str] = {
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
ALLOWED_SPECIAL_REFS: Set[str] = {"START", "END"}

ENV_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
COMMENTED_ENV_ASSIGN_RE = re.compile(r"^\s*#\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")


def collect_env_var_names(env_file: Path, include_commented: bool = True) -> List[str]:
    """Return sorted unique env var names from a .env file, optionally including commented assignments."""
    if not env_file.exists():
        return []
    names: Set[str] = set()
    for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        m = ENV_ASSIGN_RE.match(line)
        if m:
            names.add(m.group(1).strip())
            continue
        if include_commented:
            cm = COMMENTED_ENV_ASSIGN_RE.match(line)
            if cm:
                names.add(cm.group(1).strip())
    return sorted(n for n in names if n)


def _ensure_type_specific_fields(node: Dict[str, Any]) -> int:
    """Apply type-specific defaults only."""
    updates = 0
    ntype = node_type_name(node)
    if ntype == "LLM" and not isinstance(node.get("llm_config"), dict):
        node["llm_config"] = {
            "__call__": "LLMConfig",
            "provider": None,
            "class_name": None,
            "model_name": None,
            "temperature": None,
        }
        updates += 1
    if ntype == "Agent" and not isinstance(node.get("agent_type"), dict):
        node["agent_type"] = {
            "__call__": "AgentType",
            "type": ["other"],
            "other_description": "unspecified_agent_pattern",
        }
        updates += 1
    if ntype == "System" and not isinstance(node.get("system_type"), dict):
        node["system_type"] = {
            "__call__": "SystemType",
            "type": ["other"],
            "other_description": "unspecified_system_pattern",
        }
        updates += 1
    if ntype == "Tool" and not isinstance(node.get("tool_example_pairs"), list):
        node["tool_example_pairs"] = []
        updates += 1
    return updates


def _normalize_ref_list(vals: Any, *, existing_vars: Set[str], self_var: str) -> List[str]:
    """Normalize a reference list to unique valid vars, excluding self and unknown refs."""
    if not isinstance(vals, list):
        return []
    out: List[str] = []
    for item in vals:
        v = extract_var_name(item)
        if not v or v == self_var:
            continue
        if v not in existing_vars and v not in ALLOWED_SPECIAL_REFS:
            continue
        if v not in out:
            out.append(v)
    return out


def normalize_connectivity_fields(
    node: Dict[str, Any],
    *,
    self_var: str,
    existing_vars: Set[str],
    nodes_by_var: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Validate/prune connectivity fields to valid references."""
    updates = {"nodes": 0, "tool_list": 0, "internal_edges": 0, "external_connections": 0}

    # Normalize direct child references in `nodes`.
    old_nodes = node.get("nodes")
    new_nodes = _normalize_ref_list(old_nodes, existing_vars=existing_vars, self_var=self_var)
    if old_nodes != new_nodes:
        node["nodes"] = new_nodes
        updates["nodes"] += 1

    # Normalize direct child references in `tool_list`.
    old_tools = node.get("tool_list")
    new_tools = _normalize_ref_list(old_tools, existing_vars=existing_vars, self_var=self_var)
    # MCP policy: tool_list must contain Tool nodes only (never server nodes).
    ntype = node_type_name(node)
    if ntype in {"Local_MCP_server", "External_MCP_server"} and isinstance(nodes_by_var, dict):
        filtered_tools: List[str] = []
        for v in new_tools:
            cnode = nodes_by_var.get(v, {})
            if node_type_name(cnode) == "Tool":
                filtered_tools.append(v)
        new_tools = filtered_tools
    if old_tools != new_tools:
        node["tool_list"] = new_tools
        updates["tool_list"] += 1

    # Rebuild `internal_edges` so only valid edge objects between valid refs remain.
    edges = node.get("internal_edges")
    new_edges: List[Dict[str, Any]] = []
    if isinstance(edges, list):
        for e in edges:
            if not isinstance(e, dict):
                continue
            frm = extract_var_name(e.get("from_"))
            to = extract_var_name(e.get("to"))
            if not frm or not to:
                continue
            if frm == self_var or to == self_var:
                continue
            frm_ok = frm in existing_vars or frm in ALLOWED_SPECIAL_REFS
            to_ok = to in existing_vars or to in ALLOWED_SPECIAL_REFS
            if not frm_ok or not to_ok:
                continue
            new_edges.append({"from_": frm, "to": to, "description": str(e.get("description") or "")})
    if edges != new_edges:
        node["internal_edges"] = new_edges
        updates["internal_edges"] += 1

    # Normalize external connection refs and ensure the expected object shape exists.
    ec = node.get("external_connections")
    if not isinstance(ec, dict):
        ec = {"in_": [], "out": []}
        updates["external_connections"] += 1
    ec_in = _normalize_ref_list(ec.get("in_"), existing_vars=existing_vars, self_var=self_var)
    ec_out = _normalize_ref_list(ec.get("out"), existing_vars=existing_vars, self_var=self_var)
    new_ec = {"in_": ec_in, "out": ec_out}
    if ec != new_ec:
        node["external_connections"] = new_ec
        updates["external_connections"] += 1

    return updates


def resolve_detached_candidates(
    *,
    initial_candidates: List[str],
    nodes_by_var: Dict[str, Dict[str, Any]],
    var_order: List[str],
    issue_root: Path,
    root_var: str | None,
    guidance_summary: str,
    model: str,
    refresh_raw: bool,
    retriever: Any,
    rag_max_rounds: int,
    representation_shortlist_prompt: str,
    representation_pair_prompt: str,
    missing_node_decision_prompt: str,
    adding_node_parent_decision_prompt: str,
) -> List[Dict[str, Any]]:
    """Resolve detached candidates and orphan cascades using the shared orphan-resolution flow."""
    # Track resolved rows and process detached candidates breadth-first so orphan cascades are handled in order.
    rows: List[Dict[str, Any]] = []
    pending: List[str] = [str(v).strip() for v in initial_candidates if isinstance(v, str) and str(v).strip()]
    seen: Set[str] = set()
    while pending:
        var = pending.pop(0)
        if var in seen or var not in nodes_by_var:
            continue
        seen.add(var)

        # Snapshot current children before detaching/removing this node so orphan cascades can be followed later.
        children_by_parent_before, _ = build_parent_child_maps(var_order, nodes_by_var)
        children_before = list(children_by_parent_before.get(var, []))
        issue_dir = issue_root / var
        issue_dir.mkdir(parents=True, exist_ok=True)

        # First detach the candidate cleanly from its parents, then run the shared orphan-resolution flow.
        detached = detach_var_from_parents(
            var_order=var_order,
            nodes_by_var=nodes_by_var,
            child_var=var,
            parent_vars=[p for p in list(var_order) if p != var],
            drop_internal_edges=True
        )
        detach_meta = {
            "detached_var": var,
            "touched_parent_vars": list(detached.get("parents") or []),
            "parent_refs_removed": int(detached.get("parent_links_removed") or 0),
            "edge_refs_removed": int(detached.get("internal_edges_removed") or 0),
            "external_refs_removed": int(detached.get("external_refs_removed") or 0),
        }
        resolution_meta = resolve_orphan_node(
            candidate_var=var,
            root_var=root_var,
            guidance_summary=guidance_summary,
            issue_dir=issue_dir,
            model=model,
            refresh_raw=refresh_raw,
            retriever=retriever,
            rag_max_rounds=rag_max_rounds,
            nodes_by_var=nodes_by_var,
            var_order=var_order,
            representation_shortlist_prompt=representation_shortlist_prompt,
            representation_pair_prompt=representation_pair_prompt,
            missing_node_decision_prompt=missing_node_decision_prompt,
            adding_node_parent_decision_prompt=adding_node_parent_decision_prompt,
        )
        # If orphan resolution leaves the node unattached and unmerged, remove it from the graph explicitly.
        removed_from_graph = var not in nodes_by_var
        forced_remove = False
        if not removed_from_graph:
            resolve_obj = resolution_meta.get("resolve") if isinstance(resolution_meta.get("resolve"), dict) else {}
            represented_by_var = str(resolve_obj.get("represented_by_var") or "").strip()
            attached_parent_var = str(resolve_obj.get("attached_parent_var") or "").strip()
            if not represented_by_var and not attached_parent_var:
                forced_remove = True
                removed = detach_var_from_parents(
                    var_order=var_order,
                    nodes_by_var=nodes_by_var,
                    child_var=var,
                    parent_vars=list(var_order),
                    drop_internal_edges=True,
                )
                nodes_by_var.pop(var, None)
                if var in var_order:
                    var_order.remove(var)
                remove_meta = {
                    "removed_var": var,
                    "touched_parent_vars": list(removed.get("parents") or []),
                }
                removed_from_graph = True
                resolution_meta = {
                    **(resolution_meta if isinstance(resolution_meta, dict) else {}),
                    "forced_remove_unresolved": True,
                    "forced_remove_meta": remove_meta,
                }

        # Record the detach/resolution outcome for reporting and downstream review.
        row = {
            "var": var,
            "detached": detach_meta,
            "resolution": resolution_meta,
            "removed_from_graph": removed_from_graph,
            "forced_remove_unresolved": forced_remove,
        }
        rows.append(row)

        # When a node is removed, enqueue any newly orphaned direct children for the same resolution flow.
        if removed_from_graph:
            _, parents_by_child_after = build_parent_child_maps(var_order, nodes_by_var)
            for cvar in children_before:
                if cvar in nodes_by_var and not parents_by_child_after.get(cvar) and cvar not in seen:
                    pending.append(cvar)

    return rows


def apply_filtered_updates(node: Dict[str, Any], updates: Dict[str, Any], *, allowed_fields: Set[str]) -> int:
    """Apply only allowed field updates to a node and return how many fields actually changed."""
    if not isinstance(updates, dict):
        return 0
    changed = 0
    for k, v in updates.items():
        if k not in allowed_fields:
            continue
        if k == "required_keys" and isinstance(v, dict):
            v = {**v, "enabled": bool(v.get("keys"))}
        if node.get(k) != v:
            node[k] = v
            changed += 1
    return changed


def parse_group_updates(decision: Dict[str, Any], *, allowed_fields: Set[str]) -> Dict[str, Any]:
    """Extract only allowed update fields from a refinement decision, supporting both wrapped and top-level shapes."""
    if not isinstance(decision, dict):
        return {}
    updates = decision.get("updates")
    if isinstance(updates, dict):
        return {k: v for k, v in updates.items() if k in allowed_fields}
    # Fallback: accept top-level fields when model omits "updates" wrapper.
    out: Dict[str, Any] = {}
    for k in allowed_fields:
        if k in decision:
            out[k] = decision.get(k)
    return out
