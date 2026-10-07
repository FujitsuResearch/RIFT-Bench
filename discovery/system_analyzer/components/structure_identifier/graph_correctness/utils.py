#!/usr/bin/env python3
"""Helpers for graph_correctness pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
    from ..global_utils import (
        add_child_ref,
        build_parent_child_maps,
        call_with_cache,
        detach_var_from_parents,
        drop_var_refs_from_external_connections,
        drop_var_refs_from_internal_edges,
        next_var,
        node_type_name,
        read_json,
        remove_child_ref,
        run_shared_graph_candidate_decision,
        _generate_graph_summary,
        safe_name,
    )
    from ..child_creation.utils import (
        _build_child_proposals_for_parent,
        _child_field_for_parent,
        _code_refs_add,
        _judge_child_runtime_inclusion,
        _materialize_child,
        _merge_duplicate_reused_metadata,
        next_var as _cc_next_var,
        _resolve_duplicate_child_global,
    )
    from ..connectivity_pass.utils import repair_parent_connectivity_iterative
    from ..static_validation.prompts import (
        ADDING_NODE_PARENT_DECISION_PROMPT,
        MISSING_NODE_DECISION_PROMPT,
        REPRESENTATION_PAIR_PROMPT,
        REPRESENTATION_SHORTLIST_PROMPT,
    )
    from ..NodeSpec_per_file.utils import _cr_dedupe_merge_refs
except ImportError:
    from global_utils import (
        add_child_ref,
        build_parent_child_maps,
        call_with_cache,
        detach_var_from_parents,
        drop_var_refs_from_external_connections,
        drop_var_refs_from_internal_edges,
        next_var,
        node_type_name,
        read_json,
        remove_child_ref,
        run_shared_graph_candidate_decision,
        safe_name,
    )
    from child_creation.utils import (
        _build_child_proposals_for_parent,
        _child_field_for_parent,
        _code_refs_add,
        _judge_child_runtime_inclusion,
        _materialize_child,
        _merge_duplicate_reused_metadata,
        next_var as _cc_next_var,
        _resolve_duplicate_child_global,
    )
    from connectivity_pass.utils import repair_parent_connectivity_iterative
    from static_validation.prompts import (
        ADDING_NODE_PARENT_DECISION_PROMPT,
        MISSING_NODE_DECISION_PROMPT,
        REPRESENTATION_PAIR_PROMPT,
        REPRESENTATION_SHORTLIST_PROMPT,
    )
    from NodeSpec_per_file.utils import _cr_dedupe_merge_refs

MCP_SERVER_TYPES = {"Local_MCP_server", "External_MCP_server"}


def children_of(node: Dict[str, Any]) -> List[str]:
    """Return unique child vars from both `nodes` and `tool_list`."""
    out: List[str] = []
    for field in ("nodes", "tool_list"):
        vals = node.get(field)
        if not isinstance(vals, list):
            continue
        for item in vals:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict) and isinstance(item.get("__var__"), str):
                out.append(str(item["__var__"]))
    return sorted(set(x for x in out if x))


def children_in_nodes(node: Dict[str, Any]) -> List[str]:
    """Return unique child vars listed in the node's `nodes` field."""
    out: List[str] = []
    vals = node.get("nodes")
    if isinstance(vals, list):
        for item in vals:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict) and isinstance(item.get("__var__"), str):
                out.append(str(item["__var__"]))
    return sorted(set(x for x in out if x))


def children_in_tool_list(node: Dict[str, Any]) -> List[str]:
    """Return deduplicated child vars referenced in a node's `tool_list`."""
    out: List[str] = []
    vals = node.get("tool_list")
    if isinstance(vals, list):
        for item in vals:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict) and isinstance(item.get("__var__"), str):
                out.append(str(item["__var__"]))
    return sorted(set(x for x in out if x))


def build_node_catalog(var_order: List[str], nodes_by_var: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build the flat prompt-facing catalog view of the current graph nodes."""
    out: List[Dict[str, Any]] = []
    for v in var_order:
        n = nodes_by_var.get(v, {})
        out.append(
            {
                "var": v,
                "name": str(n.get("name") or ""),
                "node_type": node_type_name(n),
                "description": str(n.get("description") or "")[:500],
            }
        )
    return out


def build_summary_node_catalog(var_order: List[str], nodes_by_var: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build the richer summary catalog with each node's direct children included."""
    children_by_parent, _ = build_parent_child_maps(var_order, nodes_by_var)
    out: List[Dict[str, Any]] = []
    for v in var_order:
        n = nodes_by_var.get(v, {})
        child_vars = children_by_parent.get(v, [])
        children: List[Dict[str, Any]] = []
        for c in child_vars:
            cn = nodes_by_var.get(c, {})
            children.append(
                {
                    "var": c,
                    "name": str(cn.get("name") or ""),
                    "type": node_type_name(cn),
                }
            )
        out.append(
            {
                "var": v,
                "name": str(n.get("name") or ""),
                "type": node_type_name(n),
                "description": str(n.get("description") or "")[:500],
                "children": children,
            }
        )
    return out


def nodespec_class_fields_for_summary() -> List[str]:
    """Return the NodeSpec class fields exposed to the summary-generation prompt."""
    # NodeSpec fields only (not auxiliary classes/methods).
    return [
        "name",
        "id",
        "node_type",
        "description",
        "code_execution",
        "agent_type",
        "system_type",
        "code_references",
        "emulated_code_references",
        "inputs",
        "outputs",
        "external_connections",
        "required_keys",
        "duplicates",
        "is_graph",
        "emulated",
        "framework",
        "agency_level",
        "flows",
        "entry_point_usage_example",
        "system_summary",
        "llm_config",
        "system_prompt",
        "user_prompt_template",
        "tool_list",
        "tool_example_pairs",
        "read_internal",
        "read_external",
        "write_internal",
        "write_external",
        "is_rag_tool",
        "nodes",
        "internal_edges",
        "data_path",
        "metadata",
    ]


def run_rule_det_controller_promote_children(
    *,
    var_order: List[str],
    nodes_by_var: Dict[str, Dict[str, Any]],
    guidance_summary: str,
    raw_dir: Path,
    model: str,
    refresh_raw: bool,
    retriever: Any,
    rag_max_rounds: int,
    actions: List[Dict[str, Any]],
    unresolved: List[Dict[str, Any]],
) -> None:
    """
    Deterministic rule:
    If a Deterministic_controller has node-children, move those
    children to the controller's single parent, keep the controller but make it
    childless, and run connectivity pass on that parent.
    """
    skipped_controllers: Set[str] = set()
    while True:
        _, parents_by_child = build_parent_child_maps(var_order, nodes_by_var)
        target_controller: str | None = None

        # Find the next controller that still has direct node children to promote.
        for cvar in list(var_order):
            if cvar in skipped_controllers:
                continue
            cnode = nodes_by_var.get(cvar, {})
            if node_type_name(cnode) != "Deterministic_controller":
                continue
            if not children_in_nodes(cnode):
                continue
            target_controller = cvar
            break

        if target_controller is None:
            break

        cvar = target_controller
        cnode = nodes_by_var.get(cvar, {})

        # Gather the controller children and prepare the per-issue output directory.
        cchildren = [v for v in children_in_nodes(cnode) if isinstance(v, str) and v in nodes_by_var and v != cvar]
        issue_dir = raw_dir / "rule_det_controller_promote_children" / safe_name(cvar)
        issue_dir.mkdir(parents=True, exist_ok=True)

        parents = [p for p in (parents_by_child.get(cvar) or []) if isinstance(p, str) and p in nodes_by_var]

        # Skip controllers that cannot be safely promoted into exactly one parent.
        if len(parents) != 1:
            row = {
                "rule": "deterministic_controller_promote_children",
                "controller_var": cvar,
                "decision": "skip_controller_transfer",
                "reason": "controller does not have exactly one parent",
                "changed": False,
                "parents": sorted(parents),
                "children_before": list(cchildren),
            }
            actions.append(row)
            unresolved.append(row)
            skipped_controllers.add(cvar)
            continue

        pvar = parents[0]
        pnode = nodes_by_var.get(pvar, {})
        moved: List[str] = []

        # Reattach each controller child directly under the controller parent.
        for ch in cchildren:
            field = _child_field_for_parent(pnode, nodes_by_var.get(ch, {}))
            if add_child_ref(pnode, ch, field=field):
                moved.append(ch)

        # Keep the controller node, but clear its direct node children after the transfer.
        cnode["nodes"] = []

        # Rebuild the affected parent connectivity after moving the children.
        connectivity_meta = repair_parent_connectivity_iterative(
            parent_var=pvar,
            parent_node=nodes_by_var.get(pvar, {}),
            guidance_summary=guidance_summary,
            retriever=retriever,
            model=model,
            raw_parent_dir=issue_dir / "connectivity_pass_after_transfer",
            refresh_raw=refresh_raw,
            rag_max_rounds=rag_max_rounds,
            nodes_by_var=nodes_by_var,
            max_iterations=2,
        )

        actions.append(
            {
                "rule": "deterministic_controller_promote_children",
                "controller_var": cvar,
                "parent_var": pvar,
                "decision": "transfer_children_keep_controller_childless",
                "reason": "deterministic_controller_with_node_children",
                "changed": True,
                "children_before": list(cchildren),
                "children_moved_to_parent": moved,
                "controller_kept": True,
                "controller_nodes_after": list(children_in_nodes(nodes_by_var.get(cvar, {}))),
                "connectivity_pass_after_transfer": connectivity_meta,
            }
        )


def merge_code_references(dst_node: Dict[str, Any], src_node: Dict[str, Any]) -> int:
    """Merge source code references into the destination node with dedupe and return the effective count added."""
    dst_refs = dst_node.get("code_references") if isinstance(dst_node.get("code_references"), list) else []
    src_refs = src_node.get("code_references") if isinstance(src_node.get("code_references"), list) else []
    merged_refs = _cr_dedupe_merge_refs([*src_refs, *dst_refs])
    refs_added = max(0, len(merged_refs) - len(dst_refs))
    dst_node["code_references"] = merged_refs
    return refs_added


def merge_refs_and_children_only(dst_node: Dict[str, Any], src_node: Dict[str, Any]) -> Dict[str, int]:
    """Merge only code references and direct child links from the source node into the destination node."""
    refs_added = merge_code_references(dst_node, src_node)
    nodes_added = 0
    tool_added = 0
    for child_var in children_of({"nodes": src_node.get("nodes", []), "tool_list": []}):
        if add_child_ref(dst_node, child_var, field="nodes"):
            nodes_added += 1
    for child_var in children_of({"nodes": [], "tool_list": src_node.get("tool_list", [])}):
        if add_child_ref(dst_node, child_var, field="tool_list"):
            tool_added += 1
    return {"refs_added": refs_added, "nodes_added": nodes_added, "tool_added": tool_added}


def merge_component(parent_node: Dict[str, Any], child_node: Dict[str, Any]) -> Dict[str, int]:
    """Merge child node references and direct children into the parent node and report what was added."""

    merge_stats = merge_refs_and_children_only(parent_node, child_node)
    p_md = parent_node.get("metadata") if isinstance(parent_node.get("metadata"), dict) else {}
    merged_from = p_md.get("merged_from") if isinstance(p_md.get("merged_from"), list) else []
    c_name = str(child_node.get("name") or "")
    if c_name and c_name not in merged_from:
        merged_from.append(c_name)
    p_md["merged_from"] = merged_from
    parent_node["metadata"] = p_md
    return merge_stats


def materialize_attach_children_from_proposals(
    *,
    parent_var: str,
    parent_node: Dict[str, Any],
    proposals: List[Dict[str, Any]],
    guidance_summary: str,
    retriever: Any,
    model: str,
    raw_parent_dir: Path,
    refresh_raw: bool,
    rag_max_rounds: int,
    nodes_by_var: Dict[str, Dict[str, Any]],
    var_order: List[str],
    allowed_node_types: Set[str] | None = None,
    run_connectivity_pass: bool = False,
    skip_runtime_judge: bool = False,
) -> Dict[str, Any]:
    """Materialize child proposals, reuse or create graph nodes, attach them, and optionally repair connectivity."""
    schema_catalog_path = Path(__file__).resolve().parents[1] / "NodeSpec_schema_short.py"
    nodespec_schema_catalog = (
        schema_catalog_path.read_text(encoding="utf-8", errors="replace")
        if schema_catalog_path.exists()
        else ""
    )

    created_here: List[str] = []
    reused_here: List[Dict[str, Any]] = []
    excluded_here: List[Dict[str, Any]] = []
    attached_child_vars: List[str] = []
    created_nodes = 0
    attached_changed = False

    typed_filter = set(allowed_node_types or set())
    typed_filter_enabled = bool(typed_filter)

    for idx, proposal in enumerate([p for p in (proposals or []) if isinstance(p, dict)], start=1):
        proposal_type = str(proposal.get("node_type") or "").strip()
        if typed_filter_enabled and proposal_type and proposal_type not in typed_filter:
            excluded_here.append(
                {
                    "proposal_name": str(proposal.get("name") or ""),
                    "reason": f"filtered_node_type_not_allowed:{proposal_type}",
                }
            )
            continue

        child_raw_dir = raw_parent_dir / f"child_{idx:03d}_{safe_name(str(proposal.get('name') or 'child'))}"
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

        if typed_filter_enabled and node_type_name(child_node) not in typed_filter:
            excluded_here.append(
                {
                    "proposal_name": str(proposal.get("name") or ""),
                    "reason": f"materialized_node_type_not_allowed:{node_type_name(child_node)}",
                }
            )
            continue

        if not skip_runtime_judge:
            tentative_child_var = _cc_next_var(str(child_node.get("name") or "child"), set(nodes_by_var.keys()))
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
                existing_node.get("code_references")
                if isinstance(existing_node.get("code_references"), list)
                else [],
                child_node.get("code_references")
                if isinstance(child_node.get("code_references"), list)
                else [],
            )
            _merge_duplicate_reused_metadata(
                existing_node,
                parent_var=parent_var,
                proposal_name=str(proposal.get("name") or ""),
                reason=str(dup.get("reason") or ""),
            )
            attach_field = _child_field_for_parent(parent_node, existing_node)
            attached_changed = add_child_ref(parent_node, matched_var, field=attach_field) or attached_changed
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

        child_var = _cc_next_var(str(child_node.get("name") or "child"), set(nodes_by_var.keys()))
        nodes_by_var[child_var] = child_node
        var_order.append(child_var)
        created_nodes += 1
        created_here.append(child_var)
        attached_child_vars.append(child_var)
        attach_field = _child_field_for_parent(parent_node, child_node)
        add_child_ref(parent_node, child_var, field=attach_field)
        attached_changed = True

    connectivity_repair: Dict[str, Any] | None = None
    if run_connectivity_pass and attached_changed:
        connectivity_repair = repair_parent_connectivity_iterative(
            parent_var=parent_var,
            parent_node=parent_node,
            guidance_summary=guidance_summary,
            retriever=retriever,
            model=model,
            raw_parent_dir=raw_parent_dir / "connectivity_after_attach",
            refresh_raw=refresh_raw,
            rag_max_rounds=rag_max_rounds,
            nodes_by_var=nodes_by_var,
            max_iterations=2,
        )

    return {
        "created_children_vars": created_here,
        "reused_existing_children": reused_here,
        "excluded_children": excluded_here,
        "attached_child_vars": attached_child_vars,
        "created_nodes": created_nodes,
        "connectivity_repair": connectivity_repair,
        "changed": bool(attached_changed) or bool(connectivity_repair),
    }


def run_prioritized_parent_then_child_merges(
    *,
    source_var: str,
    parent_targets: List[str],
    child_targets: List[str],
    root_var: str | None,
    nodes_by_var: Dict[str, Dict[str, Any]],
    var_order: List[str],
    guidance_summary: str,
    issue_dir: Path,
    stem_prefix: str,
    merge_prompt: str,
    model: str,
    refresh_raw: bool,
    retriever: Any,
    rag_max_rounds: int,
) -> Dict[str, Any]:
    """Try merging a source node into parent targets first, and only fall back to child targets if no parent merge succeeds."""
    merged_into: List[str] = []
    merge_decisions: List[Dict[str, Any]] = []
    parent_merged = False
    applied_merges: Dict[str, Dict[str, Any]] = {}

    # Evaluate one merge candidate and record the decision, applying the full merge flow only when approved.
    def _eval_target(tvar: str, scope: str) -> bool:
        nonlocal parent_merged
        if tvar not in nodes_by_var or source_var not in nodes_by_var:
            return False
        pair_dec_raw, _ = call_with_cache(
            payload={
                "prompt": merge_prompt,
                "parent_var": tvar,
                "parent_node": nodes_by_var.get(tvar, {}),
                "child_var": source_var,
                "child_node": nodes_by_var.get(source_var, {}),
                "guidance_summary": guidance_summary,
                "retrieved_context": None,
            },
            payload_path=issue_dir / (str(f"{stem_prefix}_{scope}_{safe_name(tvar)}") + ".payload.json"),
            raw_path=issue_dir / (str(f"{stem_prefix}_{scope}_{safe_name(tvar)}") + ".txt"),
            model=model,
            refresh_raw=refresh_raw,
            retriever=retriever,
            rag_max_rounds=rag_max_rounds,
        )
        pair_dec = pair_dec_raw if isinstance(pair_dec_raw, dict) else {}
        pair_action = str(pair_dec.get("action") or "keep_distinct").strip()
        merge_decisions.append(
            {
                "target_var": tvar,
                "target_scope": scope,
                "decision": pair_action,
                "reason": str(pair_dec.get("reason") or ""),
            }
        )
        if pair_action != "merge_components" or tvar not in nodes_by_var or source_var not in nodes_by_var:
            return False

        merge_meta = apply_merge_decision(
            parent_var=tvar if scope == "parent" else source_var,
            merge_into_var=tvar,
            merge_from_var=source_var,
            root_var=root_var or "",
            parents_by_child=build_parent_child_maps(var_order, nodes_by_var)[1],
            nodes_by_var=nodes_by_var,
            var_order=var_order,
            guidance_summary=guidance_summary,
            retriever=retriever,
            model=model,
            issue_dir=issue_dir / f"{stem_prefix}_{scope}_{safe_name(tvar)}_full_merge",
            refresh_raw=refresh_raw,
            rag_max_rounds=rag_max_rounds,
        )
        applied_merges[tvar] = merge_meta
        if tvar not in merged_into:
            merged_into.append(tvar)
        if scope == "parent":
            parent_merged = True
        return True

    # Try parent merges first so the source collapses upward when possible.
    for tvar in parent_targets:
        if source_var not in nodes_by_var:
            break
        _eval_target(tvar, "parent")

    # Only try child merges if no parent target accepted the merge.
    if not parent_merged:
        for tvar in child_targets:
            if source_var not in nodes_by_var:
                break
            if _eval_target(tvar, "child"):
                break

    # Return both the chosen merge targets and the full per-target decision log.
    return {
        "merged_into": merged_into,
        "merge_decisions": merge_decisions,
        "parent_merged": parent_merged,
        "applied_merges": applied_merges,
    }


def _deterministic_merge_plan_for_pair(
    *,
    parent_var: str,
    child_var: str,
    parents_by_child: Dict[str, List[str]],
    nodes_by_var: Dict[str, Dict[str, Any]] | None = None,
    requested_action: str,
) -> Dict[str, Any]:
    """Apply deterministic merge guards and merge direction for one parent-child candidate pair."""
    def _count_non_llm(var: str) -> int:
        parents = [p for p in (parents_by_child.get(var) or []) if isinstance(p, str)]
        if nodes_by_var is None:
            return len(parents)
        return len([p for p in parents if node_type_name(nodes_by_var.get(p, {})) != "LLM"])

    p_count = _count_non_llm(parent_var)
    c_count = _count_non_llm(child_var)
    plan: Dict[str, Any] = {
        "requested_action": requested_action,
        "effective_action": requested_action,
        "merge_into_var": parent_var,
        "merge_from_var": child_var,
        "parent_parent_count_non_llm": p_count,
        "child_parent_count_non_llm": c_count,
        "reason": "",
    }
    if requested_action != "merge_components":
        return plan
    if p_count > 1 and c_count > 1:
        plan["effective_action"] = "keep_distinct"
        plan["reason"] = "deterministic_guard_both_multi_parent_keep_distinct"
        return plan
    if c_count > p_count:
        plan["merge_into_var"] = child_var
        plan["merge_from_var"] = parent_var
        plan["reason"] = "deterministic_guard_merge_into_higher_parent_count_node"
        return plan
    plan["reason"] = "deterministic_guard_default_merge_direction"
    return plan


def _rewire_parents_to_target(
    *,
    absorb_var: str,
    keep_var: str,
    parents_by_child: Dict[str, List[str]],
    nodes_by_var: Dict[str, Dict[str, Any]],
) -> Dict[str, int]:
    """Move parent references from the absorbed node to the kept node and update parent edge references too."""

    removed_refs = 0
    added_refs = 0
    rewired_internal_edges = 0
    rewired_external_refs = 0
    for pv in [p for p in (parents_by_child.get(absorb_var) or []) if isinstance(p, str)]:
        if pv == keep_var or pv not in nodes_by_var:
            continue
        pnode = nodes_by_var.get(pv, {})
        if remove_child_ref(pnode, absorb_var):
            removed_refs += 1
        field = _child_field_for_parent(pnode, nodes_by_var.get(keep_var, {}))
        if add_child_ref(pnode, keep_var, field=field):
            added_refs += 1

        internal_edges = pnode.get("internal_edges")
        if isinstance(internal_edges, list):
            for edge in internal_edges:
                if not isinstance(edge, dict):
                    continue
                if str(edge.get("from_") or "").strip() == absorb_var:
                    edge["from_"] = keep_var
                    rewired_internal_edges += 1
                if str(edge.get("to") or "").strip() == absorb_var:
                    edge["to"] = keep_var
                    rewired_internal_edges += 1

    return {
        "removed_refs": removed_refs,
        "added_refs": added_refs,
        "rewired_internal_edges": rewired_internal_edges
    }


def delete_node(var: str, var_order: List[str], nodes_by_var: Dict[str, Dict[str, Any]]) -> None:
    """Delete a node and remove its references from the rest of the graph."""
    detach_var_from_parents(
        var_order=var_order,
        nodes_by_var=nodes_by_var,
        child_var=var,
        parent_vars=list(var_order),
        drop_internal_edges=True
    )
    if var in nodes_by_var:
        nodes_by_var.pop(var, None)
    if var in var_order:
        var_order.remove(var)


def resolve_orphan_node(
    *,
    candidate_var: str,
    root_var: str | None,
    guidance_summary: str,
    issue_dir: Path,
    model: str,
    refresh_raw: bool,
    retriever: Any,
    rag_max_rounds: int,
    nodes_by_var: Dict[str, Dict[str, Any]],
    var_order: List[str],
    representation_shortlist_prompt: str,
    representation_pair_prompt: str,
    missing_node_decision_prompt: str,
    adding_node_parent_decision_prompt: str,
    _visited: Set[str] | None = None,
) -> Dict[str, Any]:
    """Resolve one orphan graph node by representing it elsewhere or reattaching it."""

    # Guard against recursion cycles and missing candidates.
    if _visited is None:
        _visited = set()
    if candidate_var in _visited:

        # Return the orphan-resolution result plus any follow-up connectivity repair metadata.
        resolve_meta = {
            "changed": False,
            "plausible_matches": [],
            "pair_checks": [],
            "represented_by_var": None,
            "attached_parent_var": None,
            "missing_decision": None,
            "children_resolution": {},
            "skipped_cycle": True,
            "transferred_children_to_represented": [],
            "attach_connectivity_repair": None,
        }
        return {"resolve": resolve_meta, "connectivity_repair": None, "changed": False}
    _visited.add(candidate_var)

    if candidate_var not in nodes_by_var:
        resolve_meta = {
            "changed": False,
            "plausible_matches": [],
            "pair_checks": [],
            "represented_by_var": None,
            "attached_parent_var": None,
            "missing_decision": None,
            "children_resolution": {},
            "transferred_children_to_represented": [],
            "attach_connectivity_repair": None,
        }
        return {"resolve": resolve_meta, "connectivity_repair": None, "changed": False}

    _, parents_by_child = build_parent_child_maps(var_order, nodes_by_var)
    if parents_by_child.get(candidate_var):
        resolve_meta = {
            "changed": False,
            "plausible_matches": [],
            "pair_checks": [],
            "represented_by_var": None,
            "attached_parent_var": None,
            "missing_decision": None,
            "children_resolution": {},
            "transferred_children_to_represented": [],
            "attach_connectivity_repair": None,
            "skipped_non_orphan": True,
        }
        return {"resolve": resolve_meta, "connectivity_repair": None, "changed": False}

    graph_catalog = build_node_catalog(var_order, nodes_by_var)
    graph_catalog_wo_self = [x for x in graph_catalog if str(x.get("var") or "") != candidate_var]
    allowed_graph_vars = {
        v
        for v in var_order
        if v != candidate_var and v in nodes_by_var
    }
    decision_meta = run_shared_graph_candidate_decision(
        candidate_var=candidate_var,
        candidate_node=nodes_by_var.get(candidate_var, {}),
        root_var=root_var,
        graph_catalog=graph_catalog_wo_self,
        nodes_by_var=nodes_by_var,
        decision_dir=issue_dir,
        model=model,
        refresh_raw=refresh_raw,
        retriever=retriever,
        rag_max_rounds=rag_max_rounds,
        guidance_summary=guidance_summary,
        representation_shortlist_prompt=representation_shortlist_prompt,
        representation_pair_prompt=representation_pair_prompt,
        missing_node_decision_prompt=missing_node_decision_prompt,
        adding_node_parent_decision_prompt=adding_node_parent_decision_prompt,
        allowed_match_vars=allowed_graph_vars,
        allowed_parent_vars=allowed_graph_vars,
        fallback_parent_vars=[v for v in var_order if v in allowed_graph_vars],
        allow_parent_fallback=True,
    )

    # Initialize local mutation state from the shared decision result.
    represented_by_var = str(decision_meta.get("represented_by_var") or "")
    plausible = list(decision_meta.get("plausible_matches") or [])
    pair_checks = list(decision_meta.get("pair_checks") or [])
    missing_decision = decision_meta.get("missing_decision_output") if isinstance(decision_meta.get("missing_decision_output"), dict) else None
    changed = False
    children_resolution: Dict[str, Any] = {}
    transferred_children_to_represented: List[Dict[str, Any]] = []
    attach_connectivity_repair: Dict[str, Any] | None = None

    def _attach_field_for_represented(parent_var: str, child_var: str) -> str:
        """Choose whether a transferred child should attach under `nodes` or `tool_list`."""
        ptype = node_type_name(nodes_by_var.get(parent_var, {}))
        ctype = node_type_name(nodes_by_var.get(child_var, {}))
        if ptype in {"Local_MCP_server", "External_MCP_server"} and ctype == "Tool":
            return "tool_list"
        return "nodes"

    def _transfer_direct_children_to_represented(src_var: str, dst_var: str) -> bool:
        """Move the source node's direct children onto the represented destination node."""
        nonlocal changed
        src_node = nodes_by_var.get(src_var, {})
        dst_node = nodes_by_var.get(dst_var, {})
        if not isinstance(src_node, dict) or not isinstance(dst_node, dict):
            return False
        linked_any = False
        for child_var in children_of(src_node):
            if child_var == src_var or child_var == dst_var or child_var not in nodes_by_var:
                continue
            field = _attach_field_for_represented(dst_var, child_var)
            linked = add_child_ref(dst_node, child_var, field=field)
            if linked:
                changed = True
                linked_any = True
            transferred_children_to_represented.append(
                {"child_var": child_var, "to_var": dst_var, "field": field, "linked": bool(linked)}
            )
        return linked_any

    def _resolve_children_after_parent_removed(
        removed_children: List[str],
        *,
        base_dir: Path,
        force_attach_parent_var: str | None = None,
    ) -> None:
        """Resolve children that become orphaned after their parent node is removed."""
        nonlocal changed

        # Walk each newly orphaned child and either force-reattach it or recurse.
        for child_var in removed_children:
            if child_var not in nodes_by_var or child_var in _visited:
                continue
            forced_parent = str(force_attach_parent_var or "").strip()

            # If a represented parent is known, prefer reattaching the child there directly.
            if forced_parent and forced_parent in nodes_by_var and forced_parent != child_var:
                _, parents_by_child = build_parent_child_maps(var_order, nodes_by_var)
                existing_parents = list(parents_by_child.get(child_var, []))
                already_attached = forced_parent in existing_parents
                linked = False
                if not already_attached:
                    field = _attach_field_for_represented(forced_parent, child_var)
                    linked = add_child_ref(nodes_by_var[forced_parent], child_var, field=field)
                    changed = changed or bool(linked)
                connectivity_meta: Dict[str, Any] | None = None
                if linked:
                    connectivity_meta = repair_parent_connectivity_iterative(
                        parent_var=forced_parent,
                        parent_node=nodes_by_var[forced_parent],
                        guidance_summary=guidance_summary,
                        retriever=retriever,
                        model=model,
                        raw_parent_dir=base_dir / safe_name(child_var) / "connectivity_after_forced_represented_attach",
                        refresh_raw=refresh_raw,
                        rag_max_rounds=rag_max_rounds,
                        nodes_by_var=nodes_by_var,
                        max_iterations=2,
                    )
                    changed = changed or bool(connectivity_meta)
                children_resolution[child_var] = {
                    "resolve": {
                        "changed": bool(linked),
                        "forced_attach_parent_var": forced_parent,
                        "already_attached": bool(already_attached),
                        "reason": "parent removed while candidate is represented_by another node; child is attached to represented node",
                    },
                    "connectivity_repair": connectivity_meta,
                }
                continue

            # Otherwise resolve the child as a fresh orphan through the same orphan flow.
            child_out = resolve_orphan_node(
                candidate_var=child_var,
                root_var=root_var,
                guidance_summary=guidance_summary,
                issue_dir=base_dir / safe_name(child_var),
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
                _visited=_visited,
            )
            child_meta = child_out.get("resolve") if isinstance(child_out.get("resolve"), dict) else {}
            connectivity_meta = child_out.get("connectivity_repair") if isinstance(child_out.get("connectivity_repair"), dict) else None
            changed = changed or bool(child_out.get("changed"))
            children_resolution[child_var] = {
                "resolve": child_meta,
                "connectivity_repair": connectivity_meta,
            }


    # If another node already represents this orphan, merge into it and clean up.
    attached_parent_var = ""
    if represented_by_var:
        transfer_linked = _transfer_direct_children_to_represented(candidate_var, represented_by_var)
        refs_added = merge_code_references(nodes_by_var[represented_by_var], nodes_by_var.get(candidate_var, {}))
        changed = changed or refs_added > 0
        if transfer_linked and represented_by_var in nodes_by_var:
            rep_conn = repair_parent_connectivity_iterative(
                parent_var=represented_by_var,
                parent_node=nodes_by_var[represented_by_var],
                guidance_summary=guidance_summary,
                retriever=retriever,
                model=model,
                raw_parent_dir=issue_dir / "connectivity_after_transfer_to_represented",
                refresh_raw=refresh_raw,
                rag_max_rounds=rag_max_rounds,
                nodes_by_var=nodes_by_var,
                max_iterations=2,
            )
            attach_connectivity_repair = rep_conn
            changed = changed or bool(rep_conn)
        _, parents_after_rep = build_parent_child_maps(var_order, nodes_by_var)
        if not parents_after_rep.get(candidate_var):
            removed_children = [v for v in children_of(nodes_by_var.get(candidate_var, {})) if v in nodes_by_var and v != candidate_var]
            delete_repair_meta = delete_node_with_repair(
                var=candidate_var,
                var_order=var_order,
                nodes_by_var=nodes_by_var,
                guidance_summary=guidance_summary,
                retriever=retriever,
                model=model,
                raw_base_dir=issue_dir / "delete_repair",
                refresh_raw=refresh_raw,
                rag_max_rounds=rag_max_rounds,
            )
            changed = True
            if delete_repair_meta.get("connectivity_repair"):
                changed = True
            _resolve_children_after_parent_removed(
                removed_children,
                base_dir=issue_dir / "children_after_parent_removed",
                force_attach_parent_var=represented_by_var,
            )
    else:
        # Otherwise, reattach the existing orphan under an approved parent when possible.
        approved_parents = [value for value in (decision_meta.get("approved_parents") or []) if isinstance(value, str)]
        if bool(decision_meta.get("need_to_attach")) and approved_parents and candidate_var in nodes_by_var:
            attach_parent = approved_parents[0]
            if attach_parent and attach_parent in nodes_by_var:
                linked = add_child_ref(nodes_by_var[attach_parent], candidate_var, field="nodes")
                changed = linked or changed
                if linked:
                    attach_connectivity_repair = repair_parent_connectivity_iterative(
                        parent_var=attach_parent,
                        parent_node=nodes_by_var[attach_parent],
                        guidance_summary=guidance_summary,
                        retriever=retriever,
                        model=model,
                        raw_parent_dir=issue_dir / "connectivity_after_attach",
                        refresh_raw=refresh_raw,
                        rag_max_rounds=rag_max_rounds,
                        nodes_by_var=nodes_by_var,
                        max_iterations=2,
                    )
                    changed = changed or bool(attach_connectivity_repair)
                attached_parent_var = attach_parent

    resolve_meta = {
        "changed": changed,
        "plausible_matches": plausible,
        "pair_checks": pair_checks,
        "represented_by_var": represented_by_var or None,
        "attached_parent_var": attached_parent_var or None,
        "missing_decision": missing_decision,
        "children_resolution": children_resolution,
        "transferred_children_to_represented": transferred_children_to_represented,
        "attach_connectivity_repair": attach_connectivity_repair,
    }
    connectivity_meta: Dict[str, Any] | None = attach_connectivity_repair
    if connectivity_meta is None and attached_parent_var and attached_parent_var in nodes_by_var:
        connectivity_meta = repair_parent_connectivity_iterative(
            parent_var=attached_parent_var,
            parent_node=nodes_by_var[attached_parent_var],
            guidance_summary=guidance_summary,
            retriever=retriever,
            model=model,
            raw_parent_dir=issue_dir / "connectivity_after_attach",
            refresh_raw=refresh_raw,
            rag_max_rounds=rag_max_rounds,
            nodes_by_var=nodes_by_var,
            max_iterations=2,
        )
    return {
        "resolve": resolve_meta,
        "connectivity_repair": connectivity_meta,
        "changed": bool(resolve_meta.get("changed")) or bool(connectivity_meta),
    }


def _node_mentions_var(node: Dict[str, Any], target_var: str) -> bool:
    """Return whether a node references the target var in children, edges, or external connections."""
    if target_var in children_of(node):
        return True
    internal_edges = node.get("internal_edges")
    if isinstance(internal_edges, list):
        for edge in internal_edges:
            if not isinstance(edge, dict):
                continue
            if str(edge.get("from_") or "").strip() == target_var or str(edge.get("to") or "").strip() == target_var:
                return True
    external = node.get("external_connections")
    if isinstance(external, dict):
        for key in ("in_", "out"):
            vals = external.get(key)
            if isinstance(vals, str):
                vals = [vals]
            if isinstance(vals, list) and target_var in [str(x).strip() for x in vals]:
                return True
    return False


def delete_node_with_repair(
    *,
    var: str,
    var_order: List[str],
    nodes_by_var: Dict[str, Dict[str, Any]],
    guidance_summary: str,
    retriever: Any,
    model: str,
    raw_base_dir: Path,
    refresh_raw: bool,
    rag_max_rounds: int,
    root_var: str | None = None,
    rescue_children: bool = True,
    _visited: Set[str] | None = None,
) -> Dict[str, Any]:
    """Delete a node, repair affected parent connectivity, and optionally rescue newly orphaned children."""

    # Guard against recursive delete cycles.
    if _visited is None:
        _visited = set()
    if var in _visited:
        return {
            "removed_var": var,
            "affected_vars": [],
            "connectivity_repair": {},
            "children_rescue": {"skipped_cycle": True, "children": []},
        }
    _visited.add(var)

    # Capture the node's children and every graph node that currently references it.
    node_before = nodes_by_var.get(var, {})
    removed_children = [c for c in children_of(node_before) if c in nodes_by_var and c != var]
    affected_vars = [
        pvar
        for pvar in list(var_order)
        if pvar != var and pvar in nodes_by_var and _node_mentions_var(nodes_by_var.get(pvar, {}), var)
    ]

    # Remove the node, then rebuild connectivity on affected remaining parents.
    delete_node(var, var_order, nodes_by_var)
    repairs: Dict[str, Any] = {}
    for pvar in affected_vars:
        if pvar not in nodes_by_var:
            continue
        repairs[pvar] = repair_parent_connectivity_iterative(
            parent_var=pvar,
            parent_node=nodes_by_var[pvar],
            guidance_summary=guidance_summary,
            retriever=retriever,
            model=model,
            raw_parent_dir=raw_base_dir / safe_name(pvar),
            refresh_raw=refresh_raw,
            rag_max_rounds=rag_max_rounds,
            nodes_by_var=nodes_by_var,
            max_iterations=2,
        )

    # If requested, resolve children that become orphaned because of this deletion.
    children_rescue: Dict[str, Any] = {"children": []}
    if rescue_children:
        _, parents_after = build_parent_child_maps(var_order, nodes_by_var)
        for cvar in removed_children:
            if cvar not in nodes_by_var:
                continue
            if parents_after.get(cvar):
                children_rescue["children"].append(
                    {
                        "child_var": cvar,
                        "decision": "kept_still_parented",
                        "reason": "has_remaining_parent",
                    }
                )
                continue

            child_issue_dir = raw_base_dir / "children_after_parent_removed" / safe_name(cvar)
            resolve_with_conn = resolve_orphan_node(
                candidate_var=cvar,
                root_var=root_var,
                guidance_summary=guidance_summary,
                issue_dir=child_issue_dir,
                model=model,
                refresh_raw=refresh_raw,
                retriever=retriever,
                rag_max_rounds=rag_max_rounds,
                nodes_by_var=nodes_by_var,
                var_order=var_order,
                representation_shortlist_prompt=REPRESENTATION_SHORTLIST_PROMPT,
                representation_pair_prompt=REPRESENTATION_PAIR_PROMPT,
                missing_node_decision_prompt=MISSING_NODE_DECISION_PROMPT,
                adding_node_parent_decision_prompt=ADDING_NODE_PARENT_DECISION_PROMPT,
            )
            resolve_obj = resolve_with_conn.get("resolve") if isinstance(resolve_with_conn.get("resolve"), dict) else {}
            represented_by = str(resolve_obj.get("represented_by_var") or "")
            attached_parent = str(resolve_obj.get("attached_parent_var") or "")
            child_row: Dict[str, Any] = {
                "child_var": cvar,
                "resolve": resolve_obj,
                "connectivity_repair": (
                    resolve_with_conn.get("connectivity_repair")
                    if isinstance(resolve_with_conn.get("connectivity_repair"), dict)
                    else None
                ),
            }
            if cvar not in nodes_by_var:
                child_row["decision"] = "removed_by_resolution"
            elif represented_by or attached_parent:
                child_row["decision"] = "kept_after_resolution"
            else:
                child_row["decision"] = "removed_unresolved"
                child_row["delete_repair"] = delete_node_with_repair(
                    var=cvar,
                    var_order=var_order,
                    nodes_by_var=nodes_by_var,
                    guidance_summary=guidance_summary,
                    retriever=retriever,
                    model=model,
                    raw_base_dir=child_issue_dir / "forced_remove",
                    refresh_raw=refresh_raw,
                    rag_max_rounds=rag_max_rounds,
                    root_var=root_var,
                    rescue_children=True,
                    _visited=_visited,
                )
            children_rescue["children"].append(child_row)

    return {
        "removed_var": var,
        "affected_vars": affected_vars,
        "connectivity_repair": repairs,
        "children_rescue": children_rescue,
    }


def apply_merge_decision(
    *,
    parent_var: str,
    merge_into_var: str,
    merge_from_var: str,
    root_var: str,
    parents_by_child: Dict[str, List[str]],
    nodes_by_var: Dict[str, Dict[str, Any]],
    var_order: List[str],
    guidance_summary: str,
    retriever: Any,
    model: str,
    issue_dir: Path,
    refresh_raw: bool,
    rag_max_rounds: int,
) -> Dict[str, Any]:
    """Apply one merge decision, repair the kept node connectivity, and delete the absorbed node when possible."""

    # Skip invalid merge targets and never absorb the root node.
    if (
        merge_into_var not in nodes_by_var
        or merge_from_var not in nodes_by_var
        or merge_from_var == root_var
    ):
        return {
            "changed": False,
            "removed_direct_ref": False,
            "dropped_edges": 0,
            "dropped_ext": False,
            "merge_stats": {},
            "remaining_parent_vars": [],
            "child_deleted": False,
            "delete_repair": {},
            "connectivity_repair": None,
            "rewire_stats": {"removed_refs": 0, "added_refs": 0},
        }

    # Resolve the kept and absorbed nodes and rewire other parents when needed.
    kept_var = merge_into_var
    absorbed_var = merge_from_var
    kept_node = nodes_by_var[kept_var]
    absorbed_node = nodes_by_var[absorbed_var]
    if kept_var != parent_var:
        rewire_stats = _rewire_parents_to_target(
            absorb_var=absorbed_var,
            keep_var=kept_var,
            parents_by_child=parents_by_child,
            nodes_by_var=nodes_by_var,
        )
    else:
        rewire_stats = {"removed_refs": 0, "added_refs": 0}

    # Remove stale direct references, merge content, and repair the kept node connectivity.
    removed_direct_ref = remove_child_ref(kept_node, absorbed_var)
    dropped_edges = drop_var_refs_from_internal_edges(kept_node, absorbed_var)
    dropped_ext = drop_var_refs_from_external_connections(kept_node, absorbed_var)
    merge_stats = merge_component(kept_node, absorbed_node)
    connectivity_meta = repair_parent_connectivity_iterative(
        parent_var=kept_var,
        parent_node=kept_node,
        guidance_summary=guidance_summary,
        retriever=retriever,
        model=model,
        raw_parent_dir=issue_dir / "connectivity_after_merge",
        refresh_raw=refresh_raw,
        rag_max_rounds=rag_max_rounds,
        nodes_by_var=nodes_by_var,
        max_iterations=2,
    )

    # Delete the absorbed node only if no parents still point to it.
    _, parents_after = build_parent_child_maps(var_order, nodes_by_var)
    remaining_parents = sorted(
        p for p in (parents_after.get(absorbed_var) or []) if isinstance(p, str)
    )
    child_deleted = False
    delete_meta: Dict[str, Any] = {}
    if not remaining_parents:
        delete_meta = delete_node_with_repair(
            var=absorbed_var,
            var_order=var_order,
            nodes_by_var=nodes_by_var,
            guidance_summary=guidance_summary,
            retriever=retriever,
            model=model,
            raw_base_dir=issue_dir / "delete_child_repair",
            refresh_raw=refresh_raw,
            rag_max_rounds=rag_max_rounds,
        )
        child_deleted = True

    # Report whether this merge path actually changed graph state.
    changed = bool(
        removed_direct_ref
        or dropped_edges > 0
        or dropped_ext
        or int(merge_stats.get("refs_added") or 0) > 0
        or int(merge_stats.get("nodes_added") or 0) > 0
        or int(merge_stats.get("tool_added") or 0) > 0
        or int(rewire_stats.get("removed_refs") or 0) > 0
        or int(rewire_stats.get("added_refs") or 0) > 0
        or child_deleted
        or bool(delete_meta.get("connectivity_repair"))
        or bool(connectivity_meta)
    )
    return {
        "changed": changed,
        "removed_direct_ref": removed_direct_ref,
        "dropped_edges": dropped_edges,
        "dropped_ext": dropped_ext,
        "merge_stats": merge_stats,
        "remaining_parent_vars": remaining_parents,
        "child_deleted": child_deleted,
        "delete_repair": delete_meta,
        "connectivity_repair": connectivity_meta,
        "rewire_stats": rewire_stats,
    }


def expand_system_agents_with_child_creation(
    *,
    system_var: str,
    system_node: Dict[str, Any],
    guidance_summary: str,
    retriever: Any,
    model: str,
    raw_parent_dir: Path,
    refresh_raw: bool,
    rag_max_rounds: int,
    nodes_by_var: Dict[str, Dict[str, Any]],
    var_order: List[str],
    phase_max_rounds: int = 8,
    no_change_patience: int = 2,
) -> Dict[str, Any]:
    """Discover Agent children for a System node and then materialize or reuse only Agent proposals."""

    # First discover candidate child proposals for this system.
    proposals, discover_meta = _build_child_proposals_for_parent(
        parent_var=system_var,
        parent_node=system_node,
        guidance_summary=guidance_summary,
        retriever=retriever,
        model=model,
        raw_parent_dir=raw_parent_dir,
        refresh_raw=refresh_raw,
        phase_max_rounds=phase_max_rounds,
        no_change_patience=no_change_patience,
    )

    # Then materialize or reuse only proposals that resolve to Agent nodes.
    apply_meta = materialize_attach_children_from_proposals(
        parent_var=system_var,
        parent_node=system_node,
        proposals=proposals,
        guidance_summary=guidance_summary,
        retriever=retriever,
        model=model,
        raw_parent_dir=raw_parent_dir,
        refresh_raw=refresh_raw,
        rag_max_rounds=rag_max_rounds,
        nodes_by_var=nodes_by_var,
        var_order=var_order,
        allowed_node_types={"Agent"},
        run_connectivity_pass=True,
    )

    # Return the discovery output together with the final Agent attachment results.
    return {
        "phase_a": discover_meta,
        "created_agent_vars": apply_meta.get("created_children_vars") if isinstance(apply_meta.get("created_children_vars"), list) else [],
        "reused_existing_agents": apply_meta.get("reused_existing_children") if isinstance(apply_meta.get("reused_existing_children"), list) else [],
        "excluded_children": apply_meta.get("excluded_children") if isinstance(apply_meta.get("excluded_children"), list) else [],
    }


def resolve_single_llm_child_with_child_creation_style(
    *,
    agent_var: str,
    agent_node: Dict[str, Any],
    guidance_summary: str,
    issue_dir: Path,
    model: str,
    refresh_raw: bool,
    retriever: Any,
    rag_max_rounds: int,
    nodes_by_var: Dict[str, Dict[str, Any]],
    var_order: List[str],
    llm_reuse_prompt: str,
    llm_resolution_prompt: str,
) -> Dict[str, Any]:
    """Resolve a missing LLM child for one agent by reusing an existing LLM first, ot creating one if needed."""
    issue_dir.mkdir(parents=True, exist_ok=True)

    # Build a lightweight catalog of existing LLM nodes for reuse.
    llm_catalog: List[Dict[str, Any]] = []
    for v in var_order:
        n = nodes_by_var.get(v, {})
        if node_type_name(n) != "LLM":
            continue
        llm_catalog.append(
            {
                "var": v,
                "name": str(n.get("name") or ""),
                "description": str(n.get("description") or "")[:500],
                "code_references": n.get("code_references") if isinstance(n.get("code_references"), list) else [],
            }
        )
    if llm_catalog:

        # Ask whether an existing LLM should be attached directly to this agent.
        reuse_dec_raw, _ = call_with_cache(
            payload={
                "prompt": llm_reuse_prompt,
                "agent_var": agent_var,
                "agent_node": agent_node,
                "llm_catalog": llm_catalog,
                "guidance_summary": guidance_summary,
                "retrieved_context": None,
            },
            payload_path=issue_dir / (str("reuse_existing_llm_check") + ".payload.json"),
            raw_path=issue_dir / (str("reuse_existing_llm_check") + ".txt"),
            model=model,
            refresh_raw=refresh_raw,
            retriever=retriever,
            rag_max_rounds=rag_max_rounds,
        )
        reuse_dec = reuse_dec_raw if isinstance(reuse_dec_raw, dict) else {}
        reuse_action = str(reuse_dec.get("action") or "").strip()
        selected_llm_var = str(reuse_dec.get("selected_llm_var") or "").strip()
        if (
            reuse_action == "reuse_existing"
            and selected_llm_var
            and selected_llm_var in nodes_by_var
            and node_type_name(nodes_by_var.get(selected_llm_var, {})) == "LLM"
        ):
            attached = add_child_ref(agent_node, selected_llm_var, field="nodes")
            connectivity_meta: Dict[str, Any] | None = None
            if attached:
                connectivity_meta = repair_parent_connectivity_iterative(
                    parent_var=agent_var,
                    parent_node=agent_node,
                    guidance_summary=guidance_summary,
                    retriever=retriever,
                    model=model,
                    raw_parent_dir=issue_dir / "connectivity_after_reuse_existing_llm",
                    refresh_raw=refresh_raw,
                    rag_max_rounds=rag_max_rounds,
                    nodes_by_var=nodes_by_var,
                    max_iterations=2,
                )
            return {
                "changed": bool(attached) or bool(connectivity_meta),
                "llm_var": selected_llm_var,
                "status": "reused_existing_llm_by_llm_check",
                "reuse_output": reuse_dec if isinstance(reuse_dec, dict) else {},
                "connectivity_repair": connectivity_meta,
            }

    # If reuse is not selected, ask for an LLM child proposal.
    dec_raw, _ = call_with_cache(
        payload={
            "prompt": llm_resolution_prompt,
            "agent_var": agent_var,
            "agent_node": agent_node,
            "guidance_summary": guidance_summary,
            "retrieved_context": None,
        },
        payload_path=issue_dir / (str("resolve_llm_child") + ".payload.json"),
        raw_path=issue_dir / (str("resolve_llm_child") + ".txt"),
        model=model,
        refresh_raw=refresh_raw,
        retriever=retriever,
        rag_max_rounds=rag_max_rounds,
    )
    dec = dec_raw if isinstance(dec_raw, dict) else {}
    proposal = dec.get("child_proposal") if isinstance(dec.get("child_proposal"), dict) else {}
    if not proposal:
        return {
            "changed": False,
            "llm_var": None,
            "status": "no_llm_proposal",
            "discovery_output": dec if isinstance(dec, dict) else {},
        }
    proposal_type = str(proposal.get("node_type") or "").strip()
    if proposal_type not in {"LLM", "syn_llm"}:
        return {
            "changed": False,
            "llm_var": None,
            "status": "no_llm_proposal",
            "discovery_output": dec if isinstance(dec, dict) else {},
        }
    if proposal_type == "syn_llm":

        # Handle synthetic/default LLM proposals by always creating a minimal synthetic LLM node directly.
        synthetic_name = str(proposal.get("name") or "").strip() or f"{agent_var}_llm"
        synthetic_desc = str(proposal.get("description") or "").strip() or "Framework-default/inferred LLM runtime used by the agent."
        llm_var = _cc_next_var(synthetic_name, set(nodes_by_var.keys()))
        nodes_by_var[llm_var] = {
            "name": synthetic_name,
            "node_type": {"__call__": "NodeType", "type": "LLM", "other_description": None},
            "description": synthetic_desc,
            "code_references": [],
            "inputs": [],
            "outputs": [],
            "external_connections": {"in_": [], "out": []},
            "tool_list": [],
            "nodes": [],
            "internal_edges": [],
            "metadata": {
                "synthetic_inferred_llm": True,
                "created_by": "graph_correctness",
                "creation_reason": "syn_llm_resolution_fallback",
            },
        }
        var_order.append(llm_var)

        attached = add_child_ref(agent_node, llm_var, field="nodes")
        connectivity_meta: Dict[str, Any] | None = None
        if attached:
            connectivity_meta = repair_parent_connectivity_iterative(
                parent_var=agent_var,
                parent_node=agent_node,
                guidance_summary=guidance_summary,
                retriever=retriever,
                model=model,
                raw_parent_dir=issue_dir / f"llm_child_{safe_name(synthetic_name)}" / "connectivity_after_direct_attach",
                refresh_raw=refresh_raw,
                rag_max_rounds=rag_max_rounds,
                nodes_by_var=nodes_by_var,
                max_iterations=2,
            )
        return {
            "changed": True,
            "llm_var": llm_var or None,
            "status": "created_new_llm_synthetic_direct",
            "connectivity_repair": connectivity_meta,
            "discovery_output": dec if isinstance(dec, dict) else {},
        }

    # For concrete LLM proposals, materialize or reuse them through the shared child-creation path.
    apply_meta = materialize_attach_children_from_proposals(
        parent_var=agent_var,
        parent_node=agent_node,
        proposals=[proposal],
        guidance_summary=guidance_summary,
        retriever=retriever,
        model=model,
        raw_parent_dir=issue_dir / f"llm_child_{safe_name(str(proposal.get('name') or 'llm'))}",
        refresh_raw=refresh_raw,
        rag_max_rounds=rag_max_rounds,
        nodes_by_var=nodes_by_var,
        var_order=var_order,
        allowed_node_types={"LLM"},
        run_connectivity_pass=True,
        skip_runtime_judge=True,
    )
    created = apply_meta.get("created_children_vars") if isinstance(apply_meta.get("created_children_vars"), list) else []
    reused = apply_meta.get("reused_existing_children") if isinstance(apply_meta.get("reused_existing_children"), list) else []
    excluded = apply_meta.get("excluded_children") if isinstance(apply_meta.get("excluded_children"), list) else []

    # Return the first successful outcome from reuse, creation, or exclusion.
    if reused:
        matched_var = str((reused[0] or {}).get("matched_var") or "").strip()
        return {
            "changed": bool(apply_meta.get("changed")),
            "llm_var": matched_var or None,
            "status": "reused_existing_llm",
            "apply_meta": apply_meta,
            "discovery_output": dec if isinstance(dec, dict) else {},
        }
    if created:
        child_var = str(created[0] or "").strip()
        return {
            "changed": bool(apply_meta.get("changed")),
            "llm_var": child_var or None,
            "status": "created_new_llm",
            "apply_meta": apply_meta,
            "discovery_output": dec if isinstance(dec, dict) else {},
        }
    if excluded:
        return {
            "changed": False,
            "llm_var": None,
            "status": "runtime_judge_excluded",
            "apply_meta": apply_meta,
            "discovery_output": dec if isinstance(dec, dict) else {},
        }
    return {
        "changed": bool(apply_meta.get("changed")),
        "llm_var": None,
        "status": "materialization_failed_or_non_llm",
        "apply_meta": apply_meta,
        "discovery_output": dec if isinstance(dec, dict) else {},
    }


def descendant_tool_names_from_server_children(
        parent_var: str,
        server_children: List[str],
        children_by_parent: Dict[str, Set[str]],
    ) -> Set[str]:
        """Collect nested Tool names reachable through MCP server child subtrees."""
        tool_names: Set[str] = set()
        for s in server_children:
            stack = [s]
            seen_local: Set[str] = set()
            while stack:
                cur = stack.pop()
                if cur in seen_local:
                    continue
                seen_local.add(cur)
                cnode = nodes_by_var.get(cur, {})
                if node_type_name(cnode) == "Tool":
                    tname = str(cnode.get("name") or "").strip()
                    if tname:
                        tool_names.add(tname)
                stack.extend([x for x in children_by_parent.get(cur, set()) if x != parent_var])
        return tool_names


def run_single_child_rule(
    *,
    var_order: List[str],
    nodes_by_var: Dict[str, Dict[str, Any]],
    root_var: str | None,
    guidance_summary: str,
    raw_dir: Path,
    model: str,
    refresh_raw: bool,
    retriever: Any,
    rag_max_rounds: int,
    single_child_prompt: str,
    actions: List[Dict[str, Any]],
    unresolved: List[Dict[str, Any]],
    run_suffix: str = "single_child",
    resolved_keep_distinct_pairs: Set[Tuple[str, str]] | None = None,
) -> None:

    """Process parent nodes with exactly one direct node child and either merge or keep them distinct."""
    if resolved_keep_distinct_pairs is None:
        resolved_keep_distinct_pairs = set()

    # Track single-child pairs already examined in this pass.
    seen_single_child: set[tuple[str, str]] = set()
    while True:
        _, parents_by_child = build_parent_child_maps(var_order, nodes_by_var)
        issue = None

        # Find the next eligible parent-child pair with exactly one direct node child.
        for pvar in var_order:
            pnode = nodes_by_var.get(pvar, {})
            childs = children_in_nodes(pnode)
            if len(childs) != 1:
                continue
            key = (pvar, childs[0])
            if key in resolved_keep_distinct_pairs:
                continue
            if key in seen_single_child:
                continue
            issue = key
            break

        if issue is None:
            break

        # Unpack the current single-child pair selected for resolution.
        pvar, cvar = issue
        seen_single_child.add(issue)
        pnode = nodes_by_var.get(pvar, {})
        cnode = nodes_by_var.get(cvar, {})

        issue_dir = raw_dir / run_suffix / f"{safe_name(pvar)}__{safe_name(cvar)}"
        issue_dir.mkdir(parents=True, exist_ok=True)
        p_count = len([
            p for p in (parents_by_child.get(pvar) or [])
            if isinstance(p, str) and node_type_name(nodes_by_var.get(p, {})) != "LLM"
        ])
        c_count = len([
            p for p in (parents_by_child.get(cvar) or [])
            if isinstance(p, str) and node_type_name(nodes_by_var.get(p, {})) != "LLM"
        ])

        # Apply deterministic guards first, otherwise ask the model whether to merge.
        if node_type_name(pnode) == "Agent" and node_type_name(cnode) == "LLM":
            dec = {
                "action": "keep_distinct",
                "reason": "deterministic_guard_agent_single_llm_keep_distinct",
            }
        elif p_count > 1 and c_count > 1:
            dec = {
                "action": "keep_distinct",
                "reason": "deterministic_guard_both_multi_parent_keep_distinct",
            }
        else:
            dec_raw, _ = call_with_cache(
                payload={
                    "prompt": single_child_prompt,
                    "parent_var": pvar,
                    "parent_node": pnode,
                    "child_var": cvar,
                    "child_node": cnode,
                    "guidance_summary": guidance_summary,
                    "retrieved_context": None,
                },
                payload_path=issue_dir / (str("decision") + ".payload.json"),
                raw_path=issue_dir / (str("decision") + ".txt"),
                model=model,
                refresh_raw=refresh_raw,
                retriever=retriever,
                rag_max_rounds=rag_max_rounds,
            )

            dec = dec_raw if isinstance(dec_raw, dict) else {}

        action = str(dec.get("action") or "keep_distinct").strip()
        reason = str(dec.get("reason") or "")
        plan = _deterministic_merge_plan_for_pair(
            parent_var=pvar,
            child_var=cvar,
            parents_by_child=parents_by_child,
            nodes_by_var=nodes_by_var,
            requested_action=action,
        )
        action = str(plan.get("effective_action") or action).strip()
        merge_into_var = str(plan.get("merge_into_var") or pvar).strip()
        merge_from_var = str(plan.get("merge_from_var") or cvar).strip()

        # Apply the merge path and repair connectivity when the guarded plan allows it.
        changed = False
        removed_direct_ref = False
        dropped_edges = 0
        dropped_ext = False
        merge_stats: Dict[str, Any] = {}
        remaining_parents: List[str] = []
        deleted_child = False
        connectivity_meta: Dict[str, Any] | None = None
        delete_meta: Dict[str, Any] = {}
        if (
            action == "merge_components"
            and merge_into_var in nodes_by_var
            and merge_from_var in nodes_by_var
            and merge_from_var != root_var
        ):
            merge_meta = apply_merge_decision(
                parent_var=pvar,
                merge_into_var=merge_into_var,
                merge_from_var=merge_from_var,
                root_var=root_var,
                parents_by_child=parents_by_child,
                nodes_by_var=nodes_by_var,
                var_order=var_order,
                guidance_summary=guidance_summary,
                retriever=retriever,
                model=model,
                issue_dir=issue_dir,
                refresh_raw=refresh_raw,
                rag_max_rounds=rag_max_rounds,
            )
            changed = bool(merge_meta.get("changed"))
            removed_direct_ref = bool(merge_meta.get("removed_direct_ref"))
            dropped_edges = int(merge_meta.get("dropped_edges") or 0)
            dropped_ext = bool(merge_meta.get("dropped_ext"))
            merge_stats = merge_meta.get("merge_stats") if isinstance(merge_meta.get("merge_stats"), dict) else {}
            remaining_parents = sorted(
                p for p in (merge_meta.get("remaining_parent_vars") or []) if isinstance(p, str)
            )
            deleted_child = bool(merge_meta.get("child_deleted"))
            connectivity_meta = merge_meta.get("connectivity_repair")
            delete_meta = merge_meta.get("delete_repair") if isinstance(merge_meta.get("delete_repair"), dict) else {}

        # Record the decision and preserve keep-distinct pairs for later rounds.
        entry = {"rule": "single_child_node", "parent_var": pvar, "child_var": cvar, "decision": action, "reason": reason, "changed": changed}
        entry["deterministic_merge_guard"] = plan
        if action == "merge_components":
            entry["merge"] = {
                "merge_into_var": merge_into_var,
                "merge_from_var": merge_from_var,
                "removed_direct_ref": bool(removed_direct_ref),
                "dropped_internal_edges": int(dropped_edges),
                "dropped_external_refs": bool(dropped_ext),
                "stats": merge_stats,
                "remaining_parent_vars": remaining_parents,
                "child_deleted": bool(deleted_child),
                "connectivity_repair": connectivity_meta,
            }
        if action == "keep_distinct":
            resolved_keep_distinct_pairs.add((pvar, cvar))
            entry["treated_as_resolved"] = True

        actions.append(entry)
        if not changed and action != "keep_distinct":
            unresolved.append(entry)


def run_rule_llm_no_children(
    *,
    var_order: List[str],
    nodes_by_var: Dict[str, Dict[str, Any]],
    guidance_summary: str,
    raw_dir: Path,
    model: str,
    refresh_raw: bool,
    retriever: Any,
    rag_max_rounds: int,
    llm_no_children_prompt: str,
    actions: List[Dict[str, Any]],
    unresolved: List[Dict[str, Any]],
) -> None:
    """Resolve LLM nodes that incorrectly retain direct children in the graph."""

    # Track LLM nodes already processed in this pass.
    seen_llm_no_children: Set[str] = set()
    while True:
        issue_var = None
        issue_children: List[str] = []

        # Find the next LLM node that still has direct children.
        for v in var_order:
            if v in seen_llm_no_children:
                continue
            n = nodes_by_var.get(v, {})
            if node_type_name(n) != "LLM":
                continue
            childs = children_of(n)
            if not childs:
                continue
            issue_var = v
            issue_children = childs
            break

        if not issue_var:
            break

        seen_llm_no_children.add(issue_var)
        node = nodes_by_var.get(issue_var, {})
        issue_dir = raw_dir / "llm_no_children" / safe_name(issue_var)
        issue_dir.mkdir(parents=True, exist_ok=True)

        # Ask the model whether this node should be retyped or just have its children removed.
        dec_raw, _ = call_with_cache(
            payload={
                "prompt": llm_no_children_prompt,
                "llm_var": issue_var,
                "llm_node": node,
                "direct_children": issue_children,
                "guidance_summary": guidance_summary,
                "retrieved_context": None,
            },
            payload_path=issue_dir / (str("decision") + ".payload.json"),
            raw_path=issue_dir / (str("decision") + ".txt"),
            model=model,
            refresh_raw=refresh_raw,
            retriever=retriever,
            rag_max_rounds=rag_max_rounds,
        )
        dec = dec_raw if isinstance(dec_raw, dict) else {}
        action = str(dec.get("action") or "").strip()
        reason = str(dec.get("reason") or "")
        changed = False
        removed_children: List[str] = []
        type_before = node_type_name(node)
        type_after = type_before

        # Apply the selected repair path and run connectivity repair when children are removed.
        if action == "change_type_to_agent":
            nt = node.get("node_type")
            if isinstance(nt, dict):
                nt["type"] = "Agent"
                if "other_description" not in nt:
                    nt["other_description"] = None
            else:
                node["node_type"] = {"type": "Agent", "other_description": None}
            type_after = "Agent"
            changed = True

        elif action == "remove_children":
            for cvar in issue_children:
                removed_any = remove_child_ref(node, cvar)
                dropped_edges = drop_var_refs_from_internal_edges(node, cvar)
                dropped_ext = drop_var_refs_from_external_connections(node, cvar)
                if removed_any or dropped_edges > 0 or dropped_ext:
                    changed = True
                    removed_children.append(cvar)

            if changed and issue_var in nodes_by_var:
                conn_meta = repair_parent_connectivity_iterative(
                    parent_var=issue_var,
                    parent_node=nodes_by_var[issue_var],
                    guidance_summary=guidance_summary,
                    retriever=retriever,
                    model=model,
                    raw_parent_dir=issue_dir / "connectivity_after_remove_children",
                    refresh_raw=refresh_raw,
                    rag_max_rounds=rag_max_rounds,
                    nodes_by_var=nodes_by_var,
                    max_iterations=2,
                )
                changed = changed or bool(conn_meta)

        # Record the outcome and keep unchanged cases for unresolved reporting.
        # Record the final resolution outcome for this invalid nesting pair.
        # Record the final outcome for this agent node.
        entry = {
            "rule": "llm_no_children",
            "llm_var": issue_var,
            "decision": action or "invalid_or_empty_action",
            "reason": reason,
            "children_before": issue_children,
            "children_after": children_of(node),
            "type_before": type_before,
            "type_after": type_after,
            "removed_children": sorted(set(removed_children)),
            "changed": changed,
        }
        actions.append(entry)
        if not changed:
            unresolved.append(entry)


def run_rule_orphan_no_parent(
    *,
    var_order: List[str],
    nodes_by_var: Dict[str, Dict[str, Any]],
    root_var: str | None,
    guidance_summary: str,
    raw_dir: Path,
    model: str,
    refresh_raw: bool,
    retriever: Any,
    rag_max_rounds: int,
    representation_shortlist_prompt: str,
    representation_pair_prompt: str,
    missing_node_decision_prompt: str,
    adding_node_parent_decision_prompt: str,
    actions: List[Dict[str, Any]],
    unresolved: List[Dict[str, Any]],
) -> None:
    """Resolve non-root graph nodes that currently have no parent."""

    # Track orphan candidates already attempted in this pass.
    seen_orphan_no_parent: Set[str] = set()
    while True:
        _, parents_by_child = build_parent_child_maps(var_order, nodes_by_var)
        issue_var = None

        # Find the next non-root node that has no parent reference.
        for cvar in var_order:
            if cvar == root_var:
                continue
            if cvar in seen_orphan_no_parent:
                continue
            if cvar not in nodes_by_var:
                continue
            if not parents_by_child.get(cvar):
                issue_var = cvar
                break

        if issue_var is None:
            break

        cvar = issue_var
        seen_orphan_no_parent.add(cvar)
        issue_dir = raw_dir / "orphan_no_parent" / safe_name(cvar)
        issue_dir.mkdir(parents=True, exist_ok=True)

        # Run the isolated-node attach or representation resolution flow.
        resolve_with_conn = resolve_orphan_node(
            candidate_var=cvar,
            root_var=root_var,
            guidance_summary=guidance_summary,
            issue_dir=issue_dir / "attach_pipeline",
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
        resolve_meta = resolve_with_conn.get("resolve") if isinstance(resolve_with_conn.get("resolve"), dict) else {}
        attach_parent = str(resolve_meta.get("attached_parent_var") or "")
        represented_by = str(resolve_meta.get("represented_by_var") or "")
        removed_by_resolution = cvar not in nodes_by_var
        changed = bool(resolve_with_conn.get("changed"))

        # Delete the orphan only when the attach pipeline could not resolve it.
        delete_meta: Dict[str, Any] = {}
        action = "attached_or_represented"
        reason = "Node was attached, represented, or removed by attach pipeline."
        if not attach_parent and not represented_by and not removed_by_resolution and cvar in nodes_by_var:
            delete_meta = delete_node_with_repair(
                var=cvar,
                var_order=var_order,
                nodes_by_var=nodes_by_var,
                guidance_summary=guidance_summary,
                retriever=retriever,
                model=model,
                raw_base_dir=issue_dir / "delete_orphan_repair",
                refresh_raw=refresh_raw,
                rag_max_rounds=rag_max_rounds,
            )
            changed = True
            action = "removed_unattached_orphan"
            reason = "Attach pipeline could not attach or represent node; node deleted and references repaired."

        # Record the outcome and keep unresolved cases for the final report.
        entry = {
            "rule": "orphan_no_parent",
            "node_var": cvar,
            "decision": action,
            "reason": reason,
            "changed": changed,
            "resolve": resolve_meta,
            "connectivity_repair": (
                resolve_with_conn.get("connectivity_repair")
                if isinstance(resolve_with_conn.get("connectivity_repair"), dict)
                else None
            ),
            "delete_repair": delete_meta,
        }
        actions.append(entry)
        if not changed:
            unresolved.append(entry)


def run_rule_no_agent_in_agent(
    *,
    var_order: List[str],
    nodes_by_var: Dict[str, Dict[str, Any]],
    root_var: str | None,
    guidance_summary: str,
    raw_dir: Path,
    model: str,
    refresh_raw: bool,
    retriever: Any,
    rag_max_rounds: int,
    single_child_prompt: str,
    actions: List[Dict[str, Any]],
    unresolved: List[Dict[str, Any]],
) -> None:
    """Resolve invalid cases where an Agent directly contains another Agent or System node."""

    # Track invalid containment pairs already processed in this pass.
    seen_no_agent_in_agent: Set[Tuple[str, str]] = set()
    while True:
        issue = None

        # Find the next Agent -> (Agent|System) direct containment violation.
        for pvar in var_order:
            pnode = nodes_by_var.get(pvar, {})
            if node_type_name(pnode) != "Agent":
                continue
            for cvar in children_of(pnode):
                cnode = nodes_by_var.get(cvar, {})
                if node_type_name(cnode) in {"Agent", "System"}:
                    key = (pvar, cvar)
                    if key not in seen_no_agent_in_agent:
                        issue = key
                        break
            if issue:
                break

        if issue is None:
            break
        pvar, cvar = issue
        seen_no_agent_in_agent.add(issue)

        # Gather candidate system siblings and prepare the raw output directory.
        _, parents_by_child = build_parent_child_maps(var_order, nodes_by_var)
        candidate_system_vars = [
            pv
            for pv in parents_by_child.get(pvar, [])
            if node_type_name(nodes_by_var.get(pv, {})) == "System"
        ]
        if root_var and node_type_name(nodes_by_var.get(root_var, {})) == "System" and root_var not in candidate_system_vars:
            candidate_system_vars.append(root_var)

        issue_dir = raw_dir / "no_agent_in_agent" / f"{safe_name(pvar)}__{safe_name(cvar)}"
        issue_dir.mkdir(parents=True, exist_ok=True)

        # First decide whether the two nodes should merge instead of being restructured.
        inner_type = node_type_name(nodes_by_var.get(cvar, {}))
        p_count = len([
            p for p in (parents_by_child.get(pvar) or [])
            if isinstance(p, str) and node_type_name(nodes_by_var.get(p, {})) != "LLM"
        ])
        c_count = len([
            p for p in (parents_by_child.get(cvar) or [])
            if isinstance(p, str) and node_type_name(nodes_by_var.get(p, {})) != "LLM"
        ])
        if inner_type == "System":
            merge_dec = {
                "action": "keep_distinct",
                "reason": "deterministic_guard_system_inside_agent_forbidden",
            }
        elif p_count > 1 and c_count > 1:
            merge_dec = {
                "action": "keep_distinct",
                "reason": "deterministic_guard_both_multi_parent_keep_distinct",
            }
        else:
            merge_dec_raw, _ = call_with_cache(
                payload={
                    "prompt": single_child_prompt,
                    "parent_var": pvar,
                    "parent_node": nodes_by_var.get(pvar, {}),
                    "child_var": cvar,
                    "child_node": nodes_by_var.get(cvar, {}),
                    "guidance_summary": guidance_summary,
                    "retrieved_context": None,
                },
                payload_path=issue_dir / (str("merge_decision") + ".payload.json"),
                raw_path=issue_dir / (str("merge_decision") + ".txt"),
                model=model,
                refresh_raw=refresh_raw,
                retriever=retriever,
                rag_max_rounds=rag_max_rounds,
            )
            merge_dec = merge_dec_raw if isinstance(merge_dec_raw, dict) else {}

        # Normalize the merge decision and apply deterministic merge direction.
        merge_action = str(merge_dec.get("action") or "keep_distinct").strip()
        merge_reason = str(merge_dec.get("reason") or "")
        plan = _deterministic_merge_plan_for_pair(
            parent_var=pvar,
            child_var=cvar,
            parents_by_child=parents_by_child,
            nodes_by_var=nodes_by_var,
            requested_action=merge_action,
        )
        merge_action = str(plan.get("effective_action") or merge_action).strip()
        merge_into_var = str(plan.get("merge_into_var") or pvar).strip()
        merge_from_var = str(plan.get("merge_from_var") or cvar).strip()

        # If merge is allowed, apply it and finish this issue immediately.
        changed = False
        if (
            merge_action == "merge_components"
            and merge_into_var in nodes_by_var
            and merge_from_var in nodes_by_var
            and merge_from_var != root_var
        ):
            merge_meta = apply_merge_decision(
                parent_var=pvar,
                merge_into_var=merge_into_var,
                merge_from_var=merge_from_var,
                root_var=root_var,
                parents_by_child=parents_by_child,
                nodes_by_var=nodes_by_var,
                var_order=var_order,
                guidance_summary=guidance_summary,
                retriever=retriever,
                model=model,
                issue_dir=issue_dir,
                refresh_raw=refresh_raw,
                rag_max_rounds=rag_max_rounds,
            )
            removed_direct_ref = bool(merge_meta.get("removed_direct_ref"))
            dropped_edges = int(merge_meta.get("dropped_edges") or 0)
            dropped_ext = bool(merge_meta.get("dropped_ext"))
            merge_stats = merge_meta.get("merge_stats") if isinstance(merge_meta.get("merge_stats"), dict) else {}
            remaining_parents = sorted(
                p for p in (merge_meta.get("remaining_parent_vars") or []) if isinstance(p, str)
            )
            deleted_child = bool(merge_meta.get("child_deleted"))
            delete_meta = merge_meta.get("delete_repair") if isinstance(merge_meta.get("delete_repair"), dict) else {}
            connectivity_meta = merge_meta.get("connectivity_repair")
            changed = bool(merge_meta.get("changed"))
            entry = {
                "rule": "agent_in_agent_forbidden",
                "outer_agent_var": pvar,
                "inner_agent_var": cvar,
                "merge_decision": merge_action,
                "merge_reason": merge_reason,
                "decision": "merged",
                "reason": "Merged by pair decision.",
                "changed": changed,
                "deterministic_merge_guard": plan,
                "merge": {
                    "merge_into_var": merge_into_var,
                    "merge_from_var": merge_from_var,
                    "removed_direct_ref": bool(removed_direct_ref),
                    "dropped_internal_edges": int(dropped_edges),
                    "dropped_external_refs": bool(dropped_ext),
                    "stats": merge_stats,
                    "remaining_parent_vars": remaining_parents,
                    "child_deleted": bool(deleted_child),
                    "delete_repair": delete_meta,
                    "connectivity_repair": connectivity_meta,
                },
            }
            actions.append(entry)
            if not changed:
                unresolved.append(entry)
            continue

        # Otherwise separate the nodes as siblings. Start with detaching the inner node from the outer node before applying the chosen restructuring.
        action = "make_siblings"
        reason = "agent_or_system_inside_agent_forbidden_after_non_merge"
        detached = False
        dropped_edges = 0
        dropped_ext = False
        outer_parent_connectivity_meta: Dict[str, Any] | None = None
        if pvar in nodes_by_var and cvar in nodes_by_var:
            detached = remove_child_ref(nodes_by_var[pvar], cvar)
            dropped_edges = drop_var_refs_from_internal_edges(nodes_by_var[pvar], cvar)
            dropped_ext = drop_var_refs_from_external_connections(nodes_by_var[pvar], cvar)
            changed = changed or detached or dropped_edges > 0 or dropped_ext
            if detached or dropped_edges > 0 or dropped_ext:
                outer_parent_connectivity_meta = repair_parent_connectivity_iterative(
                    parent_var=pvar,
                    parent_node=nodes_by_var[pvar],
                    guidance_summary=guidance_summary,
                    retriever=retriever,
                    model=model,
                    raw_parent_dir=issue_dir / "connectivity_after_outer_detach",
                    refresh_raw=refresh_raw,
                    rag_max_rounds=rag_max_rounds,
                    nodes_by_var=nodes_by_var,
                    max_iterations=2,
                )
                changed = changed or bool(outer_parent_connectivity_meta)

        # Recompute parents after detach so the wrapper can be inserted in the right place.
        _, parents_after_detach = build_parent_child_maps(var_order, nodes_by_var)
        target_system_var = candidate_system_vars[0] if candidate_system_vars else ""
        created_new_system = False
        connectivity_meta: Dict[str, Any] | None = None
        wrap_parent_connectivity_runs: Dict[str, Dict[str, Any]] = {}

        # Prefer reusing an existing system wrapper when one is already available.
        if target_system_var and target_system_var in nodes_by_var and cvar in nodes_by_var:
            changed = add_child_ref(nodes_by_var[target_system_var], cvar, field="nodes") or changed
            connectivity_meta = repair_parent_connectivity_iterative(
                parent_var=target_system_var,
                parent_node=nodes_by_var[target_system_var],
                guidance_summary=guidance_summary,
                retriever=retriever,
                model=model,
                raw_parent_dir=issue_dir / "connectivity_sibling_existing_system",
                refresh_raw=refresh_raw,
                rag_max_rounds=rag_max_rounds,
                nodes_by_var=nodes_by_var,
                max_iterations=2,
            )
            changed = changed or bool(connectivity_meta)
        else:

            # Otherwise create a new wrapper system to host the two nodes as siblings.
            created_new_system = True
            new_system_var = next_var(f"{pvar}_system", set(var_order))
            new_system_node: Dict[str, Any] = {
                "name": f"{pvar}_system",
                "node_type": {"__call__": "NodeType", "type": "System", "other_description": None},
                "description": f"Wrapper system created to host sibling agents {pvar} and {cvar}.",
                "code_references": [],
                "inputs": [],
                "outputs": [],
                "nodes": [],
                "tool_list": [],
                "internal_edges": [],
                "external_connections": {"in_": [], "out": []},
                "system_type": {"__call__": "SystemType", "type": ["graph_orchestrator"], "other_description": None},
                "metadata": {"created_by": "graph_correctness", "creation_reason": "agent_in_agent_make_siblings"},
            }
            nodes_by_var[new_system_var] = new_system_node
            var_order.append(new_system_var)
            changed = True
            changed = add_child_ref(new_system_node, pvar, field="nodes") or changed
            changed = add_child_ref(new_system_node, cvar, field="nodes") or changed

            # Repoint the former parents of the outer agent to the new wrapper system.
            wrap_parent_vars = [pv for pv in parents_after_detach.get(pvar, []) if pv in nodes_by_var and pv != new_system_var]
            for wvar in wrap_parent_vars:
                wnode = nodes_by_var.get(wvar, {})
                changed = remove_child_ref(wnode, pvar) or changed
                changed = add_child_ref(wnode, new_system_var, field="nodes") or changed
                if wvar in nodes_by_var:
                    wconn = repair_parent_connectivity_iterative(
                        parent_var=wvar,
                        parent_node=nodes_by_var[wvar],
                        guidance_summary=guidance_summary,
                        retriever=retriever,
                        model=model,
                        raw_parent_dir=issue_dir / f"connectivity_wrap_parent_{safe_name(wvar)}",
                        refresh_raw=refresh_raw,
                        rag_max_rounds=rag_max_rounds,
                        nodes_by_var=nodes_by_var,
                        max_iterations=2,
                    )
                    wrap_parent_connectivity_runs[wvar] = wconn
                    changed = changed or bool(wconn)

            # If the outer agent had no remaining parent, attach the wrapper under the root system.
            if not wrap_parent_vars and root_var and root_var in nodes_by_var and root_var not in {pvar, cvar, new_system_var}:
                changed = add_child_ref(nodes_by_var[root_var], new_system_var, field="nodes") or changed
                root_conn = repair_parent_connectivity_iterative(
                    parent_var=root_var,
                    parent_node=nodes_by_var[root_var],
                    guidance_summary=guidance_summary,
                    retriever=retriever,
                    model=model,
                    raw_parent_dir=issue_dir / f"connectivity_wrap_parent_{safe_name(root_var)}",
                    refresh_raw=refresh_raw,
                    rag_max_rounds=rag_max_rounds,
                    nodes_by_var=nodes_by_var,
                    max_iterations=2,
                )
                wrap_parent_connectivity_runs[root_var] = root_conn
                changed = changed or bool(root_conn)

            # Finally repair connectivity on the new wrapper itself.
            target_system_var = new_system_var
            connectivity_meta = repair_parent_connectivity_iterative(
                parent_var=target_system_var,
                parent_node=nodes_by_var[target_system_var],
                guidance_summary=guidance_summary,
                retriever=retriever,
                model=model,
                raw_parent_dir=issue_dir / "connectivity_sibling_new_system",
                refresh_raw=refresh_raw,
                rag_max_rounds=rag_max_rounds,
                nodes_by_var=nodes_by_var,
                max_iterations=2,
            )
            changed = changed or bool(connectivity_meta)

        # Record how the sibling wrapper was chosen and repaired.
        rehome_meta = {
            "mode": "make_siblings",
            "target_system_var": target_system_var or None,
            "created_new_system": created_new_system,
            "connectivity_repair": connectivity_meta,
            "outer_parent_connectivity_repair": outer_parent_connectivity_meta,
            "wrapper_parent_connectivity_repairs": wrap_parent_connectivity_runs,
        }

        entry = {
            "rule": "system_in_agent_forbidden" if inner_type == "System" else "agent_in_agent_forbidden",
            "outer_agent_var": pvar,
            "inner_agent_var": cvar,
            "merge_decision": merge_action,
            "merge_reason": merge_reason,
            "decision": action,
            "reason": reason,
            "changed": changed,
            "rehome": rehome_meta,
        }
        actions.append(entry)
        if not changed:
            unresolved.append(entry)


def run_rule_agent_requires_llm(
    *,
    var_order: List[str],
    nodes_by_var: Dict[str, Dict[str, Any]],
    root_var: str | None,
    guidance_summary: str,
    raw_dir: Path,
    model: str,
    refresh_raw: bool,
    retriever: Any,
    rag_max_rounds: int,
    agent_llm_decision_prompt: str,
    agent_llm_reuse_prompt: str,
    agent_llm_resolution_prompt: str,
    single_child_prompt: str,
    representation_shortlist_prompt: str,
    representation_pair_prompt: str,
    missing_node_decision_prompt: str,
    adding_node_parent_decision_prompt: str,
    actions: List[Dict[str, Any]],
    unresolved: List[Dict[str, Any]],
) -> None:
    """Ensure every Agent either keeps or gains LLM support, or is downgraded/removed when not truly agentic."""

    # Track agent nodes already processed in this pass.
    seen_agent_requires_llm: Set[str] = set()
    while True:

        # Find the next MCP server node that still needs structure validation.
        children_by_parent, _ = build_parent_child_maps(var_order, nodes_by_var)
        issue_agent = None
        for avar in var_order:
            if avar in seen_agent_requires_llm:
                continue
            anode = nodes_by_var.get(avar, {})
            if node_type_name(anode) != "Agent":
                continue
            child_vars = children_by_parent.get(avar, [])
            has_llm = any(node_type_name(nodes_by_var.get(c, {})) == "LLM" for c in child_vars)
            if not has_llm:
                issue_agent = avar
                break

        if issue_agent is None:
            break
        avar = issue_agent
        seen_agent_requires_llm.add(avar)

        issue_dir = raw_dir / "agent_missing_llm" / safe_name(avar)
        issue_dir.mkdir(parents=True, exist_ok=True)

        # Ask the model whether this node is truly an LLM-based agent.
        dec_raw, _ = call_with_cache(
            payload={
                "prompt": agent_llm_decision_prompt,
                "agent_var": avar,
                "agent_node": nodes_by_var.get(avar, {}),
                "guidance_summary": guidance_summary,
                "retrieved_context": None,
            },
            payload_path=issue_dir / (str("classify_agent") + ".payload.json"),
            raw_path=issue_dir / (str("classify_agent") + ".txt"),
            model=model,
            refresh_raw=refresh_raw,
            retriever=retriever,
            rag_max_rounds=rag_max_rounds,
        )
        dec = dec_raw if isinstance(dec_raw, dict) else {}
        action = str(dec.get("action") or "not_real_agent").strip()
        reason = str(dec.get("reason") or "")
        allowed_actions = {"real_llm_agent", "not_real_agent"}

        # Skip unresolved classifier outputs until a final action is available.
        if action not in allowed_actions:
            entry = {
                "rule": "agent_requires_llm",
                "agent_var": avar,
                "decision": action or "context_request",
                "reason": reason or "Non-final classification output; skipping demotion/merge until evidence-backed final action.",
                "changed": False,
                "agent_requires_llm_meta": {
                    "classify_action": action,
                    "classify_reason": reason,
                    "status": "non_final_classification_skipped",
                },
            }
            actions.append(entry)
            unresolved.append(entry)
            continue

        changed = False
        # Classify whether the node is a real LLM-backed agent or should be downgraded.
        agent_requires_llm_meta: Dict[str, Any] = {"classify_action": action, "classify_reason": reason}
        if action == "real_llm_agent":
            llm_meta = resolve_single_llm_child_with_child_creation_style(
                agent_var=avar,
                agent_node=nodes_by_var.get(avar, {}),
                guidance_summary=guidance_summary,
                issue_dir=issue_dir / "resolve_llm_child_creation_style",
                model=model,
                refresh_raw=refresh_raw,
                retriever=retriever,
                rag_max_rounds=rag_max_rounds,
                nodes_by_var=nodes_by_var,
                var_order=var_order,
                llm_reuse_prompt=agent_llm_reuse_prompt,
                llm_resolution_prompt=agent_llm_resolution_prompt,
            )
            changed = changed or bool(llm_meta.get("changed"))
            # For real agents, try to create or reuse a valid LLM child.
            agent_requires_llm_meta["llm_resolution"] = llm_meta
            if not bool(llm_meta.get("changed")):
                node = nodes_by_var.get(avar, {})
                node["node_type"] = {"__call__": "NodeType", "type": "other", "other_description": "agent_without_evidence_of_llm"}
                changed = True
        else:

            # For non-real agents, try merging into surrounding structure before downgrading.
            children_by_parent_now, parents_by_child_now = build_parent_child_maps(var_order, nodes_by_var)
            original_parent_vars = [pv for pv in parents_by_child_now.get(avar, []) if isinstance(pv, str) and pv in nodes_by_var and pv != avar]
            original_child_vars = [cv for cv in children_by_parent_now.get(avar, []) if isinstance(cv, str) and cv in nodes_by_var and cv != avar]
            # Try full merges into parents first, then into children if needed.
            merge_run = run_prioritized_parent_then_child_merges(
                source_var=avar,
                parent_targets=original_parent_vars,
                child_targets=original_child_vars,
                root_var=root_var,
                nodes_by_var=nodes_by_var,
                var_order=var_order,
                guidance_summary=guidance_summary,
                issue_dir=issue_dir / "not_real_agent_merge_pairs",
                stem_prefix="merge_pair",
                merge_prompt=single_child_prompt,
                model=model,
                refresh_raw=refresh_raw,
                retriever=retriever,
                rag_max_rounds=rag_max_rounds,
            )
            merged_into = merge_run.get("merged_into") if isinstance(merge_run.get("merged_into"), list) else []
            merge_decisions = merge_run.get("merge_decisions") if isinstance(merge_run.get("merge_decisions"), list) else []

            # Collect connectivity repairs produced by the full merge helper.
            applied_merges = merge_run.get("applied_merges") if isinstance(merge_run.get("applied_merges"), dict) else {}
            connectivity_runs: Dict[str, Any] = {
                tvar: meta.get("connectivity_repair")
                for tvar, meta in applied_merges.items()
                if isinstance(meta, dict) and meta.get("connectivity_repair") is not None
            }
            isolated_child_runs: Dict[str, Any] = {}
            delete_meta: Dict[str, Any] = {}

            # If a full merge succeeded, clean up any leftover source node state.
            # If a full merge succeeded, delete any leftover source node and resolve only its actual remaining children.
            if merged_into:
                non_merged_children: List[str] = []
                if avar in nodes_by_var and avar != root_var:
                    non_merged_children = [
                        cvar for cvar in children_of(nodes_by_var.get(avar, {}))
                        if isinstance(cvar, str) and cvar in nodes_by_var and cvar != avar
                    ]
                    delete_meta = delete_node_with_repair(
                        var=avar,
                        var_order=var_order,
                        nodes_by_var=nodes_by_var,
                        guidance_summary=guidance_summary,
                        retriever=retriever,
                        model=model,
                        raw_base_dir=issue_dir / "not_real_agent_delete_repair",
                        refresh_raw=refresh_raw,
                        rag_max_rounds=rag_max_rounds,
                    )
                changed = True

                # Resolve only children still left under the source node at delete time.
                for cvar in non_merged_children:
                    if cvar not in nodes_by_var:
                        continue
                    resolve_with_conn = resolve_orphan_node(
                        candidate_var=cvar,
                        root_var=root_var,
                        guidance_summary=guidance_summary,
                        issue_dir=issue_dir / "not_real_agent_unmerged_children" / safe_name(cvar),
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
                    iso_meta = resolve_with_conn.get("resolve") if isinstance(resolve_with_conn.get("resolve"), dict) else {}
                    connectivity_meta = resolve_with_conn.get("connectivity_repair") if isinstance(resolve_with_conn.get("connectivity_repair"), dict) else None
                    changed = changed or bool(resolve_with_conn.get("changed"))
                    isolated_child_runs[cvar] = {"resolve": iso_meta, "connectivity_repair": connectivity_meta}

            # If no merge succeeded, downgrade the node instead of deleting it.
            else:
                if avar in nodes_by_var:
                    node = nodes_by_var.get(avar, {})
                    node["node_type"] = {"__call__": "NodeType", "type": "other", "other_description": "not_real_llm_based_agent"}
                    changed = True

            # Record the merge/delete/orphan-resolution work done for non-real agents.
            agent_requires_llm_meta["merge_pair_decisions"] = merge_decisions
            agent_requires_llm_meta["merged_into"] = merged_into
            agent_requires_llm_meta["connectivity_repair"] = connectivity_runs
            if merged_into and avar != root_var:
                agent_requires_llm_meta["delete_repair"] = delete_meta
            agent_requires_llm_meta["isolated_children_resolution"] = isolated_child_runs

        entry = {
            "rule": "agent_requires_llm",
            "agent_var": avar,
            "decision": action,
            "reason": reason,
            "changed": changed,
            "agent_requires_llm_meta": agent_requires_llm_meta,
        }
        actions.append(entry)
        if not changed:
            unresolved.append(entry)


def run_rule_system_requires_agent(
    *,
    var_order: List[str],
    nodes_by_var: Dict[str, Dict[str, Any]],
    root_var: str | None,
    guidance_summary: str,
    raw_dir: Path,
    model: str,
    refresh_raw: bool,
    retriever: Any,
    rag_max_rounds: int,
    system_no_agent_prompt: str,
    single_child_prompt: str,
    representation_shortlist_prompt: str,
    representation_pair_prompt: str,
    missing_node_decision_prompt: str,
    adding_node_parent_decision_prompt: str,
    actions: List[Dict[str, Any]],
    unresolved: List[Dict[str, Any]],
) -> None:
    """Ensure every System has Agent children, or otherwise downgrade/remove it through child creation and merge cleanup."""

    # Track system nodes already processed in this pass.
    seen_system_requires_agent: Set[str] = set()
    while True:
        children_by_parent, parents_by_child = build_parent_child_maps(var_order, nodes_by_var)
        issue_system = None
        for svar in var_order:
            if svar in seen_system_requires_agent:
                continue
            snode = nodes_by_var.get(svar, {})
            if node_type_name(snode) != "System":
                continue
            child_vars = children_by_parent.get(svar, [])
            has_agent = any(node_type_name(nodes_by_var.get(c, {})) == "Agent" for c in child_vars)
            if not has_agent:
                issue_system = svar
                break

        if issue_system is None:
            break
        svar = issue_system
        seen_system_requires_agent.add(svar)
        snode = nodes_by_var.get(svar, {})
        existing_children = children_of(snode)

        issue_dir = raw_dir / "system_no_agent" / safe_name(svar)
        issue_dir.mkdir(parents=True, exist_ok=True)
        dec_raw, _ = call_with_cache(
            payload={
                "prompt": system_no_agent_prompt,
                "system_var": svar,
                "system_node": snode,
                "guidance_summary": guidance_summary,
                "retrieved_context": None,
            },
            payload_path=issue_dir / (str("decision") + ".payload.json"),
            raw_path=issue_dir / (str("decision") + ".txt"),
            model=model,
            refresh_raw=refresh_raw,
            retriever=retriever,
            rag_max_rounds=rag_max_rounds,
        )
        dec = dec_raw if isinstance(dec_raw, dict) else {}
        action = str(dec.get("action") or "not_agentic_system").strip()
        reason = str(dec.get("reason") or "")

        changed = False

        # Classify whether this system should gain agents or be treated as a non-agentic component.
        system_requires_agent_meta: Dict[str, Any] = {}

        # For agentic systems, try to create or reuse Agent children first.
        if action == "agentic_system":
            child_meta = expand_system_agents_with_child_creation(
                system_var=svar,
                system_node=nodes_by_var.get(svar, {}),
                guidance_summary=guidance_summary,
                retriever=retriever,
                model=model,
                raw_parent_dir=issue_dir / "child_creation_agents_only",
                refresh_raw=refresh_raw,
                rag_max_rounds=rag_max_rounds,
                nodes_by_var=nodes_by_var,
                var_order=var_order,
            )
            created_agents = child_meta.get("created_agent_vars") if isinstance(child_meta.get("created_agent_vars"), list) else []
            reused_agents = child_meta.get("reused_existing_agents") if isinstance(child_meta.get("reused_existing_agents"), list) else []
            has_added_agents = bool(created_agents or reused_agents)
            connectivity_meta: Dict[str, Any] | None = None

            # If Agent children were added, repair connectivity for the updated system.
            if has_added_agents:
                connectivity_meta = repair_parent_connectivity_iterative(
                    parent_var=svar,
                    parent_node=nodes_by_var.get(svar, {}),
                    guidance_summary=guidance_summary,
                    retriever=retriever,
                    model=model,
                    raw_parent_dir=issue_dir / "connectivity_pass_after_child_creation",
                    refresh_raw=refresh_raw,
                    rag_max_rounds=rag_max_rounds,
                    nodes_by_var=nodes_by_var,
                    max_iterations=2,
                )
                changed = True

            # If no Agent child could be added, either downgrade the system or delete an empty non-root system.
            else:
                if len(existing_children) > 0:
                    node = nodes_by_var.get(svar, {})
                    node["node_type"] = {"__call__": "NodeType", "type": "other", "other_description": "system_without_agents_after_child_creation"}
                    changed = True
                elif svar != root_var:
                    delete_meta = delete_node_with_repair(
                        var=svar,
                        var_order=var_order,
                        nodes_by_var=nodes_by_var,
                        guidance_summary=guidance_summary,
                        retriever=retriever,
                        model=model,
                        raw_base_dir=issue_dir / "agentic_system_delete_repair",
                        refresh_raw=refresh_raw,
                        rag_max_rounds=rag_max_rounds,
                    )
                    child_meta["delete_repair"] = delete_meta
                    changed = True

            system_requires_agent_meta = {
                "branch": "agentic_system",
                "existing_children_before": existing_children,
                "child_creation": child_meta,
                "connectivity_repair": connectivity_meta,
            }

        # For non-agentic systems, try merging into surrounding structure before downgrading.
        else:

            # Collect the surrounding parent/child targets that may absorb this non-agentic system.
            original_parent_vars = [pv for pv in parents_by_child.get(svar, []) if isinstance(pv, str) and pv in nodes_by_var and pv != svar]
            original_child_vars = [cv for cv in children_of(nodes_by_var.get(svar, {})) if isinstance(cv, str) and cv in nodes_by_var and cv != svar]
            candidate_targets: List[str] = []
            for pv in original_parent_vars:
                if pv not in candidate_targets:
                    candidate_targets.append(pv)
            for cv in original_child_vars:
                if cv not in candidate_targets:
                    candidate_targets.append(cv)

            # Run the full merge pass against parent targets first, then child targets if needed.
            merge_run = run_prioritized_parent_then_child_merges(
                source_var=svar,
                parent_targets=original_parent_vars,
                child_targets=original_child_vars,
                root_var=root_var,
                nodes_by_var=nodes_by_var,
                var_order=var_order,
                guidance_summary=guidance_summary,
                issue_dir=issue_dir / "non_agentic_merge_pairs",
                stem_prefix="merge_pair",
                merge_prompt=single_child_prompt,
                model=model,
                refresh_raw=refresh_raw,
                retriever=retriever,
                rag_max_rounds=rag_max_rounds,
            )
            merged_into = merge_run.get("merged_into") if isinstance(merge_run.get("merged_into"), list) else []
            merge_decisions = merge_run.get("merge_decisions") if isinstance(merge_run.get("merge_decisions"), list) else []

            applied_merges = merge_run.get("applied_merges") if isinstance(merge_run.get("applied_merges"), dict) else {}
            connectivity_runs: Dict[str, Any] = {
                tvar: meta.get("connectivity_repair")
                for tvar, meta in applied_merges.items()
                if isinstance(meta, dict) and meta.get("connectivity_repair") is not None
            }

            # Track any leftover child resolutions and delete repair after the merge pass.
            isolated_child_runs: Dict[str, Any] = {}
            delete_meta: Dict[str, Any] = {}
            if merged_into:
                non_merged_children: List[str] = []
                if svar in nodes_by_var and svar != root_var:
                    non_merged_children = [
                        cvar for cvar in children_of(nodes_by_var.get(svar, {}))
                        if isinstance(cvar, str) and cvar in nodes_by_var and cvar != svar
                    ]
                    delete_meta = delete_node_with_repair(
                        var=svar,
                        var_order=var_order,
                        nodes_by_var=nodes_by_var,
                        guidance_summary=guidance_summary,
                        retriever=retriever,
                        model=model,
                        raw_base_dir=issue_dir / "non_agentic_delete_repair",
                        refresh_raw=refresh_raw,
                        rag_max_rounds=rag_max_rounds,
                    )
                changed = True

                # Resolve only children still left under the source node after merge cleanup.
                for cvar in non_merged_children:
                    if cvar not in nodes_by_var:
                        continue
                    resolve_with_conn = resolve_orphan_node(
                        candidate_var=cvar,
                        root_var=root_var,
                        guidance_summary=guidance_summary,
                        issue_dir=issue_dir / "non_agentic_unmerged_children" / safe_name(cvar),
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
                    iso_meta = resolve_with_conn.get("resolve") if isinstance(resolve_with_conn.get("resolve"), dict) else {}
                    connectivity_meta = resolve_with_conn.get("connectivity_repair") if isinstance(resolve_with_conn.get("connectivity_repair"), dict) else None
                    changed = changed or bool(resolve_with_conn.get("changed"))
                    isolated_child_runs[cvar] = {"resolve": iso_meta, "connectivity_repair": connectivity_meta}

            # If no merge succeeded, downgrade the node instead of deleting it.
            else:
                node = nodes_by_var.get(svar, {})
                node["node_type"] = {"__call__": "NodeType", "type": "other", "other_description": "non_agentic_system_component"}
                changed = True

            # Record the merge/delete/orphan-resolution work done for non-agentic systems.
            system_requires_agent_meta = {
                "branch": "not_agentic_system",
                "original_parent_vars": original_parent_vars,
                "original_child_vars": original_child_vars,
                "candidate_targets": candidate_targets,
                "merge_pair_decisions": merge_decisions,
                "merged_into": merged_into,
                "delete_repair": delete_meta,
                "connectivity_repair": connectivity_runs,
                "isolated_children_resolution": isolated_child_runs,
            }

        # Record the final outcome for this system node.
        entry = {
            "rule": "system_requires_agent",
            "system_var": svar,
            "decision": action,
            "reason": reason,
            "changed": changed,
            "system_requires_agent_meta": system_requires_agent_meta,
        }
        actions.append(entry)
        if not changed:
            unresolved.append(entry)


def run_rule_mcp_server_structure(
    *,
    var_order: List[str],
    nodes_by_var: Dict[str, Dict[str, Any]],
    root_var: str | None,
    guidance_summary: str,
    raw_dir: Path,
    model: str,
    refresh_raw: bool,
    retriever: Any,
    rag_max_rounds: int,
    representation_shortlist_prompt: str,
    representation_pair_prompt: str,
    missing_node_decision_prompt: str,
    adding_node_parent_decision_prompt: str,
    actions: List[Dict[str, Any]],
    unresolved: List[Dict[str, Any]],
    external_allow_node_children: bool = False,
    external_require_tools: bool = False,
) -> None:
    """Validate and repair MCP server child structure for both local and external MCP server nodes."""

    # Track MCP server nodes already processed in this pass.
    seen_mcp_server_structure: Set[str] = set()

    def _resolve_removed_children(removed_vars: List[str], bucket_dir: Path) -> Dict[str, Any]:
        runs: Dict[str, Any] = {}
        for cvar in removed_vars:
            if cvar not in nodes_by_var:
                continue
            resolve_with_conn = resolve_orphan_node(
                candidate_var=cvar,
                root_var=root_var,
                guidance_summary=guidance_summary,
                issue_dir=bucket_dir / safe_name(cvar),
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
            resolve_meta = resolve_with_conn.get("resolve") if isinstance(resolve_with_conn.get("resolve"), dict) else {}
            connectivity_meta = resolve_with_conn.get("connectivity_repair") if isinstance(resolve_with_conn.get("connectivity_repair"), dict) else None
            runs[cvar] = {"resolve": resolve_meta, "connectivity_repair": connectivity_meta}
        return runs

    while True:
        children_by_parent, _ = build_parent_child_maps(var_order, nodes_by_var)
        issue_var = None
        for svar in var_order:
            if svar in seen_mcp_server_structure:
                continue
            snode = nodes_by_var.get(svar, {})
            stype = node_type_name(snode)
            if stype not in MCP_SERVER_TYPES:
                continue
            issue_var = svar
            break

        if issue_var is None:
            break

        # Load the current MCP server node and its immediate child structure for validation.
        svar = issue_var
        seen_mcp_server_structure.add(svar)
        snode = nodes_by_var.get(svar, {})
        stype = node_type_name(snode)
        node_children = children_in_nodes(snode)
        tool_children = children_in_tool_list(snode)
        has_nodes = bool(node_children)
        has_tools = bool(tool_children)
        changed = False

        # Capture the starting structure and active policy flags for reporting.
        mcp_server_structure_meta: Dict[str, Any] = {
            "server_type": stype,
            "nodes_before": list(node_children),
            "tool_list_before": list(tool_children),
            "external_allow_node_children": bool(external_allow_node_children),
            "external_require_tools": bool(external_require_tools),
        }

        issue_dir = raw_dir / "mcp_server_structure" / safe_name(svar)
        issue_dir.mkdir(parents=True, exist_ok=True)

        # Enforce tool_list contains only Tool nodes for all MCP server types.
        non_tool_tool_list = [
            cvar for cvar in tool_children
            if node_type_name(nodes_by_var.get(cvar, {})) != "Tool"
        ]
        removed_non_tool_tool_list: List[str] = []
        orphan_resolution_non_tool_tool_list: Dict[str, Any] = {}
        if non_tool_tool_list:
            kept_tools = [c for c in tool_children if c not in set(non_tool_tool_list)]
            snode["tool_list"] = sorted(set(kept_tools))
            removed_non_tool_tool_list = list(non_tool_tool_list)
            orphan_resolution_non_tool_tool_list = _resolve_removed_children(
                removed_non_tool_tool_list,
                issue_dir / "removed_non_tool_tool_list",
            )
            changed = True or bool(orphan_resolution_non_tool_tool_list)

        node_children = children_in_nodes(snode)
        tool_children = children_in_tool_list(snode)

        # External MCP servers enforce a stricter shape policy than local MCP servers.
        if stype == "External_MCP_server":
            removed_node_children: List[str] = []
            orphan_resolution_removed_node_children: Dict[str, Any] = {}

            if (not external_allow_node_children) and node_children:

                # External MCP policy: no direct node children unless explicitly allowed.
                snode["nodes"] = []
                removed_node_children = list(node_children)
                orphan_resolution_removed_node_children = _resolve_removed_children(
                    removed_node_children,
                    issue_dir / "removed_node_children",
                )
                changed = True or bool(orphan_resolution_removed_node_children)

            # If external servers require tools, missing tool_list keeps this issue unresolved.
            if external_require_tools and not tool_children:
                entry = {
                    "rule": "mcp_server_structure_external",
                    "server_var": svar,
                    "decision": "missing_tool_list_for_external_server",
                    "reason": "External MCP server is configured to require tool_list, but none found.",
                    "changed": changed,
                    "mcp_server_structure_meta": {
                        **mcp_server_structure_meta,
                        "removed_non_tool_tool_list": removed_non_tool_tool_list,
                        "orphan_resolution_non_tool_tool_list": orphan_resolution_non_tool_tool_list,
                        "removed_node_children": removed_node_children,
                        "orphan_resolution_removed_node_children": orphan_resolution_removed_node_children,
                    },
                }
                actions.append(entry)
                unresolved.append(entry)
                continue

            # Otherwise record the cleaned external MCP server state as resolved.
            entry = {
                "rule": "mcp_server_structure_external",
                "server_var": svar,
                "decision": "external_policy_enforced",
                "reason": "Applied external MCP server structure policy.",
                "changed": changed,
                "mcp_server_structure_meta": {
                    **mcp_server_structure_meta,
                    "removed_non_tool_tool_list": removed_non_tool_tool_list,
                    "orphan_resolution_non_tool_tool_list": orphan_resolution_non_tool_tool_list,
                    "removed_node_children": removed_node_children,
                    "orphan_resolution_removed_node_children": orphan_resolution_removed_node_children,
                    "nodes_after": children_in_nodes(snode),
                    "tool_list_after": children_in_tool_list(snode),
                },
            }
            actions.append(entry)
            continue

        # Local MCP servers may keep MCP-server children and/or tools, and can discover missing structure.
        non_mcp_nodes = [
            cvar for cvar in node_children
            if node_type_name(nodes_by_var.get(cvar, {})) not in MCP_SERVER_TYPES
        ]

        # Remove direct node children that are not themselves MCP server nodes.
        removed_non_mcp_nodes: List[str] = []
        orphan_resolution_non_mcp_nodes: Dict[str, Any] = {}
        if non_mcp_nodes:
            kept_nodes = [c for c in node_children if c not in set(non_mcp_nodes)]
            snode["nodes"] = kept_nodes
            removed_non_mcp_nodes = list(non_mcp_nodes)
            orphan_resolution_non_mcp_nodes = _resolve_removed_children(
                removed_non_mcp_nodes,
                issue_dir / "removed_non_mcp_nodes",
            )
            changed = True or bool(orphan_resolution_non_mcp_nodes)

        node_children = children_in_nodes(snode)
        tool_children = children_in_tool_list(snode)
        has_nodes = bool(node_children)
        has_tools = bool(tool_children)

        # If the local MCP server has no usable children yet, try to discover and materialize them.
        discovery_meta: Dict[str, Any] = {}
        if not has_nodes and not has_tools:
            proposals, discover_meta = _build_child_proposals_for_parent(
                parent_var=svar,
                parent_node=snode,
                guidance_summary=guidance_summary,
                retriever=retriever,
                model=model,
                raw_parent_dir=issue_dir / "discover_children",
                refresh_raw=refresh_raw,
                phase_max_rounds=8,
                no_change_patience=2,
            )
            apply_meta = materialize_attach_children_from_proposals(
                parent_var=svar,
                parent_node=snode,
                proposals=proposals,
                guidance_summary=guidance_summary,
                retriever=retriever,
                model=model,
                raw_parent_dir=issue_dir / "materialize_children",
                refresh_raw=refresh_raw,
                rag_max_rounds=rag_max_rounds,
                nodes_by_var=nodes_by_var,
                var_order=var_order,
                allowed_node_types={"Local_MCP_server", "External_MCP_server", "Tool"},
                run_connectivity_pass=True,
            )
            discovery_meta = {"phase_a": discover_meta, "apply": apply_meta}
            changed = changed or bool(apply_meta.get("changed"))

        node_children = children_in_nodes(snode)
        tool_children = children_in_tool_list(snode)

        # If both direct MCP children and direct tools remain, keep the MCP children and orphan-resolve the tools.
        both_children_decision: Dict[str, Any] = {}
        both_children_unresolved = False
        if node_children and tool_children:
            snode["tool_list"] = []
            removed_by_both_children_decision = _resolve_removed_children(
                list(tool_children),
                issue_dir / "both_children_removed_tool_list",
            )
            changed = True or bool(removed_by_both_children_decision)
            both_children_decision = {
                "action": "keep_only_nodes",
                "reason": "deterministic_local_mcp_keep_node_children_remove_tools",
                "orphan_resolution_removed_children": removed_by_both_children_decision,
            }

        node_children = children_in_nodes(snode)
        tool_children = children_in_tool_list(snode)

        # Remove duplicate top-level tools already reachable through nested MCP server children.
        removed_matching_tools: List[str] = []
        orphan_resolution_removed_matching_tools: Dict[str, Any] = {}
        if node_children and tool_children:

            # Collect tool names that are already available under descendant MCP server branches.
            tool_names_under_servers = descendant_tool_names_from_server_children(
                parent_var=svar,
                server_children=[
                    c for c in node_children
                    if node_type_name(nodes_by_var.get(c, {})) in MCP_SERVER_TYPES
                ],
                children_by_parent=children_by_parent,
            )

            # Keep only direct tools that are not duplicates of tools already provided under child MCP servers.
            kept_tool_vars: List[str] = []
            for tvar in tool_children:
                tname = str(nodes_by_var.get(tvar, {}).get("name") or "").strip()
                if tname and tname in tool_names_under_servers:
                    removed_matching_tools.append(tvar)
                else:
                    kept_tool_vars.append(tvar)

        # Duplicate direct tools are simply dropped here because the same tools are already reachable under child MCP servers.
        node_children_after = children_in_nodes(snode)
        tool_children_after = children_in_tool_list(snode)
        valid_local_state = bool(node_children_after or tool_children_after)
        if node_children_after:
            valid_local_state = valid_local_state and all(
                node_type_name(nodes_by_var.get(c, {})) in MCP_SERVER_TYPES
                for c in node_children_after
            )
        if both_children_unresolved:
            valid_local_state = False

        # Summarize whether the final local MCP structure is valid after all repairs.
        decision = "local_mcp_structure_ok" if valid_local_state else "local_mcp_structure_unresolved"
        reason = (
            "Local MCP server has valid structure: direct MCP-server child nodes and/or tool_list."
            if valid_local_state
            else "Local MCP server still has neither valid MCP-server child nodes nor tool_list."
        )
        entry = {
            "rule": "mcp_server_structure_local",
            "server_var": svar,
            "decision": decision,
            "reason": reason,
            "changed": changed,
            "mcp_server_structure_meta": {
                **mcp_server_structure_meta,
                "removed_non_tool_tool_list": removed_non_tool_tool_list,
                "orphan_resolution_non_tool_tool_list": orphan_resolution_non_tool_tool_list,
                "removed_non_mcp_nodes": removed_non_mcp_nodes,
                "orphan_resolution_non_mcp_nodes": orphan_resolution_non_mcp_nodes,
                "both_children_decision": both_children_decision,
                "removed_matching_tools": removed_matching_tools,
                "orphan_resolution_removed_matching_tools": orphan_resolution_removed_matching_tools,
                "discovery": discovery_meta,
                "nodes_after": node_children_after,
                "tool_list_after": tool_children_after,
            },
        }
        actions.append(entry)
        if not valid_local_state:
            unresolved.append(entry)


def run_rule_multi_parent(
    *,
    var_order: List[str],
    nodes_by_var: Dict[str, Dict[str, Any]],
    root_var: str | None,
    guidance_summary: str,
    raw_dir: Path,
    model: str,
    refresh_raw: bool,
    retriever: Any,
    rag_max_rounds: int,
    multi_parent_prompt: str,
    agent_system_single_parent_prompt: str,
    representation_shortlist_prompt: str,
    representation_pair_prompt: str,
    missing_node_decision_prompt: str,
    adding_node_parent_decision_prompt: str,
    actions: List[Dict[str, Any]],
    unresolved: List[Dict[str, Any]],
    resolved_parent_sets: Dict[str, Tuple[str, ...]] | None = None,
) -> None:
    """Resolve nodes that still have more than one non-LLM parent."""
    if resolved_parent_sets is None:
        resolved_parent_sets = {}
    seen_multi_parent: Set[str] = set()

    while True:

        # Recompute parent-child structure and pick the next unresolved multi-parent node.
        children_by_parent, parents_by_child = build_parent_child_maps(var_order, nodes_by_var)
        issue_child = None
        issue_parents: List[str] = []

        for cvar, parents in parents_by_child.items():
            effective_parents = [
                p for p in parents
                if isinstance(p, str) and node_type_name(nodes_by_var.get(p, {})) != "LLM"
            ]
            if len(effective_parents) <= 1 or cvar in seen_multi_parent:
                continue
            if node_type_name(nodes_by_var.get(cvar, {})) == "Deterministic_controller":
                continue
            ntype_for_skip = node_type_name(nodes_by_var.get(cvar, {}))
            cached_parents = None
            current_parents = tuple(sorted(effective_parents))
            if ntype_for_skip not in {"Agent", "System"}:
                cached_parents = resolved_parent_sets.get(cvar)
            if cached_parents is not None and cached_parents == current_parents:
                continue
            issue_child = cvar
            issue_parents = sorted(effective_parents)
            break

        if issue_child is None:
            break
        cvar = issue_child
        seen_multi_parent.add(cvar)

        # Remove deterministic controller parents first.
        pre_detachments: Dict[str, Any] = {}
        pre_pruned_controller_parents = [p for p in issue_parents if node_type_name(nodes_by_var.get(p, {})) == "Deterministic_controller"]
        pre_prune_changed = False
        for pvar in pre_pruned_controller_parents:
            pnode = nodes_by_var.get(pvar, {})
            removed = remove_child_ref(pnode, cvar)
            dropped_edges = drop_var_refs_from_internal_edges(pnode, cvar)
            dropped_ext = drop_var_refs_from_external_connections(pnode, cvar)
            connectivity_meta: Dict[str, Any] | None = None
            if pvar in nodes_by_var:
                connectivity_meta = repair_parent_connectivity_iterative(
                    parent_var=pvar,
                    parent_node=nodes_by_var[pvar],
                    guidance_summary=guidance_summary,
                    retriever=retriever,
                    model=model,
                    raw_parent_dir=raw_dir / "multi_parent" / safe_name(cvar) / "pre_prune_connectivity" / safe_name(pvar),
                    refresh_raw=refresh_raw,
                    rag_max_rounds=rag_max_rounds,
                    nodes_by_var=nodes_by_var,
                    max_iterations=2,
                )
            pre_detachments[pvar] = {
                "removed_child_ref": bool(removed),
                "dropped_internal_edges": int(dropped_edges),
                "dropped_external_refs": bool(dropped_ext),
                "connectivity_repair": connectivity_meta,
                "reason": "deterministic_controller_parent_prune",
            }
            pre_prune_changed = pre_prune_changed or removed or dropped_edges > 0 or dropped_ext or bool(connectivity_meta)

        # Recompute the surviving non-LLM parents after the deterministic-controller prune step.
        _, pre_parents_by_child = build_parent_child_maps(var_order, nodes_by_var)
        parents_after_pre_prune = (
            sorted(
                p for p in (pre_parents_by_child.get(cvar, []) if cvar in nodes_by_var else [])
                if isinstance(p, str) and node_type_name(nodes_by_var.get(p, {})) != "LLM"
            )
        )

        # If pruning leaves exactly one non-LLM parent, the multi-parent issue is resolved without an LLM decision.
        if pre_pruned_controller_parents and len(parents_after_pre_prune) == 1:
            entry = {
                "rule": "multi_parent",
                "node_var": cvar,
                "node_type": node_type_name(nodes_by_var.get(cvar, {})),
                "parents_before": issue_parents,
                "parents_after": parents_after_pre_prune,
                "requested_keep_parents": parents_after_pre_prune,
                "selection_mode": "deterministic_controller_parent_prune",
                "detachments": pre_detachments,
                "reason": "controller_parents_removed_before_llm_decision",
                "changed": pre_prune_changed,
            }
            actions.append(entry)
            ntype_after = node_type_name(nodes_by_var.get(cvar, {}))
            if ntype_after not in {"Agent", "System"}:
                resolved_parent_sets[cvar] = tuple(parents_after_pre_prune)
            if not pre_prune_changed:
                unresolved.append(entry)
            continue

        # If pruning leaves no non-LLM parent, repair the new orphan immediately.
        if pre_pruned_controller_parents and len(parents_after_pre_prune) == 0 and cvar in nodes_by_var:
            resolve_with_conn = resolve_orphan_node(
                candidate_var=cvar,
                root_var=root_var,
                guidance_summary=guidance_summary,
                issue_dir=raw_dir / "multi_parent" / safe_name(cvar) / "post_prune_orphan_repair",
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
            orphan_resolve = resolve_with_conn.get("resolve") if isinstance(resolve_with_conn.get("resolve"), dict) else {}
            orphan_connectivity = resolve_with_conn.get("connectivity_repair") if isinstance(resolve_with_conn.get("connectivity_repair"), dict) else None
            orphan_changed = bool(resolve_with_conn.get("changed"))
            _, parents_after_orphan_repair = build_parent_child_maps(var_order, nodes_by_var)
            final_parents_after_prune = sorted(parents_after_orphan_repair.get(cvar, [])) if cvar in nodes_by_var else []
            entry = {
                "rule": "multi_parent",
                "node_var": cvar,
                "node_type": node_type_name(nodes_by_var.get(cvar, {})),
                "parents_before": issue_parents,
                "parents_after": final_parents_after_prune,
                "requested_keep_parents": [],
                "selection_mode": "deterministic_controller_parent_prune_orphan_repair",
                "detachments": pre_detachments,
                "reason": "controller_parents_removed_then_orphan_repaired",
                "changed": pre_prune_changed or orphan_changed,
                "fallback": {
                    "node_resolve": orphan_resolve,
                    "node_connectivity_repair": orphan_connectivity,
                },
            }
            actions.append(entry)
            ntype_after = node_type_name(nodes_by_var.get(cvar, {}))
            if ntype_after not in {"Agent", "System"}:
                if final_parents_after_prune:
                    resolved_parent_sets[cvar] = tuple(final_parents_after_prune)
                else:
                    resolved_parent_sets.pop(cvar, None)
            if not (pre_prune_changed or orphan_changed):
                unresolved.append(entry)
            continue

        # Refresh the candidate parent set after the deterministic prune step.
        issue_parents = parents_after_pre_prune if pre_pruned_controller_parents else issue_parents
        candidate_issue_parents = [
            p for p in issue_parents
        ]
        ntype = node_type_name(nodes_by_var.get(cvar, {}))

        # Ask for the parent-selection decision for the remaining multi-parent case.
        issue_dir = raw_dir / "multi_parent" / safe_name(cvar)
        issue_dir.mkdir(parents=True, exist_ok=True)
        reason = ""
        keep_parents: List[str] = []
        selection_mode = "multi_parent_set"
        invalid_output = False
        invalid_output_reason = ""

        # Agent/System nodes must choose exactly one surviving parent, so use the dedicated prompt only.
        if ntype in {"Agent", "System"}:
            single_dec_raw, _ = call_with_cache(
                payload={
                    "prompt": agent_system_single_parent_prompt,
                    "node_var": cvar,
                    "node": nodes_by_var.get(cvar, {}),
                    "parent_vars": candidate_issue_parents,
                    "node_type": ntype,
                    "guidance_summary": guidance_summary,
                    "retrieved_context": None,
                },
                payload_path=issue_dir / (str("single_parent_decision") + ".payload.json"),
                raw_path=issue_dir / (str("single_parent_decision") + ".txt"),
                model=model,
                refresh_raw=refresh_raw,
                retriever=retriever,
                rag_max_rounds=rag_max_rounds,
            )
            single_dec = single_dec_raw if isinstance(single_dec_raw, dict) else {}
            selected_parent = str(single_dec.get("selected_parent") or "").strip()
            if selected_parent in candidate_issue_parents:
                keep_parents = [selected_parent]
                reason = str(single_dec.get("reason") or "")
            else:
                invalid_output = True
                invalid_output_reason = "invalid_selected_parent_for_agent_system"
            selection_mode = "agent_system_single_parent"

        # Other node types may legally keep multiple parents, so use the general multi-parent prompt.
        else:
            dec_raw, _ = call_with_cache(
                payload={
                    "prompt": multi_parent_prompt,
                    "node_var": cvar,
                    "node": nodes_by_var.get(cvar, {}),
                    "parent_vars": candidate_issue_parents,
                    "node_type": ntype,
                    "guidance_summary": guidance_summary,
                    "retrieved_context": None,
                },
                payload_path=issue_dir / (str("decision") + ".payload.json"),
                raw_path=issue_dir / (str("decision") + ".txt"),
                model=model,
                refresh_raw=refresh_raw,
                retriever=retriever,
                rag_max_rounds=rag_max_rounds,
            )
            dec = dec_raw if isinstance(dec_raw, dict) else {}
            reason = str(dec.get("reason") or "")
            keep_parents_raw = dec.get("keep_parents") if isinstance(dec.get("keep_parents"), list) else []
            keep_parents = [
                p
                for p in keep_parents_raw
                if isinstance(p, str) and p in candidate_issue_parents
            ]
            if not keep_parents:
                invalid_output = True
                invalid_output_reason = "empty_or_invalid_keep_parents"

        # Detach this node only from parents that were not selected to keep it.
        changed = False
        detachments: Dict[str, Any] = {}
        if not invalid_output:
            detach_targets = [p for p in issue_parents if p not in keep_parents]
            for pvar in detach_targets:
                pnode = nodes_by_var.get(pvar, {})
                removed = remove_child_ref(pnode, cvar)
                dropped_edges = drop_var_refs_from_internal_edges(pnode, cvar)
                dropped_ext = drop_var_refs_from_external_connections(pnode, cvar)
                connectivity_meta: Dict[str, Any] | None = None
                if pvar in nodes_by_var:
                    connectivity_meta = repair_parent_connectivity_iterative(
                        parent_var=pvar,
                        parent_node=nodes_by_var[pvar],
                        guidance_summary=guidance_summary,
                        retriever=retriever,
                        model=model,
                        raw_parent_dir=issue_dir / "post_detach_connectivity" / safe_name(pvar),
                        refresh_raw=refresh_raw,
                        rag_max_rounds=rag_max_rounds,
                        nodes_by_var=nodes_by_var,
                        max_iterations=2,
                    )
                detachments[pvar] = {
                    "removed_child_ref": bool(removed),
                    "dropped_internal_edges": int(dropped_edges),
                    "dropped_external_refs": bool(dropped_ext),
                    "connectivity_repair": connectivity_meta,
                }
                changed = changed or removed or dropped_edges > 0 or dropped_ext or bool(connectivity_meta)

        # If the model output is unusable, keep the current parents unchanged and record the failure.
        fallback_meta: Dict[str, Any] = {}
        if invalid_output:
            selection_mode = "invalid_output_keep_existing_parents"
            fallback_meta = {
                "invalid_output_reason": invalid_output_reason,
                "kept_existing_parents": list(issue_parents),
            }

        # Record the final parent set and the work performed for this node.
        _, parents_by_child_after = build_parent_child_maps(var_order, nodes_by_var)
        parents_after_actual = sorted(parents_by_child_after.get(cvar, [])) if cvar in nodes_by_var else []
        entry = {
            "rule": "multi_parent",
            "node_var": cvar,
            "node_type": ntype,
            "parents_before": issue_parents,
            "parents_after": parents_after_actual,
            "requested_keep_parents": keep_parents,
            "selection_mode": selection_mode,
            "detachments": {**pre_detachments, **detachments},
            "reason": reason,
            "changed": changed or pre_prune_changed,
        }
        if fallback_meta:
            entry["fallback"] = fallback_meta
        actions.append(entry)
        if ntype not in {"Agent", "System"}:
            if parents_after_actual:
                resolved_parent_sets[cvar] = tuple(parents_after_actual)
            else:
                resolved_parent_sets.pop(cvar, None)
        if not changed:
            unresolved.append(entry)


def collect_remaining_issues(
    *,
    var_order: List[str],
    nodes_by_var: Dict[str, Dict[str, Any]],
    resolved_single_child_keep_distinct_pairs: Set[Tuple[str, str]] | None = None,
    resolved_multi_parent_sets: Dict[str, Tuple[str, ...]] | None = None,
    external_allow_node_children: bool = False,
    external_require_tools: bool = False,
    root_var: str | None = None,
) -> List[Dict[str, Any]]:
    """Collect all graph nodes that still violate the current correctness rules."""

    # Normalize cached decisions so previously accepted cases are skipped.
    if resolved_single_child_keep_distinct_pairs is None:
        resolved_single_child_keep_distinct_pairs = set()
    if resolved_multi_parent_sets is None:
        resolved_multi_parent_sets = {}
    remaining_issues: List[Dict[str, Any]] = []
    children_by_parent, parents_by_child = build_parent_child_maps(var_order, nodes_by_var)

    # Scan per-parent rule violations across the current graph.
    for pvar in var_order:
        pnode = nodes_by_var.get(pvar, {})
        if node_type_name(pnode) == "Deterministic_controller":
            continue

        # Flag invalid direct Agent containment.
        if node_type_name(pnode) == "Agent":
            for cvar in children_of(pnode):
                if node_type_name(nodes_by_var.get(cvar, {})) == "Deterministic_controller":
                    continue
                if node_type_name(nodes_by_var.get(cvar, {})) == "Agent":
                    remaining_issues.append({"rule": "agent_in_agent_forbidden", "parent_var": pvar, "child_var": cvar})
        # Flag parents that still expose exactly one direct node child.
        node_children = children_in_nodes(pnode)
        if len(node_children) == 1:
            only_child = node_children[0]
            if node_type_name(nodes_by_var.get(only_child, {})) != "Deterministic_controller":
                key = (pvar, only_child)
                if key not in resolved_single_child_keep_distinct_pairs:
                    remaining_issues.append({"rule": "single_child_node", "parent_var": pvar, "only_child_var": only_child})

        # Flag systems and agents that are missing required child node types.
        if node_type_name(pnode) == "System":
            has_agent = any(node_type_name(nodes_by_var.get(c, {})) == "Agent" for c in children_by_parent.get(pvar, []))
            if not has_agent:
                remaining_issues.append({"rule": "system_requires_agent", "system_var": pvar})
        if node_type_name(pnode) == "Agent":
            has_llm = any(node_type_name(nodes_by_var.get(c, {})) == "LLM" for c in children_by_parent.get(pvar, []))
            if not has_llm:
                remaining_issues.append({"rule": "agent_requires_llm", "agent_var": pvar})

        # Flag LLM nodes that still contain children.
        if node_type_name(pnode) == "LLM":
            llm_children = children_of(pnode)
            if llm_children:
                remaining_issues.append({"rule": "llm_no_children", "llm_var": pvar, "children": llm_children})
        ptype = node_type_name(pnode)

        # Validate MCP server child layout constraints.
        if ptype == "Local_MCP_server":
            node_children = children_in_nodes(pnode)
            tool_children = children_in_tool_list(pnode)
            has_nodes = bool(node_children)
            has_tools = bool(tool_children)
            nodes_only_mcp = all(
                node_type_name(nodes_by_var.get(c, {})) in MCP_SERVER_TYPES
                for c in node_children
            )
            if (not has_nodes and not has_tools) or (has_nodes and not nodes_only_mcp):
                remaining_issues.append(
                    {
                        "rule": "mcp_server_structure_local",
                        "server_var": pvar,
                        "has_nodes": has_nodes,
                        "has_tools": has_tools,
                        "nodes_only_mcp": nodes_only_mcp,
                    }
                )
        elif ptype == "External_MCP_server":
            node_children = children_in_nodes(pnode)
            tool_children = children_in_tool_list(pnode)
            if (not external_allow_node_children) and node_children:
                remaining_issues.append(
                    {
                        "rule": "mcp_server_structure_external",
                        "server_var": pvar,
                        "issue": "has_node_children_while_disallowed",
                    }
                )
            if external_require_tools and not tool_children:
                remaining_issues.append(
                    {
                        "rule": "mcp_server_structure_external",
                        "server_var": pvar,
                        "issue": "missing_tool_list_while_required",
                    }
                )

    # Flag nodes that have no parents.
    for cvar in var_order:
        if cvar == root_var:
            continue
        if cvar in nodes_by_var and not parents_by_child.get(cvar):
            remaining_issues.append({"rule": "orphan_no_parent", "node_var": cvar})

    # Flag nodes that have multiple parents.
    for cvar, parents in parents_by_child.items():
        if len(parents) > 1:
            if node_type_name(nodes_by_var.get(cvar, {})) == "Deterministic_controller":
                continue
            ntype = node_type_name(nodes_by_var.get(cvar, {}))
            if ntype not in {"Agent", "System"}:
                cached_parents = resolved_multi_parent_sets.get(cvar)
                current_parents = tuple(sorted(parents))
                if cached_parents is not None and cached_parents == current_parents:
                    continue
            remaining_issues.append({"rule": "multi_parent", "node_var": cvar, "parents": sorted(parents)})

    return remaining_issues
