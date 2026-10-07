#!/usr/bin/env python3
"""Helpers for static validation: representation, attachment, and reconciliation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
    from ..child_creation.utils import _normalize_child_node, expand_children_from_parent
    from ..connectivity_pass.utils import repair_parent_connectivity_iterative
    from ..rag.build_retriever import load_retriever
    from ..NodeSpec_per_file.utils import _cr_dedupe_merge_refs, _cr_normalize_and_filter_refs
    from .prompts import (
        ADDING_NODE_PARENT_DECISION_PROMPT,
        MISSING_NODE_DECISION_PROMPT,
        REPRESENTATION_PAIR_PROMPT,
        REPRESENTATION_SHORTLIST_PROMPT,
    )
    from ..global_utils import (
        add_child_ref,
        call_with_cache,
        parse_json_loose,
        load_nodes_by_var,
        node_type_name,
        read_json,
        read_root_node_json,
        root_connected_set,
        resolve_root_var,
        resolve_var_name,
        run_shared_graph_candidate_decision,
        slugify,
    )
except ImportError:
    from child_creation.utils import _normalize_child_node, expand_children_from_parent
    from connectivity_pass.utils import repair_parent_connectivity_iterative
    from rag.build_retriever import load_retriever
    from NodeSpec_per_file.utils import _cr_dedupe_merge_refs, _cr_normalize_and_filter_refs
    from static_validation.prompts import (
        ADDING_NODE_PARENT_DECISION_PROMPT,
        MISSING_NODE_DECISION_PROMPT,
        REPRESENTATION_PAIR_PROMPT,
        REPRESENTATION_SHORTLIST_PROMPT,
    )
    from global_utils import (
        add_child_ref,
        call_with_cache,
        parse_json_loose,
        load_nodes_by_var,
        node_type_name,
        read_json,
        read_root_node_json,
        root_connected_set,
        resolve_root_var,
        resolve_var_name,
        run_shared_graph_candidate_decision,
        slugify,
    )


def _normalize_node_for_output(node: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a node into the shared basic in-memory shape used before static-validation mutations."""
    out = dict(node) if isinstance(node, dict) else {}
    out.pop("__var__", None)
    return _normalize_child_node(out)


def _merge_code_refs(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    """Merge source code references into a node and re-normalize the combined list."""
    base = dst.get("code_references") if isinstance(dst.get("code_references"), list) else []
    add = src.get("code_references") if isinstance(src.get("code_references"), list) else []
    dst["code_references"] = _cr_dedupe_merge_refs(_cr_normalize_and_filter_refs([*base, *add]))


def _mark_represented(node: Dict[str, Any], catalog_var: str) -> None:
    """Record that a graph node already represents a catalog node and track the catalog var mapping."""
    md = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    v = md.get("isolated_validation") if isinstance(md.get("isolated_validation"), dict) else {}
    v["represented"] = True
    mapped = v.get("mapped_catalog_vars") if isinstance(v.get("mapped_catalog_vars"), list) else []
    if catalog_var not in mapped:
        mapped.append(catalog_var)
    v["mapped_catalog_vars"] = sorted(set(str(x) for x in mapped if isinstance(x, str)))
    md["isolated_validation"] = v
    node["metadata"] = md


def _deterministic_attach_field(parent_node: Dict[str, Any]) -> str:
    """Choose whether a new child should attach under `nodes` or `tool_list` for this parent."""
    nt = node_type_name(parent_node).lower()
    return "tool_list" if "mcp_server" in nt else "nodes"


def _node_catalog_entry(var: str, node: Dict[str, Any]) -> Dict[str, Any]:
    """Build the lightweight prompt-facing summary for one graph node."""
    return {
        "var": var,
        "name": str(node.get("name") or ""),
        "node_type": node_type_name(node),
        "description": str(node.get("description") or "")[:500],
    }

def initialize_static_validation(
    *,
    nodes_py: Path,
    catalog_nodes_py: Path,
    root_json: Path,
    out_raw_dir: Path,
    model: str,
    refresh_raw: bool,
    rag_index_dir: Path,
    rag_max_rounds: int,
) -> Dict[str, Any]:
    """Load static-validation inputs and return the shared mutable pipeline state."""
    var_order, nodes_by_var = load_nodes_by_var(nodes_py)
    if not var_order:
        raise RuntimeError("Graph input has no ALL_NODES/MAIN_GRAPH_NODES.")

    catalog_order, catalog_nodes_by_var = load_nodes_by_var(catalog_nodes_py)
    if not catalog_order:
        raise RuntimeError("Static validation nodes input has no ALL_NODES/MAIN_GRAPH_NODES.")

    retriever = load_retriever(str(rag_index_dir))
    root_node = read_root_node_json(root_json, allow_flat_object=True)
    root_var = resolve_root_var(root_node, nodes_by_var) or (var_order[0] if var_order else None)
    if not root_var:
        raise RuntimeError("Could not resolve root var for static validation.")

    guidance_summary = ""
    try:
        sg_path = nodes_py.parent / "system_guidance.json"
        if sg_path.exists():
            sg = read_json(sg_path)
            if isinstance(sg.get("guidance"), dict):
                guidance_summary = str((sg.get("guidance") or {}).get("summary") or "")
            if not guidance_summary:
                guidance_summary = str(sg.get("summary") or sg.get("guidance_summary") or "")
    except Exception:
        guidance_summary = ""

    out_raw_dir.mkdir(parents=True, exist_ok=True)

    schema_catalog_path = Path(__file__).resolve().parents[1] / "NodeSpec_schema_short.py"
    nodespec_schema_catalog = (
        schema_catalog_path.read_text(encoding="utf-8", errors="replace")
        if schema_catalog_path.exists()
        else ""
    )

    report: Dict[str, Any] = {
        "mode": "static_validation",
        "root_var": root_var,
        "catalog_total": len(catalog_order),
        "decision_counts": {
            "exists_complete": 0,
            "new_add_to_graph": 0,
            "isolated_node": 0,
        },
        "node_decisions": [],
        "added_nodes": [],
        "errors": [],
        "represented_count": 0,
        "attached_count": 0,
        "isolated_count": 0,
        "missed_agentic_components": 0,
        "not_represented_graph_components": 0,
        "fixed_point_isolated_retries": [],
        "inline_child_expansion_passes": [],
    }

    return {
        "nodes_py": nodes_py,
        "out_raw_dir": out_raw_dir,
        "model": model,
        "refresh_raw": refresh_raw,
        "rag_max_rounds": rag_max_rounds,
        "retriever": retriever,
        "root_var": root_var,
        "var_order": var_order,
        "nodes_by_var": nodes_by_var,
        "catalog_order": catalog_order,
        "catalog_nodes_by_var": catalog_nodes_by_var,
        "used_vars": set(var_order),
        "isolated_new": {},
        "guidance_summary": guidance_summary,
        "nodespec_schema_catalog": nodespec_schema_catalog,
        "report": report,
    }


def catalog_node_run_dir(*, out_raw_dir: Path, catalog_var: str, index: int, retry_round: int | None = None) -> Path:
    """Return and create the raw-output directory for one catalog-node validation run."""
    if retry_round is None:
        run_dir = out_raw_dir / f"{index:03d}_{slugify(catalog_var)}"
    else:
        run_dir = out_raw_dir / f"round_{retry_round:02d}" / f"{index:03d}_{slugify(catalog_var)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _graph_catalog(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the prompt-facing catalog view of the current graph nodes."""
    nodes_by_var = state["nodes_by_var"]
    return [_node_catalog_entry(v, nodes_by_var[v]) for v in state["var_order"] if v in nodes_by_var]


def _drop_isolated_by_source(state: Dict[str, Any], catalog_var: str) -> List[str]:
    """Remove isolated nodes that were previously created from the given catalog var."""
    dropped: List[str] = []
    isolated_new = state["isolated_new"]
    for iso_var in list(isolated_new.keys()):
        inode = isolated_new.get(iso_var, {})
        if not isinstance(inode, dict):
            continue
        imd = inode.get("metadata") if isinstance(inode.get("metadata"), dict) else {}
        if str(imd.get("source_catalog_var") or "") == catalog_var:
            isolated_new.pop(iso_var, None)
            dropped.append(iso_var)
    return dropped


def _store_isolated_node(
    state: Dict[str, Any],
    *,
    catalog_var: str,
    catalog_node: Dict[str, Any],
    reason: str | None = None,
    count_metrics: bool,
) -> None:
    """Save a catalog node into the isolated output set with static-validation metadata."""
    isolated_new = state["isolated_new"]
    iso_var = resolve_var_name(catalog_var, None, set(isolated_new.keys()), {})
    iso_node = _normalize_node_for_output(catalog_node)
    iso_md = iso_node.get("metadata") if isinstance(iso_node.get("metadata"), dict) else {}
    iso_md["isolated_by"] = "static_validation"
    iso_md["source_catalog_var"] = catalog_var
    if reason:
        iso_md["not_added_reason"] = reason
    iso_node["metadata"] = iso_md
    isolated_new[iso_var] = iso_node
    if count_metrics:
        report = state["report"]
        report["isolated_count"] += 1
        report["missed_agentic_components"] += 1


def _apply_representation_match(
    state: Dict[str, Any],
    *,
    represented_vars: List[str],
    catalog_var: str,
    catalog_node: Dict[str, Any],
    drop_isolated: bool,
) -> None:
    """Mark a catalog node as already represented by existing graph nodes and merge its evidence."""
    nodes_by_var = state["nodes_by_var"]
    for represented_var in represented_vars:
        if represented_var not in nodes_by_var:
            continue
        _merge_code_refs(nodes_by_var[represented_var], catalog_node)
        _mark_represented(nodes_by_var[represented_var], catalog_var)
    if drop_isolated:
        _drop_isolated_by_source(state, catalog_var)
    state["report"]["represented_count"] += 1


def _attach_new_graph_node(
    state: Dict[str, Any],
    *,
    catalog_var: str,
    catalog_node: Dict[str, Any],
    approved_parents: List[str],
    drop_isolated: bool,
) -> tuple[str, int]:
    """Create a new graph node from a catalog node, attach it to approved parents, and record metrics."""
    new_var = resolve_var_name(catalog_var, None, state["used_vars"], {})
    new_node = _normalize_node_for_output(catalog_node)
    new_md = new_node.get("metadata") if isinstance(new_node.get("metadata"), dict) else {}
    new_md["created_by"] = "static_validation"
    new_md["source_catalog_var"] = catalog_var
    new_node["metadata"] = new_md
    nodes_by_var = state["nodes_by_var"]
    nodes_by_var[new_var] = new_node
    state["var_order"].append(new_var)
    connections_added = 0
    for parent_var in approved_parents:
        attach_field = _deterministic_attach_field(nodes_by_var.get(parent_var, {}))
        if add_child_ref(nodes_by_var[parent_var], new_var, field=attach_field):
            connections_added += 1
    _mark_represented(nodes_by_var[new_var], catalog_var)
    if drop_isolated:
        _drop_isolated_by_source(state, catalog_var)
    report = state["report"]
    report["attached_count"] += 1
    report["added_nodes"].append({"catalog_var": catalog_var, "new_var": new_var})
    return new_var, connections_added


def process_catalog_node(
    *,
    state: Dict[str, Any],
    catalog_var: str,
    child_dir: Path,
    is_retry: bool = False,
    retry_round: int | None = None,
) -> Dict[str, Any]:
    """Validate one catalog node against the current graph and apply any resulting mutations."""

    catalog_node_raw = state["catalog_nodes_by_var"].get(catalog_var)
    if not isinstance(catalog_node_raw, dict):
        raise RuntimeError(f"Missing catalog node for {catalog_var}")
    catalog_node = _normalize_node_for_output(catalog_node_raw)
    nodes_by_var = state["nodes_by_var"]
    guidance_summary = state["guidance_summary"]
    retriever = state["retriever"]
    model = state["model"]
    rag_max_rounds = state["rag_max_rounds"]
    refresh_raw = state["refresh_raw"]
    root_var = state["root_var"]

    catalog = _graph_catalog(state)
    decision_meta = run_shared_graph_candidate_decision(
        candidate_var=catalog_var,
        candidate_node=catalog_node,
        root_var=root_var,
        graph_catalog=catalog,
        nodes_by_var=nodes_by_var,
        decision_dir=child_dir,
        model=model,
        refresh_raw=refresh_raw,
        retriever=retriever,
        rag_max_rounds=rag_max_rounds,
        guidance_summary=guidance_summary,
        representation_shortlist_prompt=REPRESENTATION_SHORTLIST_PROMPT,
        representation_pair_prompt=REPRESENTATION_PAIR_PROMPT,
        missing_node_decision_prompt=MISSING_NODE_DECISION_PROMPT,
        adding_node_parent_decision_prompt=ADDING_NODE_PARENT_DECISION_PROMPT,
        allowed_match_vars=set(nodes_by_var.keys()),
        allowed_parent_vars=set(nodes_by_var.keys()),
        filter_rep_list=False,
    )
    shortlist_output = decision_meta.get("shortlist_output") if isinstance(decision_meta.get("shortlist_output"), dict) else {}
    represented_vars = list(decision_meta.get("represented_vars") or [])
    represented_reasons = dict(decision_meta.get("represented_reasons") or {})
    pair_checks = list(decision_meta.get("pair_checks") or [])
    short_src = str((decision_meta.get("sources") or {}).get("representation_shortlist") or "skipped")
    pair_src = str((decision_meta.get("sources") or {}).get("representation_pair") or "skipped")
    miss_dec_src = str((decision_meta.get("sources") or {}).get("missing_decision") or "skipped")
    conn_src = str((decision_meta.get("sources") or {}).get("adding_parent_decision") or "skipped")
    missing_decision_output = decision_meta.get("missing_decision_output") if isinstance(decision_meta.get("missing_decision_output"), dict) else None

    # Initialize the final decision state for this catalog node.
    decision = "isolated_node"
    added_var = None
    matched_var = represented_vars[0] if represented_vars else None
    matched_vars = list(represented_vars)
    connections_added = 0
    reason = str(decision_meta.get("reason") or "")

    # If representation was confirmed, merge evidence and stop here.
    if represented_vars:
        _apply_representation_match(
            state,
            represented_vars=represented_vars,
            catalog_var=catalog_var,
            catalog_node=catalog_node,
            drop_isolated=is_retry,
        )
        decision = "exists_complete"
        reason = represented_reasons.get(represented_vars[0], "") if represented_vars else ""

    # If representation was not confirmed, check for possible attachment or keep isolated.
    else:
        need_to_attach = bool(decision_meta.get("need_to_attach"))
        approved_parents = [value for value in (decision_meta.get("approved_parents") or []) if isinstance(value, str)]

        # Keep the node isolated when there is not enough evidence to attach it.
        if not need_to_attach:
            _store_isolated_node(
                state,
                catalog_var=catalog_var,
                catalog_node=catalog_node,
                reason=None,
                count_metrics=not is_retry,
            )
        else:
            # Either keep the node isolated or attach it under the approved parent.
            if not approved_parents:
                _store_isolated_node(
                    state,
                    catalog_var=catalog_var,
                    catalog_node=catalog_node,
                    reason="need_to_attach=true but parent candidate was approved by binary parent decisions.",
                    count_metrics=not is_retry,
                )
            else:
                added_var, connections_added = _attach_new_graph_node(
                    state,
                    catalog_var=catalog_var,
                    catalog_node=catalog_node,
                    approved_parents=approved_parents,
                    drop_isolated=is_retry,
                )
                matched_var = added_var
                matched_vars = [added_var]
                decision = "new_add_to_graph"

    # Record the decision and raw call outputs in the per-node report entry.
    state["report"]["decision_counts"][decision] = int(state["report"]["decision_counts"].get(decision, 0)) + 1
    call_outputs: Dict[str, Any] = {"representation_shortlist": shortlist_output}
    if pair_checks:
        call_outputs["representation_pair"] = list(pair_checks)
    if isinstance(missing_decision_output, dict):
        call_outputs["missing_decision"] = missing_decision_output

    node_decision = {
        "catalog_var": catalog_var,
        "decision_class": decision,
        "matched_var": matched_var,
        "matched_vars": matched_vars,
        "added_var": added_var,
        "connections_added": connections_added,
        "edges_added": 0,
        "edges_removed": 0,
        "reason": reason,
        "sources": {
            "representation_shortlist": short_src,
            "representation_pair": pair_src,
            "missing_decision": miss_dec_src,
        },
        "pair_checks": pair_checks,
        "call_outputs": call_outputs,
    }
    if is_retry:
        node_decision["is_retry"] = True
        node_decision["retry_round"] = retry_round
    state["report"]["node_decisions"].append(node_decision)

    return {
        "decision": decision,
        "added_seed_var": added_var if decision == "new_add_to_graph" else None,
    }


def run_inline_child_and_connectivity_pass(*, state: Dict[str, Any], seed_vars: List[str], pass_tag: str) -> Dict[str, Any]:
    """Run inline child expansion plus connectivity repair for newly attached graph seeds."""

    # Create the shared raw-output directory for this child-expansion round.
    pass_dir = state["out_raw_dir"] / "_inline_child_creation" / (slugify(pass_tag) or "pass")
    pass_dir.mkdir(parents=True, exist_ok=True)

    # Expand children from each seed parent and aggregate the per-parent runs.
    expansion_errors: List[Dict[str, Any]] = []
    created_nodes = 0
    parents_processed = 0
    created_vars: List[str] = []
    parent_runs: List[Dict[str, Any]] = []
    for seed_var in [value for value in seed_vars if isinstance(value, str) and value in state["nodes_by_var"]]:
        try:
            run = expand_children_from_parent(
                start_parent_var=seed_var,
                nodes_by_var=state["nodes_by_var"],
                var_order=state["var_order"],
                guidance_summary=state["guidance_summary"],
                nodespec_schema_catalog=state["nodespec_schema_catalog"],
                retriever=state["retriever"],
                model=state["model"],
                raw_root_dir=pass_dir,
                refresh_raw=state["refresh_raw"],
                rag_max_rounds=state["rag_max_rounds"],
                phase_a_max_rounds=8,
                phase_a_no_change_patience=2,
            )
            created_nodes += int(run.get("created_nodes") or 0)
            parents_processed += int(run.get("parents_processed") or 0)
            parent_runs.extend(run.get("parent_runs") if isinstance(run.get("parent_runs"), list) else [])
            for parent_run in run.get("parent_runs") if isinstance(run.get("parent_runs"), list) else []:
                for child_var in parent_run.get("attached_child_vars") if isinstance(parent_run, dict) and isinstance(parent_run.get("attached_child_vars"), list) else []:
                    if isinstance(child_var, str):
                        created_vars.append(child_var)
            expansion_errors.extend(run.get("errors") if isinstance(run.get("errors"), list) else [])
        except Exception as exc:
            expansion_errors.append({"parent_var": seed_var, "error": str(exc)})

    # Run connectivity repair across the current graph after child expansion.
    connectivity_dir = pass_dir / "connectivity_pass"
    connectivity_dir.mkdir(parents=True, exist_ok=True)
    connectivity_runs: List[Dict[str, Any]] = []
    total_edges_added = 0
    total_edges_removed = 0
    for parent_var in list(state["var_order"]):
        parent_node = state["nodes_by_var"].get(parent_var)
        if not isinstance(parent_node, dict):
            continue
        conn_meta = repair_parent_connectivity_iterative(
            parent_var=parent_var,
            parent_node=parent_node,
            guidance_summary=state["guidance_summary"],
            retriever=state["retriever"],
            model=state["model"],
            raw_parent_dir=connectivity_dir / (slugify(parent_var) or "parent"),
            refresh_raw=state["refresh_raw"],
            rag_max_rounds=state["rag_max_rounds"],
            nodes_by_var=state["nodes_by_var"],
            max_iterations=5,
        )
        edges_added = int(conn_meta.get("edges_added") or 0)
        edges_removed = int(conn_meta.get("edges_removed") or 0)
        total_edges_added += edges_added
        total_edges_removed += edges_removed
        connectivity_runs.append({
            "parent_var": parent_var,
            "edges_added": edges_added,
            "edges_removed": edges_removed,
            "unresolved": bool(conn_meta.get("unresolved")),
            "final_issues": conn_meta.get("final_issues"),
            "changed": bool(conn_meta.get("changed")),
        })

    # Return one combined summary for the expansion phase and the connectivity phase.
    return {
        "pass_tag": pass_tag,
        "seed_vars": [value for value in seed_vars if isinstance(value, str)],
        "parents_processed": parents_processed,
        "created_nodes": created_nodes,
        "created_vars": list(dict.fromkeys(value for value in created_vars if isinstance(value, str)).keys()),
        "parent_runs": parent_runs,
        "connectivity_pass": {
            "runs": connectivity_runs,
            "edges_added_total": total_edges_added,
            "edges_removed_total": total_edges_removed,
        },
        "errors": expansion_errors,
    }


def list_pending_retry_catalog_vars(state: Dict[str, Any]) -> List[str]:
    """Return catalog vars currently represented only by isolated nodes and pending retry."""
    return sorted(
        {
            str((node.get("metadata") or {}).get("source_catalog_var") or "")
            for node in state["isolated_new"].values()
            if isinstance(node, dict)
            and isinstance(node.get("metadata"), dict)
            and str((node.get("metadata") or {}).get("source_catalog_var") or "").strip()
        }
    )


def finalize_static_validation(state: Dict[str, Any]) -> Tuple[List[str], Dict[str, Dict[str, Any]], List[str], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Return the final main/isolated partition for static validation without modifying the existing graph."""
    report = state["report"]
    nodes_by_var = state["nodes_by_var"]
    var_order = state["var_order"]
    root_var = state["root_var"]
    isolated_new = state["isolated_new"]

    # Compute the final main graph as the root-connected portion of the current graph.
    connected = root_connected_set(root_var=root_var, var_order=var_order, nodes_by_var=nodes_by_var)
    if connected:
        main_set = connected
    elif root_var in nodes_by_var:
        main_set = {root_var}
    else:
        main_set = set()

    # Materialize the final main-graph order and node mapping.
    main_order = [value for value in var_order if value in main_set and value in nodes_by_var]
    main_nodes = {value: nodes_by_var[value] for value in main_order}

    # Count main-graph nodes that were never marked as representing a catalog node.
    helper_unattached = 0
    for value in main_order:
        node = main_nodes[value]
        metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        validation = metadata.get("isolated_validation") if isinstance(metadata.get("isolated_validation"), dict) else {}
        if not bool(validation.get("represented")):
            helper_unattached += 1
    report["not_represented_graph_components"] = helper_unattached

    # Build the final isolated output from disconnected graph nodes plus explicitly isolated catalog nodes.
    graph_isolated_order = [value for value in var_order if value not in main_set and value in nodes_by_var]
    isolated_order = [*graph_isolated_order, *sorted(isolated_new.keys())]
    isolated_nodes = {value: nodes_by_var[value] for value in graph_isolated_order}
    isolated_nodes.update(isolated_new)


    # Update the final summary counts returned to the output artifacts.
    report["isolated_count"] = len(isolated_new)
    report["missed_agentic_components"] = len(isolated_new)
    report["main_nodes"] = len(main_order)
    report["isolated_nodes"] = len(isolated_order)
    return main_order, main_nodes, isolated_order, isolated_nodes, report


__all__ = [
    "initialize_static_validation",
    "catalog_node_run_dir",
    "process_catalog_node",
    "run_inline_child_and_connectivity_pass",
    "list_pending_retry_catalog_vars",
    "finalize_static_validation",
]
