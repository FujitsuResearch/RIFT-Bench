#!/usr/bin/env python3
"""Pipeline for node_refinement component."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set

try:
    from ..global_utils import (
        call_with_cache,
        copy_nodespec_schema_to_out_dir,
        extract_var_name,
        load_env_file,
        load_nodes_by_var,
        load_guidance_summary,
        node_type_name,
        read_root_node_json,
        resolve_root_var,
        write_nodes_output,
    )
    from ..static_validation.prompts import (
        ADDING_NODE_PARENT_DECISION_PROMPT,
        MISSING_NODE_DECISION_PROMPT,
        REPRESENTATION_PAIR_PROMPT,
        REPRESENTATION_SHORTLIST_PROMPT,
    )
    from .prompts import (
        COMPONENT_EXISTENCE_CHECK_PROMPT,
        GROUP_REFINEMENT_PROMPTS,
        TYPE_CLASSIFICATION_PROMPT,
    )
    from .utils import (
        _ensure_type_specific_fields,
        apply_filtered_updates,
        collect_env_var_names,
        normalize_connectivity_fields,
        parse_group_updates,
        resolve_detached_candidates,
    )
    from ..rag.build_retriever import load_retriever
except ImportError:
    from global_utils import (
        call_with_cache,
        copy_nodespec_schema_to_out_dir,
        extract_var_name,
        load_env_file,
        load_nodes_by_var,
        node_type_name,
        load_guidance_summary,
        read_root_node_json,
        resolve_root_var,
        write_nodes_output,
    )
    from static_validation.prompts import (
        ADDING_NODE_PARENT_DECISION_PROMPT,
        MISSING_NODE_DECISION_PROMPT,
        REPRESENTATION_PAIR_PROMPT,
        REPRESENTATION_SHORTLIST_PROMPT,
    )
    from node_refinement.prompts import (
        COMPONENT_EXISTENCE_CHECK_PROMPT,
        GROUP_REFINEMENT_PROMPTS,
        TYPE_CLASSIFICATION_PROMPT,
    )
    from node_refinement.utils import (
        _ensure_type_specific_fields,
        apply_filtered_updates,
        collect_env_var_names,
        normalize_connectivity_fields,
        parse_group_updates,
        resolve_detached_candidates,
    )
    from rag.build_retriever import load_retriever

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_file", default=".env", help="Optional .env file to load.")
    ap.add_argument("--out_dir", default="stages_outputs", help="Output directory.")
    ap.add_argument("--nodes_py", default="nodes_main_graph.py", help="Input nodes file under --out_dir.")
    ap.add_argument("--root_json", default="root_nodes.json", help="Root json under --out_dir (reserved).")
    ap.add_argument("--out", default="nodes_refined.py", help="Output refined nodes file under --out_dir.")
    ap.add_argument("--report_out", default="node_refinement_report.json", help="Report output under --out_dir.")
    ap.add_argument("--raw_dir", default="node_refinement_raw", help="Raw traces folder under --out_dir.")
    ap.add_argument("--phase", default="pre", choices=["pre", "post"], help="Refinement phase label for reporting.")
    ap.add_argument(
        "--env_include_commented",
        dest="env_include_commented",
        action="store_true",
        help="Include commented env assignments (# KEY=...) as candidate names (default: on).",
    )
    ap.add_argument(
        "--env_active_only",
        dest="env_include_commented",
        action="store_false",
        help="Use only active env assignments (ignore commented lines).",
    )
    ap.add_argument("--model", default=os.environ.get("CODEX_MODEL", "gpt-5.2-codex"))
    ap.add_argument("--rag_index_dir", default="rag_index", help="RAG index path (relative to out_dir if not absolute).")
    ap.add_argument("--rag_max_rounds", type=int, default=6, help="Max context-request rounds per LLM call.")
    ap.add_argument("--refresh_raw", "--refresh-raw", dest="refresh_raw", action="store_true")
    ap.set_defaults(env_include_commented=True)
    args = ap.parse_args()

    # Load environment inputs and prepare the output workspace.
    env_path = Path(args.env_file)
    load_env_file(env_path)
    env_var_names = collect_env_var_names(env_path, include_commented=bool(args.env_include_commented))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_nodespec_schema_to_out_dir(out_dir)

    nodes_py = out_dir / Path(args.nodes_py).name
    out_py = out_dir / Path(args.out).name
    report_path = out_dir / Path(args.report_out).name
    raw_dir = out_dir / Path(args.raw_dir).name
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the retriever from the resolved RAG index when available.
    rag_index_path = Path(args.rag_index_dir)
    rag_index_dir = rag_index_path.resolve() if rag_index_path.is_absolute() else (out_dir / rag_index_path).resolve()
    retriever = None
    try:
        retriever = load_retriever(str(rag_index_dir))
    except Exception:
        retriever = None

    var_order, nodes_by_var = load_nodes_by_var(nodes_py)
    if not var_order:
        raise RuntimeError("node_refinement: input nodes file has no ALL_NODES/MAIN_GRAPH_NODES.")

    # Load guidance and resolve the graph root used by later repair flows.
    guidance_summary = load_guidance_summary(out_dir)
    root_json = out_dir / Path(args.root_json).name
    root_node = read_root_node_json(root_json, allow_flat_object=True)
    root_var = resolve_root_var(root_node, nodes_by_var) or (var_order[0] if var_order else None)

    removed_nodes: List[Dict[str, Any]] = []
    touched_vars: List[str] = []
    total_updates = 0
    to_resolve_after_existence: Set[str] = set()
    node_runs_by_var: Dict[str, Dict[str, Any]] = {}
    node_run_order: List[str] = []
    group_sequence = [
        ("group_1_identity", {"name", "description", "framework"}),
        ("group_2_io", {"inputs", "outputs"}),
        ("group_3_execution_keys", {"code_execution", "required_keys"}),
    ]
    per_type_group: Dict[str, tuple[str, Set[str]]] = {
        "LLM": ("group_6_llm", {"llm_config", "system_prompt", "user_prompt_template"}),
        "Local_MCP_server": ("group_7_mcp", {"tool_list"}),
        "Tool": ("group_8_tool_flags", {"read_internal", "read_external", "write_internal", "write_external", "is_rag_tool"}),
        "Agent": ("group_9_agent", {"agent_type", "llm_config", "system_prompt", "user_prompt_template"}),
        "System": ("group_10_system", {"system_type", "llm_config", "system_prompt", "user_prompt_template"}),
        "Database": ("group_11_database", {"data_path"}),
    }
    final_group = ("group_5_metadata", {"metadata"})

    # Configure the ordered refinement groups used for each surviving node.

    # Phase 1: Type classification (+ existence decision for `other`) for all nodes.
    for var in list(var_order):
        node = nodes_by_var.get(var)
        if not isinstance(node, dict):
            continue
        run_meta: Dict[str, Any] = {"var": var, "original_type": node_type_name(node)}
        node_runs_by_var[var] = run_meta
        node_run_order.append(var)

        # Skip node types that this stage intentionally leaves unchanged.
        if node_type_name(node) == "Deterministic_controller":
            run_meta["skipped"] = "controller_immutable"
            run_meta["final_type"] = node_type_name(node)
            run_meta["updated_fields_count"] = 0
            continue
        if node_type_name(node) == "External_MCP_server":
            run_meta["skipped"] = "external_mcp_skipped_by_policy"
            run_meta["final_type"] = node_type_name(node)
            run_meta["updated_fields_count"] = 0
            continue
        issue_dir = raw_dir / var
        issue_dir.mkdir(parents=True, exist_ok=True)
        pre_updates = 0

        # Collect direct child type hints to help the type-classification prompt.
        child_type_hints: List[Dict[str, str]] = []
        seen_child_vars: Set[str] = set()
        for ref_key in ("nodes", "tool_list"):
            refs = node.get(ref_key)
            if not isinstance(refs, list):
                continue
            for item in refs:
                cvar = extract_var_name(item)
                if not cvar or cvar in seen_child_vars:
                    continue
                seen_child_vars.add(cvar)
                cnode = nodes_by_var.get(cvar, {})
                child_type_hints.append(
                    {
                        "var_name": cvar,
                        "node_type": node_type_name(cnode),
                        "source": ref_key,
                    }
                )

        # Refine the node type.
        type_dec_raw, _ = call_with_cache(
            payload={
                "prompt": TYPE_CLASSIFICATION_PROMPT,
                "node_var": var,
                "node_block": node,
                "child_type_hints": child_type_hints,
                "guidance_summary": guidance_summary,
                "retrieved_context": None,
            },
            payload_path=issue_dir / "type_classification.payload.json",
            raw_path=issue_dir / "type_classification.txt",
            model=args.model,
            refresh_raw=bool(args.refresh_raw),
            retriever=retriever,
            rag_max_rounds=args.rag_max_rounds,
        )
        type_dec = type_dec_raw if isinstance(type_dec_raw, dict) else {}
        final_type = str(type_dec.get("final_type") or node_type_name(node) or "other").strip()
        if final_type not in {"LLM", "Tool", "Database", "System", "Local_MCP_server", "External_MCP_server", "Agent", "Deterministic_controller", "other"}:
            final_type = "other"

        type_reason = str(type_dec.get("reason") or "")
        if final_type != node_type_name(node):
            node["node_type"] = {"__call__": "NodeType", "type": final_type, "other_description": None if final_type != "other" else "retyped_to_other"}
            pre_updates += 1
        run_meta["type_classification"] = {"final_type": final_type, "reason": type_reason}

        # Decide whether `other` should remain in the graph.
        if final_type == "other":
            comp_dec_raw, _ = call_with_cache(
                payload={
                    "prompt": COMPONENT_EXISTENCE_CHECK_PROMPT,
                    "node_var": var,
                    "node_block": node,
                    "guidance_summary": guidance_summary,
                    "retrieved_context": None,
                },
                payload_path=issue_dir / "component_existence.payload.json",
                raw_path=issue_dir / "component_existence.txt",
                model=args.model,
                refresh_raw=bool(args.refresh_raw),
                retriever=retriever,
                rag_max_rounds=args.rag_max_rounds,
            )
            comp_dec = comp_dec_raw if isinstance(comp_dec_raw, dict) else {}
            action = str(comp_dec.get("action") or "keep_component").strip()
            if action == "remove_node":
                to_resolve_after_existence.add(var)
            run_meta["component_existence_check"] = {
                "action": action,
                "reason": str(comp_dec.get("reason") or ""),
            }
        run_meta["pre_refine_updates"] = pre_updates
        if pre_updates > 0:
            touched_vars.append(var)
            total_updates += pre_updates

    # Phase 2: For remove-candidates, run shared detach/resolve/forced-remove flow.
    phase2_rows = resolve_detached_candidates(
        initial_candidates=sorted(to_resolve_after_existence),
        nodes_by_var=nodes_by_var,
        var_order=var_order,
        issue_root=raw_dir / "_phase2_existence_resolution",
        root_var=root_var,
        guidance_summary=guidance_summary,
        model=args.model,
        refresh_raw=bool(args.refresh_raw),
        retriever=retriever,
        rag_max_rounds=args.rag_max_rounds,
        representation_shortlist_prompt=REPRESENTATION_SHORTLIST_PROMPT,
        representation_pair_prompt=REPRESENTATION_PAIR_PROMPT,
        missing_node_decision_prompt=MISSING_NODE_DECISION_PROMPT,
        adding_node_parent_decision_prompt=ADDING_NODE_PARENT_DECISION_PROMPT,
    )

    # Merge each Phase 2 resolution result back into the per-node run metadata and removed-node report.
    for row in phase2_rows:
        var = str(row.get("var") or "")
        if not var:
            continue
        run_meta = node_runs_by_var.get(var, {"var": var, "original_type": node_type_name(nodes_by_var.get(var, {}))})
        if bool(row.get("removed_from_graph")):
            removed_nodes.append(row)
            run_meta["removed_in_pre_refine_resolution"] = True
            run_meta["updated_fields_count"] = int(run_meta.get("pre_refine_updates") or 0)
        else:
            run_meta["kept_after_pre_refine_resolution"] = True
        run_meta["component_existence_resolution"] = row
        node_runs_by_var[var] = run_meta

    # Phase 3: Field refinement groups on remaining nodes.
    for var in list(var_order):
        node = nodes_by_var.get(var)
        if not isinstance(node, dict):
            continue
        run_meta = node_runs_by_var.get(var, {"var": var, "original_type": node_type_name(node)})
        if node_type_name(node) == "Deterministic_controller":
            run_meta["skipped"] = "controller_immutable"
            run_meta["final_type"] = node_type_name(node)
            run_meta["updated_fields_count"] = int(run_meta.get("pre_refine_updates") or 0)
            node_runs_by_var[var] = run_meta
            continue

        issue_dir = raw_dir / var
        issue_dir.mkdir(parents=True, exist_ok=True)
        pre_updates = int(run_meta.get("pre_refine_updates") or 0)

        # Run the shared group sequence, with a reduced path for external MCP servers.
        updates = pre_updates
        group_runs: List[Dict[str, Any]] = []
        effective_group_sequence = list(group_sequence)
        if node_type_name(node) == "External_MCP_server":
            # External MCP servers are refined for identity, IO, and execution/key requirements.
            effective_group_sequence = [
                ("group_1_identity", {"name", "description", "framework"}),
                ("group_2_io", {"inputs", "outputs"}),
                ("group_3_execution_keys", {"code_execution", "required_keys"}),
            ]
        for gkey, allowed in effective_group_sequence:
            g_prompt = GROUP_REFINEMENT_PROMPTS[gkey]
            g_dec_raw, _ = call_with_cache(
                payload={
                    "prompt": g_prompt,
                    "node_var": var,
                    "node_block": node,
                    "guidance_summary": guidance_summary,
                    "env_var_names": env_var_names if gkey == "group_3_execution_keys" else None,
                    "retrieved_context": None,
                },
                payload_path=issue_dir / f"refine_{gkey}.payload.json",
                raw_path=issue_dir / f"refine_{gkey}.txt",
                model=args.model,
                refresh_raw=bool(args.refresh_raw),
                retriever=retriever,
                rag_max_rounds=args.rag_max_rounds,
            )
            g_dec = g_dec_raw if isinstance(g_dec_raw, dict) else {}
            applied = apply_filtered_updates(
                node,
                parse_group_updates(g_dec, allowed_fields=allowed),
                allowed_fields=allowed,
            )
            updates += applied
            group_runs.append({"group": gkey, "applied_updates": applied})

        # Apply the type-specific refinement group after the shared groups.
        ntype = node_type_name(node)
        if ntype in per_type_group:
            tgkey, tallowed = per_type_group[ntype]
            t_dec_raw, _ = call_with_cache(
                payload={
                    "prompt": GROUP_REFINEMENT_PROMPTS[tgkey],
                    "node_var": var,
                    "node_block": node,
                    "guidance_summary": guidance_summary,
                    "retrieved_context": None,
                },
                payload_path=issue_dir / f"refine_{tgkey}.payload.json",
                raw_path=issue_dir / f"refine_{tgkey}.txt",
                model=args.model,
                refresh_raw=bool(args.refresh_raw),
                retriever=retriever,
                rag_max_rounds=args.rag_max_rounds,
            )
            t_dec = t_dec_raw if isinstance(t_dec_raw, dict) else {}
            t_applied = apply_filtered_updates(
                node,
                parse_group_updates(t_dec, allowed_fields=tallowed),
                allowed_fields=tallowed,
            )
            updates += t_applied
            group_runs.append({"group": tgkey, "applied_updates": t_applied})

        # Finish with metadata cleanup and then normalize structural defaults/connectivity.
        m_dec_raw, _ = call_with_cache(
            payload={
                "prompt": GROUP_REFINEMENT_PROMPTS[final_group[0]],
                "node_var": var,
                "node_block": node,
                "guidance_summary": guidance_summary,
                "retrieved_context": None,
            },
            payload_path=issue_dir / f"refine_{final_group[0]}.payload.json",
            raw_path=issue_dir / f"refine_{final_group[0]}.txt",
            model=args.model,
            refresh_raw=bool(args.refresh_raw),
            retriever=retriever,
            rag_max_rounds=args.rag_max_rounds,
        )
        m_dec = m_dec_raw if isinstance(m_dec_raw, dict) else {}
        m_applied = apply_filtered_updates(
            node,
            parse_group_updates(m_dec, allowed_fields=final_group[1]),
            allowed_fields=final_group[1],
        )
        updates += m_applied
        group_runs.append({"group": final_group[0], "applied_updates": m_applied})

        # Fill any missing type-specific required fields, then sanitize connectivity references before output.
        updates += _ensure_type_specific_fields(node)
        connectivity_updates = normalize_connectivity_fields(
            node,
            self_var=var,
            existing_vars=set(var_order),
            nodes_by_var=nodes_by_var,
        )
        updates += sum(int(v) for v in connectivity_updates.values())
        run_meta["group_runs"] = group_runs
        run_meta["connectivity_updates"] = connectivity_updates
        run_meta["final_type"] = node_type_name(node)
        run_meta["updated_fields_count"] = updates
        node_runs_by_var[var] = run_meta

        post_updates = max(0, updates - pre_updates)
        if post_updates > 0:
            touched_vars.append(var)
            total_updates += post_updates

    # Materialize the per-node run log in original processing order.
    node_runs: List[Dict[str, Any]] = [node_runs_by_var[v] for v in node_run_order if v in node_runs_by_var]

    # Build the stage report and write the refined graph output.
    report: Dict[str, Any] = {
        "mode": "node_refinement",
        "phase": args.phase,
        "input_nodes_file": Path(args.nodes_py).name,
        "output_nodes_file": Path(args.out).name,
        "total_nodes": len(var_order),
        "touched_nodes": len(set(touched_vars)),
        "field_updates": total_updates,
        "touched_vars": sorted(set(touched_vars)),
        "removed_nodes_count": len(removed_nodes),
        "removed_nodes": removed_nodes,
        "node_runs": node_runs,
    }

    write_nodes_output(
        out_path=out_py,
        var_order=var_order,
        nodes_by_var=nodes_by_var,
        report=report,
        report_var_name="NODE_REFINEMENT_REPORT",
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[node_refinement] wrote: {out_py}")
    print(f"[node_refinement] report: {report_path}")
    print(
        f"[node_refinement] phase={args.phase} touched={len(set(touched_vars))} "
        f"removed={len(removed_nodes)}"
    )


if __name__ == "__main__":
    main()
