#!/usr/bin/env python3
"""Pipeline for graph_correctness component."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
    from ..global_utils import (
        copy_nodespec_schema_to_out_dir,
        load_env_file,
        load_nodes_by_var,
        node_type_name,
        read_root_node_json,
        resolve_root_var,
        load_guidance_summary,
        write_nodes_output,
    )
    from .prompts import (
        AGENT_SYSTEM_SINGLE_PARENT_DECISION_PROMPT,
        AGENT_LLM_DECISION_PROMPT,
        AGENT_LLM_REUSE_PROMPT,
        AGENT_LLM_RESOLUTION_PROMPT,
        LLM_NO_CHILDREN_DECISION_PROMPT,
        MULTI_PARENT_DECISION_PROMPT,
        SINGLE_CHILD_DECISION_PROMPT,
        SYSTEM_NO_AGENT_DECISION_PROMPT,
    )
    from .utils import (
        collect_remaining_issues,
        run_rule_det_controller_promote_children,
        run_rule_orphan_no_parent,
        run_single_child_rule,
        run_rule_no_agent_in_agent,
        run_rule_agent_requires_llm,
        run_rule_system_requires_agent,
        run_rule_llm_no_children,
        run_rule_mcp_server_structure,
        run_rule_multi_parent,
    )
    from ..static_validation.prompts import (
        ADDING_NODE_PARENT_DECISION_PROMPT,
        MISSING_NODE_DECISION_PROMPT,
        REPRESENTATION_PAIR_PROMPT,
        REPRESENTATION_SHORTLIST_PROMPT,
    )
    from ..rag.build_retriever import load_retriever
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from global_utils import (
        copy_nodespec_schema_to_out_dir,
        load_env_file,
        load_nodes_by_var,
        node_type_name,
        read_root_node_json,
        resolve_root_var,
        load_guidance_summary,
        write_nodes_output,
    )
    from graph_correctness.prompts import (
        AGENT_SYSTEM_SINGLE_PARENT_DECISION_PROMPT,
        AGENT_LLM_DECISION_PROMPT,
        AGENT_LLM_REUSE_PROMPT,
        AGENT_LLM_RESOLUTION_PROMPT,
        LLM_NO_CHILDREN_DECISION_PROMPT,
        MULTI_PARENT_DECISION_PROMPT,
        SINGLE_CHILD_DECISION_PROMPT,
        SYSTEM_NO_AGENT_DECISION_PROMPT,
    )
    from graph_correctness.utils import (
        collect_remaining_issues,
        run_rule_det_controller_promote_children,
        run_rule_orphan_no_parent,
        run_single_child_rule,
        run_rule_no_agent_in_agent,
        run_rule_agent_requires_llm,
        run_rule_system_requires_agent,
        run_rule_llm_no_children,
        run_rule_mcp_server_structure,
        run_rule_multi_parent,
    )
    from static_validation.prompts import (
        ADDING_NODE_PARENT_DECISION_PROMPT,
        MISSING_NODE_DECISION_PROMPT,
        REPRESENTATION_PAIR_PROMPT,
        REPRESENTATION_SHORTLIST_PROMPT,
    )
    from rag.build_retriever import load_retriever


def main() -> None:

    # Parse CLI options for inputs, outputs, and rule behavior.
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_file", default=".env", help="Optional .env file to load.")
    ap.add_argument("--out_dir", default="stages_outputs", help="Output directory.")
    ap.add_argument("--nodes_py", default="nodes_refined.py", help="Input nodes file under --out_dir.")
    ap.add_argument("--root_json", default="root_nodes.json", help="Root json under --out_dir.")
    ap.add_argument("--out", default="nodes_graph_corrected.py", help="Output corrected nodes file under --out_dir.")
    ap.add_argument("--report_out", default="graph_correctness_report.json", help="Report output under --out_dir.")
    ap.add_argument("--raw_dir", default="graph_correctness_raw", help="Raw traces folder under --out_dir.")
    ap.add_argument("--phase", default="loop", help="Phase label for reporting.")
    ap.add_argument("--model", default=os.environ.get("CODEX_MODEL", "gpt-5.2-codex"))
    ap.add_argument(
        "--rag_index_dir",
        default="rag_index",
        help="Path to FAISS index for retrieval (relative paths resolve under --out_dir).",
    )
    ap.add_argument("--rag_max_rounds", type=int, default=8, help="Max context-request rounds per LLM call.")
    ap.add_argument("--max_rule_rounds", type=int, default=2, help="Max outer correctness rounds across Rules.")
    ap.add_argument(
        "--skip_det_controller_rule",
        action="store_true",
        help="Skip deterministic controller child-promotion rule (default: false).",
    )
    ap.add_argument(
        "--external_mcp_allow_node_children",
        action="store_true",
        help="Allow External_MCP_server to keep direct node children (default: false).",
    )

    ap.add_argument(
        "--external_mcp_require_tools",
        action="store_true",
        help="Require External_MCP_server to have at least one tool in tool_list (default: false).",
    )
    ap.add_argument("--refresh_raw", "--refresh-raw", dest="refresh_raw", action="store_true")
    args = ap.parse_args()

    # Resolve paths, load environment variables, and prepare output locations.
    load_env_file(Path(args.env_file))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_nodespec_schema_to_out_dir(out_dir)

    nodes_py = out_dir / Path(args.nodes_py).name
    root_json = out_dir / Path(args.root_json).name
    out_py = out_dir / Path(args.out).name
    report_path = out_dir / Path(args.report_out).name
    raw_dir = out_dir / Path(args.raw_dir).name
    raw_dir.mkdir(parents=True, exist_ok=True)

    rag_index_path = Path(args.rag_index_dir)
    rag_index_dir = rag_index_path.resolve() if rag_index_path.is_absolute() else (out_dir / rag_index_path).resolve()
    retriever = load_retriever(str(rag_index_dir))

    # Load the current graph, root node, and guidance summary.
    var_order, nodes_by_var = load_nodes_by_var(nodes_py)
    if not var_order:
        raise RuntimeError("graph_correctness: input nodes file has no ALL_NODES/MAIN_GRAPH_NODES.")

    root_node = read_root_node_json(root_json, allow_flat_object=True)
    root_var = resolve_root_var(root_node, nodes_by_var) or (var_order[0] if var_order else None)
    guidance_summary = load_guidance_summary(out_dir)

    # Initialize reporting containers and previously verified rule decisions.
    actions: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []
    verified_cache_path = out_dir / "graph_correctness_verified_rules.json"
    verified_cache: Dict[str, Any] = {}
    try:
        if verified_cache_path.exists():
            verified_cache = json.loads(verified_cache_path.read_text(encoding="utf-8"))
    except Exception:
        verified_cache = {}

    resolved_single_child_keep_distinct_pairs: Set[Tuple[str, str]] = set()
    raw_single_child = verified_cache.get("single_child_keep_distinct_pairs")
    if isinstance(raw_single_child, list):
        for item in raw_single_child:
            if (
                isinstance(item, list)
                and len(item) == 2
                and isinstance(item[0], str)
                and isinstance(item[1], str)
            ):
                resolved_single_child_keep_distinct_pairs.add((item[0], item[1]))

    resolved_multi_parent_sets: Dict[str, Tuple[str, ...]] = {}
    raw_multi_parent_sets = verified_cache.get("multi_parent_parent_sets")
    if isinstance(raw_multi_parent_sets, dict):
        for item_var, item_parents in raw_multi_parent_sets.items():
            if not isinstance(item_var, str):
                continue
            ntype = node_type_name(nodes_by_var.get(item_var, {}))
            if ntype in {"Agent", "System"}:
                continue
            if not isinstance(item_parents, list):
                continue
            normalized = tuple(
                sorted(
                    {
                        p
                        for p in item_parents
                        if isinstance(p, str)
                    }
                )
            )
            if normalized:
                resolved_multi_parent_sets[item_var] = normalized
    # Run the ordered correctness rules until the graph stabilizes or rounds are exhausted.
    max_rule_rounds = max(1, int(args.max_rule_rounds))
    executed_rule_rounds = 0
    for round_idx in range(1, max_rule_rounds + 1):
        executed_rule_rounds = round_idx

        # First repair deterministic-controller structure when enabled.
        if not bool(args.skip_det_controller_rule):
            run_rule_det_controller_promote_children(
                var_order=var_order,
                nodes_by_var=nodes_by_var,
                guidance_summary=guidance_summary,
                raw_dir=raw_dir / f"rules_round_{round_idx:02d}",
                model=args.model,
                refresh_raw=bool(args.refresh_raw),
                retriever=retriever,
                rag_max_rounds=args.rag_max_rounds,
                actions=actions,
                unresolved=unresolved,
            )

        # Then resolve orphan nodes before structural parent-child rules run.
        run_rule_orphan_no_parent(
            var_order=var_order,
            nodes_by_var=nodes_by_var,
            root_var=root_var,
            guidance_summary=guidance_summary,
            raw_dir=raw_dir / f"rules_round_{round_idx:02d}",
            model=args.model,
            refresh_raw=bool(args.refresh_raw),
            retriever=retriever,
            rag_max_rounds=args.rag_max_rounds,
            representation_shortlist_prompt=REPRESENTATION_SHORTLIST_PROMPT,
            representation_pair_prompt=REPRESENTATION_PAIR_PROMPT,
            missing_node_decision_prompt=MISSING_NODE_DECISION_PROMPT,
            adding_node_parent_decision_prompt=ADDING_NODE_PARENT_DECISION_PROMPT,
            actions=actions,
            unresolved=unresolved,
        )

        # Apply the single-child simplification pass before deeper structural checks.
        run_single_child_rule(
            var_order=var_order,
            nodes_by_var=nodes_by_var,
            root_var=root_var,
            guidance_summary=guidance_summary,
            raw_dir=raw_dir / f"rules_round_{round_idx:02d}",
            model=args.model,
            refresh_raw=bool(args.refresh_raw),
            retriever=retriever,
            rag_max_rounds=args.rag_max_rounds,
            single_child_prompt=SINGLE_CHILD_DECISION_PROMPT,
            actions=actions,
            unresolved=unresolved,
            run_suffix="single_child",
            resolved_keep_distinct_pairs=resolved_single_child_keep_distinct_pairs,
        )

        # Enforce LLM child restrictions and then validate higher-level containment rules.
        run_rule_llm_no_children(
            var_order=var_order,
            nodes_by_var=nodes_by_var,
            guidance_summary=guidance_summary,
            raw_dir=raw_dir / f"rules_round_{round_idx:02d}",
            model=args.model,
            refresh_raw=bool(args.refresh_raw),
            retriever=retriever,
            rag_max_rounds=args.rag_max_rounds,
            llm_no_children_prompt=LLM_NO_CHILDREN_DECISION_PROMPT,
            actions=actions,
            unresolved=unresolved,
        )

        # Fix invalid Agent-in-Agent containment.
        run_rule_no_agent_in_agent(
            var_order=var_order,
            nodes_by_var=nodes_by_var,
            root_var=root_var,
            guidance_summary=guidance_summary,
            raw_dir=raw_dir / f"rules_round_{round_idx:02d}",
            model=args.model,
            refresh_raw=bool(args.refresh_raw),
            retriever=retriever,
            rag_max_rounds=args.rag_max_rounds,
            single_child_prompt=SINGLE_CHILD_DECISION_PROMPT,
            actions=actions,
            unresolved=unresolved,
        )

        # Ensure each Agent has an LLM child.
        run_rule_agent_requires_llm(
            var_order=var_order,
            nodes_by_var=nodes_by_var,
            root_var=root_var,
            guidance_summary=guidance_summary,
            raw_dir=raw_dir / f"rules_round_{round_idx:02d}",
            model=args.model,
            refresh_raw=bool(args.refresh_raw),
            retriever=retriever,
            rag_max_rounds=args.rag_max_rounds,
            agent_llm_decision_prompt=AGENT_LLM_DECISION_PROMPT,
            agent_llm_reuse_prompt=AGENT_LLM_REUSE_PROMPT,
            agent_llm_resolution_prompt=AGENT_LLM_RESOLUTION_PROMPT,
            single_child_prompt=SINGLE_CHILD_DECISION_PROMPT,
            representation_shortlist_prompt=REPRESENTATION_SHORTLIST_PROMPT,
            representation_pair_prompt=REPRESENTATION_PAIR_PROMPT,
            missing_node_decision_prompt=MISSING_NODE_DECISION_PROMPT,
            adding_node_parent_decision_prompt=ADDING_NODE_PARENT_DECISION_PROMPT,
            actions=actions,
            unresolved=unresolved,
        )

        # Ensure each System has an Agent child.
        run_rule_system_requires_agent(
            var_order=var_order,
            nodes_by_var=nodes_by_var,
            root_var=root_var,
            guidance_summary=guidance_summary,
            raw_dir=raw_dir / f"rules_round_{round_idx:02d}",
            model=args.model,
            refresh_raw=bool(args.refresh_raw),
            retriever=retriever,
            rag_max_rounds=args.rag_max_rounds,
            system_no_agent_prompt=SYSTEM_NO_AGENT_DECISION_PROMPT,
            single_child_prompt=SINGLE_CHILD_DECISION_PROMPT,
            representation_shortlist_prompt=REPRESENTATION_SHORTLIST_PROMPT,
            representation_pair_prompt=REPRESENTATION_PAIR_PROMPT,
            missing_node_decision_prompt=MISSING_NODE_DECISION_PROMPT,
            adding_node_parent_decision_prompt=ADDING_NODE_PARENT_DECISION_PROMPT,
            actions=actions,
            unresolved=unresolved,
        )

        # Validate MCP server shape before the final single-child rerun and multi-parent cleanup.
        run_rule_mcp_server_structure(
            var_order=var_order,
            nodes_by_var=nodes_by_var,
            root_var=root_var,
            guidance_summary=guidance_summary,
            raw_dir=raw_dir / f"rules_round_{round_idx:02d}",
            model=args.model,
            refresh_raw=bool(args.refresh_raw),
            retriever=retriever,
            rag_max_rounds=args.rag_max_rounds,
            representation_shortlist_prompt=REPRESENTATION_SHORTLIST_PROMPT,
            representation_pair_prompt=REPRESENTATION_PAIR_PROMPT,
            missing_node_decision_prompt=MISSING_NODE_DECISION_PROMPT,
            adding_node_parent_decision_prompt=ADDING_NODE_PARENT_DECISION_PROMPT,
            actions=actions,
            unresolved=unresolved,
            external_allow_node_children=bool(args.external_mcp_allow_node_children),
            external_require_tools=bool(args.external_mcp_require_tools),
        )

        # Single-child rerun: recheck single-child cases after earlier mutations.
        run_single_child_rule(
            var_order=var_order,
            nodes_by_var=nodes_by_var,
            root_var=root_var,
            guidance_summary=guidance_summary,
            raw_dir=raw_dir / f"rules_round_{round_idx:02d}",
            model=args.model,
            refresh_raw=bool(args.refresh_raw),
            retriever=retriever,
            rag_max_rounds=args.rag_max_rounds,
            single_child_prompt=SINGLE_CHILD_DECISION_PROMPT,
            actions=actions,
            unresolved=unresolved,
            run_suffix="single_child_rerun",
            resolved_keep_distinct_pairs=resolved_single_child_keep_distinct_pairs,
        )

        # Resolve invalid multi-parent ownership.
        run_rule_multi_parent(
            var_order=var_order,
            nodes_by_var=nodes_by_var,
            root_var=root_var,
            guidance_summary=guidance_summary,
            raw_dir=raw_dir / f"rules_round_{round_idx:02d}",
            model=args.model,
            refresh_raw=bool(args.refresh_raw),
            retriever=retriever,
            rag_max_rounds=args.rag_max_rounds,
            multi_parent_prompt=MULTI_PARENT_DECISION_PROMPT,
            agent_system_single_parent_prompt=AGENT_SYSTEM_SINGLE_PARENT_DECISION_PROMPT,
            representation_shortlist_prompt=REPRESENTATION_SHORTLIST_PROMPT,
            representation_pair_prompt=REPRESENTATION_PAIR_PROMPT,
            missing_node_decision_prompt=MISSING_NODE_DECISION_PROMPT,
            adding_node_parent_decision_prompt=ADDING_NODE_PARENT_DECISION_PROMPT,
            actions=actions,
            unresolved=unresolved,
            resolved_parent_sets=resolved_multi_parent_sets,
        )

        # Stop early when no rule violations remain after this round.
        if not collect_remaining_issues(
            var_order=var_order,
            nodes_by_var=nodes_by_var,
            resolved_single_child_keep_distinct_pairs=resolved_single_child_keep_distinct_pairs,
            resolved_multi_parent_sets=resolved_multi_parent_sets,
            external_allow_node_children=bool(args.external_mcp_allow_node_children),
            external_require_tools=bool(args.external_mcp_require_tools),
            root_var=root_var,
        ):
            break

    # Recompute the final remaining issues and summarize them for the report.
    remaining_issues: List[Dict[str, Any]] = collect_remaining_issues(
        var_order=var_order,
        nodes_by_var=nodes_by_var,
        resolved_single_child_keep_distinct_pairs=resolved_single_child_keep_distinct_pairs,
        resolved_multi_parent_sets=resolved_multi_parent_sets,
        external_allow_node_children=bool(args.external_mcp_allow_node_children),
        external_require_tools=bool(args.external_mcp_require_tools),
        root_var=root_var,
    )
    remaining_counts: Dict[str, int] = {}
    for it in remaining_issues:
        key = str(it.get("rule") or "unknown")
        remaining_counts[key] = int(remaining_counts.get(key, 0)) + 1

    # Build the final report payload for the corrected graph output.
    report: Dict[str, Any] = {
        "mode": "graph_correctness",
        "mode_detail": "llm_rule_resolution",
        "phase": args.phase,
        "input_nodes_file": Path(args.nodes_py).name,
        "output_nodes_file": Path(args.out).name,
        "root_var": root_var,
        "total_nodes": len(var_order),
        "actions_applied": len(actions),
        "unresolved_count": len(unresolved),
        "max_rule_rounds": max_rule_rounds,
        "executed_rule_rounds": executed_rule_rounds,
        "resolved_single_child_keep_distinct_pairs_count": len(resolved_single_child_keep_distinct_pairs),
        "resolved_multi_parent_sets_count": len(resolved_multi_parent_sets),
        "remaining_issues_count": len(remaining_issues),
        "remaining_issue_counts": remaining_counts,
        "verified_issue_cache": {
            "path": str(verified_cache_path),
            "single_child_keep_distinct_pairs_count": len(
                resolved_single_child_keep_distinct_pairs
            ),
            "multi_parent_parent_sets_count": len(resolved_multi_parent_sets),
        },
        "actions": actions,
        "unresolved": unresolved,
        "remaining_issues": remaining_issues,
    }

    # Persist the verified-rule cache for reuse in later runs.
    verified_cache_out = {
        "phase_last_updated": args.phase,
        "single_child_keep_distinct_pairs": sorted(
            [[a, b] for (a, b) in resolved_single_child_keep_distinct_pairs],
            key=lambda x: (x[0], x[1]),
        ),
        "multi_parent_parent_sets": {
            k: list(v) for k, v in sorted(resolved_multi_parent_sets.items(), key=lambda x: x[0])
        },
    }
    verified_cache_path.write_text(
        json.dumps(verified_cache_out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Write the corrected nodes file and the JSON report to disk.
    write_nodes_output(
        out_path=out_py,
        var_order=var_order,
        nodes_by_var=nodes_by_var,
        report=report,
        report_var_name="GRAPH_CORRECTNESS_REPORT",
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[graph_correctness] wrote: {out_py}")
    print(f"[graph_correctness] report: {report_path}")
    print(f"[graph_correctness] actions={len(actions)} unresolved={len(unresolved)} remaining={len(remaining_issues)}")


if __name__ == "__main__":
    main()
