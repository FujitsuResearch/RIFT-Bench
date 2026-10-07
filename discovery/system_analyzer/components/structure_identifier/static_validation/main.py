#!/usr/bin/env python3
"""Pipeline for static_validation component."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from ..global_utils import (
        copy_nodespec_schema_to_out_dir,
        load_env_file,
        write_nodes_output,
    )
    from .utils import (
        catalog_node_run_dir,
        finalize_static_validation,
        initialize_static_validation,
        list_pending_retry_catalog_vars,
        process_catalog_node,
        run_inline_child_and_connectivity_pass,
    )
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from global_utils import (
        copy_nodespec_schema_to_out_dir,
        load_env_file,
        write_nodes_output,
    )
    from static_validation.utils import (
        catalog_node_run_dir,
        finalize_static_validation,
        initialize_static_validation,
        list_pending_retry_catalog_vars,
        process_catalog_node,
        run_inline_child_and_connectivity_pass,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_file", default=".env", help="Optional .env file to load.")
    ap.add_argument("--out_dir", default="stages_outputs", help="Outputs directory.")
    ap.add_argument("--nodes_py", default="nodes_connected.py", help="Connected nodes file under --out_dir.")
    ap.add_argument(
        "--nodespec_per_file_nodes_py",
        "--catalog_nodes_py",
        dest="catalog_nodes_py",
        default="nodes_catalog.py",
        help="NodeSpec-per-file combined catalog nodes file under --out_dir.",
    )
    ap.add_argument("--root_json", default="root_nodes.json", help="Root json under --out_dir.")
    ap.add_argument("--out_main", default="nodes_main_graph.py", help="Main graph output under --out_dir.")
    ap.add_argument("--out_isolated", default="nodes_isolated.py", help="Isolated graph output under --out_dir.")
    ap.add_argument("--report_out", default="isolation_report.json", help="Report output under --out_dir.")
    ap.add_argument("--raw_dir", default="static_validation_raw", help="Raw folder under --out_dir.")
    ap.add_argument("--model", default=os.environ.get("CODEX_MODEL", "gpt-5.2-codex"))
    ap.add_argument("--rag_index_dir", default="rag_index", help="Path to FAISS index for retrieval.")
    ap.add_argument("--rag_max_rounds", type=int, default=8, help="Max context-request rounds per LLM call.")
    ap.add_argument("--refresh_raw", "--refresh-raw", dest="refresh_raw", action="store_true")
    args = ap.parse_args()

    # Load environment settings and prepare the output directory structure.
    load_env_file(Path(args.env_file))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_nodespec_schema_to_out_dir(out_dir)

    nodes_py = out_dir / Path(args.nodes_py).name
    catalog_nodes_py = out_dir / Path(args.catalog_nodes_py).name
    root_json = out_dir / Path(args.root_json).name
    out_main = out_dir / Path(args.out_main).name
    out_isolated = out_dir / Path(args.out_isolated).name
    report_path = out_dir / Path(args.report_out).name
    raw_dir = out_dir / Path(args.raw_dir).name
    rag_index_path = Path(args.rag_index_dir)
    rag_index_dir = rag_index_path.resolve() if rag_index_path.is_absolute() else (out_dir / rag_index_path).resolve()

    # Load the graph, catalog, retriever, and shared mutable pipeline state.
    state = initialize_static_validation(
        nodes_py=nodes_py,
        catalog_nodes_py=catalog_nodes_py,
        root_json=root_json,
        out_raw_dir=raw_dir,
        model=args.model,
        refresh_raw=bool(args.refresh_raw),
        rag_index_dir=rag_index_dir,
        rag_max_rounds=args.rag_max_rounds,
    )

    # Process the initial catalog pass and any later retry rounds through one shared loop.
    report = state["report"]
    retry_round = 0
    while True:
        is_retry = retry_round > 0
        round_catalog = state["catalog_order"] if not is_retry else list_pending_retry_catalog_vars(state)
        if not round_catalog:
            break

        round_added = 0
        round_represented = 0
        round_still_isolated = 0
        round_errors = 0
        round_log = {
            "round": retry_round,
            "candidates": len(round_catalog),
            "added": 0,
            "represented": 0,
            "still_isolated": 0,
            "errors": 0,
            "decisions": [],
        } if is_retry else None
        added_seed_vars: list[str] = []

        # Validate each catalog node for this round and collect any newly attached seeds.
        for index, catalog_var in enumerate(round_catalog, start=1):
            child_dir = catalog_node_run_dir(
                out_raw_dir=raw_dir,
                catalog_var=catalog_var,
                index=index,
                retry_round=retry_round if is_retry else None,
            )
            try:
                result = process_catalog_node(
                    state=state,
                    catalog_var=catalog_var,
                    child_dir=child_dir,
                    is_retry=is_retry,
                    retry_round=retry_round if is_retry else None,
                )
                decision = str(result.get("decision") or "isolated_node")
                added_seed_var = result.get("added_seed_var")
                if isinstance(added_seed_var, str) and added_seed_var:
                    added_seed_vars.append(added_seed_var)
                if is_retry and isinstance(round_log, dict):
                    round_log["decisions"].append({"catalog_var": catalog_var, "decision_class": decision})
                    if decision == "new_add_to_graph":
                        round_added += 1
                    elif decision == "exists_complete":
                        round_represented += 1
                    else:
                        round_still_isolated += 1
            except Exception as exc:
                if is_retry:
                    round_errors += 1
                    report["errors"].append({"catalog_var": catalog_var, "retry_round": retry_round, "error": str(exc)})
                else:
                    report["errors"].append({"catalog_var": catalog_var, "error": str(exc)})

        # Persist retry-round metrics after the round finishes.
        if is_retry and isinstance(round_log, dict):
            round_log["added"] = round_added
            round_log["represented"] = round_represented
            round_log["still_isolated"] = round_still_isolated
            round_log["errors"] = round_errors
            report["fixed_point_isolated_retries"].append(round_log)

        # Run child expansion plus connectivity repair for any nodes added in this round.
        if added_seed_vars:
            report["inline_child_expansion_passes"].append(
                run_inline_child_and_connectivity_pass(
                    state=state,
                    seed_vars=added_seed_vars,
                    pass_tag="initial_candidates_pass" if not is_retry else f"retry_round_{retry_round:02d}",
                )
            )

        if not is_retry:
            retry_round = 1
            continue

        if round_added == 0:
            break

        retry_round += 1

    # Finalize the main-vs-isolated partition and write all output artifacts.
    main_order, main_nodes, isolated_order, isolated_nodes, report = finalize_static_validation(state)

    write_nodes_output(
        out_path=out_main,
        var_order=main_order,
        nodes_by_var=main_nodes,
        report=report,
        report_var_name="STATIC_VALIDATION_REPORT",
    )
    write_nodes_output(
        out_path=out_isolated,
        var_order=isolated_order,
        nodes_by_var=isolated_nodes,
        report=report,
        report_var_name="STATIC_VALIDATION_REPORT",
    )
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[static_validation] wrote main: {out_main}")
    print(f"[static_validation] wrote isolated: {out_isolated}")
    print(f"[static_validation] wrote report: {report_path}")
    print(f"[static_validation] summary main={len(main_order)} isolated={len(isolated_order)}")


if __name__ == "__main__":
    main()
