#!/usr/bin/env python3
"""Pipeline for connectivity_pass component."""

import argparse
import os
from collections import deque
from pathlib import Path
from typing import Any, Dict, Set

try:
    from .utils import process_parent_connectivity
    from ..global_utils import (
        copy_nodespec_schema_to_out_dir,
        load_env_file,
        load_nodes_by_var,
        read_json,
        read_root_node_json,
        resolve_root_var,
        safe_name,
        write_nodes_output,
    )
    from ..rag.build_retriever import load_retriever
except ImportError:
    from connectivity_pass.utils import process_parent_connectivity
    from global_utils import (
        copy_nodespec_schema_to_out_dir,
        load_env_file,
        load_nodes_by_var,
        read_json,
        read_root_node_json,
        resolve_root_var,
        safe_name,
        write_nodes_output,
    )
    from rag.build_retriever import load_retriever


def main() -> None:
    # Parse CLI arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_file", default=".env", help="Optional .env file to load.")
    ap.add_argument("--out_dir", default="stages_outputs", help="Output directory.")
    ap.add_argument("--nodes_py", default="nodes_with_children.py", help="Input nodes file under --out_dir.")
    ap.add_argument("--root_json", default="root_nodes.json", help="Root json under --out_dir.")
    ap.add_argument("--out", default="nodes_connected.py", help="Output connected nodes file under --out_dir.")
    ap.add_argument("--raw_dir", default="connectivity_raw", help="Raw traces folder under --out_dir.")
    ap.add_argument("--model", default=os.environ.get("CODEX_MODEL", "gpt-5.2-codex"))
    ap.add_argument("--rag_index_dir", default="rag_index", help="Path to FAISS index for retrieval.")
    ap.add_argument("--rag_max_rounds", type=int, default=6, help="Max retrieval rounds per LLM call.")
    ap.add_argument("--max_iterations", type=int, default=5, help="Max per-parent connectivity-repair iterations.")
    ap.add_argument("--refresh_raw", "--refresh-raw", dest="refresh_raw", action="store_true")
    args = ap.parse_args()

    # Resolve inputs and working directories.
    load_env_file(Path(args.env_file))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_nodespec_schema_to_out_dir(out_dir)

    nodes_py = out_dir / Path(args.nodes_py).name
    root_json = out_dir / Path(args.root_json).name
    out_py = out_dir / Path(args.out).name
    raw_dir = out_dir / Path(args.raw_dir).name
    raw_dir.mkdir(parents=True, exist_ok=True)

    rag_index_path = Path(args.rag_index_dir)
    rag_index_dir = rag_index_path.resolve() if rag_index_path.is_absolute() else (out_dir / rag_index_path).resolve()
    retriever = load_retriever(str(rag_index_dir))

    # Load graph inputs and resolve the root node.
    var_order, nodes_by_var = load_nodes_by_var(nodes_py)
    if not var_order or not nodes_by_var:
        raise RuntimeError("connectivity_pass: input nodes file has no ALL_NODES/MAIN_GRAPH_NODES.")

    root_node = read_root_node_json(root_json)
    root_var = resolve_root_var(root_node, nodes_by_var)
    if not root_var:
        raise RuntimeError("connectivity_pass: failed resolving root var from root_json.")

    guidance_json = out_dir / "system_guidance.json"
    guidance_obj = read_json(guidance_json) if guidance_json.exists() else {}
    guidance = guidance_obj.get("guidance") if isinstance(guidance_obj.get("guidance"), dict) else {}
    guidance_summary = str(guidance.get("summary") or "")

    # Initialize traversal state and run report.
    report: Dict[str, Any] = {
        "mode": "connectivity_pass",
        "root_var": root_var,
        "parents_processed": 0,
        "edges_added": 0,
        "edges_removed": 0,
        "post_cleanup": {
            "detached_nodes_count": 0,
            "orphan_deleted_count": 0,
        },
        "parent_runs": [],
        "errors": [],
    }
    q: deque[str] = deque([root_var])
    processed: Set[str] = set()

    # Traverse parents breadth-first and process each one.
    while q:
        # Skip parents already handled or removed during cleanup.
        parent_var = q.popleft()
        if parent_var in processed or parent_var not in nodes_by_var:
            continue
        parent_node = nodes_by_var[parent_var]
        processed.add(parent_var)
        report["parents_processed"] += 1

        # Create the per-parent raw trace directory.
        parent_raw_dir = raw_dir / "pass_dynamic" / safe_name(parent_var)
        parent_raw_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Run repair and cleanup for the current parent node.
            parent_result = process_parent_connectivity(
                parent_var=parent_var,
                parent_node=parent_node,
                nodes_by_var=nodes_by_var,
                var_order=var_order,
                root_var=root_var,
                guidance_summary=guidance_summary,
                retriever=retriever,
                model=args.model,
                raw_parent_dir=parent_raw_dir,
                refresh_raw=args.refresh_raw,
                rag_max_rounds=args.rag_max_rounds,
                max_iterations=args.max_iterations,
            )
            conn_meta = parent_result.get("connectivity") if isinstance(parent_result.get("connectivity"), dict) else {}
            detached_nodes = parent_result.get("detached_nodes") if isinstance(parent_result.get("detached_nodes"), list) else []
            orphan_cleanup = parent_result.get("orphan_cleanup") if isinstance(parent_result.get("orphan_cleanup"), dict) else {}

            # Accumulate connectivity changes into the run report.
            report["edges_added"] += int(conn_meta.get("edges_added") or 0)
            report["edges_removed"] += int(conn_meta.get("edges_removed") or 0)

            if bool(conn_meta.get("unresolved")):
                report["errors"].append(
                    {
                        "parent_var": parent_var,
                        "error": "unresolved_connectivity_after_iterations",
                        "final_issues": conn_meta.get("final_issues"),
                        "rejected_edges": conn_meta.get("rejected_edges"),
                    }
                )

            # Accumulate detached-node and orphan-delete cleanup counts.
            post_cleanup = report.get("post_cleanup")
            if isinstance(post_cleanup, dict):
                post_cleanup["detached_nodes_count"] = int(post_cleanup.get("detached_nodes_count") or 0) + len(detached_nodes)
                post_cleanup["orphan_deleted_count"] = int(post_cleanup.get("orphan_deleted_count") or 0) + int(orphan_cleanup.get("deleted_count") or 0)

            # Record the per-parent result and queue remaining children.
            report["parent_runs"].append(
                {
                    "parent_var": parent_var,
                    "connectivity": conn_meta,
                    "nodes_detached_after_repair": detached_nodes,
                    "orphan_cleanup": orphan_cleanup,
                }
            )

            child_vars_to_queue = parent_result.get("child_vars_to_queue") if isinstance(parent_result.get("child_vars_to_queue"), list) else []
            for child_var in child_vars_to_queue:
                if child_var not in processed:
                    q.append(child_var)
        except Exception as exc:
            report["errors"].append({"parent_var": parent_var, "error": str(exc)})

    # Persist the updated graph and report.
    write_nodes_output(
        out_path=out_py,
        var_order=var_order,
        nodes_by_var=nodes_by_var,
        report=report,
        report_var_name="CONNECTIVITY_PASS_REPORT",
    )

    # Print the final pipeline summary.
    print(f"[connectivity_pass] wrote: {out_py}")
    print(
        f"[connectivity_pass] "
        f"nodes={len(var_order)} edges_added={report.get('edges_added')}"
    )


if __name__ == "__main__":
    main()
