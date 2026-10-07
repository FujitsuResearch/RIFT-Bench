#!/usr/bin/env python3
"""Pipeline for child_creation component."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    from . import utils as child_creation_utils
    from ..global_utils import (
        copy_nodespec_schema_to_out_dir,
        load_env_file,
        read_json,
        read_root_node_json,
        write_nodes_output,
    )
    from ..rag.build_retriever import load_retriever
except ImportError:
    PACKAGE_ROOT = Path(__file__).resolve().parents[1]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from child_creation import utils as child_creation_utils
    from global_utils import (
        copy_nodespec_schema_to_out_dir,
        load_env_file,
        read_json,
        read_root_node_json,
        write_nodes_output,
    )
    from rag.build_retriever import load_retriever


DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "stages_outputs"
DEFAULT_SCHEMA_SHORT = Path(__file__).resolve().parents[1] / "NodeSpec_schema_short.py"


def main() -> None:

    # Parse CLI arguments for inputs, outputs, retrieval settings, and refresh behavior.
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_file", default=".env", help="Optional .env file to load.")
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    ap.add_argument("--root_json", default="root_nodes.json", help="Root json under --out_dir.")
    ap.add_argument("--out", default="nodes_with_children.py", help="Output child-created nodes file under --out_dir.")
    ap.add_argument("--raw_dir", default="child_creation_raw", help="Raw traces folder under --out_dir.")
    ap.add_argument("--model", default=os.environ.get("CODEX_MODEL", "gpt-5.2-codex"))
    ap.add_argument("--rag_index_dir", default="rag_index", help="RAG index path (relative to out_dir if not absolute).")
    ap.add_argument("--rag_max_rounds", type=int, default=6, help="Max retrieval rounds per LLM call.")
    ap.add_argument("--phase_a_max_rounds", type=int, default=8, help="Max outer rounds for child delta discovery.")
    ap.add_argument("--phase_a_no_change_patience", type=int, default=2, help="Stop child discovery after consecutive no-change rounds.")
    ap.add_argument(
        "--use_first_round_context_request_hint_discovery",
        dest="use_first_round_context_request_hint_discovery",
        action="store_true",
        help="In child discovery phase, force round 1 to request context when evidence is incomplete (default: enabled).",
    )
    ap.add_argument(
        "--no_use_first_round_context_request_hint_discovery",
        dest="use_first_round_context_request_hint_discovery",
        action="store_false",
        help="Disable round-1 forced context request hint in child discovery.",
    )
    ap.set_defaults(use_first_round_context_request_hint_discovery=True)
    ap.add_argument("--refresh_raw", "--refresh-raw", dest="refresh_raw", action="store_true")
    args = ap.parse_args()

    # Initialize the output workspace and load optional environment variables.
    load_env_file(Path(args.env_file))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    copy_nodespec_schema_to_out_dir(out_dir)

    # Resolve the main input/output paths for root input, generated nodes, and raw traces.
    root_json = out_dir / Path(args.root_json).name
    out_py = out_dir / Path(args.out).name
    raw_dir = out_dir / Path(args.raw_dir).name
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Load the retriever used for retrieval-backed child discovery and materialization.
    rag_index_path = Path(args.rag_index_dir)
    rag_index_dir = rag_index_path.resolve() if rag_index_path.is_absolute() else (out_dir / rag_index_path).resolve()
    retriever = load_retriever(str(rag_index_dir))

    # Load the root node, guidance summary, and short schema text used by the child-creation prompts.
    root_node = read_root_node_json(root_json)
    if not root_node:
        raise RuntimeError("child_creation: root_json did not contain a root node.")

    guidance_json = out_dir / "system_guidance.json"
    guidance_obj = read_json(guidance_json) if guidance_json.exists() else {}
    guidance = guidance_obj.get("guidance") if isinstance(guidance_obj.get("guidance"), dict) else {}
    guidance_summary = str(guidance.get("summary") or "")
    schema_catalog_path = DEFAULT_SCHEMA_SHORT.resolve()
    nodespec_schema_catalog = (
        schema_catalog_path.read_text(encoding="utf-8", errors="replace")
        if schema_catalog_path.exists()
        else ""
    )

    # Build the initial in-memory graph state from the single root node.
    root_var = child_creation_utils.next_var(str(root_node.get("name") or "root"), set())
    root_node_working = dict(root_node)
    root_node_working.setdefault("tool_list", [])
    root_node_working.setdefault("nodes", [])
    root_node_working.setdefault("internal_edges", [])

    var_order: List[str] = [root_var]
    nodes_by_var: Dict[str, Dict[str, Any]] = {root_var: root_node_working}

    # Expand the graph downward from the root by repeatedly discovering and attaching child nodes.
    expansion = child_creation_utils.expand_children_from_parent(
        start_parent_var=root_var,
        nodes_by_var=nodes_by_var,
        var_order=var_order,
        guidance_summary=guidance_summary,
        nodespec_schema_catalog=nodespec_schema_catalog,
        retriever=retriever,
        model=args.model,
        raw_root_dir=raw_dir,
        refresh_raw=bool(args.refresh_raw),
        rag_max_rounds=args.rag_max_rounds,
        phase_a_max_rounds=args.phase_a_max_rounds,
        phase_a_no_change_patience=args.phase_a_no_change_patience,
        use_first_round_context_request_hint_discovery=bool(args.use_first_round_context_request_hint_discovery),
    )

    # Build the stage report and write the final child-expanded graph output.
    report: Dict[str, Any] = {
        "mode": "child_creation",
        "root_var": root_var,
        "seed_vars": [root_var],
        "created_nodes": int(expansion.get("created_nodes") or 0),
        "parents_processed": int(expansion.get("parents_processed") or 0),
        "parent_runs": expansion.get("parent_runs") if isinstance(expansion.get("parent_runs"), list) else [],
        "errors": expansion.get("errors") if isinstance(expansion.get("errors"), list) else [],
    }

    write_nodes_output(
        out_path=out_py,
        var_order=var_order,
        nodes_by_var=nodes_by_var,
        report=report,
        report_var_name="CHILD_CREATION_REPORT",
    )

    print(f"[child_creation] wrote: {out_py}")
    print(f"[child_creation] nodes={len(var_order)} created={report.get('created_nodes')}")


if __name__ == "__main__":
    main()
