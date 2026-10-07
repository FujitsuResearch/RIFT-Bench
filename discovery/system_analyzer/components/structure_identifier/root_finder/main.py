#!/usr/bin/env python3
"""Pipeline for root_finder component."""

import argparse
import os
import sys
from pathlib import Path

try:
    from ..global_utils import load_env_file
    from ..rag.build_retriever import load_retriever
    from .utils import (
        align_candidate_to_list,
        align_root_to_catalog_node,
        build_nodes_file_map,
        call_compare_candidates,
        call_list_best_candidate,
        coerce_root_node_type,
        extract_compare_winner,
        extract_node_from_response,
        load_node_lists,
        sync_root_nodetype_to_outputs,
        write_root_node,
    )
except ImportError:
    PACKAGE_ROOT = Path(__file__).resolve().parents[1]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from global_utils import load_env_file
    from rag.build_retriever import load_retriever
    from root_finder.utils import (
        align_candidate_to_list,
        align_root_to_catalog_node,
        build_nodes_file_map,
        call_compare_candidates,
        call_list_best_candidate,
        coerce_root_node_type,
        extract_compare_winner,
        extract_node_from_response,
        load_node_lists,
        sync_root_nodetype_to_outputs,
        write_root_node,
    )


DEFAULT_ENTRY = "use_cases/use_case_1/agent_scripts/langraph_react_agent.py"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "stages_outputs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_file", default=".env", help="Optional .env file to load.")
    ap.add_argument("-f", "--entry", default=DEFAULT_ENTRY, type=str)
    ap.add_argument("--nodes_index", default="nodes_index.json")
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    ap.add_argument("--raw_dir", default="root_finder_raw")
    ap.add_argument("--model", default=os.environ.get("CODEX_MODEL", "gpt-5.2-codex"))
    ap.add_argument(
        "--rag_index_dir",
        default="rag_index",
        help="RAG index path (relative to out_dir if not absolute).",
    )
    ap.add_argument("--rag_max_rounds", type=int, default=6, help="Max context-request rounds per LLM call.")
    ap.add_argument("--refresh_raw", "--refresh-raw", dest="refresh_raw", action="store_true")
    args = ap.parse_args()

    # Prepare output locations and load prerequisite inputs.
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    load_env_file(Path(args.env_file))
    nodes_index_path = out_dir / Path(args.nodes_index).name
    nodes_file_map = build_nodes_file_map(nodes_index_path)
    node_lists = load_node_lists(nodes_file_map)

    # Load the retriever used during root-candidate selection.
    rag_index_path = Path(args.rag_index_dir)
    rag_index_dir = rag_index_path.resolve() if rag_index_path.is_absolute() else (out_dir / rag_index_path).resolve()
    retriever = load_retriever(str(rag_index_dir))

    # Prepare the raw artifact directory for model-call traces.
    root_raw_dir = out_dir / Path(args.raw_dir).name
    root_raw_dir.mkdir(parents=True, exist_ok=True)

    # Pick one best candidate per file, then compare winners across files.
    stored_candidate = None
    compare_idx = 0
    for list_idx, (source_path, nodes) in enumerate(node_lists, start=1):

        # Select the best local root candidate from the current file's node list.
        if not nodes:
            continue
        list_response = call_list_best_candidate(
            source_path=source_path,
            nodes=nodes,
            model=args.model,
            out_raw_dir=root_raw_dir,
            refresh_raw=bool(args.refresh_raw),
            retriever=retriever,
            rag_max_rounds=args.rag_max_rounds,
            idx=list_idx,
        )
        list_candidate = extract_node_from_response(list_response, "selected_node", "root_node")
        list_candidate = align_candidate_to_list(list_candidate, nodes)
        if not isinstance(list_candidate, dict):
            continue
        list_candidate = coerce_root_node_type(list_candidate)
        if stored_candidate is None:
            stored_candidate = list_candidate
            continue

        # Compare this file's winner against the running global winner.
        compare_idx += 1
        compare_response = call_compare_candidates(
            stored_candidate=stored_candidate,
            contender_candidate=list_candidate,
            model=args.model,
            out_raw_dir=root_raw_dir,
            refresh_raw=bool(args.refresh_raw),
            retriever=retriever,
            rag_max_rounds=args.rag_max_rounds,
            idx=compare_idx,
        )
        stored_candidate = coerce_root_node_type(
            extract_compare_winner(compare_response, stored_candidate, list_candidate)
        )

    # Persist the final winner and sync its chosen root type into outputs.
    if stored_candidate is None:
        raise RuntimeError("root_finder: no root candidate found from node lists.")

    root_node = coerce_root_node_type(stored_candidate)
    root_node = align_root_to_catalog_node(out_dir=out_dir, root_node=root_node)
    sync_root_nodetype_to_outputs(
        out_dir=out_dir,
        root_node=root_node,
        nodes_file_map=nodes_file_map,
    )
    write_root_node(out_dir, root_node)


if __name__ == "__main__":
    main()
