#!/usr/bin/env python3
"""Pipeline for summary_creator component."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

try:
    from ..global_utils import (
        call_with_cache,
        load_env_file,
        load_nodes_by_var,
        normalize_code_references_add,
        read_root_node_json,
        resolve_root_var,
        unique_json_items,
    )
    from ..rag.build_retriever import load_retriever
    from .prompts import SYSTEM_GUIDANCE_SUMMARY_PROMPT
except ImportError:
    PACKAGE_ROOT = Path(__file__).resolve().parents[1]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from global_utils import (
        call_with_cache,
        load_env_file,
        load_nodes_by_var,
        normalize_code_references_add,
        read_root_node_json,
        resolve_root_var,
        unique_json_items,
    )
    from rag.build_retriever import load_retriever
    from summary_creator.prompts import SYSTEM_GUIDANCE_SUMMARY_PROMPT


DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "stages_outputs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_file", default=".env", help="Optional .env file to load.")
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR), help="Output directory.")
    ap.add_argument("--root_nodes_file", default="root_nodes.json")
    ap.add_argument("--nodes_file", default="nodes_catalog.py")
    ap.add_argument("--raw_dir", default="guidance_raw")
    ap.add_argument("--model", default=os.environ.get("CODEX_MODEL", "gpt-5.2-codex"))
    ap.add_argument(
        "--rag_index_dir",
        default="rag_index",
        help="RAG index path (relative to out_dir if not absolute).",
    )
    ap.add_argument("--rag_max_rounds", type=int, default=6, help="Max context-request rounds per LLM call.")
    ap.add_argument("--refresh_raw", "--refresh-raw", dest="refresh_raw", action="store_true")
    args = ap.parse_args()

    # Prepare output paths and load required stage inputs.
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the retriever used to enrich the single-pass guidance generation.
    load_env_file(Path(args.env_file))
    rag_index_path = Path(args.rag_index_dir)
    rag_index_dir = rag_index_path.resolve() if rag_index_path.is_absolute() else (out_dir / rag_index_path).resolve()
    retriever = load_retriever(str(rag_index_dir))

    # Resolve the stage artifacts and locate the chosen root node in the catalog.
    root_json = out_dir / Path(args.root_nodes_file).name
    nodes_py = out_dir / Path(args.nodes_file).name
    raw_dir = out_dir / Path(args.raw_dir).name
    raw_dir.mkdir(parents=True, exist_ok=True)

    root_node = read_root_node_json(root_json)

    _, nodes_by_var = load_nodes_by_var(nodes_py)
    root_var = resolve_root_var(root_node, nodes_by_var)
    if root_var is None:
        raise RuntimeError("summary_creator: could not map root node to nodes list.")

    # Generate the guidance summary and normalize any added evidence references.
    refs = list(root_node.get("code_references")) if isinstance(root_node.get("code_references"), list) else []
    payload: Dict[str, Any] = {
        "prompt": SYSTEM_GUIDANCE_SUMMARY_PROMPT,
        "code_references": refs,
    }
    parsed, guidance_source = call_with_cache(
        payload,
        payload_path=raw_dir / "guidance_summary.hop1.payload.json",
        raw_path=raw_dir / "guidance_summary.hop1.txt",
        model=args.model,
        refresh_raw=bool(args.refresh_raw),
        retriever=retriever,
        rag_max_rounds=args.rag_max_rounds,
        use_first_round_context_request_hint=True,
    )
    if not isinstance(parsed, dict):
        parsed = {}
    if not isinstance(parsed.get("summary"), str):
        parsed["summary"] = ""

    refs_add = normalize_code_references_add(parsed.get("code_references_add"))
    if refs_add:
        refs = unique_json_items([*refs, *refs_add])

    guidance_obj: Dict[str, Any] = {"summary": str(parsed.get("summary") or "")}
    if refs_add:
        guidance_obj["code_references_add"] = refs_add

    # Persist the final guidance artifact for downstream stages.
    guidance_out = {
        "root_var": root_var,
        "guidance": guidance_obj,
        "source": guidance_source,
        "guidance_meta": {
            "mode": "single_pass_rag_guidance",
            "code_references_final_count": len(refs),
            "code_references_added": len(refs_add),
        },
    }
    (out_dir / "system_guidance.json").write_text(
        json.dumps(guidance_out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
