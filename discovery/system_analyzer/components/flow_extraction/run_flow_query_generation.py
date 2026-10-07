from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .flow_query_generation import (
        run_flow_query_generation,
        build_flow_counts_summary_from_existing,
    )
except ImportError:
    from flow_extraction.flow_query_generation import (
        run_flow_query_generation,
        build_flow_counts_summary_from_existing,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_dir", required=True, help="Use-case outputs directory (contains validation/traces).")
    ap.add_argument(
        "--model",
        default=os.environ.get("AZURE_OPENAI_DEPLOYMENT_SI", ""),
        help="LLM model id used by modular_imp/model_client.py",
    )
    ap.add_argument("--traces_dir", default="", help="Optional traces directory override.")
    ap.add_argument("--nodespec_file", default="", help="Optional nodespec file override.")
    ap.add_argument("--runtime_mapping_file", default="", help="Optional runtime mapping file override.")
    ap.add_argument("--query_timeout_sec", type=int, default=180, help="Timeout per generated query trace run.")
    ap.add_argument("--refresh_raw", "--refresh-raw", dest="refresh_raw", action="store_true")
    ap.add_argument("--skip_trace_execution", action="store_true", help="Do not execute generated queries to produce traces.")
    ap.add_argument(
        "--known_queries_per_flow",
        type=int,
        default=5,
        help="Queries to generate per flow (used for both known and unseen-target flows).",
    )
    ap.add_argument(
        "--report_only_from_existing",
        action="store_true",
        help="Only compute and save flow_counts_summary.json from existing artifacts; no new LLM calls or executions.",
    )
    args = ap.parse_args()

    if bool(args.report_only_from_existing):
        report_file = build_flow_counts_summary_from_existing(
            outputs_dir=Path(args.outputs_dir),
            nodespec_file=Path(args.nodespec_file) if str(args.nodespec_file).strip() else None,
            runtime_mapping_file=Path(args.runtime_mapping_file) if str(args.runtime_mapping_file).strip() else None,
        )
        print(f"[flow-query-gen] flow_counts_summary={report_file}")
        return

    res = run_flow_query_generation(
        outputs_dir=Path(args.outputs_dir),
        model=str(args.model),
        traces_dir=Path(args.traces_dir) if str(args.traces_dir).strip() else None,
        nodespec_file=Path(args.nodespec_file) if str(args.nodespec_file).strip() else None,
        runtime_mapping_file=Path(args.runtime_mapping_file) if str(args.runtime_mapping_file).strip() else None,
        known_queries_per_flow=max(1, int(args.known_queries_per_flow)),
        execute_generated_queries=not bool(args.skip_trace_execution),
        query_timeout_sec=max(30, int(args.query_timeout_sec)),
        refresh_raw=bool(args.refresh_raw),
    )

    print(f"[flow-query-gen] out_dir={res.out_dir}")
    print(f"[flow-query-gen] flow_catalog={res.flow_catalog_file}")
    print(f"[flow-query-gen] known_flow_queries={res.known_queries_file}")
    print(f"[flow-query-gen] unseen_flow_targets={res.unseen_targets_file}")
    print(f"[flow-query-gen] unseen_flow_queries={res.unseen_queries_file}")
    print(f"[flow-query-gen] flow_counts_summary={res.flow_counts_summary_file}")


if __name__ == "__main__":
    main()
