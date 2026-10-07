import argparse
import json
import os
from pathlib import Path
from typing import List

# Ensure a user agent is set before any network-bound imports execute
os.environ.setdefault(
    "USER_AGENT",
    "hierarchical-research-app/0.1 (+https://example.com; contact@example.com)",
)

from hierarchical_research_app_script import run_query_with_mlflow, setup_mlflow_logging


def _load_queries(path: Path) -> List[str]:
    with path.open() as f:
        data = json.load(f)
    if isinstance(data, dict) and "queries" in data:
        queries = data["queries"]
    elif isinstance(data, list):
        queries = data
    else:
        raise ValueError("JSON must be a list of queries or a dict with a 'queries' key.")
    return [str(q) for q in queries]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the hierarchical research app for a single query or a list of queries."
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to run. Use either --query or --queries-file.",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        help="Path to a JSON file containing either a list of queries or {'queries': [...]}",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=150,
        help="LangGraph recursion limit.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (defaults to env MLFLOW_TRACKING_URI or file://<repo>/Research_Orchestrator/mlruns).",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="hierarchical_research",
        help="MLflow experiment name.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if bool(args.query) == bool(args.queries_file):
        raise SystemExit("Please provide either --query or --queries-file (but not both).")

    mlflow_client = setup_mlflow_logging(
        experiment_name=args.mlflow_experiment, tracking_uri=args.mlflow_tracking_uri
    )

    if args.query:
        answer = run_query_with_mlflow(
            args.query, recursion_limit=args.recursion_limit
        )
        print(answer)
        return

    if not args.queries_file.exists():
        raise SystemExit(f"Queries file not found: {args.queries_file}")
    queries = _load_queries(args.queries_file)
    answers = [
        run_query_with_mlflow(
            query, recursion_limit=args.recursion_limit
        )
        for query in queries
    ]
    print(json.dumps(answers, indent=2))


if __name__ == "__main__":
    main()
