"""Run vulnerability scanning from a NodeSpec and an original or emulated code ZIP."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
AGENTIC_SYSTEMS_DIR = PROJECT_ROOT / "agentic_systems"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments"


def _resolve_system_paths(system_name: str) -> tuple[Path, Path, str, str]:
    normalized_name = system_name.replace("\\", "/").strip("/")
    if not normalized_name:
        raise ValueError("--system-name must not be empty.")

    candidates = [AGENTIC_SYSTEMS_DIR / normalized_name]
    if not normalized_name.startswith("domain_systems/"):
        candidates.append(AGENTIC_SYSTEMS_DIR / "domain_systems" / normalized_name)
    selected_path = next((path for path in candidates if path.is_dir()), candidates[0])
    if not selected_path.is_dir():
        raise FileNotFoundError(f"System path not found: {selected_path}")

    nested_repository = selected_path / selected_path.name
    if nested_repository.is_dir():
        system_root, repository = selected_path, nested_repository
    elif selected_path.parent.name == selected_path.name:
        system_root, repository = selected_path.parent, selected_path
    else:
        system_root = repository = selected_path

    relative_path = system_root.relative_to(AGENTIC_SYSTEMS_DIR).as_posix()
    return system_root, repository, relative_path, relative_path.replace("/", "__")


def _remove_image(image_tag: str) -> None:
    result = subprocess.run(
        ["docker", "image", "rm", "-f", image_tag],
        capture_output=True,
        text=True,
    )
    if result.returncode and "no such image" not in result.stderr.lower():
        raise RuntimeError(f"Could not remove Docker image {image_tag}: {result.stderr}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scanning from a NodeSpec and an original or post-emulation code ZIP."
    )
    parser.add_argument("--user-id", "--user_id", required=True, dest="user_id")
    parser.add_argument(
        "--system-name",
        "--system_name",
        required=True,
        dest="system_name",
        help="Path below agentic_systems (domain_systems/ is optional).",
    )
    parser.add_argument(
        "--node-spec-path",
        type=Path,
        help="NodeSpec JSON. Defaults to <system>_after_discovery.json.",
    )
    parser.add_argument(
        "--codebase-zip-path",
        type=Path,
        required=True,
        help="Original or post-emulation source ZIP used to build the scan image.",
    )
    parser.add_argument("--env-file-path", type=Path)
    parser.add_argument("--scan-results-dir", type=Path)
    parser.add_argument(
        "--scan-timeout-sec",
        "--scan_timeout_sec",
        type=int,
        default=600,
        dest="scan_timeout_sec",
    )
    parser.add_argument(
        "--flows-per-probe",
        "--flows_per_probe",
        type=int,
        default=3,
        dest="flows_per_probe",
    )
    parser.add_argument("--apply-defenses", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    for option in (
        "node_spec_path",
        "codebase_zip_path",
        "env_file_path",
        "scan_results_dir",
    ):
        value = getattr(args, option)
        if value is not None:
            setattr(args, option, value.resolve())

    if args.flows_per_probe <= 0:
        raise ValueError("--flows-per-probe must be a positive integer.")
    if args.scan_timeout_sec is not None and args.scan_timeout_sec <= 0:
        raise ValueError("--scan-timeout-sec must be a positive integer.")

    os.chdir(PROJECT_ROOT)
    load_dotenv(PROJECT_ROOT / "rift.env")
    system_root, repository, _, system_key = _resolve_system_paths(args.system_name)
    args.node_spec_path = args.node_spec_path or (
        system_root / f"{system_root.name}_after_discovery.json"
    )
    args.env_file_path = args.env_file_path or next(
        (
            path
            for path in (
                system_root / f"{system_root.name}.env",
                repository / f"{repository.name}.env",
            )
            if path.exists()
        ),
        system_root / f"{system_root.name}.env",
    )
    args.scan_results_dir = args.scan_results_dir or (
        DEFAULT_OUTPUT_DIR / f"{system_key}_scan_results"
    )

    for path, label in (
        (args.node_spec_path, "NodeSpec"),
        (args.codebase_zip_path, "codebase ZIP"),
        (args.env_file_path, "environment file"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label.capitalize()} not found: {path}")
    args.scan_results_dir.mkdir(parents=True, exist_ok=True)

    from environment_handler.runtime import prepare_environment
    from node_spec.structure_schema import NodeSpec
    from scanning.main import main as run_scanning_pipeline

    node_spec = NodeSpec.from_json(path=str(args.node_spec_path))
    handler = prepare_environment(
        user_id=args.user_id,
        system_name=system_key,
        envfile_path=str(args.env_file_path),
        zipfile_path=str(args.codebase_zip_path),
        execution_command={},
    )
    scan_image_tag = f"{args.user_id}_{node_spec.name.lower()}:latest"
    try:
        subprocess.run(["docker", "tag", handler.image_tag, scan_image_tag], check=True)
        run_scanning_pipeline(node_spec, args)
    finally:
        try:
            _remove_image(scan_image_tag)
        finally:
            handler.remove_image(force=True)

    print(f"Scanning completed. Results saved to: {args.scan_results_dir}")


if __name__ == "__main__":
    main()
