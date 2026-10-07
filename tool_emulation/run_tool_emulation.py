"""Emulate tools and export the resulting NodeSpec and sandbox code archive."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
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


def _ensure_default_node_spec(repository: Path) -> Path:
    named_node_specs = sorted(
        path
        for path in repository.glob("*_spec.json")
        if path.name != "final_spec_gt_spec.json"
    )
    if len(named_node_specs) == 1:
        return named_node_specs[0]
    if len(named_node_specs) > 1:
        names = ", ".join(path.name for path in named_node_specs)
        raise FileNotFoundError(f"Multiple project NodeSpecs found in {repository}: {names}")

    final_node_spec = repository / "final_spec_gt_spec.json"
    if final_node_spec.is_file():
        return final_node_spec

    named_generators = sorted(
        path
        for path in repository.glob("*_spec.py")
        if path.name != "final_spec_gt_spec.py"
    )
    if len(named_generators) > 1:
        names = ", ".join(path.name for path in named_generators)
        raise FileNotFoundError(f"Multiple project NodeSpec generators found in {repository}: {names}")
    generator = named_generators[0] if named_generators else repository / "final_spec_gt_spec.py"
    if generator.is_file():
        node_spec_path = generator.with_suffix(".json")
        subprocess.run([sys.executable, str(generator)], cwd=repository, check=True)
        if node_spec_path.is_file():
            return node_spec_path
        raise FileNotFoundError(f"Generator did not create {node_spec_path}")

    raise FileNotFoundError(
        "NodeSpec not found. Checked "
        f"for a unique *_spec.json or {final_node_spec}; matching Python generators were not found."
    )


def _create_codebase_zip(repository: Path, system_key: str) -> Path:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archive = DEFAULT_OUTPUT_DIR / f"{system_key}.zip"
    subprocess.run(["zip", "-r", str(archive), repository.name], cwd=repository.parent, check=True)
    return archive


def _zip_exported_sandbox(export_root: Path, output_zip_path: Path) -> Path:
    sandbox_directory = export_root / "sandbox"
    if not sandbox_directory.is_dir():
        raise FileNotFoundError(
            f"Sandbox export did not contain the expected directory: {sandbox_directory}"
        )
    if output_zip_path.suffix.lower() != ".zip":
        raise ValueError("--output-codebase-zip-path must end in .zip")

    output_zip_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(
        str(output_zip_path.with_suffix("")),
        "zip",
        root_dir=sandbox_directory.parent,
        base_dir=sandbox_directory.name,
    )
    return output_zip_path


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
        description="Emulate tools and export the resulting NodeSpec and sandbox code ZIP."
    )
    parser.add_argument("--user-id", "--user_id", required=True, dest="user_id")
    parser.add_argument(
        "--system-name",
        "--system_name",
        required=True,
        dest="system_name",
        help="Path below agentic_systems (domain_systems/ is optional).",
    )
    parser.add_argument("--node-spec-path", type=Path)
    parser.add_argument("--runtime-mapping-path", type=Path)
    parser.add_argument("--node-spec-py-path", type=Path)
    parser.add_argument("--env-file-path", type=Path)
    parser.add_argument(
        "--codebase-zip-path",
        type=Path,
        help="Original source ZIP. Defaults to a temporary ZIP of the system repository.",
    )
    parser.add_argument("--output-node-spec-path", type=Path)
    parser.add_argument(
        "--output-codebase-zip-path",
        type=Path,
        help="Post-emulation sandbox ZIP. Defaults below experiments_v2/.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> tuple[Path, Path]:
    args = parse_args(argv)
    for option in (
        "node_spec_path",
        "runtime_mapping_path",
        "node_spec_py_path",
        "env_file_path",
        "codebase_zip_path",
        "output_node_spec_path",
        "output_codebase_zip_path",
    ):
        value = getattr(args, option)
        if value is not None:
            setattr(args, option, value.resolve())

    os.chdir(PROJECT_ROOT)
    load_dotenv(PROJECT_ROOT / "rift.env")
    system_root, repository, relative_path, system_key = _resolve_system_paths(args.system_name)

    node_spec_path = args.node_spec_path or _ensure_default_node_spec(repository)
    runtime_mapping_path = args.runtime_mapping_path or repository / "runtime_mapping.json"
    node_spec_py_candidates = [node_spec_path.with_suffix(".py")]
    node_spec_py_candidates.extend(
        path
        for path in sorted(repository.glob("*_spec.py"))
        if path.name not in {"final_spec_gt_spec.py", node_spec_path.with_suffix(".py").name}
    )
    node_spec_py_candidates.append(repository / "final_spec_gt_spec.py")
    node_spec_py_path = args.node_spec_py_path or next(
        (path for path in node_spec_py_candidates if path.is_file()),
        node_spec_path.with_suffix(".py"),
    )
    env_file_path = args.env_file_path or next(
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
    output_node_spec_path = args.output_node_spec_path or (
        system_root / f"{system_root.name}_after_discovery.json"
    )
    output_codebase_zip_path = args.output_codebase_zip_path or (
        DEFAULT_OUTPUT_DIR / f"{system_key}_post_emulation.zip"
    )

    for path, label in (
        (node_spec_path, "NodeSpec"),
        (runtime_mapping_path, "runtime mapping"),
        (node_spec_py_path, "Python NodeSpec source"),
        (env_file_path, "environment file"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label.capitalize()} not found: {path}")

    created_input_archive = args.codebase_zip_path is None
    input_codebase_zip = args.codebase_zip_path or _create_codebase_zip(repository, system_key)
    if not input_codebase_zip.is_file():
        raise FileNotFoundError(f"Codebase ZIP not found: {input_codebase_zip}")

    from environment_handler.runtime import prepare_environment
    from runner_runtime import run_sandbox_simulation
    from node_spec.structure_schema import NodeSpec

    handler = None
    committed_image_tag = None
    try:
        node_spec = NodeSpec.from_json(path=str(node_spec_path))
        node_spec.name = system_key
        node_spec.metadata = node_spec.metadata or {}
        node_spec.metadata["runtime_mapping_path"] = str(runtime_mapping_path)
        node_spec.metadata["node_spec_py_path"] = str(node_spec_py_path)

        handler = prepare_environment(
            user_id=args.user_id,
            system_name=system_key,
            envfile_path=str(env_file_path),
            zipfile_path=str(input_codebase_zip),
            execution_command={},
        )
        committed_image_tag = f"{handler.image_tag.split(':', 1)[0]}:latest"
        with tempfile.TemporaryDirectory(prefix="tool-emulation-export-") as export_directory:
            emulated_node_spec = run_sandbox_simulation(
                handler=handler,
                node_spec=node_spec,
                sandbox_output_path=export_directory,
            )
            output_node_spec_path.parent.mkdir(parents=True, exist_ok=True)
            emulated_node_spec.to_json(path=str(output_node_spec_path))
            _zip_exported_sandbox(Path(export_directory), output_codebase_zip_path)
    finally:
        if created_input_archive and input_codebase_zip.exists():
            input_codebase_zip.unlink()
        try:
            if committed_image_tag is not None:
                _remove_image(committed_image_tag)
        finally:
            if handler is not None:
                handler.remove_image(force=True)

    print("Tool emulation completed.")
    print(f"NodeSpec: {output_node_spec_path}")
    print(f"Post-emulation code ZIP: {output_codebase_zip_path}")
    return output_node_spec_path, output_codebase_zip_path


if __name__ == "__main__":
    main()
