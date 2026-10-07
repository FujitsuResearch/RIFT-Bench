"""Run the Structure Identifier pipeline for one use case."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4
from zipfile import ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PIPELINE_PATH = Path(__file__).resolve().with_name("run_pipeline.py")


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_.-") or "testing"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Structure Identifier for one use case.")
    parser.add_argument(
        "--execution-command-example-json", dest="execution_command_example_json",
        help="Path to the JSON execution command example. Takes precedence over ZIP discovery.",
    )
    parser.add_argument(
        "--discovery-results-dir", dest="out_dir", help="Directory for pipeline artifacts."
    )
    parser.add_argument(
        "--env-file-path", dest="envfile_path", default=".env", help="Target system env file used by runtime tracing."
    )
    parser.add_argument(
        "--si-env-file-path", dest="si_envfile_path",
        default=str(PROJECT_ROOT / ".env.si"),
        help="SI model credentials file; defaults to the repository's .env.si.",
    )
    parser.add_argument(
        "--codebase-zip-path", dest="zipfile_path", help="Codebase ZIP. Defaults to use_case_1.zip in the current directory."
    )
    parser.add_argument(
        "--system-name", dest="system_name", help="System name. Defaults to the current directory name."
    )
    parser.add_argument("--model", help="Optional model override for pipeline stages.")
    parser.add_argument(
        "--prog-lang-version", dest="prog_lang_version", default="3.11", help=argparse.SUPPRESS
    )
    return parser.parse_args(argv)


def _load_execution_example(path: Path) -> dict[str, Any]:
    try:
        execution_command = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Execution command example not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in execution command example: {path}") from exc
    if not isinstance(execution_command, dict):
        raise ValueError(f"Execution command example must contain a JSON object: {path}")
    return execution_command


def _load_execution_example_from_zip(path: Path) -> dict[str, Any] | None:
    """Read an execution example included in the target code ZIP, if present."""
    with ZipFile(path, "r") as archive:
        matches = [
            name for name in archive.namelist()
            if Path(name).name == "execution_command_example.json"
        ]
        if not matches:
            return None
        if len(matches) > 1:
            raise ValueError(
                "Codebase ZIP contains multiple execution_command_example.json files; "
                "provide --execution_command_example_json to select one."
            )
        try:
            execution_command = json.loads(archive.read(matches[0]).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid execution command example in codebase ZIP: {matches[0]}"
            ) from exc
    if not isinstance(execution_command, dict):
        raise ValueError("Execution command example in codebase ZIP must be a JSON object")
    return execution_command


def _read_env_value(path: Path, key: str) -> str | None:
    """Read one value from a simple .env file without exporting other settings."""
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if text.startswith("export "):
            text = text[len("export "):].strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        name, value = text.split("=", 1)
        if name.strip() == key:
            return value.strip().strip("\"'").strip()
    return None


def _default_execution_example() -> dict[str, Any]:
    return {
        "runner": "python",
        "entrypoint": "main.py",
        "args": [
            {"name": "--task", "value": "What is 42 * 17?"},
            {"name": "--tool_file", "value": "./resources/langraph_react_tools_list.json"},
            {"name": "--exp_name", "value": "langraph_react_test"},
            {"name": "--port", "value": 5000},
        ],
        "task_arg_names": ["--task"],
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    invocation_dir = Path.cwd().resolve()

    zipfile_path = (
        Path(args.zipfile_path).resolve()
        if args.zipfile_path
        else invocation_dir / "use_case_1.zip"
    )
    if not zipfile_path.is_file():
        raise FileNotFoundError(f"Codebase ZIP not found: {zipfile_path}")

    embedded_example = False
    if args.execution_command_example_json:
        execution_command_path = Path(args.execution_command_example_json).expanduser()
        if not execution_command_path.is_absolute():
            execution_command_path = invocation_dir / execution_command_path
        execution_command_path = execution_command_path.resolve()
        execution_command = _load_execution_example(execution_command_path)
    else:
        execution_command_path = invocation_dir / "execution_command_example.json"
    if not args.execution_command_example_json and execution_command_path.is_file():
        execution_command = _load_execution_example(execution_command_path)
    elif not args.execution_command_example_json:
        execution_command = _load_execution_example_from_zip(zipfile_path)
        embedded_example = execution_command is not None
        if execution_command is None:
            execution_command = _default_execution_example()
        execution_command_path = None
    if args.execution_command_example_json or (execution_command_path is not None):
        embedded_example = False
    entrypoint = execution_command.get("entrypoint")
    if isinstance(entrypoint, dict):
        entrypoint = entrypoint.get("value")
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        raise ValueError(
            f"The execution command example must define a non-empty entrypoint: {execution_command_path}"
        )
    system_name = _safe_name(args.system_name or invocation_dir.name)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (
        invocation_dir / "test_runs" / system_name
    )
    if embedded_example:
        out_dir.mkdir(parents=True, exist_ok=True)
        execution_command_path = out_dir / "execution_command_example.json"
        execution_command_path.write_text(
            json.dumps(execution_command, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    flow_execution_example_path = execution_command_path
    if flow_execution_example_path is None:
        out_dir.mkdir(parents=True, exist_ok=True)
        flow_execution_example_path = out_dir / "execution_command_example.json"
        flow_execution_example_path.write_text(
            json.dumps(execution_command, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    envfile_path = Path(args.envfile_path)
    if not envfile_path.is_absolute():
        envfile_path = (invocation_dir / envfile_path).resolve()
    si_envfile_path = Path(args.si_envfile_path).resolve()
    model = args.model or os.environ.get("AZURE_OPENAI_DEPLOYMENT_SI") or _read_env_value(
        si_envfile_path, "AZURE_OPENAI_DEPLOYMENT_SI"
    )
    if not model:
        raise ValueError(
            "AZURE_OPENAI_DEPLOYMENT_SI is missing from the SI environment; "
            "set it in the .env.si file or pass --model."
        )

    from environment_handler.runtime import prepare_environment

    handler = prepare_environment(
        user_id=uuid4().hex,
        system_name=system_name,
        envfile_path=str(envfile_path),
        zipfile_path=str(zipfile_path),
        execution_command=execution_command,
        prog_lang_version=args.prog_lang_version,
    )
    try:
        # prepare_environment builds the sandbox image and removes its temporary
        # extraction. Re-extract the source for the local SI pipeline stages.
        codebase_dir = handler.prepare().resolve()
        entry_path = (codebase_dir / entrypoint).resolve()
        if not entry_path.is_file():
            raise FileNotFoundError(f"Entrypoint not found in codebase ZIP: {entry_path}")

        command = [
            sys.executable,
            str(PIPELINE_PATH),
            "--entrypoint-path",
            str(entry_path),
            "--codebase-zip-path",
            str(zipfile_path),
            "--env-file-path",
            str(envfile_path),
            "--discovery-results-dir",
            str(out_dir),
            "--rag-repo-root",
            str(codebase_dir),
        ]
        if execution_command_path is not None:
            command.extend([
                "--execution-command-example-json",
                str(execution_command_path),
            ])
        command.extend(["--model", model])

        out_dir.mkdir(parents=True, exist_ok=True)
        child_env = os.environ.copy()
        child_env["STRUCTURE_SI_ENV_FILE"] = str(si_envfile_path)
        child_env["STRUCTURE_TARGET_SANDBOX_IMAGE"] = str(handler.image_tag or "")
        child_env["STRUCTURE_TARGET_CODEBASE_ROOT"] = str(codebase_dir)
        subprocess.run(command, cwd=codebase_dir, check=True, env=child_env)

        final_spec = out_dir / "final_spec_gt_spec.json"
        print(f"[run_si] wrote final NodeSpec to {final_spec}")

        # Keep the execution context alive for Flow Extraction's generated-query
        # runs. The flow pipeline reads its command example from outputs_dir.
        if flow_execution_example_path is not None:
            flow_example_in_out_dir = out_dir / "execution_command_example.json"
            if flow_execution_example_path.resolve() != flow_example_in_out_dir.resolve():
                shutil.copyfile(flow_execution_example_path, flow_example_in_out_dir)

        execution_validation_dir = out_dir / "execution_validation"
        trace_dirs = sorted(
            path for path in execution_validation_dir.glob("execution_validation_runs/**/traces")
            if path.is_dir()
        )
        if not trace_dirs:
            raise FileNotFoundError(
                f"No execution-validation traces found under {execution_validation_dir}"
            )
        traces_dir = trace_dirs[0]

        os.environ["STRUCTURE_SI_ENV_FILE"] = str(si_envfile_path)
        from discovery.system_analyzer.components.flow_extraction.flow_query_generation import (
            run_flow_query_generation,
        )

        runtime_mapping_file = out_dir / "validation" / "runtime_mapping.json"
        print("[run_si] starting Flow Extraction pipeline")
        flow_result = run_flow_query_generation(
            outputs_dir=out_dir,
            model=model,
            traces_dir=traces_dir,
            nodespec_file=final_spec,
            runtime_mapping_file=runtime_mapping_file if runtime_mapping_file.is_file() else None,
            handler=handler,
        )
    finally:
        # The pipeline needs the extracted source during runtime tracing; remove
        # the temporary extraction only after all pipeline stages have finished.
        if handler.path_to_codebase is not None:
            shutil.rmtree(handler.path_to_codebase.parent, ignore_errors=True)
        handler.remove_image(force=True)

    print(f"[run_si] Flow Extraction completed: {flow_result.out_dir}")
    print(f"[run_si] wrote flow catalog to {flow_result.flow_catalog_file}")
    print(f"[run_si] wrote NodeSpec with flows to {final_spec}")
    print(f"[run_si] wrote runtime mapping to {out_dir / 'validation' / 'runtime_mapping.json'}")


if __name__ == "__main__":
    main()
