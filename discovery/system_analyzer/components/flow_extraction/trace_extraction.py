import json
import re
from pathlib import Path
from typing import Any
import os
import sys
sys.path.append(os.getcwd())



COMMAND_TEMPLATE_BY_USE_CASE = {
    1: "python /sandbox/main.py --task {query}",
    2: "python /sandbox/main.py --task {query}",
    3: "python /sandbox/run_agent.py --query {query}",
    4: "python /sandbox/run_hierarchical_research.py --query {query}",
    5: "python /sandbox/src/main.py --query {query}",
    6: "python /sandbox/main.py --task {query}",
}

DEFAULT_COMMAND_TEMPLATE = "python /sandbox/main.py --task {query}"
DEFAULT_ENVFILE_PATH = ".env"
DEFAULT_STAGES_OUTPUTS_DIR = "modular_imp/stages_outputs"
DEFAULT_HOST_TRACES_PATH = str(Path(DEFAULT_STAGES_OUTPUTS_DIR) / "traces")
DEFAULT_BACKEND_URI = "file:/sandbox/mlruns"
DEFAULT_USER_ID = "temp_user"
DEFAULT_PROG_LANG = "python"
DEFAULT_PROG_LANG_VERSION = "3.11"


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower() or "repo"


def _load_trace_json(trace_path: str) -> dict[str, Any]:
    with open(trace_path, "r", encoding="utf-8") as handle:
        trace = json.load(handle)

    if not isinstance(trace, dict):
        raise ValueError(f"Expected a JSON object trace at {trace_path}.")

    return trace


def _get_host_traces_path(out_dir: str | None = None) -> str:
    stages_outputs_dir = out_dir or DEFAULT_STAGES_OUTPUTS_DIR
    return str(Path(stages_outputs_dir) / "traces")


def _run_trace_extraction(
    zipfile_path: str,
    task: str,
    handler: Any,
    command_template: str = None,
    system_name: str = None,
    out_dir: str | None = None,
    query_timeout_sec: int | None = 180,
) -> str:

    if command_template is None:
        command_template = getattr(handler, "command_template", DEFAULT_COMMAND_TEMPLATE)
    if handler is None or not str(getattr(handler, "image_tag", "") or "").strip():
        raise ValueError("A prepared target sandbox handler is required for flow trace extraction.")

    from environment_handler.runtime import create_sandbox_session, run_command_sb
    from scanning.abstraction_executer.util import (
        copy_env_to_sandbox,
        export_single_traces_json_from_sandbox_v2,
        start_mlflow_ui_in_sandbox,
    )
    from discovery.utils import build_command

    traces_dir = Path(out_dir or DEFAULT_HOST_TRACES_PATH).resolve()
    traces_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(
        query=task,
        command_template=command_template,
        quote_query=True,
    )
    print(f"Using command template: {command_template}")
    print(f"Running flow trace in target sandbox image: {handler.image_tag}")

    with create_sandbox_session(handler=handler, verbose=True) as session:
        env_file = getattr(handler, "path_to_envfile", None)
        if env_file and Path(env_file).is_file():
            copy_env_to_sandbox(session, str(env_file))
        start_mlflow_ui_in_sandbox(session, mlruns_root="/sandbox/mlruns")
        ready = session.execute_command([
            "bash",
            "-lc",
            "python - <<'PY'\n"
            "import socket, time, sys\n"
            "deadline = time.time() + 15\n"
            "while time.time() < deadline:\n"
            "    try:\n"
            "        connection = socket.create_connection(('127.0.0.1', 5000), timeout=0.5)\n"
            "        connection.close()\n"
            "        sys.exit(0)\n"
            "    except OSError:\n"
            "        time.sleep(0.25)\n"
            "sys.exit(1)\n"
            "PY",
        ])
        ready_code = getattr(ready, "exit_code", getattr(ready, "returncode", 0))
        if int(ready_code or 0) != 0:
            raise RuntimeError("MLflow did not become ready inside the target sandbox.")
        result = run_command_sb(command, session, timeout_sec=query_timeout_sec)
        if getattr(result, "stdout", None):
            print(result.stdout[-1000:])
        if getattr(result, "stderr", None):
            print(result.stderr[-1000:])
        trace_path = export_single_traces_json_from_sandbox_v2(
            session,
            host_dest_path=str(traces_dir),
            mlruns_root="/sandbox/mlruns",
        )

    if not trace_path or not Path(trace_path).is_file():
        raise RuntimeError(f"Trace extraction failed for task: {task}")
    return str(Path(trace_path).resolve())


def extract_trace_path_from_use_case(
    use_case_number: int,
    task: str,
    handler: Any,
    out_dir: str | None = None,
    query_timeout_sec: int | None = 180,
) -> str:
    if use_case_number not in COMMAND_TEMPLATE_BY_USE_CASE:
        raise ValueError(f"Unsupported use case number: {use_case_number}")

    return _run_trace_extraction(
        zipfile_path=f"use_cases/use_case_{use_case_number}.zip",
        task=task,
        handler=handler,
        command_template=COMMAND_TEMPLATE_BY_USE_CASE[use_case_number],
        system_name=f"temp_system_{use_case_number}",
        out_dir=out_dir,
        query_timeout_sec=query_timeout_sec,
    )


def extract_trace_json_from_use_case(
    use_case_number: int,
    task: str,
    out_dir: str | None = None,
    query_timeout_sec: int | None = 180,
) -> dict[str, Any]:
    trace_path = extract_trace_path_from_use_case(
        use_case_number,
        task,
        out_dir=out_dir,
        query_timeout_sec=query_timeout_sec,
    )
    return _load_trace_json(trace_path)


def extract_trace_path_from_zip(
    zipfile_path: str,
    task: str,
    out_dir: str | None = None,
    query_timeout_sec: int | None = 180,
) -> str:
    system_name = f"temp_system_{_slugify(Path(zipfile_path).stem)}"
    return _run_trace_extraction(
        zipfile_path=zipfile_path,
        task=task,
        command_template=DEFAULT_COMMAND_TEMPLATE,
        system_name=system_name,
        out_dir=out_dir,
        query_timeout_sec=query_timeout_sec,
    )


def extract_trace_json_from_zip(
    zipfile_path: str,
    task: str,
    out_dir: str | None = None,
    query_timeout_sec: int | None = 180,
) -> dict[str, Any]:
    trace_path = extract_trace_path_from_zip(
        zipfile_path,
        task,
        out_dir=out_dir,
        query_timeout_sec=query_timeout_sec,
    )
    return _load_trace_json(trace_path)
