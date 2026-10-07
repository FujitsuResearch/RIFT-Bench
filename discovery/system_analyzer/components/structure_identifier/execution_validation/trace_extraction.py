import json
import re
from pathlib import Path
from typing import Any
import shlex
import socket
import subprocess
import sys
import time
import uuid

# ``discovery.utils`` lives at the repository root, outside the
# ``structure_identifier`` package.  The pipeline may launch this module
# while its working directory is the target agent repository, so neither a
# package-relative import nor the current working directory identifies it.
PROJECT_ROOT = str(Path(__file__).resolve().parents[5])
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from discovery.utils import build_command, _populate_missing_trace_response


DEFAULT_COMMAND_TEMPLATE = "python main.py --task {query}"
DEFAULT_ENVFILE_PATH = ".env"
DEFAULT_STAGES_OUTPUTS_DIR = "stages_outputs"
DEFAULT_HOST_TRACES_PATH = str(Path(DEFAULT_STAGES_OUTPUTS_DIR) / "traces")
DEFAULT_BACKEND_URI = "file:/sandbox/mlruns"
DEFAULT_USER_ID = "temp_user"
DEFAULT_PROG_LANG = "python"
DEFAULT_PROG_LANG_VERSION = "3.12"
DEFAULT_RUNTIME_ROOT: Path | None = None
DEFAULT_SANDBOX_IMAGE: str | None = None


def _slugify(value: str) -> str:
    """Convert a free-form string into a simple lowercase underscore slug."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower() or "repo"


def _get_host_traces_path(out_dir: str | None = None) -> str:
    """Return the host-side directory used to store exported trace JSON files."""
    stages_outputs_dir = out_dir or DEFAULT_STAGES_OUTPUTS_DIR
    return str(Path(stages_outputs_dir) / "traces")


def _run_trace_extraction_in_sandbox(
    *,
    command: str,
    traces_dir: Path,
    env_file: Path,
    query_timeout_sec: int | None,
) -> str:
    """Execute a target query in its prepared Docker image and export its trace."""
    from types import SimpleNamespace

    from environment_handler.runtime import create_sandbox_session, run_command_sb
    from scanning.abstraction_executer.util import export_single_traces_json_from_sandbox_v2

    image_tag = str(DEFAULT_SANDBOX_IMAGE or "").strip()
    if not image_tag:
        raise RuntimeError("Target sandbox image is not configured for trace extraction.")

    handler = SimpleNamespace(image_tag=image_tag, prog_language=DEFAULT_PROG_LANG)
    with create_sandbox_session(handler=handler, verbose=False) as session:
        if env_file.is_file():
            session.copy_to_runtime(src=str(env_file), dest="/sandbox/.env")

        start_server = (
            "mkdir -p /sandbox/mlruns && "
            "(MLFLOW_ALLOW_FILE_STORE=true mlflow server --host 0.0.0.0 --port 5000 "
            "--backend-store-uri file:/sandbox/mlruns "
            ">/tmp/structure_si_mlflow.log 2>&1 < /dev/null &)"
        )
        start_result = session.execute_command(["bash", "-lc", start_server])
        start_code = getattr(start_result, "exit_code", getattr(start_result, "returncode", 0))
        if int(start_code or 0) != 0:
            raise RuntimeError(f"Could not start MLflow in target sandbox: {getattr(start_result, 'stderr', '')}")

        wait_for_server = """python - <<'PY'
import socket
import sys
import time

deadline = time.time() + 15
while time.time() < deadline:
    try:
        connection = socket.create_connection(("127.0.0.1", 5000), timeout=0.5)
        connection.close()
        raise SystemExit(0)
    except OSError:
        time.sleep(0.25)
raise SystemExit(1)
PY"""
        ready_result = session.execute_command(["bash", "-lc", wait_for_server])
        ready_code = getattr(ready_result, "exit_code", getattr(ready_result, "returncode", 0))
        if int(ready_code or 0) != 0:
            raise RuntimeError("MLflow did not become ready inside the target sandbox.")

        result = run_command_sb(command, session, timeout_sec=query_timeout_sec)
        return_code = getattr(result, "exit_code", getattr(result, "returncode", 0))
        try:
            return_code = int(return_code)
        except (TypeError, ValueError):
            return_code = 0
        if getattr(result, "stdout", None):
            print(f"DEBUG:trace_extraction:sandbox_stdout_tail {result.stdout[-1000:]!r}")
        if getattr(result, "stderr", None):
            print(f"DEBUG:trace_extraction:sandbox_stderr_tail {result.stderr[-1000:]!r}")

        trace_path = export_single_traces_json_from_sandbox_v2(
            session,
            host_dest_path=str(traces_dir),
            mlruns_root="/sandbox/mlruns",
        )
        if not trace_path:
            raise RuntimeError(
                "No MLflow trace was exported from the target sandbox "
                f"(query return code {return_code})."
            )
        return trace_path


def _run_trace_extraction(
    task: str,
    command_template: str,
    out_dir: str | None = None,
    query_timeout_sec: int | None = 180,
) -> str:
    """Run one query, export the newest MLflow trace, and return its host JSON path."""

    # Store exported trace JSON files under the caller-provided artifact root.
    traces_dir = Path(_get_host_traces_path(out_dir))
    traces_dir.mkdir(parents=True, exist_ok=True)
    print(f"DEBUG:trace_extraction:start task={task!r} out_dir={out_dir!r} traces_dir={str(traces_dir)!r}")

    # Prefer the caller's target root; the working directory belongs to the SI
    # package when stages are launched through run_pipeline.py.
    runtime_root = DEFAULT_RUNTIME_ROOT or Path.cwd()
    mlruns_root = runtime_root / "mlruns"
    backend_uri = f"file:{mlruns_root}"
    env_file = Path(DEFAULT_ENVFILE_PATH).expanduser()
    if not env_file.is_absolute():
        env_file = runtime_root / env_file
    print(f"DEBUG:trace_extraction:runtime runtime_root={str(runtime_root)!r} mlruns_root={str(mlruns_root)!r} env_file={str(env_file)!r}")

    def _port_is_open(host: str, port: int) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        try:
            return sock.connect_ex((host, port)) == 0
        finally:
            sock.close()

    def _command_runner() -> str:
        parts = shlex.split(command_template)
        runner = parts[0] if parts else "python"
        if runner in {"python", "python3"}:
            return runner
        runner_path = Path(runner)
        if runner_path.exists():
            return str(runner_path)
        return sys.executable

    def _ensure_mlflow_server() -> None:
        # Reuse an existing local MLflow server when present, otherwise launch one in the active runtime root.
        if _port_is_open("127.0.0.1", 5000):
            print("DEBUG:trace_extraction:mlflow reuse_existing_server port=5000")
            return

        server_log = traces_dir / "mlflow_server.log"
        print(f"DEBUG:trace_extraction:mlflow starting_server backend_uri={backend_uri!r} log={str(server_log)!r}")
        with server_log.open("ab") as log_handle:
            subprocess.Popen(
                [
                    _command_runner(),
                    "-m",
                    "mlflow",
                    "server",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "5000",
                    "--backend-store-uri",
                    backend_uri,
                ],
                cwd=str(runtime_root),
                env={**dict(**__import__('os').environ), "MLFLOW_ALLOW_FILE_STORE": __import__('os').environ.get("MLFLOW_ALLOW_FILE_STORE", "true")},
                stdout=log_handle,
                stderr=log_handle,
            )

        deadline = time.time() + 10
        while time.time() < deadline:
            if _port_is_open("127.0.0.1", 5000):
                return
            time.sleep(0.25)

        log_tail = ""
        try:
            log_tail = server_log.read_text(encoding="utf-8", errors="replace")[-2000:]
        except Exception:
            pass
        raise RuntimeError(f"Failed to start MLflow server. Runtime root: {runtime_root}.\n{log_tail}".strip())

    def _list_trace_candidates() -> list[tuple[float, str, Path]]:
        # Discover candidate traces from the configured MLflow artifact store after a query run.
        root = mlruns_root
        if not root.exists():
            return []

        candidates: list[tuple[float, str, Path]] = []
        exp_dirs = [
            d for d in root.iterdir()
            if d.is_dir() and d.name.isdigit() and d.name != "0"
        ]

        for exp in exp_dirs:
            traces_root = exp / "traces"
            if not traces_root.exists():
                continue

            for json_path in traces_root.glob("tr-*/artifacts/traces.json"):
                try:
                    candidates.append((json_path.stat().st_mtime, "legacy_json", json_path))
                except OSError:
                    pass

            for trace_dir in traces_root.iterdir():
                if not trace_dir.is_dir():
                    continue
                trace_info = trace_dir / "trace_info.yaml"
                metadata_dir = trace_dir / "request_metadata"
                tags_dir = trace_dir / "tags"
                if not trace_info.exists() and not metadata_dir.exists() and not tags_dir.exists():
                    continue
                try:
                    mtime = trace_info.stat().st_mtime if trace_info.exists() else trace_dir.stat().st_mtime
                    candidates.append((mtime, "trace_dir", trace_dir))
                except OSError:
                    pass

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates

    def _export_trace_candidate(candidate: tuple[float, str, Path]) -> str:
        # Export one chosen trace candidate into the stable host-side JSON format.
        _, kind, target = candidate

        if kind == "legacy_json":
            payload = json.loads(target.read_text(encoding="utf-8"))
        else:
            export_script = f"""
import json
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

target = Path({str(target)!r})
root = Path({str(mlruns_root)!r})

def _read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_bytes().decode("utf-8", errors="replace")

trace_info_path = target / "trace_info.yaml"
trace_info_yaml = _read_file(trace_info_path) if trace_info_path.exists() else ""

def _extract_trace_id(trace_info_text: str) -> str:
    for key in ("trace_id", "request_id"):
        prefix = f"{{key}}:"
        for line in trace_info_text.splitlines():
            stripped = line.strip()
            if stripped.startswith(prefix):
                value = stripped.split(":", 1)[1].strip().strip("'").strip(chr(34))
                if value:
                    return value
    return ""

trace_id = _extract_trace_id(trace_info_yaml) or target.name

payload = None
for tracking_uri in ("http://127.0.0.1:5000", f"file:{{root}}"):
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        payload = client.get_trace(trace_id).to_dict()
        break
    except Exception:
        pass

if payload is None:
    print("")
    raise SystemExit(0)

if "spans" not in payload:
    data = payload.get("data", {{}})
    if isinstance(data, dict) and isinstance(data.get("spans"), list):
        payload["spans"] = data["spans"]

print(json.dumps(payload, ensure_ascii=False))
"""
            result = subprocess.run(
                [_command_runner(), "-c", export_script],
                cwd=str(runtime_root),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0 or not result.stdout.strip():
                raise RuntimeError(
                    f"Failed to export MLflow trace from {target}.\nSTDERR:\n{result.stderr.strip()}".strip()
                )
            payload = json.loads(result.stdout)

        payload = _populate_missing_trace_response(payload)
        dest_path = traces_dir / f"trace_{uuid.uuid4().hex}.json"
        dest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(dest_path)

    # Build the exact runtime command that will execute the query.
    command = build_command(
        query=task,
        command_template=command_template,
        quote_query=True,
    )
    print(f"DEBUG:trace_extraction:command_template {command_template!r}")
    print(f"DEBUG:trace_extraction:command_built {command!r}")

    if DEFAULT_SANDBOX_IMAGE:
        return _run_trace_extraction_in_sandbox(
            command=command,
            traces_dir=traces_dir,
            env_file=env_file,
            query_timeout_sec=query_timeout_sec,
        )

    # Capture the pre-run trace state so the new trace can be identified afterward.
    _ensure_mlflow_server()

    before = _list_trace_candidates()
    before_keys = {(kind, str(path)) for _, kind, path in before}
    before_latest_mtime = max((mtime for mtime, _, _ in before), default=-1.0)

    timeout_prefix = ""
    if isinstance(query_timeout_sec, int) and query_timeout_sec > 0:
        timeout_prefix = f"timeout {int(query_timeout_sec)}s "

    # Run the query with the expected environment and MLflow tracing enabled.
    shell_cmd = (
        f"if [ -f {shlex.quote(str(env_file))} ]; then set -a; . {shlex.quote(str(env_file))}; set +a; fi; "
        + "export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}; "
        + "export MLFLOW_ENABLE_ASYNC_TRACE_LOGGING=${MLFLOW_ENABLE_ASYNC_TRACE_LOGGING:-false}; "
        + f"{timeout_prefix}{command}"
    )
    print(f"DEBUG:trace_extraction:shell_cmd {shell_cmd!r}")
    query_run = subprocess.run(
        ["bash", "-lc", shell_cmd],
        cwd=str(runtime_root),
        capture_output=True,
        text=True,
    )
    print(f"DEBUG:trace_extraction:query_returncode {query_run.returncode}")
    if query_run.stdout:
        print(f"DEBUG:trace_extraction:query_stdout_tail {query_run.stdout[-1000:]!r}")
    if query_run.stderr:
        print(f"DEBUG:trace_extraction:query_stderr_tail {query_run.stderr[-1000:]!r}")

    # Compare the pre/post MLflow state to identify the trace created by this query.
    after = _list_trace_candidates()
    candidate = next(
        (
            item for item in after
            if (item[1], str(item[2])) not in before_keys or item[0] > before_latest_mtime
        ),
        None,
    )

    print(f"DEBUG:trace_extraction:candidate_counts before={len(before)} after={len(after)} candidate_found={candidate is not None}")
    if candidate is None:
        raise RuntimeError(
            (
                f"Trace extraction failed for task: {task}\n"
                f"Runtime root: {runtime_root}\n"
                f"Command: {command}\n"
                f"Return code: {query_run.returncode}\n"
                f"STDOUT:\n{query_run.stdout[-4000:]}\n"
                f"STDERR:\n{query_run.stderr[-4000:]}"
            ).strip()
        )

    # Export the selected trace candidate into the host trace directory.
    print(f"DEBUG:trace_extraction:export_candidate kind={candidate[1]!r} path={str(candidate[2])!r}")
    trace_path = _export_trace_candidate(candidate)
    print(f"DEBUG:trace_extraction:exported_trace_path {trace_path!r}")
    if not trace_path:
        raise RuntimeError(f"Trace extraction failed for task: {task}")

    return trace_path
