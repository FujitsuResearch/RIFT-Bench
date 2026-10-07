from __future__ import annotations
import json
import shlex
import subprocess
from llm_sandbox.core.session_base import BaseSession
from pathlib import Path
import shutil
from typing import Any, Union, Optional
import uuid
from collections import defaultdict

LineSelector = Union[int, tuple[int, int], list[int], list[tuple[int, int]]]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _container_pid(container_id: str, use_sudo: bool = True) -> Optional[int]:
    base = ["sudo"] if use_sudo else []
    r = _run(base + ["docker", "inspect", "-f", "{{.State.Pid}}", container_id])
    if r.returncode != 0:
        return None
    try:
        pid = int(r.stdout.strip())
        return pid if pid > 1 else None
    except ValueError:
        return None


def force_remove_container(container_id: str, use_sudo: bool = True, should_use: bool = False) -> bool:
    
    if not should_use:
        return None

    base = ["sudo"] if use_sudo else []

    # 1) normal fast path
    r = _run(base + ["docker", "rm", "-f", container_id])
    if r.returncode == 0:
        return True

    # 2) fallback: kill host pid, then retry remove
    pid = _container_pid(container_id, use_sudo=use_sudo)
    if pid:
        _run(base + ["kill", "-9", str(pid)])

    r2 = _run(base + ["docker", "rm", "-f", container_id])
    return r2.returncode == 0


def load_python_file(file_path: str | Path) -> str:
    """
    Load a Python file and return its contents as a string.

    Args:
        file_path (str | Path): Path to the Python file.

    Return:
        str: Full file content.
    """
    path = Path(file_path)
    return path.read_text(encoding="utf-8")


def duplicate_path(src_path: str, dst_path: str, overwrite: bool = False) -> None:
    """
    Duplicate a file or directory from source path to destination path.

    Args:
        src_path (str): Source file or directory path.
        dst_path (str): Destination file or directory path.
        overwrite (bool): Whether to overwrite the destination if it already exists.
    """
    src = Path(src_path)
    dst = Path(dst_path)

    if not src.exists():
        raise FileNotFoundError(f"Source path not found: {src}")

    if dst.exists():
        if overwrite:
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        else:
            raise FileExistsError(
                f"Destination already exists: {dst}. "
                "Use overwrite=True to replace it."
            )

    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def copy_file_from_sandbox(
    session: BaseSession, src_in_sandbox: str, dst_on_host: str = "./debugg"
) -> None:
    """
    Copy a file from inside the sandbox to the host machine.

    Args:
        session (BaseSession): Active sandbox session.
        src_in_sandbox (str): Source file path inside the sandbox.
        dst_on_host (str): Destination file path on the host machine.
    """
    dst_path = Path(dst_on_host)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    session.copy_from_runtime(src_in_sandbox, str(dst_path))


def normalize_line_numbers(line_numbers: LineSelector) -> list[int]:
    """
    Normalize line selectors into a sorted list of line numbers.

    Args:
        line_numbers (LineSelector): Single line number or mixed line/range selectors.

    Return:
        list[int]: Sorted list of concrete line numbers.
    """
    if isinstance(line_numbers, int):
        return [line_numbers]

    # single range
    if (
        isinstance(line_numbers, (tuple, list))
        and len(line_numbers) == 2
        and all(isinstance(x, int) for x in line_numbers)
    ):
        start, end = line_numbers
        return list(range(start, end + 1))

    # iterable of ints/ranges
    expanded = set()
    for item in line_numbers:
        if isinstance(item, int):
            expanded.add(item)
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            start, end = item
            expanded.update(range(start, end + 1))
        else:
            raise ValueError(f"Unsupported line number format: {item!r}")
    return sorted(list(expanded))


def copy_env_to_sandbox(
    session: BaseSession, local_env_path: str, sb_env_path: str = "/sandbox/.env"
) -> None:
    """
    Copy host-side environment variables into sandbox so the analyzed system can start with production-like configuration.
    Args:
        session (BaseSession): The sandbox session in which to copy the .env file.
        local_env_path (str): The path to the local .env file on the host.
        sb_env_path (str): The destination path for the .env file inside the sandbox (default: "/sandbox/.env").
    Returns:
        None: Writes `.env` content into sandbox path.
    """
    src = str(local_env_path)
    dst_quoted = shlex.quote(sb_env_path)
    session.execute_command(["bash", "-lc", f"mkdir -p $(dirname {dst_quoted})"])
    session.copy_to_runtime(src=src, dest=sb_env_path)


def run_command_sb(
    command: str, session: BaseSession, timeout_sec: int | None = None
):
    """
    Execute a full command string inside sandbox.

    Args:
        command (str): Full shell command to execute in the sandbox.
        session (BaseSession): Active sandbox session.
        timeout_sec (int | None): Optional timeout in seconds enforced via `timeout`.

    Example:
        python main.py --task "do something"
    """
    env_bootstrap = (
        "if [ -f /sandbox/.env ]; then "
        "while IFS= read -r __line || [ -n \"$__line\" ]; do "
        "case \"$__line\" in ''|\\#*) continue ;; esac; "
        "__line=\"${__line#export }\"; "
        "if [[ \"$__line\" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=(.*)$ ]]; then "
        "__k=\"${BASH_REMATCH[1]}\"; __v=\"${BASH_REMATCH[2]}\"; "
        "__v=\"${__v#${__v%%[![:space:]]*}}\"; "
        "export \"$__k=$__v\"; "
        "fi; "
        "done < /sandbox/.env; "
        "fi; "
    )

    timeout_prefix = ""
    if isinstance(timeout_sec, int) and timeout_sec > 0:
        timeout_prefix = f"timeout {int(timeout_sec)}s "

    result = session.execute_command([
        "bash",
        "-lc",
        (
            env_bootstrap
            + "export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}; "
            + "export MLFLOW_ENABLE_ASYNC_TRACE_LOGGING=${MLFLOW_ENABLE_ASYNC_TRACE_LOGGING:-false}; "
            + timeout_prefix
            + command
        ),
    ])
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result


def start_mlflow_ui_in_sandbox(
    session: BaseSession,
    mlruns_root: str = "/sandbox/mlruns",
) -> None:
    """
    Start MLflow UI inside sandbox.

    Args:
        session (BaseSession): Active sandbox session.
        mlruns_root (str): Root directory for MLflow local file store in sandbox.
    """
    backend_store_uri = f"file:{mlruns_root}"
    launch_cmd = (
        f"mkdir -p {shlex.quote(mlruns_root)} && "
        f"MLFLOW_ALLOW_FILE_STORE=true "
        f"mlflow server --host 0.0.0.0 --port 5000 "
        f"--backend-store-uri {shlex.quote(backend_store_uri)} "
        f"> /tmp/mlflow_ui.log 2>&1 < /dev/null & "
        f"echo MLFLOW_RUNNING_WITH_BACKEND {shlex.quote(backend_store_uri)}"
    )
    result = session.execute_command(["bash", "-lc", launch_cmd])
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)


def export_single_traces_json_from_sandbox(
    session: BaseSession,
    host_dest_path: str,
    mlruns_root: str = "/sandbox/mlruns",
    ignore_experiment_id_zero: bool = True,
) -> str:
    """
    Export the newest trace payload from MLflow in sandbox.

    Args:
        session (BaseSession): Active sandbox session.
        host_dest_path (str): Host directory path for saving the exported trace file.
        mlruns_root (str): MLflow runs root path inside sandbox.
        ignore_experiment_id_zero (bool): Whether to ignore the default experiment ID `0`.

    Return:
        str: Exported trace file path on host, or empty string when export is not possible.
    """
    try:
        cmd_find_and_prepare_trace = f"""
import json
import uuid
import os
from pathlib import Path

root = Path({mlruns_root!r})
ignore_zero = {ignore_experiment_id_zero}

if not root.exists():
    print("")
    raise SystemExit(0)

exp_dirs = [d for d in root.iterdir() if d.is_dir() and d.name.isdigit()]
if ignore_zero:
    exp_dirs = [d for d in exp_dirs if d.name != "0"]

candidates = []
for exp in exp_dirs:
    traces_dir = exp / "traces"
    if not traces_dir.exists():
        continue

    # Legacy layout: traces/tr-*/artifacts/traces.json
    for json_path in traces_dir.glob("tr-*/artifacts/traces.json"):
        try:
            mtime = json_path.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, "legacy_json", json_path))

    # Newer layout: traces/<trace_id>/trace_info.yaml + request_metadata + tags
    for trace_dir in traces_dir.iterdir():
        if not trace_dir.is_dir():
            continue

        trace_info = trace_dir / "trace_info.yaml"
        metadata_dir = trace_dir / "request_metadata"
        tags_dir = trace_dir / "tags"

        if not trace_info.exists() and not metadata_dir.exists() and not tags_dir.exists():
            continue

        try:
            mtime = trace_info.stat().st_mtime if trace_info.exists() else trace_dir.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, "trace_dir", trace_dir))

if not candidates:
    print("")
    raise SystemExit(0)

candidates.sort(key=lambda item: item[0], reverse=True)
_, kind, target = candidates[0]

if kind == "legacy_json":
    print(str(target))
    raise SystemExit(0)

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
                value = stripped.split(":", 1)[1].strip().strip("'\\\"")
                if value:
                    return value
    return ""

trace_id = _extract_trace_id(trace_info_yaml) or target.name

import mlflow
from mlflow.tracking import MlflowClient

tracking_uri_candidates = []
env_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
if env_uri.startswith(("http://", "https://")):
    tracking_uri_candidates.append(env_uri)
tracking_uri_candidates.append("http://localhost:5000")
tracking_uri_candidates.append("http://127.0.0.1:5000")
tracking_uri_candidates = list(dict.fromkeys(tracking_uri_candidates))

payload = None
last_err = None
for tracking_uri in tracking_uri_candidates:
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        trace = client.get_trace(trace_id)
        payload = trace.to_dict()
        break
    except Exception as err:
        last_err = err

if payload is None:
    if last_err is not None:
        print(f"TRACE_EXPORT_ERROR: {{last_err}}")
    print("")
    raise SystemExit(0)

# Backward compatibility for evaluators expecting top-level "spans".
if isinstance(payload, dict) and "spans" not in payload:
    data = payload.get("data", {{}})
    if isinstance(data, dict) and isinstance(data.get("spans"), list):
        payload["spans"] = data["spans"]

export_path = Path("/tmp") / f"trace_export_{{uuid.uuid4().hex}}.json"
export_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(str(export_path))
"""
        trace_out = session.run(cmd_find_and_prepare_trace)
        candidate_paths = [
            line.strip()
            for line in str(trace_out.stdout).splitlines()
            if line.strip().startswith("/")
        ]
        if not candidate_paths:
            return ""

        src_in_sandbox = candidate_paths[-1]
        src_suffix = Path(src_in_sandbox).suffix or ".json"
        unique_name = f"trace_{uuid.uuid4().hex}{src_suffix}"
        dest_path = Path(host_dest_path) / unique_name
        session.copy_from_runtime(src_in_sandbox, str(dest_path))

        return str(dest_path)

    except Exception:
        return ""


## New trace extraction


def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def _extract_response_candidate(span: dict[str, Any]) -> Any | None:
    attributes = span.get("attributes", {})
    raw_outputs = attributes.get("mlflow.spanOutputs")
    if raw_outputs in (None, ""):
        return None

    span_type = _safe_json_loads(attributes.get("mlflow.spanType", '"UNKNOWN"'))
    outputs = _safe_json_loads(raw_outputs)

    if span_type == "AGENT" and isinstance(outputs, dict):
        message = outputs.get("message")
        if isinstance(message, dict) and message.get("role") == "function":
            return message.get("content")
        if isinstance(message, dict):
            return message.get("content")
        return message

    if span_type == "TOOL":
        return outputs

    if span_type == "LLM" and isinstance(outputs, dict):
        choices = outputs.get("choices") or []
        if choices:
            message = choices[0].get("message", {})
            return message.get("content")

    return outputs



def _populate_missing_trace_response(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload

    data = payload.get("data")
    if not isinstance(data, dict) or data.get("response") not in (None, ""):
        return payload

    spans = data.get("spans")
    if not isinstance(spans, list):
        return payload

    response = None
    for span in reversed(spans):
        candidate = _extract_response_candidate(span)
        if candidate in (None, "", "TERMINATE"):
            continue
        response = json.dumps(candidate, ensure_ascii=False)
        break

    if response is None:
        return payload

    data["response"] = response

    info = payload.get("info")
    if isinstance(info, dict):
        request_metadata = info.get("request_metadata")
        if isinstance(request_metadata, dict) and not request_metadata.get(
            "mlflow.traceOutputs"
        ):
            request_metadata["mlflow.traceOutputs"] = response

    return payload



def export_single_traces_json_from_sandbox_v2(
    session: BaseSession,
    host_dest_path: str,
    mlruns_root: str = "/sandbox/mlruns",
    ignore_experiment_id_zero: bool = True,
) -> str:
    """
    Export the newest trace artifact from MLflow in sandbox.

    Args:
        session (BaseSession): Active sandbox session.
        host_dest_path (str): Host directory path for saving the exported trace file.
        mlruns_root (str): MLflow runs root path inside sandbox.
        ignore_experiment_id_zero (bool): Whether to ignore the default experiment ID `0`.

    Return:
        str: Exported trace file path on host, or empty string when export is not possible.
    """
    try:
        Path(host_dest_path).mkdir(parents=True, exist_ok=True)

        cmd_find_and_prepare_trace = f"""
import json
import uuid
import os
from pathlib import Path

root = Path({mlruns_root!r})
ignore_zero = {ignore_experiment_id_zero}

if not root.exists():
    print("")
    raise SystemExit(0)

exp_dirs = [d for d in root.iterdir() if d.is_dir() and d.name.isdigit()]
if ignore_zero:
    exp_dirs = [d for d in exp_dirs if d.name != "0"]

candidates = []
for exp in exp_dirs:
    traces_dir = exp / "traces"
    if not traces_dir.exists():
        continue

    # Legacy layout: traces/tr-*/artifacts/traces.json
    for json_path in traces_dir.glob("tr-*/artifacts/traces.json"):
        try:
            mtime = json_path.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, "legacy_json", json_path))

    # Newer layout: traces/<trace_id>/trace_info.yaml + request_metadata + tags
    for trace_dir in traces_dir.iterdir():
        if not trace_dir.is_dir():
            continue

        trace_info = trace_dir / "trace_info.yaml"
        metadata_dir = trace_dir / "request_metadata"
        tags_dir = trace_dir / "tags"

        if not trace_info.exists() and not metadata_dir.exists() and not tags_dir.exists():
            continue

        try:
            mtime = trace_info.stat().st_mtime if trace_info.exists() else trace_dir.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, "trace_dir", trace_dir))

if not candidates:
    print("")
    raise SystemExit(0)

candidates.sort(key=lambda item: item[0], reverse=True)
_, kind, target = candidates[0]

if kind == "legacy_json":
    print(str(target))
    raise SystemExit(0)

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
                value = stripped.split(":", 1)[1].strip().strip("'\\\"")
                if value:
                    return value
    return ""

trace_id = _extract_trace_id(trace_info_yaml) or target.name

import mlflow
from mlflow.tracking import MlflowClient

tracking_uri_candidates = []
env_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
if env_uri.startswith(("http://", "https://")):
    tracking_uri_candidates.append(env_uri)
tracking_uri_candidates.append("http://localhost:5000")
tracking_uri_candidates.append("http://127.0.0.1:5000")
tracking_uri_candidates = list(dict.fromkeys(tracking_uri_candidates))

payload = None
last_err = None
for tracking_uri in tracking_uri_candidates:
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        trace = client.get_trace(trace_id)
        payload = trace.to_dict()
        break
    except Exception as err:
        last_err = err

if payload is None:
    if last_err is not None:
        print(f"TRACE_EXPORT_ERROR: {{last_err}}")
    print("")
    raise SystemExit(0)

# Backward compatibility for evaluators expecting top-level "spans".
if isinstance(payload, dict) and "spans" not in payload:
    data = payload.get("data", {{}})
    if isinstance(data, dict) and isinstance(data.get("spans"), list):
        payload["spans"] = data["spans"]

export_path = Path("/tmp") / f"trace_export_{{uuid.uuid4().hex}}.json"
export_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(str(export_path))
"""
        trace_out = session.run(cmd_find_and_prepare_trace)
        candidate_paths = [
            line.strip()
            for line in str(trace_out.stdout).splitlines()
            if line.strip().startswith("/")
        ]
        if not candidate_paths:
            return ""

        src_in_sandbox = candidate_paths[-1]
        src_suffix = Path(src_in_sandbox).suffix or ".json"
        unique_name = f"trace_{uuid.uuid4().hex}{src_suffix}"
        dest_path = Path(host_dest_path) / unique_name
        session.copy_from_runtime(src_in_sandbox, str(dest_path))

        payload = json.loads(dest_path.read_text(encoding="utf-8"))
        payload = _populate_missing_trace_response(payload)
        dest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return str(dest_path)

    except Exception as e:
        print(f"Error while extracting trace: {str(e)}")
        return ""


def regroup_dict(data: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """
    Regroup tool entries into regular, local MCP, and external MCP buckets.

    Args:
        data (dict[str, list[dict[str, Any]]]): Flat tool mapping keyed by source prefix.

    Return:
        dict[str, Any]: Regrouped tool mapping with `regular`, `local`, and `external` keys.
    """
    result = {
        "regular": [],
        "local": {},
        "external": {},
    }

    for key, value in data.items():
        if key.startswith("local_"):
            suffix = key[len("local_") :]
            result["local"][suffix] = value
        elif key.startswith("external_"):
            suffix = key[len("external_") :]
            result["external"][suffix] = value
        else:
            result["regular"].extend(value)

    return result


def get_all_tools(node: Any) -> dict[str, Any]:
    """
    Collect tool metadata from a node graph.

    Args:
        node (Any): Root node specification object.

    Return:
        dict[str, Any]: Tool metadata grouped by regular, local MCP, and external MCP scopes.
    """
    tools = defaultdict(list)

    def get_all_tools_rec(node: Any, parent_node: str = "general") -> None:
        if node.node_type.type == "Tool":
            if node.duplicates.exists:
                current_tools_ids = [i["id"] for i in tools[parent_node]]
                instances_ids = [i["id"] for i in node.duplicates.instances]
                if set(current_tools_ids) & set(instances_ids):
                    return
            
            if node.metadata:
                entry = {"id": node.id, "name": node.metadata.get('component_handle', node.name)}
            else:
                entry = {"id": node.id, "name": node.name}
            if node.description:
                entry["desc"] = node.description
            assignment_ref = None
            definition_ref = None
            for tool_code_ref in node.code_references or []:
                if tool_code_ref.kind == "assignment":
                    assignment_ref = tool_code_ref
                if tool_code_ref.kind == "definition":
                    definition_ref = tool_code_ref
            if assignment_ref:
                entry["file_path"] = assignment_ref.file
                entry["assignment_lines"] = normalize_line_numbers(assignment_ref.line)
                entry["assignment_snippet"] = assignment_ref.snippet
            if entry.get("file_path", None) is None and assignment_ref:
                entry["file_path"] = definition_ref.file
                entry["assignment_lines"] = normalize_line_numbers(definition_ref.line)
                entry["assignment_snippet"] = definition_ref.snippet

            inputs = None
            outputs = None
            tool_example_pairs = None
            if node.inputs:
                inputs = node.inputs
            if node.outputs:
                outputs = node.outputs
            if node.tool_example_pairs:
                tool_example_pairs = node.tool_example_pairs

            entry["inputs"] = inputs
            entry["outputs"] = outputs
            entry["tool_example_pairs"] = tool_example_pairs

            tools[parent_node].append(entry)

        if node.nodes:
            for inner_node in node.nodes:
                get_all_tools_rec(inner_node)

        if node.tool_list:
            parent_node = node.id
            for tool in node.tool_list:
                if node.node_type.type == "Local_MCP_server":
                    get_all_tools_rec(tool, f"local_{parent_node}")
                elif node.node_type.type == "External_MCP_server":
                    get_all_tools_rec(tool, f"external_{parent_node}")
                else:
                    get_all_tools_rec(tool, parent_node)

    get_all_tools_rec(node)
    tools = regroup_dict(tools)
    return tools