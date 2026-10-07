#!/usr/bin/env python3
"""Pipeline orchestrator from file_finding through final post-processing."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Sequence

try:
    from .global_utils import (
        load_env_file,
        run_subprocess,
        validate_json_artifact,
        validate_nodes_py_artifact,
        validate_root_node_artifact,
    )
    from .rag.build_index import build_faiss
except ImportError:
    PACKAGE_PARENT = Path(__file__).resolve().parent.parent
    if str(PACKAGE_PARENT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_PARENT))
    from structure_identifier.global_utils import (
        load_env_file,
        run_subprocess,
        validate_json_artifact,
        validate_nodes_py_artifact,
        validate_root_node_artifact,
    )
    from structure_identifier.rag.build_index import build_faiss


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

# Per-step wall-clock latency records, populated by _run during the pipeline run.
_STEP_TIMINGS: List[Dict[str, Any]] = []
# Output dir for the latency report, set once in main() so _run can flush incrementally.
_TIMINGS_OUT_DIR: Path | None = None
_STAGE_VALIDATIONS: List[Dict[str, Any]] = []


def _step_label(cmd: List[str]) -> str:
    """Derive a readable step label from a subprocess command line."""
    module = cmd[2] if len(cmd) > 2 and cmd[1] == "-m" else (cmd[0] if cmd else "step")
    name = module.split(".")[-2] if module.endswith(".main") and "." in module else module
    # Disambiguate stages that run multiple times (e.g. graph_correctness) by their --phase.
    if "--phase" in cmd:
        try:
            name = f"{name}:{cmd[cmd.index('--phase') + 1]}"
        except (ValueError, IndexError):
            pass
    return name


def _task_arg_names(exec_example: Dict[str, Any] | None) -> set[str]:
    """Return the CLI argument names that should be treated as task/query inputs."""
    if not isinstance(exec_example, dict):
        return {"--task", "--query", "--prompt"}
    raw = exec_example.get("task_arg_names")
    names: set[str] = set()
    if isinstance(raw, list):
        for x in raw:
            txt = str(x or "").strip()
            if txt:
                names.add(txt)
    if not names:
        names = {"--task", "--query", "--prompt"}
    return names


def _is_task_arg(arg_name: str | None, task_arg_names: set[str]) -> bool:
    """Return whether one argument name belongs to the task-input argument set."""
    if not isinstance(arg_name, str):
        return False
    return arg_name.strip() in task_arg_names


def _build_exec_example_hints(exec_example: Dict[str, Any] | None, out_dir: Path) -> Path | None:
    """Persist extra reachable-file hints derived from the execution command example."""
    if not isinstance(exec_example, dict):
        return None
    task_names = _task_arg_names(exec_example)
    extra_py_files: List[str] = []
    extra_json_files: List[str] = []

    ep = exec_example.get("entrypoint")
    if isinstance(ep, dict):
        ep_val = str(ep.get("value") or "").strip()
        if ep_val.endswith(".py"):
            extra_py_files.append(ep_val)
        elif ep_val.endswith(".json"):
            extra_json_files.append(ep_val)

    args = exec_example.get("args")
    if isinstance(args, list):
        for item in args:
            if not isinstance(item, dict):
                continue
            nm = str(item.get("name") or "").strip() or None
            if _is_task_arg(nm, task_names):
                continue
            val = item.get("value")
            if isinstance(val, str):
                sval = val.strip()
                if sval.endswith(".py"):
                    extra_py_files.append(sval)
                elif sval.endswith(".json"):
                    extra_json_files.append(sval)

    if not extra_py_files and not extra_json_files:
        return None

    hints_obj = {
        "extra_py_files": sorted(set(extra_py_files)),
        "extra_json_files": sorted(set(extra_json_files)),
    }
    hints_path = out_dir / "execution_command_reachable_hints.json"
    hints_path.write_text(json.dumps(hints_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return hints_path


def _write_step_timings(out_dir: Path | None, *, announce: bool = False) -> None:
    """Write the collected per-step latency records and running total to disk.

    Called after every step (incremental flush) and once more at the end. Always
    rewrites the full file so the report reflects the latest completed steps.
    """
    if out_dir is None or not _STEP_TIMINGS:
        return
    total = round(sum(float(rec.get("seconds") or 0.0) for rec in _STEP_TIMINGS), 3)
    payload = {
        "total_seconds": total,
        "steps": _STEP_TIMINGS,
    }
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "pipeline_latency.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if announce:
            print(f"\n[run_pipeline] wrote latency report: {out_dir / 'pipeline_latency.json'} (total {total:.2f}s)")
    except Exception as exc:
        print(f"[run_pipeline] failed to write latency report: {exc}", file=sys.stderr)


def _run(cmd: List[str], *, label: str | None = None) -> None:
    """Run one pipeline subprocess from the repository root, recording its wall-clock latency."""
    label = label or _step_label(cmd)
    ok = False
    start = time.perf_counter()
    try:
        run_subprocess(cmd, repo_root=REPO_ROOT)
        ok = True
    finally:
        elapsed = time.perf_counter() - start
        _STEP_TIMINGS.append({
            "step": label,
            "seconds": round(elapsed, 3),
            "ok": ok,
        })
        print(f"[run_pipeline] step '{label}' took {elapsed:.2f}s")
        # Flush after each step so the latency report is current even mid-run.
        _write_step_timings(_TIMINGS_OUT_DIR)


def _validate_file_exists(path: Path) -> Dict[str, Any]:
    """Validate that a file artifact exists on disk."""
    result: Dict[str, Any] = {
        "ok": False,
        "kind": "file_exists",
        "path": str(path),
        "reason": "",
        "details": {},
    }
    if not path.exists():
        result["reason"] = "missing_file"
        return result
    result["ok"] = True
    result["reason"] = "ok"
    result["details"] = {"size_bytes": path.stat().st_size}
    return result


def _validate_artifact(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to the correct artifact validator for one stage output spec."""
    path = Path(str(spec.get("path") or "")).resolve()
    kind = str(spec.get("kind") or "").strip()
    if kind == "json":
        return validate_json_artifact(path, required_keys=spec.get("required_keys"))
    if kind == "root_json":
        return validate_root_node_artifact(path, allow_flat_object=bool(spec.get("allow_flat_object")))
    if kind == "nodes_py":
        return validate_nodes_py_artifact(path)
    if kind == "file_exists":
        return _validate_file_exists(path)
    raise ValueError(f"Unsupported artifact validator kind: {kind}")


def _validate_stage_outputs(specs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate the critical output artifacts for one pipeline stage."""
    results = [_validate_artifact(spec) for spec in specs]
    failures = [res for res in results if not bool(res.get("ok"))]
    return {
        "ok": not failures,
        "results": results,
        "failures": failures,
    }


def _cmd_with_refresh_raw(cmd: List[str]) -> List[str]:
    """Return a command line that forces raw refresh when supported by the stage."""
    if "--refresh_raw" in cmd or "--refresh-raw" in cmd:
        return list(cmd)
    return [*cmd, "--refresh_raw"]


def _run_validated_stage(
    *,
    name: str,
    cmd: List[str],
    required_outputs: List[Dict[str, Any]],
    max_retries: int = 1,
    supports_refresh_raw: bool = False,
    failure_policy: str = "required",
    fallback_outputs: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Run a stage, validate its required outputs, and optionally retry or fallback."""
    attempts: List[Dict[str, Any]] = []
    retry_count = max(0, int(max_retries))
    for attempt_idx in range(retry_count + 1):
        attempt_cmd = _cmd_with_refresh_raw(cmd) if attempt_idx > 0 and supports_refresh_raw else list(cmd)
        run_error: str | None = None
        try:
            _run(attempt_cmd, label=name)
        except Exception as exc:
            run_error = str(exc)
        validation = (
            _validate_stage_outputs(required_outputs)
            if run_error is None
            else {
                "ok": False,
                "results": [],
                "failures": [{
                    "ok": False,
                    "kind": "stage_run",
                    "path": "",
                    "reason": "subprocess_failed",
                    "details": {"error": run_error},
                }],
            }
        )
        attempts.append({
            "attempt": attempt_idx + 1,
            "cmd": list(attempt_cmd),
            "run_error": run_error,
            "validation": validation,
        })
        if validation["ok"]:
            result = {
                "stage": name,
                "status": "success",
                "attempts": attempts,
                "selected_outputs": [str(Path(str(spec.get("path") or "")).name) for spec in required_outputs],
            }
            _STAGE_VALIDATIONS.append(result)
            return result
        if attempt_idx < retry_count:
            failure_reasons = [str(item.get("reason") or "validation_failed") for item in validation["failures"]]
            print(f"[run_pipeline] validation failed for stage '{name}' on attempt {attempt_idx + 1}; retrying with refreshed raw outputs: {failure_reasons}")

    result = {
        "stage": name,
        "status": "failed",
        "attempts": attempts,
        "selected_outputs": [],
    }
    if failure_policy == "fallback_to_input":
        fallback_validation = _validate_stage_outputs(fallback_outputs or [])
        result["fallback_validation"] = fallback_validation
        if not fallback_validation["ok"]:
            _STAGE_VALIDATIONS.append(result)
            raise RuntimeError(
                f"Stage '{name}' failed validation after retries and fallback outputs are invalid: "
                f"{fallback_validation['failures']}"
            )
        result["status"] = "fallback"
        result["selected_outputs"] = [str(Path(str(spec.get("path") or "")).name) for spec in (fallback_outputs or [])]
        _STAGE_VALIDATIONS.append(result)
        print(f"[run_pipeline] stage '{name}' failed validation after retries; continuing with fallback inputs: {result['selected_outputs']}")
        return result

    _STAGE_VALIDATIONS.append(result)
    last_failures = attempts[-1]["validation"]["failures"] if attempts else []
    raise RuntimeError(f"Stage '{name}' failed validation after retries: {last_failures}")


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    """Read a JSON object from disk when present, otherwise return an empty dict."""
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _parse_graph_correctness_summary_phases(raw: str) -> Set[str]:
    """Parse the selected graph-correctness summary-pass phases from CLI text."""
    txt = str(raw or "").strip().lower()
    if not txt or txt == "none":
        return set()
    if txt == "all":
        return {"before_refine", "after_refine", "after_execution_validation_actions"}
    vals = {x.strip().lower() for x in txt.split(",") if x.strip()}
    allowed = {"before_refine", "after_refine", "after_execution_validation_actions"}
    return {x for x in vals if x in allowed}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse pipeline options using the same CLI naming style as run_scanning."""
    ap = argparse.ArgumentParser(
        description="Discover a NodeSpec from source code and execution artifacts."
    )
    ap.add_argument("--python", default=sys.executable, help="Python interpreter to use.")
    ap.add_argument(
        "--env-file-path", dest="env_file", default=".env", help="Optional .env file to load and pass to stage entrypoints."
    )
    ap.add_argument(
        "--entrypoint-path", dest="entry", default="use_cases/use_case_1/main.py", help="Entrypoint python file."
    )
    ap.add_argument(
        "--codebase-zip-path", dest="zip_file",
        default="",
        help=(
            "Optional zip file for sandbox execution/tracing. "
            "If not provided, defaults to <entry_parent>.zip for backward compatibility."
        ),
    )
    ap.add_argument(
        "--discovery-results-dir", dest="out_dir", default="stages_outputs", help="Outputs directory."
    )
    ap.add_argument(
        "--root-out", dest="root_out", default="root_record.json", help="Root record filename under --discovery-results-dir."
    )
    ap.add_argument(
        "--reachable-out", dest="reachable_out", default="reachable_files.json", help="Reachable files filename under --discovery-results-dir."
    )
    ap.add_argument(
        "--nodes-index", dest="nodes_index", default="nodes_index.json", help="Per-file nodes index filename under --discovery-results-dir."
    )
    ap.add_argument(
        "--nodes-catalog", dest="nodes_catalog", default="nodes_catalog.py", help="Combined catalog filename under --discovery-results-dir."
    )
    ap.add_argument(
        "--root-nodes", dest="root_nodes", default="root_nodes.json", help="Root nodes filename under --discovery-results-dir."
    )
    ap.add_argument(
        "--guidance-file", dest="guidance_file", default="system_guidance.json", help="Guidance filename under --discovery-results-dir."
    )
    ap.add_argument(
        "--nodes-with-children", dest="nodes_with_children", default="nodes_with_children.py", help="Child-expanded nodes filename under --discovery-results-dir."
    )
    ap.add_argument(
        "--nodes-connected", dest="nodes_connected", default="nodes_connected.py", help="Connected nodes filename under --discovery-results-dir."
    )
    ap.add_argument(
        "--rag-repo-root", dest="rag_repo_root", default=".", help="Repo root to index for RAG."
    )
    ap.add_argument(
        "--rag-index-dir", dest="rag_index_dir", default="rag_index", help="RAG FAISS index directory (relative paths are resolved under --discovery-results-dir)."
    )
    ap.add_argument(
        "--rag-max-rounds", dest="rag_max_rounds", type=int, default=8, help="Max context-request rounds per LLM call."
    )
    ap.add_argument("--model", default="gpt-5.2-codex", help="Optional model override passed to subcomponents.")
    ap.add_argument(
        "--execution-validation-max-rounds", dest="execution_validation_max_rounds", type=int, default=3, help="Max execution_validation+correctness rounds."
    )
    ap.add_argument(
        "--stop-early", dest="stop_early",
        action="store_true",
        help="Exit early at the current development checkpoint in the pipeline.",
    )
    ap.add_argument(
        "--final-nodespec", dest="final_nodespec", default="final_nodespec.py", help="Final code-reference-refined nodes filename under --discovery-results-dir."
    )
    ap.add_argument(
        "--final-report", dest="final_report", default="final_report.json", help="Final code-reference-refine report filename under --discovery-results-dir."
    )
    ap.add_argument(
        "--final-output-stem", dest="final_output_stem", default="final_spec_gt_spec", help="Base output name for final post-processing .py/.json outputs."
    )
    ap.add_argument(
        "--hints-file", dest="hints_file", action="append", default=[], help="Optional hints JSON path for file_finding."
    )
    ap.add_argument(
        "--execution-command-example-json", dest="execution_command_example_json",
        default="",
        help=(
            "Optional path to a JSON file describing execution command example "
            "(runner/entrypoint/args/task_arg_names). Stored in run artifacts for downstream stages."
        ),
    )
    ap.add_argument(
        "--graph-correctness-system-summary-pass-phases", dest="graph_correctness_system_summary_pass_phases",
        default="none",
        help=(
            "Comma-separated phases to enable graph_correctness system-summary pass: "
            "before_refine,after_refine,after_execution_validation_actions,all,none"
        ),
    )
    ap.add_argument(
        "--run-static-validation", dest="run_static_validation", type=lambda x: str(x).strip().lower() in {"1", "true", "yes", "y"}, default=True, help="Whether to run static validation before the first graph-correctness pass."
    )
    ap.add_argument(
        "--run-execution-validation", dest="run_execution_validation", type=lambda x: str(x).strip().lower() in {"1", "true", "yes", "y"}, default=True, help="Whether to run execution validation, apply its actions, and run graph correctness again."
    )
    ap.add_argument(
        "--run-refinement", dest="run_refinement", type=lambda x: str(x).strip().lower() in {"1", "true", "yes", "y"}, default=True, help="Whether to run node refinement, graph correctness, and code reference refine."
    )
    ap.add_argument(
        "--run-mcp-tool-discovery", dest="run_mcp_tool_discovery", type=lambda x: str(x).strip().lower() in {"1", "true", "yes", "y"}, default=True, help="Whether to connect to live External_MCP_server constructors and replace their tool_list with real discovered tools."
    )
    ap.add_argument("--refresh-raw", dest="refresh_raw", action="store_true")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the pipeline from file discovery through final post-processing."""
    args = parse_args(argv)
    load_env_file(Path(args.env_file))

    print("[run_pipeline] parsed arguments")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run_pipeline] output directory ready: {out_dir}")
    # Record the latency report location so steps can flush incrementally, and
    # ensure it is still written even if a later step raises.
    global _TIMINGS_OUT_DIR
    _TIMINGS_OUT_DIR = out_dir
    atexit.register(_write_step_timings, out_dir)
    rag_repo_root = Path(args.rag_repo_root).resolve()
    rag_index_path = Path(args.rag_index_dir)
    rag_index_dir = rag_index_path.resolve() if rag_index_path.is_absolute() else (out_dir / rag_index_path).resolve()
    print(f"[run_pipeline] resolved repo root: {REPO_ROOT}")
    print(f"[run_pipeline] resolved rag repo root: {rag_repo_root}")
    print(f"[run_pipeline] resolved rag index dir: {rag_index_dir}")

    execution_command_example: Dict[str, Any] | None = None
    execution_command_example_source_path: Path | None = None
    if isinstance(args.execution_command_example_json, str) and args.execution_command_example_json.strip():
        print("[run_pipeline] loading execution command example json")
        ec_path = Path(args.execution_command_example_json).resolve()
        if not ec_path.exists():
            raise FileNotFoundError(f"execution_command_example_json not found: {ec_path}")
        try:
            ec_obj = json.loads(ec_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"Invalid JSON in execution_command_example_json: {ec_path}") from e
        if not isinstance(ec_obj, dict):
            raise ValueError("execution_command_example_json must contain a JSON object")
        execution_command_example = ec_obj
        execution_command_example_source_path = ec_path
        print("[run_pipeline] loaded execution command example json")

    print("[run_pipeline] building stage path configuration")
    package_name = Path(__file__).resolve().parent.name
    stage_modules = {
        "file_finding": f"{package_name}.file_finding.main",
        "nodespec_per_file": f"{package_name}.NodeSpec_per_file.main",
        "root_finder": f"{package_name}.root_finder.main",
        "summary_creator": f"{package_name}.summary_creator.main",
        "child_creation": f"{package_name}.child_creation.main",
        "connectivity_pass": f"{package_name}.connectivity_pass.main",
        "static_validation": f"{package_name}.static_validation.main",
        "graph_correctness": f"{package_name}.graph_correctness.main",
        "execution_validation": f"{package_name}.execution_validation.main",
        "node_refinement": f"{package_name}.node_refinement.main",
        "code_reference_refine": f"{package_name}.code_reference_refine.main",
        "mcp_tool_discovery": f"{package_name}.mcp_tool_discovery.main",
        "post_processing": f"{package_name}.post_processing.main",
    }

    artifacts = {
        "static_validation": {
            "main_nodes": "static_validation_nodes_main_graph.py",
            "isolated_nodes": "static_validation_nodes_isolated.py",
            "report": "static_validation_isolation_report.json",
            "raw_dir": "static_validation_raw",
        },
        "after_static_graph_correctness": {
            "nodes": "post_static_validation_graph_corrected.py",
            "report": "post_static_validation_graph_correctness_report.json",
            "raw_dir": "graph_correctness_raw_post_static_validation",
        },
        "execution_validation": {
            "report": "execution_validation_report_post_static_validation.json",
        },
        "after_execution_validation": {
            "actions_nodes": "post_execution_validation_actions.py",
            "actions_report": "post_execution_validation_action_apply_report.json",
            "corrected_nodes": "post_execution_validation_graph_corrected.py",
            "correctness_report": "post_execution_validation_graph_correctness_report.json",
            "raw_dir": "graph_correctness_raw_post_execution_validation",
        },
        "refinement": {
            "nodes": "post_refinement_nodes.py",
            "report": "post_refinement_node_refinement_report.json",
            "corrected_nodes": "post_refinement_graph_corrected.py",
            "correctness_report": "post_refinement_graph_correctness_report.json",
            "raw_dir": "graph_correctness_raw_post_refinement",
        },
        "code_reference_refine": {
            "report": "code_reference_refine_report.json",
            "raw_dir": "code_reference_refine_raw",
        },
        "mcp_tool_discovery": {
            "nodes": "mcp_tool_discovery_nodes.py",
            "report": "mcp_tool_discovery_report.json",
            "raw_dir": "mcp_tool_discovery_raw",
        },
    }
    execution_validation_report_path = out_dir / "validation" / artifacts["execution_validation"]["report"]
    print("[run_pipeline] stage and artifact configuration ready")

    run_info = {
        "entry": str(Path(args.entry).resolve()),
        "out_dir": str(out_dir),
        "rag_repo_root": str(rag_repo_root),
        "rag_index_dir": str(rag_index_dir),
        "execution_command_example": execution_command_example,
    }
    print("[run_pipeline] writing run_info.json")
    (out_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("[run_pipeline] wrote run_info.json")
    if execution_command_example is not None:
        (out_dir / "execution_command_example.json").write_text(
            json.dumps(execution_command_example, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("[run_pipeline] writing execution_command_example.json")
        os.environ["MODULAR_IMP_EXECUTION_COMMAND_EXAMPLE_JSON"] = str(
            (out_dir / "execution_command_example.json").resolve()
        )
    print("[run_pipeline] execution command example setup done")
    summary_pass_phases = _parse_graph_correctness_summary_phases(
        args.graph_correctness_system_summary_pass_phases
    )

    print(f"[run_pipeline] summary pass phases: {sorted(summary_pass_phases)}")
    exec_hints_path = _build_exec_example_hints(execution_command_example, out_dir)
    print(f"[run_pipeline] execution hints path: {exec_hints_path}")
    print("[run_pipeline] setup complete; starting stages")

    # Discover the entrypoint root and the reachable project files.
    file_cmd = [
        args.python,
        "-m", stage_modules["file_finding"],
        "--env_file",
        args.env_file,
        "--entry",
        args.entry,
        "--out_dir",
        str(out_dir),
        "--root_out",
        Path(args.root_out).name,
        "--reachable_out",
        Path(args.reachable_out).name,
    ]
    for hint in args.hints_file:
        if isinstance(hint, str) and hint.strip():
            file_cmd.extend(["--hints_file", hint])
    if exec_hints_path is not None:
        file_cmd.extend(["--hints_file", str(exec_hints_path)])
    _run_validated_stage(
        name="file_finding",
        cmd=file_cmd,
        required_outputs=[
            {"path": out_dir / Path(args.root_out).name, "kind": "json", "required_keys": ["project_root", "entry_abs"]},
            {"path": out_dir / Path(args.reachable_out).name, "kind": "json", "required_keys": ["project_root", "entry_abs", "reachable_py_abs", "reachable_json_abs"]},
        ],
    )

    print(f"\n[rag] building index from: {rag_repo_root}")
    print(f"[rag] saving index to: {rag_index_dir}")
    _rag_start = time.perf_counter()
    build_faiss(
        str(rag_repo_root),
        str(rag_index_dir),
        reachable_record=str(out_dir / Path(args.reachable_out).name),
    )
    _STEP_TIMINGS.append({
        "step": "rag_build_index",
        "seconds": round(time.perf_counter() - _rag_start, 3),
        "ok": True,
    })
    _write_step_timings(_TIMINGS_OUT_DIR)

    # Extract node candidates per file and build the combined node catalog.
    nodespec_cmd = [
        args.python,
        "-m", stage_modules["nodespec_per_file"],
        "--env_file",
        args.env_file,
        "--out_dir",
        str(out_dir),
        "--reachable_record",
        Path(args.reachable_out).name,
        "--index_file",
        Path(args.nodes_index).name,
        "--combined_out",
        Path(args.nodes_catalog).name,
    ]
    if execution_command_example is not None:
        nodespec_cmd.extend(["--execution_command_example_json", str((out_dir / "execution_command_example.json").resolve())])
    if args.model.strip():
        nodespec_cmd.extend(["--model", args.model.strip()])
    if args.refresh_raw:
        nodespec_cmd.append("--refresh_raw")
    _run_validated_stage(
        name="nodespec_per_file",
        cmd=nodespec_cmd,
        required_outputs=[
            {"path": out_dir / Path(args.nodes_index).name, "kind": "file_exists"},
            {"path": out_dir / Path(args.nodes_catalog).name, "kind": "nodes_py"},
        ],
        max_retries=1,
        supports_refresh_raw=True,
    )

    # Resolve the graph root from the extracted node catalog.
    root_cmd = [
        args.python,
        "-m", stage_modules["root_finder"],
        "--env_file",
        args.env_file,
        "--entry",
        args.entry,
        "--out_dir",
        str(out_dir),
        "--nodes_index",
        Path(args.nodes_index).name,
        "--raw_dir",
        "root_finder_raw",
        "--rag_index_dir",
        str(rag_index_dir),
        "--rag_max_rounds",
        str(args.rag_max_rounds),
    ]
    if args.model.strip():
        root_cmd.extend(["--model", args.model.strip()])
    if args.refresh_raw:
        root_cmd.append("--refresh_raw")
    _run_validated_stage(
        name="root_finder",
        cmd=root_cmd,
        required_outputs=[
            {"path": out_dir / "root_nodes.json", "kind": "root_json"},
        ],
        max_retries=1,
        supports_refresh_raw=True,
    )

    requested_root_json = Path(args.root_nodes).name
    canonical_root_json = out_dir / "root_nodes.json"
    requested_root_path = out_dir / requested_root_json
    if requested_root_json != "root_nodes.json" and canonical_root_json.exists():
        shutil.copy2(canonical_root_json, requested_root_path)

    # Generate the system guidance summary used by later repair steps.
    summary_cmd = [
        args.python,
        "-m", stage_modules["summary_creator"],
        "--env_file",
        args.env_file,
        "--out_dir",
        str(out_dir),
        "--root_nodes_file",
        requested_root_json,
        "--nodes_file",
        Path(args.nodes_catalog).name,
        "--raw_dir",
        "guidance_raw",
        "--rag_index_dir",
        str(rag_index_dir),
        "--rag_max_rounds",
        str(args.rag_max_rounds),
    ]
    if args.model.strip():
        summary_cmd.extend(["--model", args.model.strip()])
    if args.refresh_raw:
        summary_cmd.append("--refresh_raw")
    _run_validated_stage(
        name="summary_creator",
        cmd=summary_cmd,
        required_outputs=[
            {"path": out_dir / Path(args.guidance_file).name, "kind": "json", "required_keys": ["guidance"]},
        ],
        max_retries=1,
        supports_refresh_raw=True,
    )

    # Create and attach missing child nodes beneath the root graph.
    child_cmd = [
        args.python,
        "-m", stage_modules["child_creation"],
        "--env_file",
        args.env_file,
        "--out_dir",
        str(out_dir),
        "--root_json",
        requested_root_json,
        "--out",
        Path(args.nodes_with_children).name,
        "--raw_dir",
        "child_creation_raw",
        "--rag_index_dir",
        str(rag_index_dir),
        "--rag_max_rounds",
        str(args.rag_max_rounds),
    ]
    if args.model.strip():
        child_cmd.extend(["--model", args.model.strip()])
    if args.refresh_raw:
        child_cmd.append("--refresh_raw")
    _run_validated_stage(
        name="child_creation",
        cmd=child_cmd,
        required_outputs=[
            {"path": out_dir / Path(args.nodes_with_children).name, "kind": "nodes_py"},
        ],
        max_retries=1,
        supports_refresh_raw=True,
    )

    # Repair and complete graph connectivity on the expanded node graph.
    conn_cmd = [
        args.python,
        "-m", stage_modules["connectivity_pass"],
        "--env_file",
        args.env_file,
        "--out_dir",
        str(out_dir),
        "--nodes_py",
        Path(args.nodes_with_children).name,
        "--root_json",
        requested_root_json,
        "--out",
        Path(args.nodes_connected).name,
        "--raw_dir",
        "connectivity_raw",
        "--rag_index_dir",
        str(rag_index_dir),
        "--rag_max_rounds",
        str(args.rag_max_rounds),
    ]
    if args.model.strip():
        conn_cmd.extend(["--model", args.model.strip()])
    if args.refresh_raw:
        conn_cmd.append("--refresh_raw")
    _run_validated_stage(
        name="connectivity_pass",
        cmd=conn_cmd,
        required_outputs=[
            {"path": out_dir / Path(args.nodes_connected).name, "kind": "nodes_py"},
        ],
        max_retries=1,
        supports_refresh_raw=True,
    )

    # Optionally run static validation, then always run graph correctness on
    # the best available graph for this point in the pipeline.
    latest_correctness_name = Path(args.nodes_connected).name
    static_pass_report: Dict[str, Any] = {
        "skipped": False,
        "input_nodes_file": Path(args.nodes_connected).name,
    }

    static_input_nodes_name = Path(args.nodes_connected).name
    graph_correctness_input_name = static_input_nodes_name
    if args.run_static_validation:
        static_main_name = artifacts["static_validation"]["main_nodes"]
        static_isolated_name = artifacts["static_validation"]["isolated_nodes"]
        static_report_name = artifacts["static_validation"]["report"]
        static_raw_dir = artifacts["static_validation"]["raw_dir"]

        # Static validation adds missing isolated nodes back into the main graph when needed.
        static_validation_cmd = [
            args.python,
            "-m", stage_modules["static_validation"],
            "--env_file",
            args.env_file,
            "--out_dir",
            str(out_dir),
            "--nodes_py",
            static_input_nodes_name,
            "--nodespec_per_file_nodes_py",
            Path(args.nodes_catalog).name,
            "--root_json",
            requested_root_json,
            "--out_main",
            static_main_name,
            "--out_isolated",
            static_isolated_name,
            "--report_out",
            static_report_name,
            "--raw_dir",
            static_raw_dir,
            "--rag_index_dir",
            str(rag_index_dir.resolve()),
            "--rag_max_rounds",
            str(args.rag_max_rounds),
        ]
        if args.model.strip():
            static_validation_cmd.extend(["--model", args.model.strip()])
        if args.refresh_raw:
            static_validation_cmd.append("--refresh_raw")
        _run_validated_stage(
            name="static_validation",
            cmd=static_validation_cmd,
            required_outputs=[
                {"path": out_dir / static_main_name, "kind": "nodes_py"},
            ],
            max_retries=1,
            supports_refresh_raw=True,
        )

        static_report = _read_json_if_exists(out_dir / static_report_name)
        added_nodes = static_report.get("added_nodes") if isinstance(static_report.get("added_nodes"), list) else []
        added_seed_vars = [
            str(x.get("new_var") or "").strip()
            for x in added_nodes
            if isinstance(x, dict) and str(x.get("new_var") or "").strip()
        ]
        static_pass_report = {
            "mode": "static_validation_single_pass",
            "input_nodes_file": static_input_nodes_name,
            "main_output_file": static_main_name,
            "isolated_output_file": static_isolated_name,
            "report_file": static_report_name,
            "added_nodes_count": len(added_nodes),
            "added_seed_vars": added_seed_vars,
            "main_nodes_file": static_main_name,
        }
        graph_correctness_input_name = static_main_name
        if not (out_dir / graph_correctness_input_name).exists():
            raise FileNotFoundError(f"Expected static-validation main graph not found: {out_dir / graph_correctness_input_name}")
    else:
        static_pass_report = {
            "skipped": True,
            "reason": "static_validation_disabled_by_run_pipeline_flag",
            "input_nodes_file": static_input_nodes_name,
            "main_nodes_file": static_input_nodes_name,
        }
        graph_correctness_input_name = static_input_nodes_name

    # Graph correctness normalizes the graph after connectivity/static-validation changes.
    graph_after_static_cmd = [
        args.python,
        "-m", stage_modules["graph_correctness"],
        "--env_file",
        args.env_file,
        "--out_dir",
        str(out_dir),
        "--nodes_py",
        graph_correctness_input_name,
        "--root_json",
        requested_root_json,
        "--out",
        artifacts["after_static_graph_correctness"]["nodes"],
        "--report_out",
        artifacts["after_static_graph_correctness"]["report"],
        "--raw_dir",
        artifacts["after_static_graph_correctness"]["raw_dir"],
        "--phase",
        "before_refine",
        "--model",
        args.model.strip() if args.model.strip() else "gpt-5.2-codex",
        "--rag_index_dir",
        str(rag_index_dir),
        "--rag_max_rounds",
        str(args.rag_max_rounds),
        "--skip_det_controller_rule",
    ]
    if args.refresh_raw:
        graph_after_static_cmd.append("--refresh_raw")
    if "before_refine" in summary_pass_phases:
        graph_after_static_cmd.append("--use_system_summary_pass")
    graph_after_static_result = _run_validated_stage(
        name="graph_correctness:before_refine",
        cmd=graph_after_static_cmd,
        required_outputs=[
            {"path": out_dir / artifacts["after_static_graph_correctness"]["nodes"], "kind": "nodes_py"},
        ],
        max_retries=1,
        supports_refresh_raw=True,
        failure_policy="fallback_to_input",
        fallback_outputs=[
            {"path": out_dir / graph_correctness_input_name, "kind": "nodes_py"},
        ],
    )
    latest_correctness_name = (
        artifacts["after_static_graph_correctness"]["nodes"]
        if graph_after_static_result.get("status") == "success"
        else graph_correctness_input_name
    )

    post_static_pipeline: Dict[str, Any] = {
        "latest_correctness_after_static": latest_correctness_name,
    }

    # Optionally validate execution-grounded existence, apply the resulting
    # actions, then run graph correctness again on the updated graph.
    if args.run_execution_validation:
        entry_path = Path(args.entry).resolve()

        # Execution validation checks the current graph against the runtime trace artifacts.
        execution_validation_cmd = [
            args.python,
            "-m", stage_modules["execution_validation"],
            "--env_file",
            args.env_file,
            "--mode",
            "existence_pass",
            "--out_dir",
            str(out_dir),
            "--entry",
            str(entry_path),
            "--nodes_file",
            latest_correctness_name,
            "--report_out",
            str(execution_validation_report_path),
            "--out",
            str(out_dir / artifacts["after_execution_validation"]["actions_nodes"]),
            "--apply_report_out",
            str(out_dir / artifacts["after_execution_validation"]["actions_report"]),
            "--max_component_iterations",
            str(args.execution_validation_max_rounds),
        ]
        if execution_command_example_source_path is not None:
            execution_validation_cmd.extend([
                "--execution_command_example_json",
                str(execution_command_example_source_path),
            ])
        if args.model.strip():
            execution_validation_cmd.extend(["--model", args.model.strip()])
        if args.refresh_raw:
            execution_validation_cmd.append("--refresh_raw")
        _run_validated_stage(
            name="execution_validation",
            cmd=execution_validation_cmd,
            required_outputs=[
                {"path": out_dir / artifacts["after_execution_validation"]["actions_nodes"], "kind": "nodes_py"},
            ],
            max_retries=1,
            supports_refresh_raw=True,
        )

        # Re-run graph correctness because execution_validation now writes the updated graph directly.
        graph_post_execution_cmd = [
            args.python,
            "-m", stage_modules["graph_correctness"],
            "--env_file",
            args.env_file,
            "--out_dir",
            str(out_dir),
            "--nodes_py",
            artifacts["after_execution_validation"]["actions_nodes"],
            "--root_json",
            requested_root_json,
            "--out",
            artifacts["after_execution_validation"]["corrected_nodes"],
            "--report_out",
            artifacts["after_execution_validation"]["correctness_report"],
            "--raw_dir",
            artifacts["after_execution_validation"]["raw_dir"],
            "--phase",
            "after_execution_validation_actions",
            "--model",
            args.model.strip() if args.model.strip() else "gpt-5.2-codex",
            "--rag_index_dir",
            str(rag_index_dir),
            "--rag_max_rounds",
            str(args.rag_max_rounds),
        ]
        if args.refresh_raw:
            graph_post_execution_cmd.append("--refresh_raw")
        if "after_execution_validation_actions" in summary_pass_phases:
            graph_post_execution_cmd.append("--use_system_summary_pass")
        graph_post_execution_result = _run_validated_stage(
            name="graph_correctness:after_execution_validation_actions",
            cmd=graph_post_execution_cmd,
            required_outputs=[
                {"path": out_dir / artifacts["after_execution_validation"]["corrected_nodes"], "kind": "nodes_py"},
            ],
            max_retries=1,
            supports_refresh_raw=True,
            failure_policy="fallback_to_input",
            fallback_outputs=[
                {"path": out_dir / artifacts["after_execution_validation"]["actions_nodes"], "kind": "nodes_py"},
            ],
        )
        latest_correctness_name = (
            artifacts["after_execution_validation"]["corrected_nodes"]
            if graph_post_execution_result.get("status") == "success"
            else artifacts["after_execution_validation"]["actions_nodes"]
        )
        post_static_pipeline["execution_validation_existence_pass"] = _read_json_if_exists(execution_validation_report_path)
        post_static_pipeline["execution_validation_action_apply"] = _read_json_if_exists(out_dir / artifacts["after_execution_validation"]["actions_report"])
        post_static_pipeline["graph_correctness_after_execution_validation"] = _read_json_if_exists(out_dir / artifacts["after_execution_validation"]["correctness_report"])
    else:
        post_static_pipeline["execution_validation_existence_pass"] = {
            "skipped": True,
            "reason": "disabled_by_run_pipeline_flag",
        }
        post_static_pipeline["execution_validation_action_apply"] = {
            "skipped": True,
            "reason": "disabled_by_run_pipeline_flag",
        }
        post_static_pipeline["graph_correctness_after_execution_validation"] = {
            "skipped": True,
            "reason": "disabled_by_run_pipeline_flag",
        }

    post_static_pipeline["latest_correctness_after_execution"] = latest_correctness_name

    # Optionally refine nodes, re-run graph correctness, refine code references.
    post_processing_input_nodes_name = latest_correctness_name
    if args.run_refinement:
        # Node refinement updates node fields on top of the latest corrected graph.
        refine_cmd = [
            args.python,
            "-m", stage_modules["node_refinement"],
            "--env_file",
            args.env_file,
            "--out_dir",
            str(out_dir),
            "--nodes_py",
            latest_correctness_name,
            "--root_json",
            requested_root_json,
            "--out",
            artifacts["refinement"]["nodes"],
            "--report_out",
            artifacts["refinement"]["report"],
            "--raw_dir",
            "node_refinement_raw_post",
            "--model",
            args.model.strip() if args.model.strip() else "gpt-5.2-codex",
            "--phase",
            "post",
        ]
        _run_validated_stage(
            name="node_refinement",
            cmd=refine_cmd,
            required_outputs=[
                {"path": out_dir / artifacts["refinement"]["nodes"], "kind": "nodes_py"},
            ],
            max_retries=1,
            supports_refresh_raw=False,
        )

        # Re-run graph correctness because node refinement can affect graph structure.
        graph_post_refine_cmd = [
            args.python,
            "-m", stage_modules["graph_correctness"],
            "--env_file",
            args.env_file,
            "--out_dir",
            str(out_dir),
            "--nodes_py",
            artifacts["refinement"]["nodes"],
            "--root_json",
            requested_root_json,
            "--out",
            artifacts["refinement"]["corrected_nodes"],
            "--report_out",
            artifacts["refinement"]["correctness_report"],
            "--raw_dir",
            artifacts["refinement"]["raw_dir"],
            "--phase",
            "after_refine",
            "--model",
            args.model.strip() if args.model.strip() else "gpt-5.2-codex",
            "--rag_index_dir",
            str(rag_index_dir),
            "--rag_max_rounds",
            str(args.rag_max_rounds),
        ]
        if args.refresh_raw:
            graph_post_refine_cmd.append("--refresh_raw")
        if "after_refine" in summary_pass_phases:
            graph_post_refine_cmd.append("--use_system_summary_pass")
        graph_post_refine_result = _run_validated_stage(
            name="graph_correctness:after_refine",
            cmd=graph_post_refine_cmd,
            required_outputs=[
                {"path": out_dir / artifacts["refinement"]["corrected_nodes"], "kind": "nodes_py"},
            ],
            max_retries=1,
            supports_refresh_raw=True,
            failure_policy="fallback_to_input",
            fallback_outputs=[
                {"path": out_dir / artifacts["refinement"]["nodes"], "kind": "nodes_py"},
            ],
        )
        latest_correctness_name = (
            artifacts["refinement"]["corrected_nodes"]
            if graph_post_refine_result.get("status") == "success"
            else artifacts["refinement"]["nodes"]
        )

        # Code-reference refinement improves the final node evidence before export.
        code_reference_refine_cmd = [
            args.python,
            "-m", stage_modules["code_reference_refine"],
            "--env_file",
            args.env_file,
            "--out_dir",
            str(out_dir),
            "--nodes_py",
            latest_correctness_name,
            "--out",
            Path(args.final_nodespec).name,
            "--report_out",
            artifacts["code_reference_refine"]["report"],
            "--final_report_out",
            Path(args.final_report).name,
            "--raw_dir",
            artifacts["code_reference_refine"]["raw_dir"],
            "--rag_index_dir",
            str(rag_index_dir),
            "--rag_max_rounds",
            str(args.rag_max_rounds),
            "--model",
            args.model.strip() if args.model.strip() else "gpt-5.2-codex",
        ]
        if args.refresh_raw:
            code_reference_refine_cmd.append("--refresh_raw")
        _run_validated_stage(
            name="code_reference_refine",
            cmd=code_reference_refine_cmd,
            required_outputs=[
                {"path": out_dir / Path(args.final_nodespec).name, "kind": "nodes_py"},
            ],
            max_retries=1,
            supports_refresh_raw=True,
        )
        post_processing_input_nodes_name = Path(args.final_nodespec).name
        post_static_pipeline["node_refinement_post"] = _read_json_if_exists(out_dir / artifacts["refinement"]["report"])
        post_static_pipeline["graph_correctness_after_refinement"] = _read_json_if_exists(out_dir / artifacts["refinement"]["correctness_report"])
        post_static_pipeline["code_reference_refine"] = _read_json_if_exists(out_dir / artifacts["code_reference_refine"]["report"])

        # Live MCP tool discovery connects to each External_MCP_server's real constructor and
        # replaces its tool_list with the actually-discovered tools; never fails the pipeline
        # over one unreachable server (per-node failures are caught and reported internally).
        if args.run_mcp_tool_discovery:
            mcp_tool_discovery_cmd = [
                args.python,
                "-m", stage_modules["mcp_tool_discovery"],
                "--env_file",
                args.env_file,
                "--out_dir",
                str(out_dir),
                "--nodes_py",
                post_processing_input_nodes_name,
                "--out",
                artifacts["mcp_tool_discovery"]["nodes"],
                "--report_out",
                artifacts["mcp_tool_discovery"]["report"],
                "--raw_dir",
                artifacts["mcp_tool_discovery"]["raw_dir"],
                "--entry",
                str(Path(args.entry).resolve()),
            ]
            if execution_command_example_source_path is not None:
                mcp_tool_discovery_cmd.extend([
                    "--execution_command_example_json",
                    str(execution_command_example_source_path),
                ])
            if args.model.strip():
                mcp_tool_discovery_cmd.extend(["--model", args.model.strip()])
            if args.refresh_raw:
                mcp_tool_discovery_cmd.append("--refresh_raw")
            _run_validated_stage(
                name="mcp_tool_discovery",
                cmd=mcp_tool_discovery_cmd,
                required_outputs=[
                    {"path": out_dir / artifacts["mcp_tool_discovery"]["nodes"], "kind": "nodes_py"},
                ],
                max_retries=0,
                supports_refresh_raw=False,
                failure_policy="fallback_to_input",
                fallback_outputs=[
                    {"path": out_dir / post_processing_input_nodes_name, "kind": "nodes_py"},
                ],
            )
            post_processing_input_nodes_name = artifacts["mcp_tool_discovery"]["nodes"]
            post_static_pipeline["mcp_tool_discovery"] = _read_json_if_exists(out_dir / artifacts["mcp_tool_discovery"]["report"])
        else:
            post_static_pipeline["mcp_tool_discovery"] = {
                "skipped": True,
                "reason": "disabled_by_run_pipeline_flag",
            }
    else:
        post_static_pipeline["node_refinement_post"] = {
            "skipped": True,
            "reason": "disabled_by_run_pipeline_flag",
        }
        post_static_pipeline["graph_correctness_after_refinement"] = {
            "skipped": True,
            "reason": "disabled_by_run_pipeline_flag",
        }
        post_static_pipeline["code_reference_refine"] = {
            "skipped": True,
            "reason": "disabled_by_run_pipeline_flag",
        }
        post_static_pipeline["mcp_tool_discovery"] = {
            "skipped": True,
            "reason": "disabled_by_run_pipeline_flag",
        }

    # Final export: convert the latest graph into the final validated NodeSpec script/json output.
    post_processing_cmd = [
        args.python,
        "-m", stage_modules["post_processing"],
        "--env_file",
        args.env_file,
        "--out_dir",
        str(out_dir),
        "--nodes_py",
        post_processing_input_nodes_name,
        "--root_json",
        requested_root_json,
        "--output_stem",
        Path(args.final_output_stem).name,
    ]
    if execution_command_example_source_path is not None:
        post_processing_cmd.extend([
            "--execution_command_example_json",
            str(execution_command_example_source_path),
        ])
    _run_validated_stage(
        name="post_processing",
        cmd=post_processing_cmd,
        required_outputs=[
            {"path": out_dir / f"{Path(args.final_output_stem).name}.py", "kind": "file_exists"},
        ],
        max_retries=0,
        supports_refresh_raw=False,
    )

    final_spec_py = out_dir / f"{Path(args.final_output_stem).name}.py"
    final_spec_json = out_dir / f"{Path(args.final_output_stem).name}.json"
    _run_validated_stage(
        name="final_spec_json_export",
        cmd=[args.python, str(final_spec_py)],
        required_outputs=[
            {"path": final_spec_json, "kind": "file_exists"},
        ],
        max_retries=0,
        supports_refresh_raw=False,
    )

    post_static_pipeline["latest_correctness_name"] = latest_correctness_name
    post_static_pipeline["post_processing_input_nodes_file"] = post_processing_input_nodes_name
    post_static_pipeline["final_post_processing_output_py"] = f"{Path(args.final_output_stem).name}.py"
    post_static_pipeline["final_post_processing_output_json"] = f"{Path(args.final_output_stem).name}.json"
    post_static_pipeline["final_nodespec_file"] = Path(args.final_nodespec).name
    post_static_pipeline["final_report_file"] = Path(args.final_report).name

    pipeline_report: Dict[str, Any] = {
        "static_validation_and_graph_correctness": {
            **static_pass_report,
            "graph_correctness_after_static_validation": _read_json_if_exists(out_dir / artifacts["after_static_graph_correctness"]["report"]),
        },
        "post_static_pipeline": post_static_pipeline,
        "stage_validation": {
            "max_retries_per_stage": 1,
            "stages": _STAGE_VALIDATIONS,
        },
    }

    (out_dir / "pipeline_report.json").write_text(
        json.dumps(pipeline_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n[run_pipeline] wrote: {out_dir / 'pipeline_report.json'}")

    _write_step_timings(out_dir, announce=True)


if __name__ == "__main__":
    main()
