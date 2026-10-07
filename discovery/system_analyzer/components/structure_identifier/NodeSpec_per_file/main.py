#!/usr/bin/env python3
"""NodeSpec_per_file pipeline."""

import argparse
import ast
import json
import os
import sys
import time

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
from pathlib import Path
from typing import Any, Dict, List

try:
    from ..global_utils import (
        load_env_file,
        load_identity_map,
        load_nodes_list_from_python,
        read_json,
        resolve_var_name,
        save_identity_map,
        sort_node_dicts,
        validate_minimal_nodes,
    )
    from ..model_client import call_model
    from .prompts import DEFAULT_PROMPT
    from .utils import (
        _fallback_empty_nodes_file,
        _validate_nodespec_per_file_output_text,
        _read_file as read_file,
        enrich_node_code_references,
        ensure_missing_evidence_from_code_refs,
        fmt_seconds,
        normalize_ref_file_path,
        prefer_file_scoped_var_name,
        render_nodespec,
    )
except ImportError:
    from global_utils import (
        load_env_file,
        load_identity_map,
        load_nodes_list_from_python,
        read_json,
        resolve_var_name,
        save_identity_map,
        sort_node_dicts,
        validate_minimal_nodes,
    )
    from model_client import call_model
    from prompts import DEFAULT_PROMPT
    from utils import (
        _fallback_empty_nodes_file,
        _validate_nodespec_per_file_output_text,
        _read_file as read_file,
        enrich_node_code_references,
        ensure_missing_evidence_from_code_refs,
        fmt_seconds,
        normalize_ref_file_path,
        prefer_file_scoped_var_name,
        render_nodespec,
    )


def main() -> None:
    """Run the per-file extraction stage and combine the generated node outputs."""
    default_out_dir = Path(__file__).resolve().parents[1] / "stages_outputs"

    # Parse CLI inputs and normalize the output artifact locations for this stage.
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--env_file",
        default=".env",
        help="Optional .env file to load.",
    )
    ap.add_argument(
        "--out_dir",
        default=str(default_out_dir),
        help="Outputs directory.",
    )
    ap.add_argument(
        "--reachable_record",
        default="reachable_files.json",
        help="Reachable files JSON (filename under --out_dir).",
    )
    ap.add_argument(
        "--index_file",
        default="nodes_index.json",
        help="Index JSON filename under --out_dir.",
    )
    ap.add_argument(
        "--combined_out",
        default="nodes_catalog.py",
        help="Combined python file output filename under --out_dir.",
    )
    ap.add_argument(
        "--identity_map",
        default="nodes_identity_map.json",
        help="Node identity mapping filename under --out_dir.",
    )
    ap.add_argument("--model", default=os.environ.get("CODEX_MODEL", "gpt-5.2-codex"))
    ap.add_argument(
        "--execution_command_example_json",
        default="",
        help="Optional path to execution command example JSON for argument disambiguation context.",
    )
    ap.add_argument(
        "--max_file_bytes",
        type=int,
        default=50000,
        help="Max bytes per file.",
    )
    ap.add_argument(
        "--max_output_retries",
        type=int,
        default=3,
        help="Max retries to regenerate a valid Python NEW_NODES output per file when output is malformed.",
    )
    ap.add_argument(
        "--prompt",
        default="",
        help="Optional prompt file to use.",
    )
    ap.add_argument(
        "--refresh_raw",
        "--refresh-raw",
        dest="refresh_raw",
        action="store_true",
        help="Ignore cached per-file raw outputs and call model again.",
    )
    args = ap.parse_args()

    # Load environment defaults and resolve the core stage file paths.
    load_env_file(Path(args.env_file))

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reachable_record = out_dir / Path(args.reachable_record).name
    nodes_out_dir = out_dir / "nodes_per_file_outputs"
    raw_artifacts_dir = out_dir / "nodes_per_file_raw"
    index_path = out_dir / Path(args.index_file).name
    combined_path = out_dir / Path(args.combined_out).name
    identity_map_path = out_dir / Path(args.identity_map).name
    stage_report_path = out_dir / "nodes_per_file_report.json"
    identity_map = load_identity_map(identity_map_path)

    # Load reachable-file inputs and any optional execution-command context for model disambiguation.
    reachable = read_json(reachable_record)
    execution_command_example: Dict[str, Any] | None = None
    if isinstance(args.execution_command_example_json, str) and args.execution_command_example_json.strip():
        ec_path = Path(args.execution_command_example_json).resolve()
        if ec_path.exists():
            try:
                ec_obj = json.loads(ec_path.read_text(encoding="utf-8"))
                if isinstance(ec_obj, dict):
                    execution_command_example = ec_obj
            except Exception:
                execution_command_example = None
    project_root = Path(str(reachable.get("project_root") or "")).resolve()
    if not str(project_root):
        project_root = Path.cwd().resolve()
    reachable_py = reachable.get("reachable_py_abs", []) or []
    reachable_json = reachable.get("reachable_json_abs", []) or []

    schema_path = Path(__file__).resolve().parents[1] / "NodeSpec_schema_short.py"
    schema_text = schema_path.read_text(encoding="utf-8", errors="replace") if schema_path.exists() else ""

    # Load the schema and prompt text that will be sent for per-file extraction.
    prompt = DEFAULT_PROMPT
    if args.prompt:
        prompt = Path(args.prompt).read_text(encoding="utf-8")

    nodes_out_dir.mkdir(parents=True, exist_ok=True)
    raw_artifacts_dir.mkdir(parents=True, exist_ok=True)
    combined_path.parent.mkdir(parents=True, exist_ok=True)

    # Track per-file outputs and emit a progress index as the extraction loop runs.
    index_entries: List[Dict[str, Any]] = []
    file_reports: List[Dict[str, Any]] = []

    stage_start = time.perf_counter()
    files = sorted(reachable_py + reachable_json)
    total_files = len(files)
    print(f"[NodeSpec_per_file] start: {total_files} files")
    iterator = tqdm(files, desc="Processing files") if tqdm else files
    per_file_outputs: List[Path] = []

    # Generate one `NEW_NODES` file per reachable source file, with repair/fallback guards for bad model output.
    for i, fp in enumerate(iterator, start=1):
        file_start = time.perf_counter()
        file_name = Path(fp).name
        file_tag = f"{i:03d}_{file_name}"
        out_path = nodes_out_dir / f"{file_tag}.py"
        raw_file_dir = raw_artifacts_dir / file_tag
        raw_file_dir.mkdir(parents=True, exist_ok=True)
        mode = "cached"
        guard_status = "ok"
        guard_attempts = 0
        invalid_reason = ""
        llm_calls = 0
        if out_path.exists() and not args.refresh_raw:
            result_text = out_path.read_text(encoding="utf-8", errors="replace")
            if tqdm:
                iterator.set_postfix_str(f"{i}/{total_files} cached")
        else:
            mode = "llm"
            file_text = read_file(fp, args.max_file_bytes)
            if not file_text:
                elapsed_file = time.perf_counter() - file_start
                total_elapsed = time.perf_counter() - stage_start
                done = i
                avg = total_elapsed / done if done else 0.0
                eta = avg * (total_files - done)
                print(
                    f"[NodeSpec_per_file] [{i}/{total_files}] skipped {Path(fp).name} "
                    f"in {fmt_seconds(elapsed_file)} | elapsed={fmt_seconds(total_elapsed)} "
                    f"eta={fmt_seconds(eta)}"
                )
                file_report = {
                    "file_index": i,
                    "source_file": fp,
                    "output_file": str(out_path),
                    "raw_dir": str(raw_file_dir),
                    "mode": "skipped",
                    "reason": "file_read_failed_or_empty",
                    "guard_status": guard_status,
                    "guard_attempts": guard_attempts,
                    "invalid_reason": invalid_reason,
                    "llm_calls": llm_calls,
                }
                (raw_file_dir / "meta.json").write_text(
                    json.dumps(file_report, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                index_entries.append(
                    {
                        "file_index": i,
                        "source_file": fp,
                        "output_file": str(out_path),
                        "mode": "skipped",
                        "reason": "file_read_failed_or_empty",
                    }
                )
                file_reports.append(file_report)
                index_path.parent.mkdir(parents=True, exist_ok=True)
                index_path.write_text(json.dumps(index_entries, indent=2, ensure_ascii=False), encoding="utf-8")
                continue
            payload = {
                "prompt": prompt,
                "nodespec_schema": schema_text,
                "current_file": {"path": fp, "content": file_text},
                "execution_command_example": execution_command_example,
            }
            (raw_file_dir / "attempt00.payload.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            llm_calls += 1
            result_text = call_model(payload, args.model)
            (raw_file_dir / "attempt00.txt").write_text(result_text, encoding="utf-8")

        # Guardrail: ensure per-file output is valid Python and contains NEW_NODES.
        valid, guard_err = _validate_nodespec_per_file_output_text(result_text)
        invalid_reason = str(guard_err or "")
        while not valid and guard_attempts < max(1, args.max_output_retries):
            guard_attempts += 1
            mode = "llm_repair"
            repair_payload = {
                "prompt": (
                    prompt
                    + "\n\nSTRICT OUTPUT REPAIR:\n"
                    + "- Return valid Python code only.\n"
                    + "- Must include top-level assignment: NEW_NODES = [...].\n"
                    + "- No markdown fences and no prose.\n"
                    + "- Keep schema-compatible NodeSpec construction.\n"
                ),
                "nodespec_schema": schema_text,
                "current_file": {"path": fp, "content": read_file(fp, args.max_file_bytes) or ""},
                "execution_command_example": execution_command_example,
                "previous_invalid_output": result_text,
                "invalid_reason": guard_err,
            }
            (raw_file_dir / f"attempt{guard_attempts:02d}_repair.payload.json").write_text(
                json.dumps(repair_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            llm_calls += 1
            result_text = call_model(repair_payload, args.model)
            (raw_file_dir / f"attempt{guard_attempts:02d}_repair.txt").write_text(
                result_text,
                encoding="utf-8",
            )
            valid, guard_err = _validate_nodespec_per_file_output_text(result_text)
            invalid_reason = str(guard_err or "")
        if not valid:
            guard_status = "fallback_empty_nodes"
            mode = "fallback"
            result_text = _fallback_empty_nodes_file()
        elif guard_attempts > 0:
            guard_status = "repaired"

        out_path.write_text(result_text, encoding="utf-8")

        elapsed_file = time.perf_counter() - file_start
        file_report = {
            "file_index": i,
            "source_file": fp,
            "output_file": str(out_path),
            "raw_dir": str(raw_file_dir),
            "mode": mode,
            "guard_status": guard_status,
            "guard_attempts": guard_attempts,
            "invalid_reason": invalid_reason,
            "llm_calls": llm_calls,
            "elapsed_seconds": elapsed_file,
            "output_bytes": len(result_text.encode("utf-8", errors="replace")),
        }
        (raw_file_dir / "meta.json").write_text(
            json.dumps(file_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        total_elapsed = time.perf_counter() - stage_start
        done = i
        avg = total_elapsed / done if done else 0.0
        eta = avg * (total_files - done)
        print(
            f"[NodeSpec_per_file] [{i}/{total_files}] {mode} {Path(fp).name} -> {out_path.name} "
            f"in {fmt_seconds(elapsed_file)} | elapsed={fmt_seconds(total_elapsed)} "
            f"eta={fmt_seconds(eta)}"
        )
        per_file_outputs.append(out_path)
        index_entries.append(
            {
                "file_index": i,
                "source_file": fp,
                "output_file": str(out_path),
                "mode": mode,
                "guard_status": guard_status,
                "guard_attempts": guard_attempts,
                "output_bytes": len(result_text.encode("utf-8", errors="replace")),
                "output_preview": result_text[:200],
            }
        )
        file_reports.append(file_report)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(index_entries, indent=2, ensure_ascii=False), encoding="utf-8")

    # Re-parse, normalize, and combine all per-file node outputs into one deterministic catalog.
    combine_start = time.perf_counter()
    print(f"[NodeSpec_per_file] combine start: {len(per_file_outputs)} per-file outputs")
    combined_path.write_text(
        "from NodeSpec_schema import NodeSpec, NodeType, CodeReference, InputPort, OutputPort\n\n",
        encoding="utf-8",
    )

    all_nodes: List[Dict[str, Any]] = []
    for out_path in per_file_outputs:
        try:
            parsed_nodes = load_nodes_list_from_python(out_path, list_name="NEW_NODES")
        except Exception as exc:
            print(f"[NodeSpec_per_file] failed to parse {out_path}: {exc}", file=sys.stderr)
            continue

        for node_idx, node_dict in enumerate(parsed_nodes, start=1):
            if not isinstance(node_dict, dict):
                continue
            node_dict.setdefault("code_references", [])
            node_dict.setdefault("inputs", [])
            node_dict.setdefault("outputs", [])
            refs = list(node_dict.get("code_references") or [])
            cleaned_refs: List[Dict[str, Any]] = []
            for ref in refs:
                if not isinstance(ref, dict):
                    continue
                file_val = ref.get("file")
                line_val = ref.get("line")
                snippet_val = ref.get("snippet")
                if not file_val:
                    continue
                if isinstance(file_val, str) and file_val.strip():
                    file_val = normalize_ref_file_path(file_val, project_root)
                    ref["file"] = file_val
                cleaned_refs.append(ref)
            node_dict["code_references"] = cleaned_refs
            node_dict["code_references"] = enrich_node_code_references(
                node_dict,
                project_root=str(project_root),
                reachable_py=reachable_py,
            )
            node_dict = ensure_missing_evidence_from_code_refs(node_dict)
            all_nodes.append(node_dict)

    # Guardrail: normalize malformed parsed entries instead of crashing combine.
    required_fields = ["name", "node_type", "description", "code_references", "inputs", "outputs"]
    valid_nodes: List[Dict[str, Any]] = []
    dropped_nodes = 0
    normalized_nodes = 0
    for idx, node in enumerate(all_nodes, start=1):
        candidate = node if isinstance(node, dict) else None
        # Common bad shape from JSON wrappers: {"root_node": {...}}
        if (
            isinstance(candidate, dict)
            and "root_node" in candidate
            and isinstance(candidate.get("root_node"), dict)
        ):
            candidate = candidate["root_node"]
        if not isinstance(candidate, dict):
            dropped_nodes += 1
            print(f"[NodeSpec_per_file] dropping malformed node #{idx}: not a dict", file=sys.stderr)
            continue

        changed = False
        if not isinstance(candidate.get("name"), str) or not candidate.get("name", "").strip():
            candidate["name"] = f"unknown_node_{idx}"
            changed = True
        if candidate.get("node_type") is None:
            candidate["node_type"] = {"type": "other"}
            changed = True
        elif isinstance(candidate.get("node_type"), str):
            candidate["node_type"] = {"type": str(candidate.get("node_type"))}
            changed = True
        if not isinstance(candidate.get("description"), str) or not candidate.get("description", "").strip():
            candidate["description"] = "No description available."
            changed = True
        if not isinstance(candidate.get("code_references"), list):
            candidate["code_references"] = []
            changed = True
        if not isinstance(candidate.get("inputs"), list):
            candidate["inputs"] = []
            changed = True
        if not isinstance(candidate.get("outputs"), list):
            candidate["outputs"] = []
            changed = True

        if changed:
            normalized_nodes += 1
            print(f"[NodeSpec_per_file] normalized malformed node #{idx}", file=sys.stderr)
        valid_nodes.append(candidate)

    all_nodes = sort_node_dicts(valid_nodes)
    validate_minimal_nodes(all_nodes, required_fields)

    # Emit stable variable names and write the final Python catalog plus `ALL_NODES` list.
    with combined_path.open("a", encoding="utf-8") as fh:
        used_names: set[str] = set()
        emitted: List[tuple[str, Dict[str, Any]]] = []
        for item in all_nodes:
            file_path = ""
            refs = item.get("code_references")
            if isinstance(refs, list) and refs:
                first = refs[0]
                if isinstance(first, dict) and isinstance(first.get("file"), str):
                    file_path = first["file"]
            prefer_file_scoped_var_name(item.get("name"), file_path, used_names, identity_map)
            var = resolve_var_name(item.get("name"), file_path, used_names, identity_map)
            emitted.append((var, item))

        emitted.sort(key=lambda x: x[0])
        var_names: List[str] = []
        for var, item in emitted:
            fh.write(f"{var} = ")
            fh.write(render_nodespec(item).lstrip())
            fh.write("\n\n")
            var_names.append(var)
        fh.write("ALL_NODES = [\n")
        for var in var_names:
            fh.write(f"    {var},\n")
        fh.write("]\n")

    if schema_path.exists():
        (out_dir / "NodeSpec_schema.py").write_text(schema_text, encoding="utf-8")

    # Persist the identity map and summarize the stage outcome for downstream steps.
    save_identity_map(identity_map_path, identity_map)
    stage_report_path.write_text(
        json.dumps({"files": file_reports}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    total_elapsed = time.perf_counter() - stage_start
    combine_elapsed = time.perf_counter() - combine_start
    print(
        f"[NodeSpec_per_file] done: files={total_files}, outputs={len(per_file_outputs)} "
        f"combine={fmt_seconds(combine_elapsed)} total={fmt_seconds(total_elapsed)} "
        f"all_nodes={len(all_nodes)} normalized_nodes={normalized_nodes} dropped_nodes={dropped_nodes}"
    )
    print(f"[NodeSpec_per_file] raw artifacts: {raw_artifacts_dir}")

if __name__ == "__main__":
    main()
