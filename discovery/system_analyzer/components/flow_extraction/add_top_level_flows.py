#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from node_spec.structure_schema import FlowSpec, NodeSpec

try:
    from .global_utils import (
        load_nodes_with_vars_from_python,
        render_nodespec_assignment,
    )
    from .flow_query_generation import _flow_signature
    from .save_simple_flow import build_simple_flow
    from .trace_parsing import parse_trace_file
except ImportError:
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[4]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from discovery.system_analyzer.components.flow_extraction.global_utils import (
        load_nodes_with_vars_from_python,
        render_nodespec_assignment,
    )
    from discovery.system_analyzer.components.flow_extraction.flow_query_generation import _flow_signature
    from discovery.system_analyzer.components.flow_extraction.save_simple_flow import build_simple_flow
    from discovery.system_analyzer.components.flow_extraction.trace_parsing import parse_trace_file


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _execution_validation_runs_root(out_dir: Path) -> Path:
    candidates = (
        out_dir / "validation" / "execution_validation_runs",
        out_dir / "execution_validation" / "execution_validation_runs",
    )
    return next((path for path in candidates if path.is_dir()), candidates[0])


def _events_from_llm_span_pipeline_file(path: Path) -> List[Dict[str, Any]]:
    obj = _read_json(path)
    if not isinstance(obj, dict):
        return []
    for key in ("canonical_events", "events", "parsed_events"):
        rows = obj.get(key)
        if isinstance(rows, list):
            return [r for r in rows if isinstance(r, dict)]
    for outer in ("result", "data", "output"):
        box = obj.get(outer)
        if not isinstance(box, dict):
            continue
        for key in ("canonical_events", "events", "parsed_events"):
            rows = box.get(key)
            if isinstance(rows, list):
                return [r for r in rows if isinstance(r, dict)]
    return []


def _relabel_input_only_agent_events_to_system(parsed_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flow-only relabeling to reduce prompt-scaffolding noise:
    relabel agent events to system only when all guards pass:
    - type == agent
    - path is under spanInputs
    - no tool_calls
    - same span_id has at least one spanOutputs event
    """
    if not isinstance(parsed_obj, dict):
        return parsed_obj
    events = parsed_obj.get("events")
    if not isinstance(events, list):
        return parsed_obj

    output_span_ids: Set[str] = set()
    for e in events:
        if not isinstance(e, dict):
            continue
        p = str(e.get("path") or "")
        sid = str(e.get("span_id") or "").strip()
        if sid and "spanOutputs" in p:
            output_span_ids.add(sid)

    relabeled: List[Dict[str, Any]] = []
    for e in events:
        if not isinstance(e, dict):
            relabeled.append(e)
            continue
        ee = dict(e)
        et = str(ee.get("type") or "").strip().lower()
        p = str(ee.get("path") or "")
        sid = str(ee.get("span_id") or "").strip()
        tc = ee.get("tool_calls")
        has_tool_calls = isinstance(tc, list) and len(tc) > 0

        if (
            et == "agent"
            and "spanInputs" in p
            and not has_tool_calls
            and sid
            and sid in output_span_ids
        ):
            ee["type"] = "system"
        relabeled.append(ee)

    out = dict(parsed_obj)
    out["events"] = relabeled
    return out


def _load_inventory_trace_names_from_execution_validation(out_dir: Path) -> Set[str]:
    """
    Returns trace basenames that correspond to inventory-target validation runs.
    Source of truth: run_XXX.target_component.json + run_XXX.trace_path.txt
    under validation/execution_validation_runs/**.
    """
    run_root = _execution_validation_runs_root(out_dir)
    if not run_root.exists():
        return set()

    inventory_traces: Set[str] = set()
    for target_file in sorted(run_root.glob("**/run_*.target_component.json")):
        trace_ref = target_file.with_name(target_file.name.replace(".target_component.json", ".trace_path.txt"))
        if not trace_ref.exists():
            continue
        target_obj = _read_json(target_file)
        if not isinstance(target_obj, dict):
            continue
        comp_kind = str(target_obj.get("component_kind") or "").strip().lower()
        if comp_kind != "inventory":
            continue
        trace_path_text = trace_ref.read_text(encoding="utf-8", errors="replace").strip()
        if not trace_path_text:
            continue
        trace_name = Path(trace_path_text).name
        if trace_name:
            inventory_traces.add(trace_name)
    return inventory_traces


def _normalize_flow_obj(flow_obj: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(flow_obj)
    out.pop("nodespec_file", None)
    if not isinstance(out.get("input_args"), dict):
        out["input_args"] = {}
    if not isinstance(out.get("events"), list):
        out["events"] = []
    if not isinstance(out.get("invoked_tools"), list):
        out["invoked_tools"] = []
    if not isinstance(out.get("invoked_agents"), list):
        out["invoked_agents"] = []
    out["parsed_trace_file"] = str(out.get("parsed_trace_file") or "")
    out["source_trace_file"] = str(out.get("source_trace_file") or "")
    # Assigned later as deterministic integer IDs grouped by flow structure.
    out["flow_id"] = 0
    return out


def _flow_trace_name_from_source(source_trace_file: str, fallback_stem: str) -> str:
    src = str(source_trace_file or "").strip()
    if src:
        return f"flow_{Path(src).stem}.json"
    return f"flow_{fallback_stem}.json"


def _load_existing_flow_trace_files(out_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    files: List[Path] = []
    files.extend(sorted(_execution_validation_runs_root(out_dir).glob("**/flow_trace_*.json")))
    files.extend(
        sorted(
            (
                out_dir
                / "flow_extraction"
                / "flow_query_generation"
                / "generated_runs"
                / "parsed_traces"
            ).glob("flow_trace_*.json")
        )
    )
    out: List[Tuple[Path, Dict[str, Any]]] = []
    for p in files:
        obj = _read_json(p)
        if isinstance(obj, dict):
            if not str(obj.get("source_trace_file") or "").strip():
                stem = p.stem
                if stem.startswith("flow_trace_"):
                    obj["source_trace_file"] = "trace_" + stem[len("flow_trace_") :] + ".json"
            out.append((p, obj))
    return out


def _collect_validation_simple_flows(
    *,
    out_dir: Path,
    nodespec_path: Optional[Path],
    node_spec: NodeSpec | None,
    runtime_mapping_file: Optional[Path],
    seen_keys: Set[Tuple[str, str]],
    write_flow_trace_files: bool,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    ev_root = _execution_validation_runs_root(out_dir)
    if not ev_root.exists():
        return out

    for llm_path in sorted(ev_root.glob("**/run_*.llm_span_pipeline.json")):
        run_stem = llm_path.name.replace(".llm_span_pipeline.json", "")
        trace_path_txt = llm_path.with_name(run_stem + ".trace_path.txt")
        source_trace_file = ""
        trace_path_abs: Optional[Path] = None
        if trace_path_txt.exists():
            trace_path_raw = trace_path_txt.read_text(encoding="utf-8", errors="replace").strip()
            if trace_path_raw:
                source_trace_file = Path(trace_path_raw).name
                cand = Path(trace_path_raw)
                if cand.is_absolute() and cand.exists():
                    trace_path_abs = cand.resolve()
                else:
                    trace_roots = (
                        out_dir / "validation" / "traces",
                        ev_root,
                    )
                    for trace_root in trace_roots:
                        matches = sorted(trace_root.glob(f"**/{source_trace_file}")) if trace_root.exists() else []
                        if matches:
                            trace_path_abs = matches[0].resolve()
                            break
        if not source_trace_file:
            source_trace_file = f"{run_stem}.json"

        dedupe_key = (source_trace_file, str(llm_path.resolve()))
        if dedupe_key in seen_keys:
            continue

        # Prefer flow_extraction parser for validation traces to ensure consistent
        # deterministic name mapping (e.g., assistant -> canonical agent).
        parsed_obj: Dict[str, Any] | None = None
        if trace_path_abs is not None and trace_path_abs.exists():
            try:
                parsed_obj = parse_trace_file(
                    trace_file=trace_path_abs,
                    node_spec=node_spec,
                    nodespec_file=nodespec_path,
                    runtime_mapping_file=runtime_mapping_file,
                )
            except Exception:
                parsed_obj = None

        # Fallback to llm_span_pipeline events only when trace parsing is unavailable.
        if parsed_obj is None:
            events = _events_from_llm_span_pipeline_file(llm_path)
            if not events:
                continue
            parsed_obj = {"trace_file": source_trace_file, "events": events}
        parsed_obj = _relabel_input_only_agent_events_to_system(parsed_obj)

        try:
            flow_obj = build_simple_flow(
                parsed_obj=parsed_obj,
                parsed_path=llm_path,
                nodespec_path=nodespec_path,
                node_spec=node_spec,
            )
        except Exception:
            # Compatibility fallback for legacy final_nodespec variants that cannot
            # be fully validated/imported by strict schema loading.
            flow_obj = build_simple_flow(
                parsed_obj=parsed_obj,
                parsed_path=llm_path,
                nodespec_path=None,
                node_spec=node_spec,
            )
        flow_obj = _normalize_flow_obj(flow_obj)
        flow_obj["parsed_trace_file"] = str(llm_path.resolve())
        flow_obj["source_trace_file"] = source_trace_file
        if write_flow_trace_files:
            out_file = llm_path.with_name(
                _flow_trace_name_from_source(source_trace_file, run_stem)
            )
            out_file.write_text(
                json.dumps(flow_obj, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        seen_keys.add(dedupe_key)
        out.append(flow_obj)
    return out


def _collect_generated_simple_flows(
    *,
    out_dir: Path,
    nodespec_path: Optional[Path],
    node_spec: NodeSpec | None,
    seen_keys: Set[Tuple[str, str]],
    write_flow_trace_files: bool,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    parsed_root = (
        out_dir
        / "flow_extraction"
        / "flow_query_generation"
        / "generated_runs"
        / "parsed_traces"
    )
    if not parsed_root.exists():
        return out

    for parsed_path in sorted(parsed_root.glob("parsed_trace_*.json")):
        if parsed_path.name.endswith(".debug.json"):
            continue
        parsed_obj = _read_json(parsed_path)
        if not isinstance(parsed_obj, dict):
            continue
        parsed_obj = _relabel_input_only_agent_events_to_system(parsed_obj)
        source_trace_file = str(parsed_obj.get("trace_file") or "").strip()
        if not source_trace_file:
            source_trace_file = parsed_path.name.replace("parsed_", "")
        dedupe_key = (source_trace_file, str(parsed_path.resolve()))
        if dedupe_key in seen_keys:
            continue

        try:
            flow_obj = build_simple_flow(
                parsed_obj=parsed_obj,
                parsed_path=parsed_path,
                nodespec_path=nodespec_path,
                node_spec=node_spec,
            )
        except Exception:
            # Compatibility fallback for legacy final_nodespec variants that cannot
            # be fully validated/imported by strict schema loading.
            flow_obj = build_simple_flow(
                parsed_obj=parsed_obj,
                parsed_path=parsed_path,
                nodespec_path=None,
                node_spec=node_spec,
            )
        flow_obj = _normalize_flow_obj(flow_obj)
        if write_flow_trace_files:
            out_file = parsed_path.with_name(
                _flow_trace_name_from_source(source_trace_file, parsed_path.stem)
            )
            out_file.write_text(
                json.dumps(flow_obj, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        seen_keys.add(dedupe_key)
        out.append(flow_obj)
    return out


def _pick_top_level_var(var_order: List[str], nodes_by_var: Dict[str, Dict[str, Any]]) -> str:
    all_vars = set(var_order)
    referenced: Set[str] = set()
    for var in var_order:
        node = nodes_by_var.get(var) or {}
        for fld in ("nodes", "tool_list"):
            vals = node.get(fld)
            if not isinstance(vals, list):
                continue
            for x in vals:
                if isinstance(x, str) and x in all_vars:
                    referenced.add(x)

    candidates = [v for v in var_order if v not in referenced]
    if not candidates:
        candidates = list(var_order)

    graph_candidates = [
        v for v in candidates if bool((nodes_by_var.get(v) or {}).get("is_graph", False))
    ]
    if graph_candidates:
        candidates = graph_candidates

    for tname in ("System", "Agent"):
        for v in candidates:
            n = nodes_by_var.get(v) or {}
            nt = n.get("node_type") if isinstance(n.get("node_type"), dict) else {}
            if str(nt.get("type") or "") == tname:
                return v
    return candidates[0]


def _rewrite_nodes_file(
    *,
    nodes_path: Path,
    var_order: List[str],
    nodes_by_var: Dict[str, Dict[str, Any]],
) -> None:
    source = nodes_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(nodes_path))
    lines = source.splitlines(keepends=True)

    first_node_start = None
    all_nodes_end = None
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            if (
                len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id in set(var_order)
                and first_node_start is None
            ):
                first_node_start = int(stmt.lineno)
            if any(isinstance(t, ast.Name) and t.id == "ALL_NODES" for t in stmt.targets):
                all_nodes_end = int(getattr(stmt, "end_lineno", stmt.lineno))

    if first_node_start is None:
        first_node_start = 1
    if all_nodes_end is None:
        all_nodes_end = len(lines)

    header = "".join(lines[: first_node_start - 1])
    suffix = "".join(lines[all_nodes_end:])

    out_parts: List[str] = [header]
    for var in var_order:
        node = dict(nodes_by_var[var])
        node.pop("__var__", None)
        out_parts.append(render_nodespec_assignment(var, node))
        out_parts.append("\n\n")

    out_parts.append("ALL_NODES = [\n")
    for var in var_order:
        out_parts.append(f"    {var},\n")
    out_parts.append("]\n")
    if suffix and not suffix.startswith("\n"):
        out_parts.append("\n")
    out_parts.append(suffix)
    nodes_path.write_text("".join(out_parts), encoding="utf-8")


def _resolve_default_nodespec(out_dir: Path) -> Path:
    m = re.search(r"use_case_(\d+)_outputs_\d+$", out_dir.name)
    if m:
        uc = m.group(1)
        spec_cand = (Path("use_cases") / f"use_case_{uc}" / f"use_case_{uc}_spec.py").resolve()
        if spec_cand.exists():
            return spec_cand
    spec_hits = sorted(out_dir.glob("*_spec.py"))
    if spec_hits:
        return spec_hits[0].resolve()
    for name in ("nodespec.py", "final_nodespec.py"):
        cand = (out_dir / name).resolve()
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Could not resolve default nodespec/spec file for {out_dir}")


def add_top_level_flows_from_out_dir(
    *,
    out_dir: Path,
    nodes_file: str = "",
    node_spec: NodeSpec | None = None,
    exclude_inventory_queries: bool = True,
    write_flow_trace_files: bool = True,
    min_examples_per_flow_id: int = 1,
) -> Dict[str, Any]:
    out_dir = Path(out_dir).resolve()
    nodes_path: Optional[Path] = None
    if node_spec is None:
        if str(nodes_file or "").strip():
            np = Path(nodes_file)
            nodes_path = np.resolve() if np.is_absolute() else (out_dir / np).resolve()
        else:
            nodes_path = _resolve_default_nodespec(out_dir)
        if not nodes_path.exists():
            raise FileNotFoundError(f"nodes file not found: {nodes_path}")
    runtime_mapping_file = (out_dir / "validation" / "runtime_mapping.json").resolve()
    if not runtime_mapping_file.exists():
        runtime_mapping_file = None

    seen_keys: Set[Tuple[str, str]] = set()
    flows: List[Dict[str, Any]] = []
    inventory_trace_names = _load_inventory_trace_names_from_execution_validation(out_dir)

    flows.extend(
        _collect_validation_simple_flows(
            out_dir=out_dir,
            nodespec_path=nodes_path,
            node_spec=node_spec,
            runtime_mapping_file=runtime_mapping_file,
            seen_keys=seen_keys,
            write_flow_trace_files=write_flow_trace_files,
        )
    )
    flows.extend(
        _collect_generated_simple_flows(
            out_dir=out_dir,
            nodespec_path=nodes_path,
            node_spec=node_spec,
            seen_keys=seen_keys,
            write_flow_trace_files=write_flow_trace_files,
        )
    )

    # Fallback only: keep pre-existing flow_trace files when canonical sources
    # are not available for a trace.
    seen_sources: Set[str] = set()
    for f in flows:
        if not isinstance(f, dict):
            continue
        src = str(f.get("source_trace_file") or "").strip()
        if src:
            seen_sources.add(src)
    for p, obj in _load_existing_flow_trace_files(out_dir):
        norm = _normalize_flow_obj(obj)
        src = str(norm.get("source_trace_file") or "").strip()
        if src and src in seen_sources:
            continue
        dedupe_key = (str(norm.get("source_trace_file", "")), str(norm.get("parsed_trace_file", "") or p.resolve()))
        if dedupe_key in seen_keys:
            continue
        if src:
            seen_sources.add(src)
        seen_keys.add(dedupe_key)
        flows.append(norm)

    filtered: List[Dict[str, Any]] = []
    skipped_inventory = 0
    for f in flows:
        src_trace = str(f.get("source_trace_file") or "").strip()
        if (
            exclude_inventory_queries
            and src_trace
            and src_trace in inventory_trace_names
        ):
            skipped_inventory += 1
            continue
        filtered.append(f)

    min_examples = max(1, int(min_examples_per_flow_id))
    key_by_row: List[str] = [
        str(_flow_signature(f.get("events") if isinstance(f.get("events"), list) else [])[0])
        for f in filtered
    ]

    if min_examples > 1:
        counts: Dict[str, int] = {}
        for k in key_by_row:
            counts[k] = int(counts.get(k, 0)) + 1
        keep_keys = {k for k, c in counts.items() if int(c) >= min_examples}
        kept_rows: List[Dict[str, Any]] = []
        kept_keys: List[str] = []
        for f, k in zip(filtered, key_by_row):
            if k in keep_keys:
                kept_rows.append(f)
                kept_keys.append(k)
        filtered = kept_rows
        key_by_row = kept_keys

    # Assign deterministic integer flow IDs (0..N-1) by unique flow structure.
    # Same structure across different traces => same flow_id.
    flow_key_to_id: Dict[str, int] = {}
    unique_keys = sorted(set(key_by_row))
    for idx, k in enumerate(unique_keys):
        flow_key_to_id[k] = idx
    for f, k in zip(filtered, key_by_row):
        f["flow_id"] = int(flow_key_to_id.get(k, 0))

    filtered.sort(
        key=lambda x: (
            int(x.get("flow_id") or 0),
            str(x.get("source_trace_file") or ""),
            str(x.get("parsed_trace_file") or ""),
        )
    )

    top_var = ""
    if node_spec is not None:
        node_spec.flows = [FlowSpec(**f) for f in filtered]
    else:
        assert nodes_path is not None
        var_order, nodes = load_nodes_with_vars_from_python(nodes_path, list_name="ALL_NODES")
        nodes_by_var = {v: dict(n) for v, n in zip(var_order, nodes)}
        top_var = _pick_top_level_var(var_order, nodes_by_var)
        top_node = nodes_by_var.get(top_var)
        if not isinstance(top_node, dict):
            raise RuntimeError(f"Failed resolving top-level node var: {top_var}")
        top_node["flows"] = filtered
        nodes_by_var[top_var] = top_node
        _rewrite_nodes_file(nodes_path=nodes_path, var_order=var_order, nodes_by_var=nodes_by_var)

    unique_ids = sorted({int(f.get("flow_id") or 0) for f in filtered})
    return {
        "nodes_file": str(nodes_path) if nodes_path is not None else "",
        "top_level_var": top_var,
        "node_spec_updated": bool(node_spec is not None),
        "flows_written": len(filtered),
        "unique_flow_ids": len(unique_ids),
        "skipped_inventory_queries": skipped_inventory,
        "min_examples_per_flow_id": min_examples,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Populate top-level nodespec/spec flows using simple-flow artifacts from "
            "execution validation and generated flow traces."
        )
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--nodes_file", default="", help="Optional nodespec/spec file path. Default: auto-resolve (prefer spec.py).")
    ap.add_argument("--include_inventory_queries", action="store_true")
    ap.add_argument("--no_write_flow_trace_files", action="store_true")
    ap.add_argument("--min_examples_per_flow_id", type=int, default=1)
    args = ap.parse_args()

    res = add_top_level_flows_from_out_dir(
        out_dir=Path(args.out_dir),
        nodes_file=args.nodes_file,
        exclude_inventory_queries=not bool(args.include_inventory_queries),
        write_flow_trace_files=not bool(args.no_write_flow_trace_files),
        min_examples_per_flow_id=max(1, int(args.min_examples_per_flow_id)),
    )
    print(
        "[add_top_level_flows] "
        f"nodes_file={res['nodes_file']} "
        f"top_level_var={res['top_level_var']} "
        f"flows_written={res['flows_written']} "
        f"unique_flow_ids={res['unique_flow_ids']} "
        f"min_examples_per_flow_id={res['min_examples_per_flow_id']} "
        f"skipped_inventory_queries={res['skipped_inventory_queries']}"
    )


if __name__ == "__main__":
    main()
