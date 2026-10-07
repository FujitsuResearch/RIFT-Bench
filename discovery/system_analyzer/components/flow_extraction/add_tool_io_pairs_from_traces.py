#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Tuple
import sys
from node_spec.structure_schema import NodeSpec, ToolIOPair

try:
    from .global_utils import load_nodes_with_vars_from_python, render_nodespec_assignment
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
from .global_utils import load_nodes_with_vars_from_python, render_nodespec_assignment


MAX_TEXT_CHARS = 1000


def _norm_name(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _truncate_text(v: Any, max_chars: int = MAX_TEXT_CHARS) -> Any:
    if not isinstance(v, str):
        return v
    if len(v) <= max_chars:
        return v
    return v[:max_chars]


def _truncate_recursive(v: Any, max_chars: int = MAX_TEXT_CHARS) -> Any:
    if isinstance(v, str):
        return _truncate_text(v, max_chars=max_chars)
    if isinstance(v, list):
        return [_truncate_recursive(x, max_chars=max_chars) for x in v]
    if isinstance(v, dict):
        return {str(k): _truncate_recursive(val, max_chars=max_chars) for k, val in v.items()}
    return v


def _ensure_output_obj(v: Any, *, truncate_output: bool = True) -> Dict[str, Any]:
    if isinstance(v, dict):
        return _truncate_recursive(v) if truncate_output else v
    out = {"value": "" if v is None else str(v)}
    return _truncate_recursive(out) if truncate_output else out


def _normalize_pair_obj(p: Any, *, truncate_output: bool = True) -> Dict[str, Any] | None:
    if not isinstance(p, dict):
        return None
    inp = p.get("input")
    out = p.get("output")
    if inp is None and "inputs" in p:
        inp = p.get("inputs")
    if out is None and "outputs" in p:
        out = p.get("outputs")
    if inp is None and out is None:
        return None
    if not isinstance(inp, dict):
        inp = {"value": "" if inp is None else str(inp)}
    return {"input": inp, "output": _ensure_output_obj(out, truncate_output=truncate_output)}


def _canonical_pair_key(pair: Dict[str, Any]) -> str:
    return json.dumps(pair, sort_keys=True, ensure_ascii=False, default=str)


def _canonical_pairs_key(pairs: List[Dict[str, Any]]) -> str:
    return json.dumps(pairs, sort_keys=True, ensure_ascii=False, default=str)


def _parse_tool_calls_from_event(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    tcs = event.get("tool_calls")
    if not isinstance(tcs, list):
        return []
    out: List[Dict[str, Any]] = []
    for tc in tcs:
        if not isinstance(tc, dict):
            continue
        name = str(tc.get("name") or "").strip()
        if not name:
            continue
        args = tc.get("arguments")
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                if isinstance(parsed, dict):
                    args = parsed
            except Exception:
                pass
        if not isinstance(args, dict):
            args = {"value": "" if args is None else str(args)}
        out.append(
            {
                "name": name,
                "id": str(tc.get("id") or "").strip(),
                "arguments": args,
            }
        )
    return out


def _extract_pairs_from_trace_obj(obj: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    events = obj.get("events")
    if not isinstance(events, list):
        return []

    pending_by_id: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    pending_by_name: Dict[str, Deque[Dict[str, Any]]] = defaultdict(deque)
    pairs: List[Tuple[str, Dict[str, Any]]] = []

    for ev in events:
        if not isinstance(ev, dict):
            continue

        for tc in _parse_tool_calls_from_event(ev):
            tool_name = tc["name"]
            tool_id = tc["id"]
            args = tc["arguments"]
            if tool_id:
                pending_by_id[tool_id] = (tool_name, args)
            pending_by_name[_norm_name(tool_name)].append(args)

        if str(ev.get("type") or "").strip().lower() != "tool":
            continue

        tool_name = str(ev.get("name") or "").strip()
        if not tool_name:
            continue
        tool_call_id = str(ev.get("tool_call_id") or "").strip()

        inp: Dict[str, Any] = {}
        if tool_call_id and tool_call_id in pending_by_id:
            _tn, inp = pending_by_id.pop(tool_call_id)
        else:
            q = pending_by_name.get(_norm_name(tool_name))
            if q:
                inp = q.popleft()

        out_obj = _ensure_output_obj(ev.get("content"))
        pairs.append((tool_name, {"input": inp if isinstance(inp, dict) else {"value": str(inp)}, "output": out_obj}))

    return pairs


def _collect_trace_pairs(traces_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    by_tool: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if not traces_dir.exists():
        return by_tool
    for p in sorted(traces_dir.glob("*.json")):
        if p.name.endswith(".debug.json"):
            continue
        obj = _read_json(p)
        if not isinstance(obj, dict):
            continue
        for tool_name, pair in _extract_pairs_from_trace_obj(obj):
            by_tool[_norm_name(tool_name)].append(pair)
    return by_tool


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


def _resolve_paths(out_dir: Path, nodes_file: str, traces_dir: str | None) -> Tuple[Path, Path]:
    if str(nodes_file or "").strip():
        np = Path(nodes_file)
        nodes_path = np.resolve() if np.is_absolute() else (out_dir / np).resolve()
    else:
        nodes_path = _resolve_default_nodespec(out_dir)
    if traces_dir and str(traces_dir).strip():
        tr = Path(traces_dir)
        traces_path = tr.resolve() if tr.is_absolute() else (out_dir / tr).resolve()
    else:
        traces_path = (out_dir / "flow_extraction" / "flow_query_generation" / "generated_runs" / "parsed_traces").resolve()
    return nodes_path, traces_path


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


def add_tool_io_pairs_from_out_dir(
    *,
    out_dir: Path,
    nodes_file: str = "",
    node_spec: NodeSpec | None = None,
    traces_dir: str | None = None,
    max_pairs_per_tool: int = 20,
    dry_run: bool = False,
) -> Dict[str, Any]:
    out_dir = Path(out_dir).resolve()
    traces_path: Path
    if traces_dir and str(traces_dir).strip():
        tr = Path(traces_dir)
        traces_path = tr.resolve() if tr.is_absolute() else (out_dir / tr).resolve()
    else:
        traces_path = (out_dir / "flow_extraction" / "flow_query_generation" / "generated_runs" / "parsed_traces").resolve()

    nodes_path: Path | None = None
    if node_spec is None:
        nodes_path, traces_path = _resolve_paths(out_dir, nodes_file, traces_dir)

    if not traces_path.exists():
        raise FileNotFoundError(f"parsed traces dir not found: {traces_path}")

    trace_pairs_by_tool = _collect_trace_pairs(traces_path)

    changed_tools: List[Tuple[str, int, int]] = []
    if node_spec is not None:
        for tool_node in node_spec.iter_descendants(include_self=True):
            if str(getattr(getattr(tool_node, "node_type", None), "type", "") or "") != "Tool":
                continue
            tool_name = str(getattr(tool_node, "name", "") or "").strip()
            if not tool_name:
                continue
            existing_raw = tool_node.tool_example_pairs or []
            existing_no_trunc: List[Dict[str, Any]] = []
            existing: List[Dict[str, Any]] = []
            for p in existing_raw:
                row = p.model_dump(mode="python") if hasattr(p, "model_dump") else p
                norm_no_trunc = _normalize_pair_obj(row, truncate_output=False)
                norm = _normalize_pair_obj(row, truncate_output=True)
                if norm_no_trunc is not None:
                    existing_no_trunc.append(norm_no_trunc)
                if norm is not None:
                    existing.append(norm)
            cap = max(1, int(max_pairs_per_tool))
            if len(existing) > cap:
                existing = existing[:cap]
            seen = {_canonical_pair_key(p) for p in existing}
            before = len(existing)
            if len(existing) < cap:
                for p in trace_pairs_by_tool.get(_norm_name(tool_name), []):
                    if len(existing) >= cap:
                        break
                    k = _canonical_pair_key(p)
                    if k in seen:
                        continue
                    existing.append(p)
                    seen.add(k)
            normalized_changed = _canonical_pairs_key(existing) != _canonical_pairs_key(existing_no_trunc)
            if (len(existing) != before) or normalized_changed:
                tool_node.tool_example_pairs = [ToolIOPair(**x) for x in existing]
                changed_tools.append((str(getattr(tool_node, "id", "") or tool_name), before, len(existing)))
    else:
        assert nodes_path is not None
        if not nodes_path.exists():
            raise FileNotFoundError(f"nodes file not found: {nodes_path}")
        var_order, nodes = load_nodes_with_vars_from_python(nodes_path, list_name="ALL_NODES")
        nodes_by_var = {v: dict(n) for v, n in zip(var_order, nodes)}
        for var in var_order:
            node = nodes_by_var.get(var)
            if not isinstance(node, dict):
                continue
            nt = node.get("node_type") if isinstance(node.get("node_type"), dict) else {}
            if str(nt.get("type") or "") != "Tool":
                continue
            tool_name = str(node.get("name") or "").strip()
            if not tool_name:
                continue
            existing_raw = node.get("tool_example_pairs") if isinstance(node.get("tool_example_pairs"), list) else []
            existing_no_trunc: List[Dict[str, Any]] = []
            existing: List[Dict[str, Any]] = []
            for p in existing_raw:
                norm_no_trunc = _normalize_pair_obj(p, truncate_output=False)
                norm = _normalize_pair_obj(p, truncate_output=True)
                if norm_no_trunc is not None:
                    existing_no_trunc.append(norm_no_trunc)
                if norm is not None:
                    existing.append(norm)
            cap = max(1, int(max_pairs_per_tool))
            if len(existing) > cap:
                existing = existing[:cap]
            seen = {_canonical_pair_key(p) for p in existing}
            before = len(existing)
            if len(existing) < cap:
                for p in trace_pairs_by_tool.get(_norm_name(tool_name), []):
                    if len(existing) >= cap:
                        break
                    k = _canonical_pair_key(p)
                    if k in seen:
                        continue
                    existing.append(p)
                    seen.add(k)
            normalized_changed = _canonical_pairs_key(existing) != _canonical_pairs_key(existing_no_trunc)
            if (len(existing) != before) or normalized_changed:
                node["tool_example_pairs"] = existing
                nodes_by_var[var] = node
                changed_tools.append((var, before, len(existing)))
        if (not dry_run) and changed_tools:
            _rewrite_nodes_file(nodes_path=nodes_path, var_order=var_order, nodes_by_var=nodes_by_var)

    return {
        "nodes_file": str(nodes_path) if nodes_path is not None else "",
        "node_spec_updated": bool(node_spec is not None),
        "traces_dir": str(traces_path),
        "tools_updated": len(changed_tools),
        "changes": [{"node_var": v, "before": b, "after": a} for v, b, a in changed_tools],
        "wrote": bool((node_spec is None) and (not dry_run) and bool(changed_tools)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Add tool_example_pairs to nodespec/spec file from flow-extraction parsed traces.")
    ap.add_argument("--out_dir", required=True, help="Use-case output folder (contains flow_extraction/...).")
    ap.add_argument("--nodes_file", default="", help="Optional NodeSpec/spec file path. Default: auto-resolve (prefer spec.py).")
    ap.add_argument("--traces_dir", default="", help="Optional parsed_traces dir override (relative to out_dir if not absolute).")
    ap.add_argument("--max_pairs_per_tool", type=int, default=20, help="Cap total stored pairs per tool node.")
    ap.add_argument("--dry_run", action="store_true", help="Print planned changes without writing file.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    result = add_tool_io_pairs_from_out_dir(
        out_dir=out_dir,
        nodes_file=args.nodes_file,
        traces_dir=args.traces_dir or None,
        max_pairs_per_tool=max(1, int(args.max_pairs_per_tool)),
        dry_run=bool(args.dry_run),
    )
    print(f"[add_tool_io_pairs] nodes_file={result['nodes_file']}")
    print(f"[add_tool_io_pairs] traces_dir={result['traces_dir']}")
    print(f"[add_tool_io_pairs] tools_updated={result['tools_updated']}")
    for row in result["changes"]:
        print(f"  - {row['node_var']}: {row['before']} -> {row['after']}")
    if args.dry_run:
        return
    print(f"[add_tool_io_pairs] {'wrote' if result['wrote'] else 'no changes'}")


if __name__ == "__main__":
    main()
