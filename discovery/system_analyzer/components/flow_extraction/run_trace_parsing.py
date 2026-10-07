from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .trace_parsing import parse_trace_file, save_parsed_outputs
except ImportError:
    from discovery.system_analyzer.components.flow_extraction.trace_parsing import parse_trace_file, save_parsed_outputs


def _infer_outputs_dir_from_traces_dir(traces_dir: Path) -> Path | None:
    td = traces_dir.resolve()
    # Expected pattern: <outputs_dir>/validation/traces
    if td.name == "traces" and td.parent.name == "validation":
        return td.parent.parent
    return None


def _default_out_dir_from_traces_dir(traces_dir: Path) -> Path:
    td = traces_dir.resolve()
    # Expected pattern: <outputs_dir>/validation/traces
    if td.name == "traces" and td.parent.name == "validation":
        outputs_dir = td.parent.parent
        return outputs_dir / "flow_extraction" / "parsed_traces"
    # Generic fallback keeps outputs colocated under provided traces directory.
    return td / "flow_extraction" / "parsed_traces"


def _resolve_runtime_mapping_file(
    *,
    traces_dir: Path,
    nodespec_file: Path | None,
    runtime_mapping_file: Path | None,
) -> Path | None:
    if runtime_mapping_file is not None:
        return runtime_mapping_file.resolve()
    outputs_dir = _infer_outputs_dir_from_traces_dir(traces_dir)
    if outputs_dir is None:
        return None
    cand = (outputs_dir / "validation" / "runtime_mapping.json").resolve()
    if cand.exists():
        return cand
    # Auto-induce mapping when absent to keep single-script UX.
    try:
        from discovery.system_analyzer.components.flow_extraction.induce_runtime_mapping import induce_runtime_mapping

        mapping = induce_runtime_mapping(outputs_dir=outputs_dir, nodespec_file=nodespec_file)
        cand.parent.mkdir(parents=True, exist_ok=True)
        import json

        cand.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[flow_extraction] auto-generated runtime mapping at {cand}")
        return cand
    except Exception as exc:
        print(f"[flow_extraction] warning: failed to auto-generate runtime mapping: {exc}")
        return None


def run(
    traces_dir: Path,
    out_dir: Path | None,
    pattern: str,
    recursive: bool,
    nodespec_file: Path | None = None,
    runtime_mapping_file: Path | None = None,
) -> None:
    traces_dir = traces_dir.resolve()
    out_dir = (out_dir.resolve() if out_dir is not None else _default_out_dir_from_traces_dir(traces_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_mapping_file = _resolve_runtime_mapping_file(
        traces_dir=traces_dir,
        nodespec_file=nodespec_file,
        runtime_mapping_file=runtime_mapping_file,
    )

    glob_fn = traces_dir.rglob if recursive else traces_dir.glob
    trace_files = sorted(p for p in glob_fn(pattern) if p.is_file())

    if not trace_files:
        print(f"[flow_extraction] no trace files found in {traces_dir} with pattern={pattern}")
        return

    for tf in trace_files:
        parsed = parse_trace_file(tf, nodespec_file=nodespec_file, runtime_mapping_file=runtime_mapping_file)
        parsed_out = out_dir / f"parsed_{tf.stem}.json"
        debug_out = out_dir / f"parsed_{tf.stem}.debug.json"
        save_parsed_outputs(parsed=parsed, parsed_out_file=parsed_out, debug_out_file=debug_out)
        print(f"[flow_extraction] parsed {tf.name} -> {parsed_out.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces_dir", required=True, help="Directory containing trace JSON files.")
    ap.add_argument(
        "--out_dir",
        default="",
        help="Optional output directory. Default: <outputs_dir>/flow_extraction/parsed_traces when traces_dir is <outputs_dir>/validation/traces.",
    )
    ap.add_argument("--pattern", default="*.json", help="Glob pattern for trace files.")
    ap.add_argument("--recursive", action="store_true", help="Recursively scan traces_dir.")
    ap.add_argument("--nodespec_file", default="", help="Optional path to nodespec/spec file for deterministic name mapping.")
    ap.add_argument("--runtime_mapping_file", default="", help="Optional path to runtime_mapping.json for deterministic alias mapping.")
    args = ap.parse_args()

    run(
        traces_dir=Path(args.traces_dir),
        out_dir=Path(args.out_dir).resolve() if str(args.out_dir).strip() else None,
        pattern=args.pattern,
        recursive=bool(args.recursive),
        nodespec_file=Path(args.nodespec_file).resolve() if str(args.nodespec_file).strip() else None,
        runtime_mapping_file=Path(args.runtime_mapping_file).resolve() if str(args.runtime_mapping_file).strip() else None,
    )


if __name__ == "__main__":
    main()
