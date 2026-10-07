#!/usr/bin/env python3
"""Standalone file_finding component."""
import argparse
import ast
import json
import os
from collections import deque
from pathlib import Path

try:
    from ..global_utils import load_env_file
    from . import utils as file_finding_utils
except ImportError:
    from global_utils import load_env_file
    import utils as file_finding_utils


def main() -> None:
    """Build the root record and reachable-files record for a single Python entrypoint."""
    default_out_dir = Path(__file__).resolve().parents[1] / "stages_outputs"

    # Parse CLI inputs and normalize output artifact locations for this stage.
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_file", default=".env", help="Optional .env file to load.")
    ap.add_argument(
        "--entry",
        default="agent_scripts/langraph_react_agent.py",
        type=str,
        help="Path to the entry Python file to analyze (preferably repo-relative).",
    )
    ap.add_argument(
        "--out_dir",
        default=str(default_out_dir),
        type=str,
        help="Output directory for file_finding artifacts.",
    )
    ap.add_argument(
        "--root_out",
        default="root_record.json",
        type=str,
        help="Root record filename (written under --out_dir).",
    )
    ap.add_argument(
        "--reachable_out",
        default="reachable_files.json",
        type=str,
        help="Reachable record filename (written under --out_dir).",
    )
    ap.add_argument(
        "--hints_file",
        action="append",
        default=[],
        help=(
            "Optional hints JSON file path. Can be passed multiple times. "
            "If omitted, no predefined hints are applied."
        ),
    )
    args = ap.parse_args()
    load_env_file(Path(args.env_file))

    out_dir = Path(args.out_dir)
    root_out_path = out_dir / Path(args.root_out).name
    reachable_out_path = out_dir / Path(args.reachable_out).name

    # Resolve the entrypoint into a stable root-record that downstream stages can reuse.
    root_record = file_finding_utils.build_root_record(args.entry)
    project_root = Path(root_record["project_root"]).resolve()
    entry_abs = Path(root_record["entry_abs"]).resolve()
    root_abs_prefix = str(project_root) + os.sep

    root_out_path.parent.mkdir(parents=True, exist_ok=True)
    root_out_path.write_text(json.dumps(root_record, indent=2), encoding="utf-8")
    print(f"[file_finding] wrote root: {root_out_path}")
    print(f"[file_finding] project_root: {root_record['project_root']}")

    # Seed the reachable-file traversal with the entrypoint and shared lookup structures.
    py_index = file_finding_utils.build_py_file_index(str(project_root))
    visited_py = set()
    visited_json = set()
    queue_py = deque([entry_abs])
    unresolved_imports = []

    # Walk reachable Python files, expanding through path literals, imports, and dynamic sibling scans.
    while queue_py:
        py_file = os.path.abspath(str(queue_py.popleft()))
        if py_file in visited_py or not os.path.isfile(py_file) or not py_file.endswith(".py"):
            continue
        if not py_file.startswith(root_abs_prefix):
            continue

        visited_py.add(py_file)

        json_paths, py_paths = file_finding_utils.extract_paths_from_code(py_file, str(project_root))

        for jp in sorted(json_paths):
            jp = os.path.abspath(jp)
            if not jp.startswith(root_abs_prefix):
                continue
            if os.path.isfile(jp) and jp not in visited_json:
                visited_json.add(jp)
                for pp in file_finding_utils.extract_py_files_from_json(jp, str(project_root)):
                    pp = os.path.abspath(pp)
                    if pp.startswith(root_abs_prefix) and os.path.isfile(pp) and pp not in visited_py:
                        queue_py.append(pp)

        for pp in sorted(py_paths):
            pp = os.path.abspath(pp)
            if pp.startswith(root_abs_prefix) and os.path.isfile(pp) and pp not in visited_py:
                queue_py.append(pp)

        for pp in sorted(
            file_finding_utils.extract_import_like_modules_from_string_literals(
                py_file, str(project_root), py_index
            )
        ):
            pp = os.path.abspath(pp)
            if pp.startswith(root_abs_prefix) and os.path.isfile(pp) and pp not in visited_py:
                queue_py.append(pp)

        for pp in sorted(file_finding_utils.extract_iterdir_dynamic_sibling_py_files(py_file, str(project_root))):
            pp = os.path.abspath(pp)
            if pp.startswith(root_abs_prefix) and os.path.isfile(pp) and pp not in visited_py:
                queue_py.append(pp)

        # Parse the current file AST to resolve standard imports and record unresolved local-looking ones.
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                src = f.read()
            tree = ast.parse(src)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modname = alias.name
                    resolved = file_finding_utils.resolve_import_to_local_py(
                        modname, py_file, str(project_root), py_index, level=0
                    )
                    if resolved:
                        resolved = os.path.abspath(resolved)
                        if resolved.startswith(root_abs_prefix) and resolved not in visited_py:
                            queue_py.append(resolved)
                    elif file_finding_utils.looks_like_local_import(modname, 0, str(project_root)):
                        unresolved_imports.append(
                            {"file": py_file, "import": modname, "lineno": getattr(node, "lineno", None)}
                        )

            elif isinstance(node, ast.ImportFrom):
                level = getattr(node, "level", 0) or 0
                modname = node.module or ""

                if any(a.name == "*" for a in node.names):
                    resolved = file_finding_utils.resolve_import_to_local_py(
                        modname, py_file, str(project_root), py_index, level=level
                    )
                    if resolved:
                        resolved = os.path.abspath(resolved)
                        if resolved.startswith(root_abs_prefix) and resolved not in visited_py:
                            queue_py.append(resolved)
                    elif file_finding_utils.looks_like_local_import(modname or ".", level, str(project_root)):
                        unresolved_imports.append(
                            {
                                "file": py_file,
                                "import": f"from {'.' * level}{modname} import *",
                                "lineno": getattr(node, "lineno", None),
                            }
                        )
                    continue

                resolved = file_finding_utils.resolve_import_to_local_py(
                    modname, py_file, str(project_root), py_index, level=level
                )
                if resolved:
                    resolved = os.path.abspath(resolved)
                    if resolved.startswith(root_abs_prefix) and resolved not in visited_py:
                        queue_py.append(resolved)
                else:
                    if file_finding_utils.looks_like_local_import(modname or ".", level, str(project_root)):
                        unresolved_imports.append(
                            {
                                "file": py_file,
                                "import": f"from {'.' * level}{modname} import {', '.join(a.name for a in node.names)}",
                                "lineno": getattr(node, "lineno", None),
                            }
                        )
                    for alias in node.names:
                        alias_name = getattr(alias, "name", "")
                        if not isinstance(alias_name, str) or not alias_name or alias_name == "*":
                            continue
                        if not modname:
                            continue
                        sub_mod = f"{modname}.{alias_name}"
                        sub_resolved = file_finding_utils.resolve_import_to_local_py(
                            sub_mod, py_file, str(project_root), py_index, level=level
                        )
                        if sub_resolved:
                            sub_resolved = os.path.abspath(sub_resolved)
                            if sub_resolved.startswith(root_abs_prefix) and sub_resolved not in visited_py:
                                queue_py.append(sub_resolved)

    # Apply optional manual hints for files the automatic discovery heuristics may miss.
    hints_paths = [Path(x) for x in args.hints_file if isinstance(x, str) and x.strip()]
    hints = file_finding_utils.load_reachable_hints(project_root, hints_paths)
    for py_abs in hints.get("extra_py_abs", []):
        if os.path.isfile(py_abs) and py_abs.startswith(root_abs_prefix):
            visited_py.add(py_abs)
    for json_abs in hints.get("extra_json_abs", []):
        if os.path.isfile(json_abs) and json_abs.startswith(root_abs_prefix):
            visited_json.add(json_abs)

    # Persist the final reachable-file inventory for the rest of the pipeline.
    reachable = {
        "project_root": str(project_root),
        "entry_abs": str(entry_abs),
        "reachable_py_abs": sorted(str(p) for p in visited_py),
        "reachable_json_abs": sorted(str(p) for p in visited_json),
        "unresolved_imports": unresolved_imports,
        "reachable_hints": hints,
        "counts": {"py": len(visited_py), "json": len(visited_json)},
    }
    reachable_out_path.parent.mkdir(parents=True, exist_ok=True)
    reachable_out_path.write_text(json.dumps(reachable, indent=2), encoding="utf-8")
    print(f"[file_finding] wrote reachable: {reachable_out_path}")
    print(f"[file_finding] reachable: {reachable['counts']}")


if __name__ == "__main__":
    main()
