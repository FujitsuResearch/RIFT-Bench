#!/usr/bin/env python3
"""Utility functions for root_finder."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from ..global_utils import (
        call_with_cache,
        load_nodes_list_from_python,
        load_nodes_with_vars_from_python,
        primary_file,
        render_nodespec_assignment,
        safe_file_tag,
        to_json_compatible,
    )
    from .prompts import (
        ROOT_CANDIDATE_COMPARE_PROMPT,
        ROOT_LIST_CANDIDATE_PROMPT,
    )
except ImportError:
    from global_utils import (
        call_with_cache,
        load_nodes_list_from_python,
        load_nodes_with_vars_from_python,
        primary_file,
        render_nodespec_assignment,
        safe_file_tag,
        to_json_compatible,
    )
    from root_finder.prompts import (
        ROOT_CANDIDATE_COMPARE_PROMPT,
        ROOT_LIST_CANDIDATE_PROMPT,
    )


def _schema_module_and_path() -> Tuple[str, Path]:
    """Return the schema module name and file path used when emitting root node Python outputs."""
    return "NodeSpec_schema_short", Path(__file__).resolve().parents[1] / "NodeSpec_schema_short.py"


def _nodespec_import_header() -> str:
    """Build the import line required at the top of generated root node Python files."""
    module_name, _ = _schema_module_and_path()
    return f"from {module_name} import NodeSpec, NodeType, CodeReference, InputPort, OutputPort"


def coerce_root_node_type(root_node: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure a selected root node has an allowed root-level node_type, defaulting to System."""
    out = dict(root_node) if isinstance(root_node, dict) else {}
    node_type = out.get("node_type")
    current = ""
    if isinstance(node_type, dict) and isinstance(node_type.get("type"), str):
        current = str(node_type.get("type"))
    if current not in {"Agent", "System"}:
        out["node_type"] = {"type": "System", "other_description": None}
    return out


def write_root_node(out_dir: Path, root_node: Dict[str, Any]) -> None:
    """Write the final chosen root node as both JSON and Python output artifacts."""
    out_path = out_dir / "root_nodes.json"
    safe_root = to_json_compatible({"root_node": root_node})
    out_path.write_text(
        json.dumps(safe_root, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )

    root_nodes_py = out_dir / "root_nodes.py"
    root_var = safe_var_base(str((root_node or {}).get("name") or "root_node"))
    root_node_for_py = dict(root_node) if isinstance(root_node, dict) else {}
    root_node_for_py.pop("__var__", None)
    py_text = (
        _nodespec_import_header()
        + "\n\n"
        + render_nodespec_assignment(root_var, root_node_for_py)
        + f"\n\nALL_NODES = [\n    {root_var},\n]\n"
    )
    root_nodes_py.write_text(py_text, encoding="utf-8")


def safe_var_base(name: str) -> str:
    """Normalize a node name into a safe Python variable base name for generated files."""
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", (name or "").strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "root_node"
    if text[0].isdigit():
        text = f"n_{text}"
    return text


def load_nodes_index_rows(nodes_index_path: Path) -> List[Dict[str, Any]]:
    """Load normalized row dictionaries from the nodes index file."""
    if not nodes_index_path.exists():
        return []
    try:
        obj = json.loads(nodes_index_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(obj, dict):
        rows = obj.get("files") if isinstance(obj.get("files"), list) else []
    elif isinstance(obj, list):
        rows = obj
    else:
        rows = []
    return [row for row in rows if isinstance(row, dict)]


def build_nodes_file_map(nodes_index_path: Path) -> Dict[str, Path]:
    """Build a map from each source file path to its generated per-file nodes output path."""
    out: Dict[str, Path] = {}
    for row in load_nodes_index_rows(nodes_index_path):
        source_file = row.get("source_file") or row.get("file_path")
        output_file = row.get("output_file")
        if isinstance(source_file, str) and source_file and isinstance(output_file, str) and output_file:
            out[str(Path(source_file).resolve())] = Path(output_file).resolve()
    return out


def write_nodes_file(nodes_file: Path, nodes: List[Dict[str, Any]]) -> None:
    """Rewrite a per-file Python nodes output file from the provided node list."""
    lines = [_nodespec_import_header(), ""]
    lines.append("NEW_NODES = [")
    for node in nodes:
        rendered = render_nodespec_assignment("_node", node).splitlines()
        if not rendered:
            continue
        first = rendered[0]
        if " = " in first:
            rendered[0] = first.split(" = ", 1)[1]
        rendered[-1] = rendered[-1] + ","
        lines.extend(f"    {line}" for line in rendered)
    lines.append("]")
    lines.append("")
    nodes_file.write_text("\n".join(lines), encoding="utf-8")


def sync_root_nodetype_to_outputs(
    *,
    out_dir: Path,
    root_node: Dict[str, Any],
    nodes_file_map: Dict[str, Path],
) -> None:
    """Propagate the chosen root node_type into the per-file node output and combined catalog entry."""
    root_name = str(root_node.get("name") or "")
    root_file = primary_file(root_node)
    root_type = root_node.get("node_type")
    if not isinstance(root_type, dict):
        return

    if root_name and root_file:
        src_abs = str(Path(root_file).resolve())
        nodes_file = nodes_file_map.get(src_abs)
        if nodes_file and nodes_file.exists():
            changed = False
            try:
                nodes = load_nodes_list_from_python(nodes_file, list_name="NEW_NODES")
            except Exception as exc:
                print(f"[root_finder] warning: skipping node_type sync for malformed file {nodes_file}: {exc}")
                nodes = []
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                if str(node.get("name") or "") != root_name:
                    continue
                node_file = primary_file(node)
                if node_file and str(Path(node_file).resolve()) != src_abs:
                    continue
                node["node_type"] = dict(root_type)
                changed = True
                break
            if changed:
                write_nodes_file(nodes_file, nodes)

    combined_path = out_dir / "nodes_catalog.py"
    if not combined_path.exists():
        return
    try:
        var_order, existing_nodes = load_nodes_with_vars_from_python(combined_path, list_name="ALL_NODES")
    except Exception as exc:
        print(f"[root_finder] warning: skipping combined node_type sync for malformed file {combined_path}: {exc}")
        return

    by_var: Dict[str, Dict[str, Any]] = {}
    for var, node in zip(var_order, existing_nodes):
        if isinstance(node, dict):
            by_var[var] = dict(node)

    changed = False
    for var in var_order:
        cur = by_var.get(var, {})
        if str(cur.get("name") or "") != root_name:
            continue
        if root_file:
            cur_file = primary_file(cur)
            if cur_file and str(Path(cur_file).resolve()) != str(Path(root_file).resolve()):
                continue
        cur["node_type"] = dict(root_type)
        by_var[var] = cur
        changed = True
        break
    if not changed:
        return

    header = _nodespec_import_header() + "\n\n"
    combined_path.write_text(header, encoding="utf-8")
    with combined_path.open("a", encoding="utf-8") as fh:
        for var in var_order:
            node = dict(by_var[var])
            node.pop("__var__", None)
            fh.write(render_nodespec_assignment(var, node))
            fh.write("\n\n")
        fh.write("ALL_NODES = [\n")
        for var in var_order:
            fh.write(f"    {var},\n")
        fh.write("]\n")


def align_root_to_catalog_node(*, out_dir: Path, root_node: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer the enriched node from nodes_catalog.py for the selected root.

    Root selection currently runs over per-file NEW_NODES lists. This helper
    remaps the chosen root to the matching node in ALL_NODES (same name/file),
    preferring the candidate with the richest code_references.
    """
    if not isinstance(root_node, dict):
        return root_node

    combined_path = out_dir / "nodes_catalog.py"
    if not combined_path.exists():
        return root_node

    try:
        _vars, catalog_nodes = load_nodes_with_vars_from_python(combined_path, list_name="ALL_NODES")
    except Exception:
        return root_node

    root_name = str(root_node.get("name") or "").strip()
    root_file = primary_file(root_node)
    root_file_abs = str(Path(root_file).resolve()) if root_file else ""

    matches: List[Dict[str, Any]] = []
    for node in catalog_nodes:
        if not isinstance(node, dict):
            continue
        if str(node.get("name") or "").strip() != root_name:
            continue
        node_file = primary_file(node)
        node_file_abs = str(Path(node_file).resolve()) if node_file else ""
        if root_file_abs and node_file_abs and node_file_abs != root_file_abs:
            continue
        matches.append(dict(node))

    if not matches:
        return root_node

    def _score(n: Dict[str, Any]) -> Tuple[int, int]:
        refs = n.get("code_references") if isinstance(n.get("code_references"), list) else []
        # Prefer more references; tie-break by description length.
        return (len(refs), len(str(n.get("description") or "")))

    best = max(matches, key=_score)
    # Keep selected node_type if judge/coercion already set it explicitly.
    if isinstance(root_node.get("node_type"), dict):
        best["node_type"] = dict(root_node["node_type"])
    return best


def load_node_lists(nodes_file_map: Dict[str, Path]) -> List[Tuple[Path, List[Dict[str, Any]]]]:
    """Load each source file's generated NEW_NODES list and pair it with that source path."""
    out: List[Tuple[Path, List[Dict[str, Any]]]] = []
    for source_str in sorted(nodes_file_map.keys()):
        source_path = Path(source_str).resolve()
        nodes_file = nodes_file_map[source_str]
        if not nodes_file.exists():
            out.append((source_path, []))
            continue
        try:
            nodes = load_nodes_list_from_python(nodes_file, list_name="NEW_NODES")
        except Exception as exc:
            print(f"[root_finder] warning: failed loading {nodes_file}: {exc}")
            nodes = []
        clean_nodes = [dict(node) for node in nodes if isinstance(node, dict)]
        out.append((source_path, clean_nodes))
    return out


def node_signature(node: Dict[str, Any]) -> Tuple[str, str]:
    """Return a stable matching key for a node using its name and primary file path."""
    node_file = primary_file(node)
    return (
        str(node.get("name") or ""),
        str(Path(node_file).resolve()) if node_file else "",
    )


def extract_node_from_response(response: Dict[str, Any], *keys: str) -> Dict[str, Any] | None:
    """Return the first node-like dict found under the requested response keys."""
    if not isinstance(response, dict):
        return None
    for key in keys:
        val = response.get(key)
        if isinstance(val, dict):
            return val
        if key == "root_node":
            roots = response.get("root_nodes")
            if isinstance(roots, list):
                items = [x for x in roots if isinstance(x, dict)]
                if items:
                    return items[0]
    return None


def align_candidate_to_list(candidate: Dict[str, Any] | None, nodes: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Map a model-selected candidate back to the exact node object from the source list."""
    if not isinstance(candidate, dict) or not nodes:
        return None
    by_sig: Dict[Tuple[str, str], Dict[str, Any]] = {}
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        sig = node_signature(node)
        by_sig[sig] = node
        name = str(node.get("name") or "").strip()
        if name:
            by_name.setdefault(name, []).append(node)

    cand_sig = node_signature(candidate)
    if cand_sig in by_sig:
        return by_sig[cand_sig]
    cand_name = str(candidate.get("name") or "").strip()
    if cand_name and len(by_name.get(cand_name, [])) == 1:
        return by_name[cand_name][0]
    return None


def extract_compare_winner(
    response: Dict[str, Any],
    stored_candidate: Dict[str, Any],
    contender_candidate: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve the comparison response into either the stored or contender candidate object."""
    winner = str(response.get("winner") or "").strip().lower() if isinstance(response, dict) else ""
    if winner in {"stored", "a", "candidate_a", "current", "existing"}:
        return stored_candidate
    if winner in {"contender", "b", "candidate_b", "new", "incoming"}:
        return contender_candidate

    winner_node = extract_node_from_response(response, "winner_node", "selected_node", "root_node")
    if isinstance(winner_node, dict):
        if node_signature(winner_node) == node_signature(contender_candidate):
            return contender_candidate
        if node_signature(winner_node) == node_signature(stored_candidate):
            return stored_candidate
    return stored_candidate


def call_list_best_candidate(
    *,
    source_path: Path,
    nodes: List[Dict[str, Any]],
    model: str,
    out_raw_dir: Path,
    refresh_raw: bool,
    retriever: Any,
    rag_max_rounds: int,
    idx: int,
) -> Dict[str, Any]:
    """Ask the model to choose the strongest root candidate from one file's node list."""
    payload = {
        "prompt": ROOT_LIST_CANDIDATE_PROMPT,
        "current_filepath": str(source_path.resolve()),
        "nodes": nodes,
    }
    safe = safe_file_tag(f"root_list_best_{idx:03d}_{source_path.name}")
    parsed, _ = call_with_cache(
        payload,
        payload_path=out_raw_dir / f"{safe}.payload.json",
        raw_path=out_raw_dir / f"{safe}.txt",
        model=model,
        refresh_raw=refresh_raw,
        retriever=retriever,
        rag_max_rounds=rag_max_rounds,
    )
    return parsed if isinstance(parsed, dict) else {}


def call_compare_candidates(
    *,
    stored_candidate: Dict[str, Any],
    contender_candidate: Dict[str, Any],
    model: str,
    out_raw_dir: Path,
    refresh_raw: bool,
    retriever: Any,
    rag_max_rounds: int,
    idx: int,
) -> Dict[str, Any]:
    """Ask the model to choose the stronger root candidate between two file-level winners."""
    payload = {
        "prompt": ROOT_CANDIDATE_COMPARE_PROMPT,
        "stored_candidate": stored_candidate,
        "contender_candidate": contender_candidate,
    }
    safe = safe_file_tag(f"root_compare_{idx:03d}")
    parsed, _ = call_with_cache(
        payload,
        payload_path=out_raw_dir / f"{safe}.payload.json",
        raw_path=out_raw_dir / f"{safe}.txt",
        model=model,
        refresh_raw=refresh_raw,
        retriever=retriever,
        rag_max_rounds=rag_max_rounds,
    )
    return parsed if isinstance(parsed, dict) else {}


__all__ = [
    "align_candidate_to_list",
    "align_root_to_catalog_node",
    "build_nodes_file_map",
    "call_compare_candidates",
    "call_list_best_candidate",
    "coerce_root_node_type",
    "extract_compare_winner",
    "extract_node_from_response",
    "load_nodes_index_rows",
    "load_node_lists",
    "node_signature",
    "safe_var_base",
    "sync_root_nodetype_to_outputs",
    "write_nodes_file",
    "write_root_node",
]
