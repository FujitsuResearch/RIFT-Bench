#!/usr/bin/env python3
"""Export a graph-correctness or code-reference-refine nodes file into a validated Python spec script."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import ValidationError

try:
    from ..NodeSpec_schema import NodeSpecTop  # type: ignore
    from ..execution_validation.utils import observe_components
    from ..global_utils import build_parent_child_maps, copy_nodespec_schema_to_out_dir, extract_var_name, load_env_file, load_nodes_by_var, norm, read_root_node_json, resolve_root_var
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    CLEAN_CODE_ROOT = Path(__file__).resolve().parents[1]
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    if str(CLEAN_CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CLEAN_CODE_ROOT))
    from execution_validation.utils import observe_components
    from global_utils import build_parent_child_maps, copy_nodespec_schema_to_out_dir, extract_var_name, load_env_file, load_nodes_by_var, norm, read_root_node_json, resolve_root_var
    from NodeSpec_schema import NodeSpecTop  # type: ignore


SHAREABLE_RUNTIME_TYPES = {"Tool", "LLM", "Database", "Local_MCP_server", "External_MCP_server"}


def _materialize_connections_from_edges(var_order: List[str], nodes_by_var: Dict[str, Dict[str, Any]]) -> None:
    """Build `external_connections` deterministically from the current internal edges."""
    # Reuse the graph containment maps so both `nodes` and `tool_list` children
    # participate in parent ownership.
    _children_by_parent, raw_parents_by_child = build_parent_child_maps(var_order, nodes_by_var)
    parents_by_child: Dict[str, Set[str]] = {
        cvar: {pvar for pvar in parents if pvar in nodes_by_var and pvar != cvar}
        for cvar, parents in raw_parents_by_child.items()
    }
    tool_child_vars: Set[str] = set()
    for pvar in var_order:
        pnode = nodes_by_var.get(pvar, {})
        for tvar in [extract_var_name(x) for x in (pnode.get("tool_list") if isinstance(pnode.get("tool_list"), list) else [])]:
            if tvar and tvar in nodes_by_var and tvar != pvar:
                tool_child_vars.add(tvar)

    existing_vars = set(var_order)
    incoming: Dict[str, Set[str]] = {v: set() for v in var_order}
    outgoing: Dict[str, Set[str]] = {v: set() for v in var_order}
    conn_by_parent: Dict[str, Dict[str, Dict[str, Set[str]]]] = {v: {} for v in var_order}

    for pvar in var_order:
        pnode = nodes_by_var.get(pvar, {})
        edges = pnode.get("internal_edges") if isinstance(pnode.get("internal_edges"), list) else []

        # Normalize each parent's edge list first so later connection derivation
        # works from one clean edge set.
        # Deduplicate each node's edge list before deriving connections from it.
        seen_edges: Set[Tuple[str, str]] = set()
        clean_edges: List[Dict[str, Any]] = []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            frm = extract_var_name(edge.get("from_"))
            to = extract_var_name(edge.get("to"))
            if not frm or not to or frm == to:
                continue
            frm_ok = frm in existing_vars or frm in {"START", "END"}
            to_ok = to in existing_vars or to in {"START", "END"}
            if not frm_ok or not to_ok:
                continue
            key = (frm, to)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            clean_edges.append({"from_": frm, "to": to})

            if frm in outgoing:
                outgoing[frm].add(to)
            if to in incoming:
                incoming[to].add(frm)
            if frm in conn_by_parent:
                slot = conn_by_parent[frm].setdefault(pvar, {"in_": set(), "out": set()})
                slot["out"].add(to)
            if to in conn_by_parent:
                slot = conn_by_parent[to].setdefault(pvar, {"in_": set(), "out": set()})
                slot["in_"].add(frm)

        pnode["internal_edges"] = clean_edges

    for var in var_order:
        node = nodes_by_var.get(var)
        if not isinstance(node, dict):
            continue

        # Prefer parent-scoped connectivity when the node participates in
        # one or more parent-owned edge scopes.
        parent_scope = conn_by_parent.get(var, {})
        conn_list: List[Dict[str, Any]] = []
        if parent_scope:
            for parent_var in sorted(parent_scope.keys()):
                io = parent_scope[parent_var]
                conn_list.append({
                    "parent": parent_var,
                    "in_": sorted(io.get("in_", set())),
                    "out": sorted(io.get("out", set())),
                })
        else:
            conn_list.append({
                "parent": "",
                "in_": sorted(incoming.get(var, set())),
                "out": sorted(outgoing.get(var, set())),
            })

        # If the node is alone inside every parent scope, materialize explicit
        # start/end boundaries for that otherwise-empty connection scope.
        parent_vars = sorted(parents_by_child.get(var, set()))
        if parent_vars:
            alone_in_scope = True
            for pvar in parent_vars:
                pnode = nodes_by_var.get(pvar, {})
                sibs = {
                    extract_var_name(x)
                    for x in ((pnode.get("nodes") if isinstance(pnode.get("nodes"), list) else []) + (pnode.get("tool_list") if isinstance(pnode.get("tool_list"), list) else []))
                    if extract_var_name(x)
                }
                sibs.discard(var)
                if sibs:
                    alone_in_scope = False
                    break
        else:
            alone_in_scope = True

        # If no explicit in/out survived for a singleton scope, make that
        # boundary explicit so the exported graph stays traversable.
        if alone_in_scope:
            for conn in conn_list:
                in_vals = [extract_var_name(x) or str(x) for x in (conn.get("in_") if isinstance(conn.get("in_"), list) else [])]
                out_vals = [extract_var_name(x) or str(x) for x in (conn.get("out") if isinstance(conn.get("out"), list) else [])]
                if not in_vals and not out_vals:
                    if str(conn.get("parent") or "") == "":
                        conn["in_"] = ["START"]
                        conn["out"] = ["END"]
                    else:
                        conn["in_"] = [var]
                        conn["out"] = [var]

        # Tools attached through a parent's tool_list do not keep external connectivity.
        if var in tool_child_vars:
            conn_list = []

        node["external_connections"] = conn_list


def _node_type_name(node: Dict[str, Any]) -> str:
    """Return the normalized node type string from a node-like mapping."""
    nt = node.get("node_type")
    if isinstance(nt, dict):
        return str(nt.get("type") or "")
    if isinstance(nt, str):
        return nt
    return ""


def _norm(s: str | None) -> str:
    """Normalize a string for loose identity matching."""
    return "".join(ch for ch in str(s or "").lower() if ch.isalnum())


def _as_conn_list(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return external connections as a normalized list of dicts."""
    conns = node.get("external_connections")
    if isinstance(conns, dict):
        conns = [conns]
    if not isinstance(conns, list):
        return []
    return [c for c in conns if isinstance(c, dict)]


def _var_suffix_for_name(name: str, var: str) -> str:
    """Build a readable name suffix from a node var, stripping generic trailing endings."""
    base_name = str(name or "").strip()
    suffix = str(var or "").strip()
    prefix = f"{base_name}_" if base_name else ""
    if prefix and suffix.startswith(prefix):
        suffix = suffix[len(prefix):]
    suffix = re.sub(r"(?:_(?:py|json|ya?ml|txt))+$$", "", suffix)
    suffix = re.sub(r"_\d+$$", "", suffix)
    suffix = re.sub(r"_+", "_", suffix).strip("_")
    return suffix


def _disambiguate_sibling_names(var_order: List[str], nodes_by_var: Dict[str, Dict[str, Any]]) -> None:
    """Make sibling child names unique within each parent using readable var-derived suffixes."""
    for pvar in var_order:
        pnode = nodes_by_var.get(pvar, {})
        child_vars: List[str] = []
        for field in ("nodes", "tool_list"):
            vals = pnode.get(field) if isinstance(pnode.get(field), list) else []
            for item in vals:
                cvar = extract_var_name(item)
                if cvar and cvar in nodes_by_var and cvar != pvar and cvar not in child_vars:
                    child_vars.append(cvar)

        groups: Dict[str, List[str]] = {}
        for cvar in child_vars:
            cname = str(nodes_by_var.get(cvar, {}).get("name") or "").strip()
            if cname:
                groups.setdefault(cname, []).append(cvar)

        for cname, dup_vars in groups.items():
            if len(dup_vars) <= 1:
                continue
            used_names = {
                str(nodes_by_var.get(cvar, {}).get("name") or "").strip()
                for cvar in child_vars
                if cvar not in dup_vars
            }
            used_names.add(cname)
            for cvar in dup_vars[1:]:
                node = nodes_by_var.get(cvar, {})
                suffix = _var_suffix_for_name(cname, cvar)
                candidate = f"{cname}_{suffix}" if suffix else cname
                if not suffix or candidate in used_names:
                    i = 2
                    candidate = f"{cname}_{i}"
                    while candidate in used_names:
                        i += 1
                        candidate = f"{cname}_{i}"
                node["name"] = candidate
                used_names.add(candidate)


def _split_multi_parent_shareables(var_order: List[str], nodes_by_var: Dict[str, Dict[str, Any]]) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Clone shareable runtime nodes per parent when one runtime is attached to multiple parents."""
    out_order = list(var_order)
    out_nodes = {k: deepcopy(v) for k, v in nodes_by_var.items()}

    # Match connection `parent` strings back to concrete parent vars.
    parent_identity: Dict[str, Set[str]] = {}
    for pvar in out_order:
        pnode = out_nodes.get(pvar, {})
        pnm = str(pnode.get("name") or "")
        parent_identity[pvar] = {_norm(pvar), _norm(pnm)}

    split_map: Dict[str, Dict[str, str]] = {}
    used_vars: Set[str] = set(out_order)

    # Scan runtime nodes and detect the ones whose connection records imply
    # that more than one parent owns the same flat runtime node.
    for cvar in list(out_order):
        cnode = out_nodes.get(cvar, {})
        ctype = _node_type_name(cnode)
        if ctype not in SHAREABLE_RUNTIME_TYPES:
            continue
        conns = _as_conn_list(cnode)
        if not conns:
            continue

        # Infer the owning parent vars by matching each connection's
        # `parent` string back to a concrete parent node in the graph.
        parent_vars_for_conn: Set[str] = set()
        for conn in conns:
            raw_parent = str(conn.get("parent") or "").strip()
            if not raw_parent:
                continue
            nparent = _norm(raw_parent)
            for pvar, keys in parent_identity.items():
                if nparent in keys:
                    parent_vars_for_conn.add(pvar)
        if len(parent_vars_for_conn) <= 1:
            continue

        # Create one clone per owning parent and keep only that parent's
        # connection records on each clone.
        per_parent_clone: Dict[str, str] = {}
        for pvar in sorted(parent_vars_for_conn):
            base = f"{cvar}__for__{pvar}"
            new_var = base
            suffix = 2
            while new_var in used_vars:
                new_var = f"{base}_{suffix}"
                suffix += 1
            used_vars.add(new_var)
            cloned = deepcopy(cnode)
            pkeys = parent_identity.get(pvar, set())
            kept: List[Dict[str, Any]] = []
            for conn in conns:
                cp = str(conn.get("parent") or "").strip()
                if cp and _norm(cp) in pkeys:
                    kept.append(deepcopy(conn))
            cloned["external_connections"] = kept if kept else None
            out_nodes[new_var] = cloned
            out_order.append(new_var)
            per_parent_clone[pvar] = new_var

        split_map[cvar] = per_parent_clone

    # Rewire each parent so it points to its own per-parent clone instead
    # of the original shared runtime var.
    for pvar in list(out_order):
        pnode = out_nodes.get(pvar, {})
        # Replace flat child var references with fully expanded nested child
        # objects under both containment fields.
        for field in ("nodes", "tool_list"):
            vals = pnode.get(field)
            if not isinstance(vals, list):
                continue
            new_vals: List[Any] = []
            changed = False
            for item in vals:
                cvar = extract_var_name(item)
                if not cvar or cvar not in split_map:
                    new_vals.append(item)
                    continue
                target = split_map[cvar].get(pvar)
                if target:
                    new_vals.append(target)
                    changed = True
                else:
                    new_vals.append(item)
            if changed:
                pnode[field] = new_vals

    # Drop the original shared runtime once every parent points to a clone.
    referenced: Set[str] = set()
    for pvar in out_order:
        pnode = out_nodes.get(pvar, {})
        for field in ("nodes", "tool_list"):
            vals = pnode.get(field)
            if not isinstance(vals, list):
                continue
            for item in vals:
                cvar = extract_var_name(item)
                if cvar:
                    referenced.add(cvar)

    # If the original shared runtime is no longer referenced after the
    # rewiring step, remove it from the exported graph.
    for orig_var in split_map:
        if orig_var in referenced:
            continue
        out_nodes.pop(orig_var, None)
        if orig_var in out_order:
            out_order.remove(orig_var)

    return out_order, out_nodes


def _expand_tree(root_var: str, nodes_by_var: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Expand a flat var-keyed graph into a nested `NodeSpec`-shaped tree."""
    def _rewrite_ref(value: Any, rename_map: Dict[str, str]) -> Any:
        """Rewrite references after a child name changes during expansion."""
        if isinstance(value, str):
            return rename_map.get(value, value)
        if isinstance(value, list):
            return [rename_map.get(x, x) if isinstance(x, str) else x for x in value]
        return value

    def _finalize_rename_map(raw_map: Dict[str, Set[str]]) -> Dict[str, str]:
        """Keep only deterministic child-name rewrites."""
        out: Dict[str, str] = {}
        for old, new_set in raw_map.items():
            if len(new_set) == 1:
                out[old] = next(iter(new_set))
        return out

    def walk(v: str, stack: Set[str]) -> Dict[str, Any]:
        """Inline one node and recursively expand its children."""
        if v in stack:
            return {}
        # Track the active expansion path so a cycle cannot recurse forever.
        next_stack = set(stack)
        next_stack.add(v)
        node = deepcopy(nodes_by_var.get(v, {}))
        ntype = _node_type_name(node)
        metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        is_synth_controller = ntype == "Deterministic_controller" and bool(metadata.get("synthetic_controller_for"))
        if is_synth_controller:
            # Keep the graph var as the exported name for synthetic controllers.
            node["name"] = v

        # Child names can change after expansion, so collect old->new name
        # candidates and rewrite local references once children are inlined.
        local_rename_raw: Dict[str, Set[str]] = {}
        for field in ("nodes", "tool_list"):
            vals = node.get(field)
            out: List[Dict[str, Any]] = []
            if isinstance(vals, list):
                for item in vals:
                    cvar = extract_var_name(item)
                    if cvar and cvar in nodes_by_var:
                        src_child = nodes_by_var.get(cvar, {})
                        old_child_name = str(src_child.get("name") or "")
                        child = walk(cvar, next_stack)
                        if isinstance(child, dict) and child:
                            new_child_name = str(child.get("name") or "")
                            if old_child_name and new_child_name and old_child_name != new_child_name:
                                local_rename_raw.setdefault(old_child_name, set()).add(new_child_name)
                            out.append(child)
            node[field] = out

        # Only apply renames that resolve to exactly one child name so we do
        # not rewrite references ambiguously.
        rename_map = _finalize_rename_map(local_rename_raw)
        if rename_map:
            # Update edge and connection references if child names changed after
            # inlining nested nodes.
            edges = node.get("internal_edges")
            if isinstance(edges, list):
                for edge in edges:
                    if not isinstance(edge, dict):
                        continue
                    if isinstance(edge.get("from_"), str):
                        edge["from_"] = _rewrite_ref(edge.get("from_"), rename_map)
                    if isinstance(edge.get("to"), str):
                        edge["to"] = _rewrite_ref(edge.get("to"), rename_map)

            # Child-local external connection records may still point at the old
            # child names, so rewrite those as well.
            for field in ("nodes", "tool_list"):
                vals = node.get(field)
                if not isinstance(vals, list):
                    continue
                for child in vals:
                    if not isinstance(child, dict):
                        continue
                    conns = child.get("external_connections")
                    if isinstance(conns, dict):
                        conns = [conns]
                    if not isinstance(conns, list):
                        continue
                    for conn in conns:
                        if not isinstance(conn, dict):
                            continue
                        if isinstance(conn.get("parent"), str):
                            conn["parent"] = _rewrite_ref(conn.get("parent"), rename_map)
                        conn["in_"] = _rewrite_ref(conn.get("in_"), rename_map)
                        conn["out"] = _rewrite_ref(conn.get("out"), rename_map)
        return node

    return walk(root_var, set())


def _get_parent_and_key(root: Any, loc: Tuple[Any, ...]) -> Tuple[Any, Any]:
    """Resolve the parent container and final key/index for a nested location."""
    cur = root
    for part in loc[:-1]:
        if isinstance(part, int) and isinstance(cur, list) and 0 <= part < len(cur):
            cur = cur[part]
        elif isinstance(part, str) and isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None, None
    return cur, loc[-1] if loc else None


def _get_value_at(root: Any, loc: Tuple[Any, ...]) -> Any:
    """Read a nested value by pydantic-style location path."""
    cur = root
    for part in loc:
        if isinstance(part, int) and isinstance(cur, list) and 0 <= part < len(cur):
            cur = cur[part]
        elif isinstance(part, str) and isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _set_value_at(root: Any, loc: Tuple[Any, ...], value: Any) -> bool:
    """Set a nested value when the target container exists."""
    parent, key = _get_parent_and_key(root, loc)
    if parent is None:
        return False
    if isinstance(key, int) and isinstance(parent, list) and 0 <= key < len(parent):
        parent[key] = value
        return True
    if isinstance(key, str) and isinstance(parent, dict):
        parent[key] = value
        return True
    return False


def _apply_other_fix(root: Dict[str, Any], err: Dict[str, Any]) -> bool:
    """Convert unsupported enum-like values into explicit `other` variants or add required descriptions to existing `other` fields."""
    msg = str(err.get("msg") or "")
    loc = tuple(err.get("loc") or ())
    if not loc:
        return False

    if loc[-1] == "other_description" and "required" in msg.lower():
        return _set_value_at(root, loc, "unknown")

    if loc[-1] == "other_kind_description" and "required" in msg.lower():
        return _set_value_at(root, loc, "unknown")

    current = _get_value_at(root, loc)
    if isinstance(current, dict):
        type_val = current.get("type")
        if current.get("framework") == "other" and current.get("other_description") in (None, ""):
            current["other_description"] = "unknown"
            return True
        if current.get("kind") == "other" and current.get("other_kind_description") in (None, ""):
            current["other_kind_description"] = "unknown"
            return True
        if type_val == "other" and current.get("other_description") in (None, ""):
            current["other_description"] = "unknown"
            return True
        if isinstance(type_val, (list, tuple, set)):
            vals = [str(x) for x in type_val]
            if len(vals) == 1 and vals[0] == "other" and current.get("other_description") in (None, ""):
                current["other_description"] = "unknown"
                return True

    if "Input should be" not in msg or "'other'" not in msg:
        return False

    invalid = _get_value_at(root, loc)
    invalid_s = str(invalid) if invalid not in (None, "") else "unknown"

    if loc[-1] in {"framework", "type", "kind"}:
        if _set_value_at(root, loc, "other"):
            container_loc = loc[:-1]
            if loc[-1] == "kind":
                _set_value_at(root, container_loc + ("other_kind_description",), invalid_s)
            else:
                _set_value_at(root, container_loc + ("other_description",), invalid_s)
            return True

    if len(loc) >= 2 and isinstance(loc[-1], int) and loc[-2] == "type":
        container_loc = loc[:-2]
        if _set_value_at(root, container_loc + ("type",), ["other"]):
            _set_value_at(root, container_loc + ("other_description",), invalid_s)
            return True
    return False


def _apply_tool_example_fix(root: Dict[str, Any], err: Dict[str, Any]) -> bool:
    """Remove only the failing tool example pair when schema validation rejects it."""
    msg = str(err.get("msg") or "")
    loc = tuple(err.get("loc") or ())
    if "Tool node" not in msg:
        return False
    if ("example.input" not in msg) and ("tool_example_pairs" not in msg):
        return False
    if not loc:
        return False

    try:
        pair_field_idx = loc.index("tool_example_pairs")
    except ValueError:
        return False
    if pair_field_idx + 1 >= len(loc) or not isinstance(loc[pair_field_idx + 1], int):
        return False

    pairs_loc = loc[: pair_field_idx + 1]
    failing_pair_idx = loc[pair_field_idx + 1]
    pairs = _get_value_at(root, pairs_loc)
    if not isinstance(pairs, list) or not (0 <= failing_pair_idx < len(pairs)):
        return False

    pairs.pop(failing_pair_idx)
    return True


def _apply_required_keys_fix(root: Dict[str, Any], err: Dict[str, Any]) -> bool:
    """Realign required_keys.enabled with whether keys is non-empty."""
    msg = str(err.get("msg") or "")
    loc = tuple(err.get("loc") or ())
    if not loc or loc[-1] != "required_keys":
        return False
    current = _get_value_at(root, loc)
    if not isinstance(current, dict):
        return False
    if "keys must be provided and non-empty" in msg and current.get("enabled"):
        current["enabled"] = False
        return True
    if "keys must be None or empty" in msg and not current.get("enabled"):
        current["enabled"] = True
        return True
    return False


def _drop_empty_frameworks(root: Any) -> bool:
    """Replace empty framework objects with `None` recursively."""
    changed = False
    if isinstance(root, dict):
        if root.get("framework") == {}:
            root["framework"] = None
            changed = True
        for value in root.values():
            changed = _drop_empty_frameworks(value) or changed
    elif isinstance(root, list):
        for value in root:
            changed = _drop_empty_frameworks(value) or changed
    return changed


def _validate_with_auto_fix(root_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the exported tree and apply a small set of safe auto-fixes."""

    work = deepcopy(root_dict)
    _drop_empty_frameworks(work)
    for _ in range(8):
        try:
            closed = NodeSpecTop.model_validate(work, context={"run_root_pass": True})
            return closed.model_dump(mode="python")
        except ValidationError as exc:
            changed = False
            # Apply only narrow, deterministic fixes tied to reported errors.
            for err in exc.errors():
                changed = _apply_tool_example_fix(work, err) or changed
                changed = _apply_other_fix(work, err) or changed
                changed = _apply_required_keys_fix(work, err) or changed
            changed = _drop_empty_frameworks(work) or changed
            if not changed:
                raise
    closed = NodeSpecTop.model_validate(work, context={"run_root_pass": True})
    return closed.model_dump(mode="python")


MAX_TOOL_EXAMPLES_PER_TOOL = 20


def _collect_observed_tool_io_pairs(out_dir: Path) -> List[Dict[str, Any]]:
    """Aggregate deduped tool-call/tool-result IO pairs from every parsed execution-validation trace."""
    run_root = out_dir / "execution_validation" / "execution_validation_runs"
    files = sorted(run_root.glob("*/run_*.llm_span_pipeline.json"))
    seen: Set[str] = set()
    pairs: List[Dict[str, Any]] = []
    for f in files:
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        events = obj.get("events") if isinstance(obj.get("events"), list) else []
        observed = observe_components(events)
        for row in observed.tool_io_pairs:
            # Dedup across files/runs -- the same tool call is often re-observed verbatim
            # across repeated validation prompts.
            key = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(row)
    return pairs


def _tool_ancestor_index(root_dict: Dict[str, Any]) -> Dict[str, List[Tuple[Set[str], Dict[str, Any]]]]:
    """Map each normalized Tool name to its (ancestor-name-set, node) occurrences in the tree.

    Ownership in a parsed trace is attributed to the calling *agent*, which is often several
    levels above the Tool's structural parent (e.g. Agent -> workbench -> Local_MCP_server ->
    Tool), so matching must walk the full ancestor chain, not just the immediate parent.
    """
    by_name: Dict[str, List[Tuple[Set[str], Dict[str, Any]]]] = {}

    def walk(node: Any, ancestors: Tuple[str, ...]) -> None:
        if not isinstance(node, dict):
            return
        node_type = node.get("node_type")
        ntype = node_type.get("type") if isinstance(node_type, dict) else ""
        name = str(node.get("name") or "")
        if ntype == "Tool" and name:
            ancestor_norms = {norm(a) for a in ancestors if a}
            by_name.setdefault(norm(name), []).append((ancestor_norms, node))
        next_ancestors = ancestors + (name,)
        for key in ("nodes", "tool_list"):
            for child in node.get(key) or []:
                walk(child, next_ancestors)

    walk(root_dict, ())
    return by_name


def _coerce_tool_output(value: Any) -> Dict[str, Any]:
    """Coerce an observed raw tool output into ToolIOPair's required Dict[str, Any] shape."""
    if isinstance(value, dict):
        return value
    text = str(value or "")
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    return {"result": text}


def _input_aligned_with_schema(observed_input: Any, inputs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the observed input dict if its keys exactly align with the tool's declared inputs, else None.

    Strict on purpose: an observed key the schema doesn't know about (or a missing required one)
    is a sign this might not really be the same tool/version, so the whole example is skipped
    rather than silently dropping the mismatched keys and forcing a fit.
    """
    if not isinstance(observed_input, dict):
        return None
    allowed = {str(p.get("name")) for p in inputs if isinstance(p, dict) and p.get("name")}
    required = {str(p.get("name")) for p in inputs if isinstance(p, dict) and p.get("name") and p.get("required")}
    observed_keys = set(observed_input.keys())
    if not observed_keys.issubset(allowed):
        return None
    if not required.issubset(observed_keys):
        return None
    return observed_input


def _attach_tool_example_pairs(root_dict: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """Attach real observed tool-call examples (from parsed execution-validation traces) to matching Tool nodes."""
    observed_pairs = _collect_observed_tool_io_pairs(out_dir)
    if not observed_pairs:
        return root_dict

    tool_index = _tool_ancestor_index(root_dict)
    attached = 0
    skipped_misaligned = 0
    skipped_no_match = 0

    for row in observed_pairs:
        owner = str(row.get("owner") or "").strip()
        tool = str(row.get("tool") or "").strip()
        if not owner or not tool:
            continue
        candidates = tool_index.get(norm(tool)) or []
        owner_norm = norm(owner)
        # A tool may appear under several parents/aliases; attach to every occurrence this
        # owner can actually reach, not just the first match.
        matched_nodes = [node for ancestors, node in candidates if owner_norm in ancestors]
        if not matched_nodes:
            skipped_no_match += 1
            continue

        for node in matched_nodes:
            inputs = node.get("inputs") if isinstance(node.get("inputs"), list) else []
            aligned_input = _input_aligned_with_schema(row.get("input"), inputs)
            if aligned_input is None:
                skipped_misaligned += 1
                continue
            example = {"input": aligned_input, "output": _coerce_tool_output(row.get("output"))}

            examples = node.get("tool_example_pairs")
            if not isinstance(examples, list):
                examples = []
                node["tool_example_pairs"] = examples
            key = json.dumps(example, sort_keys=True, ensure_ascii=False, default=str)
            existing_keys = {json.dumps(e, sort_keys=True, ensure_ascii=False, default=str) for e in examples}
            if key in existing_keys:
                continue
            if len(examples) >= MAX_TOOL_EXAMPLES_PER_TOOL:
                continue
            examples.append(example)
            attached += 1

    print(
        "[post_processing] tool_example_pairs: "
        f"attached={attached} skipped_misaligned_input={skipped_misaligned} skipped_no_matching_tool={skipped_no_match}"
    )
    return root_dict


def _attach_runtime_mapping(root_dict: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """Attach the induced runtime alias registry (if present) to the root node only."""
    runtime_mapping_path = out_dir / "validation" / "runtime_mapping.json"
    if not runtime_mapping_path.exists():
        return root_dict
    try:
        root_dict["runtime_mapping"] = json.loads(runtime_mapping_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return root_dict


def _load_project_root(out_dir: Path) -> str | None:
    """Load the original project root, if it was recorded in the output directory."""
    root_record_path = out_dir / "root_record.json"
    if not root_record_path.exists():
        return None
    try:
        payload = json.loads(root_record_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    project_root = payload.get("project_root")
    if not isinstance(project_root, str) or not project_root.strip():
        return None
    return project_root.rstrip("/")


def _normalize_string_path_refs(value: str, project_root: str) -> str:
    """Strip the recorded project-root prefix from a string path."""
    prefix = f"{project_root}/"
    if prefix in value:
        return value.replace(prefix, "")
    return value


def _normalize_paths_in_obj(obj: Any, project_root: str) -> Any:
    """Recursively strip the project-root prefix from nested objects."""
    if isinstance(obj, str):
        return _normalize_string_path_refs(obj, project_root)
    if isinstance(obj, list):
        return [_normalize_paths_in_obj(x, project_root) for x in obj]
    if isinstance(obj, dict):
        return {k: _normalize_paths_in_obj(v, project_root) for k, v in obj.items()}
    return obj


def _normalize_paths(root_dump: Dict[str, Any], project_root: str | None) -> Dict[str, Any]:
    """Normalize stored path fragments in known path-bearing fields, excluding flows."""
    if not project_root:
        return root_dump

    def walk(node: Any) -> None:
        """Traverse the exported tree and rewrite known path-bearing fields."""
        if isinstance(node, dict):
            if "code_references" in node:
                node["code_references"] = _normalize_paths_in_obj(node.get("code_references"), project_root)
            if "data_path" in node:
                node["data_path"] = _normalize_paths_in_obj(node.get("data_path"), project_root)
            if "metadata" in node:
                node["metadata"] = _normalize_paths_in_obj(node.get("metadata"), project_root)
            if "required_keys" in node:
                node["required_keys"] = _normalize_paths_in_obj(node.get("required_keys"), project_root)
            if "tool_example_pairs" in node:
                node["tool_example_pairs"] = _normalize_paths_in_obj(node.get("tool_example_pairs"), project_root)
            if "entry_point_usage_example" in node:
                node["entry_point_usage_example"] = _normalize_paths_in_obj(node.get("entry_point_usage_example"), project_root)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(root_dump)
    return root_dump


def _load_execution_command_example(path_str: str) -> Dict[str, Any] | None:
    """Load one execution-command example JSON object from disk."""
    p = str(path_str or "").strip()
    if not p:
        return None
    path = Path(p).resolve()
    if not path.exists():
        raise FileNotFoundError(f"execution_command_example_json not found: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON in execution_command_example_json: {path}") from exc
    if not isinstance(obj, dict):
        raise ValueError("execution_command_example_json must contain a JSON object")
    obj.setdefault("__source_path__", str(path))
    return obj


def _task_arg_names(exec_example: Dict[str, Any] | None) -> Set[str]:
    """Return the argument names treated as task/query-bearing inputs."""
    if not isinstance(exec_example, dict):
        return {"--task", "--query", "--prompt"}
    raw = exec_example.get("task_arg_names")
    names: Set[str] = set()
    if isinstance(raw, list):
        for item in raw:
            txt = str(item or "").strip()
            if txt:
                names.add(txt)
    return names or {"--task", "--query", "--prompt"}


def _entrypoint_value(exec_example: Dict[str, Any] | None) -> str:
    """Extract the configured entrypoint value from an execution example."""
    if not isinstance(exec_example, dict):
        return ""
    ep = exec_example.get("entrypoint")
    if isinstance(ep, str):
        return ep.strip()
    if isinstance(ep, dict):
        return str(ep.get("value") or "").strip()
    return ""


def _usage_example_from_exec_example(exec_example: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """Convert an execution-command example payload into a NodeSpec UsageExample dict."""
    if not isinstance(exec_example, dict):
        return None

    entrypoint = _entrypoint_value(exec_example).replace("\\", "/")
    if not entrypoint:
        return None

    sandbox_root = "/sandbox"
    if entrypoint != sandbox_root and not entrypoint.startswith(f"{sandbox_root}/"):
        parts = entrypoint.split("/")
        while parts and parts[0] in ("", "."):
            parts.pop(0)
        parts.insert(0, sandbox_root)
        script = "/".join(parts)
    else:
        script = entrypoint

    task_names = _task_arg_names(exec_example)
    args_payload = exec_example.get("args") if isinstance(exec_example.get("args"), list) else []
    rendered_args: List[Dict[str, Any]] = []
    for item in args_payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        value = item.get("value")
        rendered_args.append({
            "name": name.lstrip("-") or name,
            "type": "str",
            "required": bool(value is None),
            "example": "" if value is None else str(value),
            "is_task_input": name in task_names,
        })

    return {
        "script": script,
        "arguments": rendered_args,
    }


def _render_value(value: Any, *, indent: int, key_hint: str | None = None) -> str:
    """Render a Python value back into source text for `NodeSpec_schema`."""
    pad = " " * indent
    class_by_field = {
        "node_type": "NodeType",
        "required_keys": "RequiredKeys",
        "duplicates": "Duplication",
        "framework": "FrameworkType",
        "agent_type": "AgentType",
        "system_type": "SystemType",
        "llm_config": "LLMConfig",
        "entry_point_usage_example": "UsageExample",
    }
    list_class_by_field = {
        "inputs": "InputPort",
        "outputs": "OutputPort",
        "code_references": "CodeReference",
        "external_connections": "Connection",
        "internal_edges": "Edge",
        "tool_example_pairs": "ToolIOPair",
    }

    def _render_class_call(class_name: str, payload: Dict[str, Any], *, indent_inner: int) -> str:
        """Render a schema helper class call such as `NodeType(...)` or `Edge(...)`."""
        inner_pad = " " * indent_inner
        parts: List[str] = []
        for key, item in payload.items():
            if key == "__call__" or item is None:
                continue
            parts.append(f"{key}={_render_value(item, indent=indent_inner + 4, key_hint=key)}")
        if not parts:
            return f"{class_name}()"
        if len(parts) == 1 and "\n" not in parts[0]:
            return f"{class_name}({parts[0]})"
        return f"{class_name}(\n{inner_pad}    " + f",\n{inner_pad}    ".join(parts) + f",\n{inner_pad})"

    if isinstance(value, str):
        return repr(value)
    if value is None or isinstance(value, (int, float, bool)):
        return repr(value)
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        if key_hint in {"nodes", "tool_list"}:
            parts = ",\n".join(f"{pad}    {_render_nodespec_obj(x, indent=indent + 4)}" for x in value)
        elif key_hint in list_class_by_field:
            cls = list_class_by_field[key_hint]
            parts = ",\n".join(
                f"{pad}    {_render_class_call(cls, x, indent_inner=indent + 4) if isinstance(x, dict) else _render_value(x, indent=indent + 4)}"
                for x in value
            )
        else:
            parts = ",\n".join(f"{pad}    {_render_value(x, indent=indent + 4)}" for x in value)
        return "[\n" + parts + f"\n{pad}]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        if "__call__" in value and isinstance(value.get("__call__"), str):
            return _render_class_call(str(value["__call__"]), value, indent_inner=indent)
        if key_hint in class_by_field:
            return _render_class_call(class_by_field[key_hint], value, indent_inner=indent)
        parts = []
        for key, item in value.items():
            if item is None:
                continue
            parts.append(f"{pad}    {repr(key)}: {_render_value(item, indent=indent + 4)}")
        if not parts:
            return "{}"
        return "{\n" + ",\n".join(parts) + f"\n{pad}" + "}"
    return repr(value)


def _render_nodespec_obj(node: Dict[str, Any], *, indent: int) -> str:
    """Render one nested node dict as a `NodeSpec(...)` constructor call."""
    if not isinstance(node, dict):
        return _render_value(node, indent=indent)

    preferred = [
        "name",
        "id",
        "is_graph",
        "emulated",
        "node_type",
        "description",
        "code_execution",
        "code_references",
        "inputs",
        "outputs",
        "external_connections",
        "required_keys",
        "duplicates",
        "framework",
        "agency_level",
        "flows",
        "entry_point_usage_example",
        "runtime_mapping",
        "agent_type",
        "system_type",
        "llm_config",
        "system_prompt",
        "user_prompt_template",
        "routing_logic",
        "tool_list",
        "tool_example_pairs",
        "nodes",
        "internal_edges",
        "metadata",
    ]
    keys = [k for k in preferred if k in node] + [k for k in node.keys() if k not in preferred]
    pad = " " * indent
    lines: List[str] = ["NodeSpec("]
    for key in keys:
        value = node.get(key)
        if value is None:
            continue
        lines.append(f"{pad}    {key}={_render_value(value, indent=indent + 4, key_hint=key)},")
    lines.append(f"{pad})")
    return "\n".join(lines)


def _render_output_script(root_dump: Dict[str, Any], *, output_stem: str) -> str:
    """Render the final standalone Python output script."""
    spec_txt = _render_nodespec_obj(root_dump, indent=0)
    json_name = f"{output_stem}.json"
    return (
        "from __future__ import annotations\n\n"
        "from pathlib import Path\n\n"
        "try:\n"
        "    from NodeSpec_schema import (\n"
        "        AgentType,\n"
        "        CodeReference,\n"
        "        Connection,\n"
        "        Duplication,\n"
        "        Edge,\n"
        "        FlowSpec,\n"
        "        FrameworkType,\n"
        "        InputPort,\n"
        "        LLMConfig,\n"
        "        NodeSpec,\n"
        "        NodeSpecTop,\n"
        "        NodeType,\n"
        "        OutputPort,\n"
        "        RequiredKeys,\n"
        "        SystemType,\n"
        "        ToolIOPair,\n"
        "        UsageExample,\n"
        "    )\n"
        "except ModuleNotFoundError:\n"
        "    import sys\n"
        "    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / \"modular_imp\"))\n"
        "    from NodeSpec_schema import (\n"
        "        AgentType,\n"
        "        CodeReference,\n"
        "        Connection,\n"
        "        Duplication,\n"
        "        Edge,\n"
        "        FlowSpec,\n"
        "        FrameworkType,\n"
        "        InputPort,\n"
        "        LLMConfig,\n"
        "        NodeSpec,\n"
        "        NodeSpecTop,\n"
        "        NodeType,\n"
        "        OutputPort,\n"
        "        RequiredKeys,\n"
        "        SystemType,\n"
        "        ToolIOPair,\n"
        "        UsageExample,\n"
        "    )\n\n"
        f"agent_spec = {spec_txt}\n\n"
        "agent_spec = NodeSpecTop.model_validate(\n"
        "    agent_spec.model_dump(mode=\"python\"),\n"
        "    context={\"run_root_pass\": True},\n"
        ")\n\n"
        f"agent_spec.to_json(path=str(Path(__file__).with_name(\"{json_name}\")))\n"
        f"loaded_agent_spec = NodeSpec.from_json(path=str(Path(__file__).with_name(\"{json_name}\")))\n\n"
        "print(f\" Tools: {loaded_agent_spec.list_tools()}\")\n"
        "print(f\" Agents: {loaded_agent_spec.list_agents()}\")\n"
        "print(f\" LLMs: {loaded_agent_spec.list_llms()}\")\n"
        "print(f\" Systems: {loaded_agent_spec.list_systems()}\")\n"
        "print(f\" MCP servers: {loaded_agent_spec.list_mcp_servers()}\")\n"
    )


def _resolve_nodes_path(out_dir: Path, nodes_py: str) -> Path:
    """Resolve the required input nodes file under the selected output directory."""
    p = (out_dir / Path(nodes_py).name).resolve()
    if p.exists():
        return p
    raise FileNotFoundError(f"Input nodes file not found: {p}")


def main() -> None:
    """Load a corrected graph file, validate it, and export a Python spec script."""
    ap = argparse.ArgumentParser(description="Export a validated Python spec script from graph-correctness or code-reference-refine output.")
    ap.add_argument("--env_file", default=".env", help="Optional .env file to load.")
    ap.add_argument("--out_dir", required=True, help="Directory containing the input nodes file.")
    ap.add_argument("--nodes_py", required=True, help="Input nodes file under --out_dir.")
    ap.add_argument("--root_json", default="root_nodes.json", help="Root json under --out_dir.")
    ap.add_argument("--output_stem", default="final_spec_gt_spec", help="Base output name used for both the .py and .json files.")
    ap.add_argument("--execution_command_example_json", default="", help="Optional execution command example JSON used to populate entry_point_usage_example.")
    args = ap.parse_args()
    load_env_file(Path(args.env_file))

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = _resolve_nodes_path(out_dir, args.nodes_py)
    root_json = out_dir / Path(args.root_json).name
    out_py = out_dir / f"{Path(args.output_stem).name}.py"

    # Load the flat var-keyed graph and copy the schema alongside the export.
    copy_nodespec_schema_to_out_dir(out_dir)
    var_order, nodes_by_var = load_nodes_by_var(nodes_path)
    if not var_order or not nodes_by_var:
        raise RuntimeError(f"No nodes parsed from: {nodes_path}")

    # Make sibling child names unique before any cloning/validation pass that
    # derives nested ids from parent_id + child.name.
    _disambiguate_sibling_names(var_order, nodes_by_var)

    # Materialize `external_connections` from the current edge structure before
    # any logic that depends on parent-scoped connection ownership.
    _materialize_connections_from_edges(var_order, nodes_by_var)

    # Split shareable runtime nodes across parents, then resolve the known
    # root from the saved root-json artifact used by earlier stages.
    var_order, nodes_by_var = _split_multi_parent_shareables(var_order, nodes_by_var)
    root_node = read_root_node_json(root_json, allow_flat_object=True)
    root_var = resolve_root_var(root_node, nodes_by_var)
    if not root_var:
        raise RuntimeError("post_processing: failed resolving root var from root_json.")

    # Expand the flat graph into a nested tree, validate it, then normalize
    # persisted paths before rendering the final Python file.
    root_dict = _expand_tree(root_var, nodes_by_var)
    root_dict = _attach_runtime_mapping(root_dict, out_dir)
    root_dict = _attach_tool_example_pairs(root_dict, out_dir)
    exec_example = _load_execution_command_example(args.execution_command_example_json)
    usage_example = _usage_example_from_exec_example(exec_example)
    if usage_example is not None:
        root_dict["entry_point_usage_example"] = usage_example
    project_root = _load_project_root(out_dir)
    root_dump = _validate_with_auto_fix(root_dict)
    root_dump = _normalize_paths(root_dump, project_root)

    out_py.write_text(_render_output_script(root_dump, output_stem=Path(args.output_stem).name), encoding="utf-8")
    print(f"[post_processing] input: {nodes_path}")
    print(f"[post_processing] wrote: {out_py}")


if __name__ == "__main__":
    main()
