"""Induce a deterministic runtime name-mapping (alias registry) from execution-validation artifacts."""

from __future__ import annotations

import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from .utils import load_expected_components
    from ..global_utils import norm
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    CLEAN_CODE_ROOT = Path(__file__).resolve().parents[1]
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    if str(CLEAN_CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CLEAN_CODE_ROOT))
    from utils import load_expected_components
    from global_utils import norm


def _norm_name(s: Any) -> str:
    """Normalize a name for loose matching by lowercasing and keeping only alphanumerics."""
    return norm(str(s or ""))


def _name_keys(name: str) -> List[str]:
    """Deterministic lookup keys for a name, including its dotted/slash/colon-separated tokens."""
    raw = str(name or "").strip()
    if not raw:
        return []
    keys: List[str] = []
    seen: set = set()

    def _add(v: str) -> None:
        """Append the normalized form of `v` to `keys` if non-empty and not already seen."""
        n = _norm_name(v)
        if n and n not in seen:
            seen.add(n)
            keys.append(n)

    _add(raw)
    for token in raw.replace("/", ".").replace(":", ".").split("."):
        t = token.strip()
        if t:
            _add(t)
    return keys


def _iter_run_files(execution_validation_out_dir: Path) -> List[Path]:
    """Return every canonicalized run file produced by the execution-validation stage."""
    run_root = execution_validation_out_dir / "execution_validation_runs"
    files = sorted(run_root.glob("*/run_*.llm_span_pipeline.json"))
    return [p for p in files if p.is_file()]


def _iter_name_mapping_pairs(execution_validation_out_dir: Path) -> List[Tuple[Path, Path]]:
    """Return (payload, raw) pairs for every per-event name-mapping decision written during validation."""
    run_root = execution_validation_out_dir / "execution_validation_runs"
    payloads = sorted(run_root.glob("*/name_mapping_expected/run_*/name_mapping_expected/event_*.payload.json"))
    out: List[Tuple[Path, Path]] = []
    for pld in payloads:
        raw = pld.with_suffix("").with_suffix(".raw.txt")
        if raw.exists():
            out.append((pld, raw))
    return out


def _iter_trace_files(execution_validation_out_dir: Path) -> List[Path]:
    """Return every raw exported trace JSON file collected while running execution validation."""
    run_root = execution_validation_out_dir / "execution_validation_runs"
    return [p for p in sorted(run_root.glob("*/traces/*.json")) if p.is_file()]


def _expected_norm_map(names: List[str]) -> Dict[str, List[str]]:
    """Map every normalized sub-token key of each expected name back to that expected name."""
    out: Dict[str, List[str]] = {}
    for n in names:
        for k in _name_keys(n):
            out.setdefault(k, []).append(n)
    return out


def _expected_full_norm_map(names: List[str]) -> Dict[str, List[str]]:
    """Map the full normalized form of each expected name back to that expected name."""
    out: Dict[str, List[str]] = {}
    for n in names:
        k = _norm_name(n)
        if not k:
            continue
        out.setdefault(k, []).append(n)
    return out


def _norm_variants(observed: str) -> List[str]:
    """Return normalized candidate forms of an observed name, unwrapping common runtime wrapper prefixes/suffixes."""
    raw = str(observed or "").strip()
    base = _norm_name(raw)
    vals: List[str] = []
    seen: set = set()

    def _add(v: str) -> None:
        """Append the normalized form of `v` to `vals` if non-empty and not already seen."""
        vv = _norm_name(v)
        if vv and vv not in seen:
            seen.add(vv)
            vals.append(vv)

    _add(raw)
    if "." in raw:
        _add(raw.split(".")[-1])
    if "::" in raw:
        _add(raw.split("::")[-1])
    if "/" in raw:
        _add(raw.split("/")[-1])
    # unwrap known wrapper prefixes seen in runtime traces
    if base.startswith("functions"):
        _add(base[len("functions") :])
    if base.startswith("mcp"):
        _add(base[len("mcp") :])
    if "api" in base:
        i = base.rfind("api")
        if i >= 0 and i + 3 < len(base):
            _add(base[i + 3 :])
    return vals


def _candidate_from_expected(
    observed: str,
    expected_norm_map: Dict[str, List[str]],
    expected_full_norm_map: Dict[str, List[str]],
) -> str:
    """Deterministically resolve one observed name to a single expected name, or "" if ambiguous/unresolved."""
    # Sub-token key match against expected names (e.g. a dotted/prefixed observed name).
    keys = _name_keys(observed)
    for k in keys:
        cands = sorted(set(expected_norm_map.get(k) or []))
        if len(cands) == 1:
            return cands[0]
    variants = _norm_variants(observed)
    # Exact normalized match against full expected names.
    for v in variants:
        cands = sorted(set(expected_full_norm_map.get(v) or []))
        if len(cands) == 1:
            return cands[0]
    # Suffix containment fallback for prefixed runtime labels
    # (e.g. mcp_financialdatasets_ai_api_get_stock_price -> get_stock_price).
    suffix_hits: List[str] = []
    for v in variants:
        if len(v) < 6:
            continue
        for exp_norm, exp_names in expected_full_norm_map.items():
            if len(exp_norm) < 6:
                continue
            if v.endswith(exp_norm) or exp_norm.endswith(v):
                suffix_hits.extend(exp_names)
    uniq = sorted(set(suffix_hits))
    if len(uniq) == 1:
        return uniq[0]
    return ""


def _best_expected_name(observed: str, expected_names: List[str]) -> str:
    """Return the expected name with the highest fuzzy-similarity score to an observed name."""
    obs = str(observed or "").strip()
    if not obs or not expected_names:
        return ""
    obs_norm = _norm_name(obs)
    if not obs_norm:
        return ""
    scored: List[Tuple[float, str]] = []
    for exp in expected_names:
        exp_norm = _norm_name(exp)
        if not exp_norm:
            continue
        scored.append((difflib.SequenceMatcher(None, obs_norm, exp_norm).ratio(), exp))
    if not scored:
        return ""
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _add_alias_vote(votes: Dict[str, Counter], alias: str, canonical: str) -> None:
    """Record one vote that a normalized alias maps to a canonical expected name."""
    a = _norm_name(alias)
    c = str(canonical or "").strip()
    if not a or not c:
        return
    votes.setdefault(a, Counter())[c] += 1


def _resolve_votes(votes: Dict[str, Counter]) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
    """Resolve per-alias vote counters into a final alias->canonical mapping plus any unresolved conflicts."""
    mapping: Dict[str, str] = {}
    conflicts: Dict[str, Dict[str, int]] = {}
    for alias, counter in votes.items():
        if not counter:
            continue
        ranked = counter.most_common()
        # A single candidate is an unambiguous match.
        if len(ranked) == 1:
            mapping[alias] = ranked[0][0]
            continue
        # Multiple candidates: only accept the leader if it has a clear majority.
        top_name, top_cnt = ranked[0]
        second_cnt = ranked[1][1]
        if top_cnt >= 2 and top_cnt >= 2 * second_cnt:
            mapping[alias] = top_name
        else:
            conflicts[alias] = dict(counter)
    return mapping, conflicts


def _resolve_cross_type_conflicts(
    *,
    agent_alias_map: Dict[str, str],
    tool_alias_map: Dict[str, str],
) -> Dict[str, Dict[str, str]]:
    """Detect and drop aliases that resolved to both an agent and a tool, since that mapping is ambiguous."""
    cross: Dict[str, Dict[str, str]] = {}
    for alias, agent_target in list(agent_alias_map.items()):
        tool_target = tool_alias_map.get(alias)
        if not tool_target:
            continue
        # Same normalized alias being used for both agent and tool is ambiguous by default.
        cross[alias] = {"agent": agent_target, "tool": tool_target}
    # Drop conflicting aliases from final registry to avoid incorrect deterministic mapping.
    for alias in cross:
        agent_alias_map.pop(alias, None)
        tool_alias_map.pop(alias, None)
    return cross


def _walk(obj: Any, path: str = "") -> Iterable[Tuple[str, Any]]:
    """Recursively yield every (dotted/bracketed path, node) pair in a nested dict/list structure."""
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            yield from _walk(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{path}[{i}]"
            yield from _walk(v, p)


def _cpath(path: str) -> str:
    """Collapse list indices in a walked path so paths across different events compare equal."""
    return re.sub(r"\[\d+\]", "[*]", path)


def _induce_pointer_profile(trace_files: List[Path]) -> Dict[str, Any]:
    """Infer common attribute-payload pointer paths (name/role/content/tool-call) from raw traces."""
    name_paths = Counter()
    role_paths = Counter()
    content_paths = Counter()
    tool_name_paths = Counter()
    message_list_paths = Counter()

    # Scan every raw trace file's spans for the attribute roots that carry event payloads.
    for tf in trace_files:
        try:
            tr = json.loads(tf.read_text(encoding="utf-8"))
        except Exception:
            continue
        spans = ((tr.get("data") if isinstance(tr.get("data"), dict) else tr).get("spans") or [])
        for s in spans:
            attrs = s.get("attributes") if isinstance(s.get("attributes"), dict) else {}
            roots = {
                "spanInputs": attrs.get("mlflow.spanInputs"),
                "spanOutputs": attrs.get("mlflow.spanOutputs"),
                "metadata": attrs.get("metadata"),
            }
            # Attribute roots are often JSON-encoded strings; parse them where possible.
            parsed_roots: Dict[str, Any] = {}
            for rk, rv in roots.items():
                if isinstance(rv, str):
                    try:
                        parsed_roots[rk] = json.loads(rv)
                    except Exception:
                        parsed_roots[rk] = rv
                else:
                    parsed_roots[rk] = rv
            # Walk each parsed root and vote for the paths that look like name/role/content/tool-call carriers.
            for rk, root in parsed_roots.items():
                for p, node in _walk(root, rk):
                    cp = _cpath(p)
                    if isinstance(node, list) and node and all(isinstance(x, dict) for x in node[:3]):
                        keys = set()
                        for it in node[:5]:
                            keys.update({str(k).lower() for k in it.keys()})
                        if {"content", "text", "message", "raw"} & keys and {"role", "type", "source"} & keys:
                            message_list_paths[cp] += 1
                    if isinstance(node, dict):
                        for k in node.keys():
                            kl = str(k).lower()
                            if kl in {"name", "source"}:
                                name_paths[f"{cp}.{kl}"] += 1
                            if kl in {"role", "type"}:
                                role_paths[f"{cp}.{kl}"] += 1
                            if kl in {"content", "text", "message", "raw"}:
                                content_paths[f"{cp}.{kl}"] += 1
                            if kl == "tool_calls":
                                tool_name_paths[f"{cp}.tool_calls[*].name"] += 1

    # Return the most common candidate paths of each kind, most frequent first.
    return {
        "message_list_paths": [p for p, _ in message_list_paths.most_common(20)],
        "name_paths": [p for p, _ in name_paths.most_common(30)],
        "role_paths": [p for p, _ in role_paths.most_common(20)],
        "content_paths": [p for p, _ in content_paths.most_common(20)],
        "tool_name_paths": [p for p, _ in tool_name_paths.most_common(20)],
    }


def induce_runtime_mapping(*, out_dir: Path, nodes_py: Path) -> Dict[str, Any]:
    """Induce the runtime alias registry for one pipeline run from its execution-validation artifacts."""
    out_dir = out_dir.resolve()
    nodes_py = nodes_py.resolve()
    execution_validation_out_dir = (out_dir / "execution_validation").resolve()

    # Load the expected agent/tool names from the NodeSpec and index them for lookup.
    expected = load_expected_components(nodes_py)
    expected_agents = sorted(expected.agents)
    expected_tools = sorted({row.tool_name for row in expected.tools_by_owner})
    exp_agent_norm = _expected_norm_map(expected_agents)
    exp_tool_norm = _expected_norm_map(expected_tools)
    exp_agent_full_norm = _expected_full_norm_map(expected_agents)
    exp_tool_full_norm = _expected_full_norm_map(expected_tools)

    agent_votes: Dict[str, Counter] = {}
    tool_votes: Dict[str, Counter] = {}

    # Expand from canonical run files deterministically: every agent/tool event name,
    # and every tool_calls entry, casts one vote for its resolved expected name.
    for rf in _iter_run_files(execution_validation_out_dir):
        try:
            obj = json.loads(rf.read_text(encoding="utf-8"))
        except Exception:
            continue
        for ev in obj.get("events") or []:
            if not isinstance(ev, dict):
                continue
            et = str(ev.get("type") or "").strip().lower()
            nm = str(ev.get("name") or "").strip()
            if et == "agent" and nm:
                c = _candidate_from_expected(nm, exp_agent_norm, exp_agent_full_norm)
                if not c:
                    c = _best_expected_name(nm, expected_agents)
                if c:
                    _add_alias_vote(agent_votes, nm, c)
            elif et == "tool" and nm:
                c = _candidate_from_expected(nm, exp_tool_norm, exp_tool_full_norm)
                if not c:
                    c = _best_expected_name(nm, expected_tools)
                if c:
                    _add_alias_vote(tool_votes, nm, c)
            tcs = ev.get("tool_calls")
            if isinstance(tcs, list):
                for tc in tcs:
                    if not isinstance(tc, dict):
                        continue
                    tnm = str(tc.get("name") or "").strip()
                    if not tnm:
                        continue
                    c = _candidate_from_expected(tnm, exp_tool_norm, exp_tool_full_norm)
                    if not c:
                        c = _best_expected_name(tnm, expected_tools)
                    if c:
                        _add_alias_vote(tool_votes, tnm, c)

    # Ingest explicit per-event name-mapping decisions recorded during execution validation.
    # These are stronger than lexical candidates because they include LLM-reviewed evidence.
    # Keep only conservative rows:
    # - decision=map_to_expected
    # - alias_scope=specific
    for payload_file, raw_file in _iter_name_mapping_pairs(execution_validation_out_dir):
        try:
            payload = json.loads(payload_file.read_text(encoding="utf-8"))
            raw_obj = json.loads(raw_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict) or not isinstance(raw_obj, dict):
            continue
        decision = str(raw_obj.get("decision") or "").strip().lower()
        mapped_name = str(raw_obj.get("mapped_name") or "").strip()
        alias_scope = str(raw_obj.get("alias_scope") or "").strip().lower()
        if decision != "map_to_expected" or not mapped_name:
            continue
        if alias_scope and alias_scope != "specific":
            # skip generic aliases to avoid over-broad mappings
            continue
        ctype = str(payload.get("component_type") or "").strip().lower()
        observed_name = str(payload.get("observed_name") or "").strip()
        if ctype == "agent":
            _add_alias_vote(agent_votes, observed_name, mapped_name)
        elif ctype == "tool":
            _add_alias_vote(tool_votes, observed_name, mapped_name)

    # Resolve the accumulated votes into final alias registries, then drop any alias
    # that ambiguously resolved to both an agent and a tool.
    agent_alias_map, agent_conflicts = _resolve_votes(agent_votes)
    tool_alias_map, tool_conflicts = _resolve_votes(tool_votes)
    cross_type_conflicts = _resolve_cross_type_conflicts(
        agent_alias_map=agent_alias_map,
        tool_alias_map=tool_alias_map,
    )

    pointer_profile = _induce_pointer_profile(_iter_trace_files(execution_validation_out_dir))

    # Assemble the final runtime mapping document written to runtime_mapping.json.
    return {
        "version": 1,
        "out_dir": str(out_dir),
        "nodes_file": str(nodes_py),
        "expected_agent_names": expected_agents,
        "expected_tool_names": expected_tools,
        "pointer_profile": pointer_profile,
        "alias_registry": {
            "agent": agent_alias_map,
            "tool": tool_alias_map,
        },
        "stats": {
            "num_expected_agents": len(expected_agents),
            "num_expected_tools": len(expected_tools),
            "num_agent_aliases": len(agent_alias_map),
            "num_tool_aliases": len(tool_alias_map),
            "num_agent_conflicts": len(agent_conflicts),
            "num_tool_conflicts": len(tool_conflicts),
            "num_cross_type_conflicts": len(cross_type_conflicts),
        },
        "conflicts": {
            "agent": agent_conflicts,
            "tool": tool_conflicts,
            "cross_type": cross_type_conflicts,
        },
    }
