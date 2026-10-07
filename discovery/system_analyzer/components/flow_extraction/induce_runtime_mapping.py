from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .trace_parsing import (
        _name_keys,
        _norm_name,
        load_expected_names_from_nodespec,
    )
except ImportError:
    from flow_extraction.trace_parsing import (
        _name_keys,
        _norm_name,
        load_expected_names_from_nodespec,
    )


def _resolve_default_nodespec(outputs_dir: Path) -> Path:
    m = re.search(r"use_case_(\d+)_outputs_\d+$", outputs_dir.name)
    if m:
        uc = m.group(1)
        spec_cand = Path("use_cases") / f"use_case_{uc}" / f"use_case_{uc}_spec.py"
        if spec_cand.exists():
            return spec_cand.resolve()
    spec_hits = sorted(outputs_dir.glob("*_spec.py"))
    if spec_hits:
        return spec_hits[0].resolve()
    for name in ("nodespec.py", "final_nodespec.py"):
        cand = outputs_dir / name
        if cand.exists():
            return cand.resolve()
    raise FileNotFoundError(f"No nodespec/spec file found for {outputs_dir}")


def _iter_run_files(validation_dir: Path) -> List[Path]:
    run_root = validation_dir / "execution_validation_runs"
    files = sorted(run_root.glob("**/run_*.llm_span_pipeline.json"))
    return [p for p in files if p.is_file()]


def _iter_name_mapping_pairs(validation_dir: Path) -> List[Tuple[Path, Path]]:
    run_root = validation_dir / "execution_validation_runs"
    payloads = sorted(run_root.glob("**/name_mapping_expected/event_*.payload.json"))
    out: List[Tuple[Path, Path]] = []
    for pld in payloads:
        raw = pld.with_suffix("").with_suffix(".raw.txt")
        if raw.exists():
            out.append((pld, raw))
    return out


def _expected_norm_map(names: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for n in names:
        for k in _name_keys(n):
            out.setdefault(k, []).append(n)
    return out


def _candidate_from_expected(observed: str, expected_norm_map: Dict[str, List[str]]) -> str:
    keys = _name_keys(observed)
    for k in keys:
        cands = sorted(set(expected_norm_map.get(k) or []))
        if len(cands) == 1:
            return cands[0]
    return ""


def _add_alias_vote(votes: Dict[str, Counter], alias: str, canonical: str) -> None:
    a = _norm_name(alias)
    c = str(canonical or "").strip()
    if not a or not c:
        return
    votes.setdefault(a, Counter())[c] += 1


def _resolve_votes(votes: Dict[str, Counter]) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
    mapping: Dict[str, str] = {}
    conflicts: Dict[str, Dict[str, int]] = {}
    for alias, counter in votes.items():
        if not counter:
            continue
        ranked = counter.most_common()
        if len(ranked) == 1:
            mapping[alias] = ranked[0][0]
            continue
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
    return re.sub(r"\[\d+\]", "[*]", path)


def _induce_pointer_profile(traces_dir: Path) -> Dict[str, Any]:
    name_paths = Counter()
    role_paths = Counter()
    content_paths = Counter()
    tool_name_paths = Counter()
    message_list_paths = Counter()

    for tf in sorted(traces_dir.glob("*.json")):
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
            parsed_roots: Dict[str, Any] = {}
            for rk, rv in roots.items():
                if isinstance(rv, str):
                    try:
                        parsed_roots[rk] = json.loads(rv)
                    except Exception:
                        parsed_roots[rk] = rv
                else:
                    parsed_roots[rk] = rv
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

    return {
        "message_list_paths": [p for p, _ in message_list_paths.most_common(20)],
        "name_paths": [p for p, _ in name_paths.most_common(30)],
        "role_paths": [p for p, _ in role_paths.most_common(20)],
        "content_paths": [p for p, _ in content_paths.most_common(20)],
        "tool_name_paths": [p for p, _ in tool_name_paths.most_common(20)],
    }


def induce_runtime_mapping(
    outputs_dir: Path,
    nodespec_file: Path | None = None,
    node_spec: Any = None,
) -> Dict[str, Any]:
    outputs_dir = outputs_dir.resolve()
    validation_dir = outputs_dir / "validation"
    traces_dir = validation_dir / "traces"
    report_file = validation_dir / "execution_validation_report_after_refine.json"
    if node_spec is None and nodespec_file is None:
        nodespec_file = _resolve_default_nodespec(outputs_dir)
    expected = load_expected_names_from_nodespec(nodespec_file, node_spec=node_spec)
    expected_agents = expected.get("agents") or []
    expected_tools = expected.get("tools") or []
    exp_agent_norm = _expected_norm_map(expected_agents)
    exp_tool_norm = _expected_norm_map(expected_tools)

    agent_votes: Dict[str, Counter] = {}
    tool_votes: Dict[str, Counter] = {}

    # Seed from existing validation report mappings (if present).
    if report_file.exists():
        try:
            rep = json.loads(report_file.read_text(encoding="utf-8"))
        except Exception:
            rep = {}
        om = rep.get("overall_name_mappings") if isinstance(rep.get("overall_name_mappings"), dict) else {}
        for row in om.get("agents_expected_to_trace") or []:
            if not isinstance(row, dict):
                continue
            exp = str(row.get("expected_name") or "").strip()
            if not exp:
                continue
            _add_alias_vote(agent_votes, exp, exp)
            for alias in row.get("trace_names") or []:
                _add_alias_vote(agent_votes, str(alias), exp)
        for row in om.get("tools_expected_to_trace") or []:
            if not isinstance(row, dict):
                continue
            exp = str(row.get("expected_name") or "").strip()
            if not exp:
                continue
            _add_alias_vote(tool_votes, exp, exp)
            for alias in row.get("trace_names") or []:
                _add_alias_vote(tool_votes, str(alias), exp)

    # Expand from canonical run files deterministically.
    for rf in _iter_run_files(validation_dir):
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
                c = _candidate_from_expected(nm, exp_agent_norm)
                if c:
                    _add_alias_vote(agent_votes, nm, c)
            elif et == "tool" and nm:
                c = _candidate_from_expected(nm, exp_tool_norm)
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
                    c = _candidate_from_expected(tnm, exp_tool_norm)
                    if c:
                        _add_alias_vote(tool_votes, tnm, c)

    # Ingest explicit per-event name mapping decisions from validation runs.
    # These are stronger than lexical candidates because they include LLM-reviewed evidence.
    # Keep only conservative rows:
    # - decision=map_to_expected
    # - alias_scope=specific
    for payload_file, raw_file in _iter_name_mapping_pairs(validation_dir):
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

    agent_alias_map, agent_conflicts = _resolve_votes(agent_votes)
    tool_alias_map, tool_conflicts = _resolve_votes(tool_votes)
    cross_type_conflicts = _resolve_cross_type_conflicts(
        agent_alias_map=agent_alias_map,
        tool_alias_map=tool_alias_map,
    )

    pointer_profile = {}
    if traces_dir.exists():
        pointer_profile = _induce_pointer_profile(traces_dir)
    if not pointer_profile or not pointer_profile.get("message_list_paths"):
        alt_dirs = [
            outputs_dir / "flow_extraction" / "flow_query_generation" / "generated_runs" / "traces",
            outputs_dir / "traces",
        ]
        for alt in alt_dirs:
            if alt.exists():
                pointer_profile = _induce_pointer_profile(alt)
                if pointer_profile.get("message_list_paths"):
                    break

    return {
        "version": 1,
        "outputs_dir": str(outputs_dir),
        "nodespec_file": str(nodespec_file) if nodespec_file else "",
        "expected_component_source": str(expected.get("source") or ""),
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_dir", required=True, help="Use-case outputs directory containing validation/ and nodespec/spec file.")
    ap.add_argument("--out_file", default="", help="Output runtime_mapping.json path. Default: <outputs_dir>/validation/runtime_mapping.json")
    ap.add_argument("--nodespec_file", default="", help="Optional explicit nodespec/spec file path.")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir).resolve()
    nodespec_file = Path(args.nodespec_file).resolve() if str(args.nodespec_file).strip() else None
    out_file = Path(args.out_file).resolve() if str(args.out_file).strip() else (outputs_dir / "validation" / "runtime_mapping.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    mapping = induce_runtime_mapping(outputs_dir=outputs_dir, nodespec_file=nodespec_file)
    out_file.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[flow_extraction] wrote {out_file}")


if __name__ == "__main__":
    main()
