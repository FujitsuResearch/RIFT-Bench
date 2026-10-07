from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from .global_utils import load_nodes_by_var
except ImportError:
    from global_utils import load_nodes_by_var


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


def _safe_name(v: Any, fallback: str = "unknown") -> str:
    s = str(v or "").strip()
    return s if s else fallback


def _extract_flows_from_nodespec_py(nodespec_file: Path) -> List[Dict[str, Any]]:
    # Use the active top-level node graph instead of scanning all NodeSpec()
    # literals in file text, which can include non-active/legacy objects.
    var_order, nodes_by_var = load_nodes_by_var(nodespec_file)
    if not var_order:
        return []
    top_var = var_order[-1]
    top_node = nodes_by_var.get(top_var) if isinstance(nodes_by_var, dict) else None
    if not isinstance(top_node, dict):
        return []
    raw_flows = top_node.get("flows")
    if not isinstance(raw_flows, list):
        return []
    flows: List[Dict[str, Any]] = []
    for item in raw_flows:
        if isinstance(item, dict):
            flows.append(item)
    return flows


def _signature_tokens(events: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        et = str(e.get("type") or "").strip().lower()
        if et == "user":
            out.append("user:user")
            continue
        if et == "system":
            out.append("system:system")
            continue
        if et == "error":
            out.append("error:error")
            continue
        if et == "tool":
            out.append(f"tool:{_safe_name(e.get('name'))}")
            continue
        if et == "agent":
            tc = e.get("tool_calls")
            if isinstance(tc, list) and tc:
                for c in tc:
                    if isinstance(c, dict):
                        nm = _safe_name(c.get("name") or c.get("tool_id"))
                        out.append(f"agent:agent.tool_call:{nm}")
                    else:
                        out.append("agent:agent")
            else:
                out.append("agent:agent")
    return out


@dataclass
class FlowAgg:
    example_count: int = 0
    event_count: int = 0
    error_events: int = 0
    tool_call_count: int = 0
    agent_invocation_count: int = 0
    tool_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    agent_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    signature_counter: Counter = field(default_factory=Counter)


def _collect_metrics(flows: List[Dict[str, Any]]) -> Dict[int, FlowAgg]:
    by_flow: Dict[int, FlowAgg] = {}

    for f in flows:
        try:
            flow_id = int(f.get("flow_id"))
        except Exception:
            continue

        events = f.get("events")
        if not isinstance(events, list):
            events = []

        agg = by_flow.setdefault(flow_id, FlowAgg())
        agg.example_count += 1
        agg.event_count += len(events)

        sig = tuple(_signature_tokens(events))
        agg.signature_counter[sig] += 1

        for e in events:
            if not isinstance(e, dict):
                continue
            et = str(e.get("type") or "").strip().lower()
            if et == "error":
                agg.error_events += 1
                continue
            if et == "agent":
                aname = _safe_name(e.get("name"), "agent")
                agg.agent_invocation_count += 1
                agg.agent_counts[aname] += 1
                tc = e.get("tool_calls")
                if isinstance(tc, list):
                    for c in tc:
                        if not isinstance(c, dict):
                            continue
                        tname = _safe_name(c.get("name") or c.get("tool_id"))
                        agg.tool_call_count += 1
                        agg.tool_counts[tname] += 1
                continue
            if et == "tool":
                # Keep tool visibility even if a malformed trace missed agent.tool_calls.
                tname = _safe_name(e.get("name"))
                agg.tool_counts.setdefault(tname, agg.tool_counts.get(tname, 0))

    return by_flow


def _inverse_counts(by_flow: Dict[int, FlowAgg], mode: str) -> List[Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for fid, agg in by_flow.items():
        src = agg.tool_counts if mode == "tool" else agg.agent_counts
        for name, cnt in src.items():
            if cnt <= 0:
                continue
            entry = stats.setdefault(
                name,
                {"name": name, "flows_invoked": 0, "total_invocations": 0, "max_invocations_in_flow": 0},
            )
            entry["flows_invoked"] += 1
            entry["total_invocations"] += int(cnt)
            entry["max_invocations_in_flow"] = max(int(entry["max_invocations_in_flow"]), int(cnt))

    rows: List[Dict[str, Any]] = []
    for _, v in stats.items():
        flows_invoked = int(v["flows_invoked"])
        total_invocations = int(v["total_invocations"])
        avg_invocations = (total_invocations / flows_invoked) if flows_invoked else 0.0
        rows.append(
            {
                f"{mode}_name": v["name"],
                "flows_invoked": flows_invoked,
                "total_invocations": total_invocations,
                "avg_invocations_per_flow": round(avg_invocations, 4),
                "max_invocations_in_flow": int(v["max_invocations_in_flow"]),
            }
        )
    rows.sort(key=lambda r: (-int(r["flows_invoked"]), -int(r["total_invocations"]), str(r[f"{mode}_name"])))
    return rows


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _build_summary(by_flow: Dict[int, FlowAgg], tools_rows: List[Dict[str, Any]], agents_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not by_flow:
        return {
            "total_flows": 0,
            "total_examples": 0,
            "note": "No flow data found.",
        }

    total_flows = len(by_flow)
    total_examples = sum(v.example_count for v in by_flow.values())
    flows_with_error = sum(1 for v in by_flow.values() if v.error_events > 0)

    avg_examples_per_flow = total_examples / total_flows
    avg_unique_tools_per_flow = sum(len(v.tool_counts) for v in by_flow.values()) / total_flows
    avg_tool_calls_per_flow = sum(v.tool_call_count for v in by_flow.values()) / total_flows
    avg_unique_agents_per_flow = sum(len(v.agent_counts) for v in by_flow.values()) / total_flows
    avg_agent_invocations_per_flow = sum(v.agent_invocation_count for v in by_flow.values()) / total_flows

    return {
        "total_flows": total_flows,
        "total_examples": total_examples,
        "avg_examples_per_flow": round(avg_examples_per_flow, 4),
        "flows_with_error": flows_with_error,
        "pct_flows_with_error": round((flows_with_error / total_flows) * 100.0, 2),
        "avg_unique_tools_per_flow": round(avg_unique_tools_per_flow, 4),
        "avg_tool_calls_per_flow": round(avg_tool_calls_per_flow, 4),
        "avg_unique_agents_per_flow": round(avg_unique_agents_per_flow, 4),
        "avg_agent_invocations_per_flow": round(avg_agent_invocations_per_flow, 4),
        "top_tools_by_flow_coverage": tools_rows[:10],
        "top_agents_by_flow_coverage": agents_rows[:10],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze flow metrics per use-case outputs directory.")
    ap.add_argument("--outputs_dir", required=True, help="Use-case outputs directory (e.g., modular_imp/use_case_1_outputs_0)")
    ap.add_argument("--nodespec_file", default="", help="Optional nodespec/spec file path. Default: auto-resolve (prefer spec.py).")
    ap.add_argument(
        "--out_dir",
        default="",
        help="Output directory for CSV/JSON reports (default: <outputs_dir>/flow_extraction/flow_query_generation).",
    )
    ap.add_argument("--prefix", default="flow_metrics", help="Output filename prefix.")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir).resolve()
    if str(args.nodespec_file).strip():
        raw = Path(str(args.nodespec_file))
        nodespec_file = raw.resolve() if raw.is_absolute() else (outputs_dir / raw).resolve()
    else:
        nodespec_file = _resolve_default_nodespec(outputs_dir)
    if not nodespec_file.exists():
        raise FileNotFoundError(f"Missing nodespec file: {nodespec_file}")

    out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else (outputs_dir / "flow_extraction" / "flow_query_generation")
    out_dir.mkdir(parents=True, exist_ok=True)

    flows = _extract_flows_from_nodespec_py(nodespec_file)
    by_flow = _collect_metrics(flows)

    per_flow_rows: List[Dict[str, Any]] = []
    for fid in sorted(by_flow.keys()):
        agg = by_flow[fid]
        rep_sig = list(agg.signature_counter.most_common(1)[0][0]) if agg.signature_counter else []
        per_flow_rows.append(
            {
                "flow_id": fid,
                "example_count": agg.example_count,
                "event_count_total": agg.event_count,
                "error_events_total": agg.error_events,
                "has_error": 1 if agg.error_events > 0 else 0,
                "unique_tools": len(agg.tool_counts),
                "total_tool_calls": agg.tool_call_count,
                "unique_agents": len(agg.agent_counts),
                "total_agent_invocations": agg.agent_invocation_count,
                "flow_signature": " -> ".join(rep_sig),
            }
        )

    tools_rows = _inverse_counts(by_flow, mode="tool")
    agents_rows = _inverse_counts(by_flow, mode="agent")
    summary = _build_summary(by_flow, tools_rows, agents_rows)

    prefix = str(args.prefix).strip() or "flow_metrics"
    per_flow_csv = out_dir / f"{prefix}_per_flow.csv"
    by_tool_csv = out_dir / f"{prefix}_by_tool.csv"
    by_agent_csv = out_dir / f"{prefix}_by_agent.csv"
    summary_json = out_dir / f"{prefix}_summary.json"

    _write_csv(
        per_flow_csv,
        [
            "flow_id",
            "example_count",
            "event_count_total",
            "error_events_total",
            "has_error",
            "unique_tools",
            "total_tool_calls",
            "unique_agents",
            "total_agent_invocations",
            "flow_signature",
        ],
        per_flow_rows,
    )
    _write_csv(
        by_tool_csv,
        ["tool_name", "flows_invoked", "total_invocations", "avg_invocations_per_flow", "max_invocations_in_flow"],
        tools_rows,
    )
    _write_csv(
        by_agent_csv,
        ["agent_name", "flows_invoked", "total_invocations", "avg_invocations_per_flow", "max_invocations_in_flow"],
        agents_rows,
    )
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[flow-metrics] nodespec={nodespec_file}")
    print(f"[flow-metrics] per_flow={per_flow_csv}")
    print(f"[flow-metrics] by_tool={by_tool_csv}")
    print(f"[flow-metrics] by_agent={by_agent_csv}")
    print(f"[flow-metrics] summary={summary_json}")
    print(f"[flow-metrics] flows={len(per_flow_rows)}")


if __name__ == "__main__":
    main()
