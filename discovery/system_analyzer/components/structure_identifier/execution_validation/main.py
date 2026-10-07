#!/usr/bin/env python3
"""Execution validation main pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
    from ..global_utils import load_env_file, load_nodes_by_var
    from . import trace_extraction
    from .induce_runtime_mapping import induce_runtime_mapping
    from .prompts import agent_prompt_variants, tool_prompt_variants
    from .utils import (
                command_template_from_exec_example,
        load_execution_command_example,
        apply_validation_additions,
        contains_name,
        contains_pair,
        execute_validation_attempt,
        find_tool_io_pairs_for_validated,
        inventory_context,
        is_generic_agent_name,
        load_expected_components,
        merge_tool_candidates,
        norm,
        record_attempt_key,
        tool_row_from_expected,
    )
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    CLEAN_CODE_ROOT = Path(__file__).resolve().parents[1]
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    if str(CLEAN_CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CLEAN_CODE_ROOT))
    import trace_extraction
    from global_utils import load_env_file, load_nodes_by_var
    from induce_runtime_mapping import induce_runtime_mapping
    from prompts import agent_prompt_variants, tool_prompt_variants
    from utils import (
        command_template_from_exec_example,
        load_execution_command_example,
        apply_validation_additions,
        contains_name,
        contains_pair,
        execute_validation_attempt,
        find_tool_io_pairs_for_validated,
        inventory_context,
        is_generic_agent_name,
        load_expected_components,
        merge_tool_candidates,
        norm,
        record_attempt_key,
        tool_row_from_expected,
    )


INVENTORY_QUERY = (
    "What agents and tools do you have right now? "
    "List every agent name and every tool name you can see. "
    "For each tool include its owner and a short description in plain language."
)


def main() -> None:
    """Parse CLI args and run the execution-validation pipeline inline."""
    # Parse execution-validation CLI inputs.
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="existence_pass", choices=["existence_pass"], help="Execution validation mode.")
    ap.add_argument("--env_file", default=".env", help="Optional env file.")
    ap.add_argument("--out_dir", default="stages_outputs", help="Stages outputs directory.")
    ap.add_argument("--model", default=os.environ.get("CODEX_MODEL", "gpt-5.2-codex"))
    ap.add_argument("--refresh_raw", "--refresh-raw", dest="refresh_raw", action="store_true")
    ap.add_argument("--use_case", type=int, default=0, help="Optional use case number for trace extraction defaults.")
    ap.add_argument("--entry", default="", help="Optional source entry file path used to resolve runtime execution config.")
    ap.add_argument("--execution_command_example_json", default="", help="Optional execution command example JSON.")
    ap.add_argument("--nodes_file", default="nodes_graph_corrected_preloop.py", help="Node graph file under out_dir to validate against.")
    ap.add_argument("--max_component_iterations", type=int, default=2, help="Retained for compatibility; the simplified flow uses two prompt variants per target.")
    ap.add_argument("--python_version", default="", help="Optional python version override for trace extraction sandbox.")
    ap.add_argument("--trace_timeout_sec", type=int, default=180, help="Per-query timeout in seconds for sandbox trace extraction.")
    ap.add_argument("--report_out", default="", help="Optional explicit report path.")
    ap.add_argument("--out", default="", help="Optional output nodes file for the applied execution-validation graph.")
    ap.add_argument("--apply_report_out", default="", help="Optional explicit apply-report path.")
    args = ap.parse_args()
    load_env_file(Path(args.env_file))

    # Resolve the output locations and the input NodeSpec graph for this run.
    out_dir = Path(args.out_dir).resolve()
    execution_validation_out_dir = (out_dir / "execution_validation").resolve()
    default_uco = f"use_case_{args.use_case}_outputs" if int(args.use_case or 0) > 0 else str(out_dir.parent)
    out_dir = Path(args.out_dir or default_uco).resolve()
    nodes_py = (out_dir / args.nodes_file).resolve()
    if not nodes_py.exists():
        raise FileNotFoundError(f"nodes file not found: {nodes_py}")

    # Load the expected agents/tools/servers from the existing NodeSpec graph.
    expected = load_expected_components(nodes_py)
    _var_order, nodes_by_var = load_nodes_by_var(nodes_py)
    external_servers = {
        str(node.get("name") or "").strip()
        for node in nodes_by_var.values()
        if str(node.get("name") or "").strip() and str(((node.get("node_type") or {}) if isinstance(node.get("node_type"), dict) else {}).get("type") or "") == "External_MCP_server"
    }

    # Freeze the initial expected component lists used for validation bookkeeping.
    expected_agents = sorted(expected.agents)
    expected_tools = sorted(expected.tools_by_owner, key=lambda row: (row.owner_name, row.tool_name))
    expected_tool_names_dynamic: Set[str] = {row.tool_name for row in expected.tools_by_owner}
    validated: Set[Tuple[Any, ...]] = set()
    attempts: Dict[Tuple[str, str, str], int] = {}

    # Accumulate all runtime observations, alias mappings, and per-run diagnostics.
    observed_agents_all: Set[str] = set()
    observed_actor_names_all: Set[str] = set()
    observed_tools_by_owner_all: Set[Tuple[str, str]] = set()
    observed_tool_names_all: Set[str] = set()
    observed_tool_io_all: List[Dict[str, Any]] = []
    observed_inventory_tools_all: List[Dict[str, Any]] = []
    global_agent_alias_by_norm: Dict[str, str] = {}
    global_tool_alias_by_norm: Dict[str, str] = {}
    discovered_validated_agents: Set[str] = set()
    discovered_validated_tool_pairs_norm: Set[Tuple[str, str]] = set()
    rounds: List[Dict[str, Any]] = []
    failed_attempts: List[Dict[str, Any]] = []

    # Create a run directory for prompts, traces, parsed outputs, and reports.
    run_label = f"use_case_{args.use_case}" if int(args.use_case or 0) > 0 else "custom"
    run_dir = (execution_validation_out_dir / "execution_validation_runs" / run_label).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_counter = 0

    # Prepare trace-extraction configuration that is shared across all validation attempts.
    trace_extraction.DEFAULT_ENVFILE_PATH = args.env_file
    sandbox_image = str(os.environ.get("STRUCTURE_TARGET_SANDBOX_IMAGE") or "").strip()
    target_root = str(os.environ.get("STRUCTURE_TARGET_CODEBASE_ROOT") or "").strip()
    if target_root:
        trace_extraction.DEFAULT_RUNTIME_ROOT = Path(target_root).resolve()
    elif args.entry:
        trace_extraction.DEFAULT_RUNTIME_ROOT = Path(args.entry).resolve().parent
    trace_extraction.DEFAULT_SANDBOX_IMAGE = sandbox_image or None
    exec_example = load_execution_command_example(args)
    if args.entry:
        entry_path = Path(args.entry).resolve()
        source_root = Path(target_root).resolve() if target_root else entry_path.parent
        try:
            relative_entry = entry_path.relative_to(source_root)
        except ValueError:
            relative_entry = Path(entry_path.name)
        # In the target image, the codebase is mounted at /sandbox. Resolve the
        # entrypoint against that root instead of the copied example JSON file.
        if not isinstance(exec_example, dict):
            exec_example = {"runner": "python", "args": []}
        exec_example["entrypoint"] = relative_entry.as_posix()
        exec_example.pop("__source_path__", None)
    command_template = command_template_from_exec_example(exec_example)

    # First ask the runtime for its own inventory before validating specific targets.
    run_counter, inventory_result = execute_validation_attempt(
        run_counter=run_counter,
        out_dir=execution_validation_out_dir,
        run_dir=run_dir,
        args=args,
        phase="inventory",
        command_template=command_template,
        trace_timeout_sec=int(args.trace_timeout_sec or 180),
        prompt_variant=1,
        query_text=INVENTORY_QUERY,
        target_component={"component_kind": "inventory", "name": "global_inventory"},
        parse_inventory=True,
        allow_unknown_inventory_owners=True,
        expected_agent_names=expected_agents + sorted(discovered_validated_agents),
        expected_tool_names=sorted(expected_tool_names_dynamic),
        expected_tools_by_owner_all=expected.tools_by_owner,
        expected_server_to_ancestor_agents=expected.server_to_ancestor_agents,
        expected_server_to_ancestor_servers=expected.server_to_ancestor_servers,
        validated=validated,
        global_agent_alias_by_norm=global_agent_alias_by_norm,
        global_tool_alias_by_norm=global_tool_alias_by_norm,
        observed_agents_all=observed_agents_all,
        observed_actor_names_all=observed_actor_names_all,
        observed_tools_by_owner_all=observed_tools_by_owner_all,
        observed_tool_names_all=observed_tool_names_all,
        observed_tool_io_all=observed_tool_io_all,
        observed_inventory_tools_all=observed_inventory_tools_all,
        rounds=rounds,
        failed_attempts=failed_attempts,
    )
    # Seed dynamic tool-name coverage from whatever the runtime reported in its inventory reply.
    inventory_tools = inventory_result.get("inventory_tools") if isinstance(inventory_result, dict) and isinstance(inventory_result.get("inventory_tools"), list) else []
    for row in inventory_tools:
        if isinstance(row, dict):
            tname = str(row.get("tool") or "").strip()
            if tname:
                expected_tool_names_dynamic.add(tname)

    # Build lightweight prompt context from the inventory reply and collect extra agent candidates.
    inventory_context_text = inventory_context(inventory_tools)
    inventory_agent_candidates = {
        str(row.get("owner") or "").strip()
        for row in inventory_tools
        if isinstance(row, dict)
        and str(row.get("owner") or "").strip()
        and str(row.get("owner") or "").strip() not in expected.servers
        and not is_generic_agent_name(str(row.get("owner") or "").strip())
    }
    inventory_agent_candidates.update(
        a for a in observed_actor_names_all if a and not is_generic_agent_name(a) and a not in expected.servers
    )
    candidate_agents = sorted(set(expected_agents) | inventory_agent_candidates)

    # Merge NodeSpec tools with inventory-discovered tools to form the validation candidate list.
    candidate_tools = merge_tool_candidates(
        [tool_row_from_expected(row) for row in expected.tools_by_owner],
        [dict(row, from_nodespec=False) for row in inventory_tools if isinstance(row, dict)],
    )

    # Validate every candidate agent with a direct prompt and then a more flexible prompt.
    for agent_name in candidate_agents:
        is_expected = agent_name in expected.agents
        key = record_attempt_key("agent", "", agent_name)
        attempts.setdefault(key, 0)
        for prompt_variant, prompt_text in enumerate(agent_prompt_variants(agent_name, inventory_context_text), start=1):
            # Stop early once this expected or discovered agent has already been validated.
            if is_expected and ("agent", agent_name) in validated:
                break
            if (not is_expected) and agent_name in discovered_validated_agents:
                break
            attempts[key] += 1
            # Execute one validation attempt, parse its trace, and fold the observations into global state.
            run_counter, result = execute_validation_attempt(
                run_counter=run_counter,
                out_dir=execution_validation_out_dir,
                run_dir=run_dir,
                args=args,
                phase="agent_validation",
                command_template=command_template,
                trace_timeout_sec=int(args.trace_timeout_sec or 180),
                prompt_variant=prompt_variant,
                query_text=prompt_text,
                target_component={"component_kind": "agent", "name": agent_name, "from_nodespec": is_expected},
                parse_inventory=False,
                allow_unknown_inventory_owners=False,
                expected_agent_names=expected_agents + sorted(discovered_validated_agents),
                expected_tool_names=sorted(expected_tool_names_dynamic),
                expected_tools_by_owner_all=expected.tools_by_owner,
                expected_server_to_ancestor_agents=expected.server_to_ancestor_agents,
                expected_server_to_ancestor_servers=expected.server_to_ancestor_servers,
                validated=validated,
                global_agent_alias_by_norm=global_agent_alias_by_norm,
                global_tool_alias_by_norm=global_tool_alias_by_norm,
                observed_agents_all=observed_agents_all,
                observed_actor_names_all=observed_actor_names_all,
                observed_tools_by_owner_all=observed_tools_by_owner_all,
                observed_tool_names_all=observed_tool_names_all,
                observed_tool_io_all=observed_tool_io_all,
                observed_inventory_tools_all=observed_inventory_tools_all,
                rounds=rounds,
                failed_attempts=failed_attempts,
            )
            if not isinstance(result, dict):
                continue
            observed = result["observed"]
            if not is_expected and (contains_name(agent_name, observed.agents) or contains_name(agent_name, observed.actor_names)):
                discovered_validated_agents.add(agent_name)

    # Validate every owner/tool candidate with the same two-prompt strategy.
    for row in candidate_tools:
        owner = str(row.get("owner") or "").strip()
        tool = str(row.get("tool") or "").strip()
        if not owner or not tool:
            continue
        expected_tool_names_dynamic.add(tool)
        is_expected = bool(row.get("from_nodespec"))
        key = record_attempt_key("tool", owner, tool)
        attempts.setdefault(key, 0)
        for prompt_variant, prompt_text in enumerate(tool_prompt_variants(owner, tool, str(row.get("description") or "").strip(), inventory_context_text), start=1):
            # Stop early once this expected or discovered tool-owner pair has already been validated.
            if is_expected and ("tool", owner, tool) in validated:
                break
            if (not is_expected) and (norm(owner), norm(tool)) in discovered_validated_tool_pairs_norm:
                break
            attempts[key] += 1
            run_counter, result = execute_validation_attempt(
                run_counter=run_counter,
                out_dir=execution_validation_out_dir,
                run_dir=run_dir,
                args=args,
                phase="tool_validation",
                command_template=command_template,
                trace_timeout_sec=int(args.trace_timeout_sec or 180),
                prompt_variant=prompt_variant,
                query_text=prompt_text,
                target_component={
                    "component_kind": "tool",
                    "owner": owner,
                    "owner_kind": str(row.get("owner_kind") or "agent_or_server"),
                    "name": tool,
                    "from_nodespec": is_expected,
                },
                parse_inventory=False,
                allow_unknown_inventory_owners=False,
                expected_agent_names=expected_agents + sorted(discovered_validated_agents),
                expected_tool_names=sorted(expected_tool_names_dynamic),
                expected_tools_by_owner_all=expected.tools_by_owner,
                expected_server_to_ancestor_agents=expected.server_to_ancestor_agents,
                expected_server_to_ancestor_servers=expected.server_to_ancestor_servers,
                validated=validated,
                global_agent_alias_by_norm=global_agent_alias_by_norm,
                global_tool_alias_by_norm=global_tool_alias_by_norm,
                observed_agents_all=observed_agents_all,
                observed_actor_names_all=observed_actor_names_all,
                observed_tools_by_owner_all=observed_tools_by_owner_all,
                observed_tool_names_all=observed_tool_names_all,
                observed_tool_io_all=observed_tool_io_all,
                observed_inventory_tools_all=observed_inventory_tools_all,
                rounds=rounds,
                failed_attempts=failed_attempts,
            )
            if not isinstance(result, dict):
                continue
            observed = result["observed"]
            if not is_expected and contains_pair(owner, tool, observed.tools_by_owner):
                discovered_validated_tool_pairs_norm.add((norm(owner), norm(tool)))

    # Split expected components into identified vs not-identified sets for the final report.
    identified_agents = sorted([name for name in expected_agents if ("agent", name) in validated])
    not_identified_agents = sorted([name for name in expected_agents if ("agent", name) not in validated])
    identified_tools = sorted(
        [
            {"owner": row.owner_name, "owner_kind": row.owner_kind, "tool": row.tool_name}
            for row in expected.tools_by_owner
            if ("tool", row.owner_name, row.tool_name) in validated
        ],
        key=lambda row: (row["owner"], row["tool"]),
    )
    not_identified_tools = sorted(
        [
            {"owner": row.owner_name, "owner_kind": row.owner_kind, "tool": row.tool_name}
            for row in expected.tools_by_owner
            if ("tool", row.owner_name, row.tool_name) not in validated
        ],
        key=lambda row: (row["owner"], row["tool"]),
    )

    # Index inventory metadata and filter discovered validated components that are not already in the NodeSpec.
    expected_agent_norms = {norm(name) for name in expected_agents}
    expected_tool_pair_norms = {(norm(row.owner_name), norm(row.tool_name)) for row in expected.tools_by_owner}
    inventory_meta_by_pair: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in inventory_tools:
        if not isinstance(row, dict):
            continue
        owner = str(row.get("owner") or "").strip()
        tool = str(row.get("tool") or "").strip()
        if owner and tool:
            inventory_meta_by_pair[(norm(owner), norm(tool))] = row

    discovered_agents_for_add = sorted(
        {
            name
            for name in discovered_validated_agents
            if norm(name) not in expected_agent_norms and name not in expected.servers
        }
    )
    discovered_tools_for_add = sorted(
        [
            row
            for row in candidate_tools
            if (norm(str(row.get("owner") or "")), norm(str(row.get("tool") or ""))) in discovered_validated_tool_pairs_norm
            and (norm(str(row.get("owner") or "")), norm(str(row.get("tool") or ""))) not in expected_tool_pair_norms
        ],
        key=lambda row: (str(row.get("owner") or ""), str(row.get("tool") or "")),
    )

    # Convert newly validated components into graph-addition candidates.
    external_mcp_tool_nodes = []
    add_candidates = [
        {
            "component_type": "agent",
            "name": name,
            "owner": "",
            "description": "Observed during execution validation and validated by runtime trace.",
            "reason": "validated_new_agent",
        }
        for name in discovered_agents_for_add
    ]
    for row in discovered_tools_for_add:
        owner = str(row.get("owner") or "").strip()
        tool = str(row.get("tool") or "").strip()
        description = str(row.get("description") or inventory_meta_by_pair.get((norm(owner), norm(tool)), {}).get("description") or "").strip()
        if owner in external_servers:
            external_mcp_tool_nodes.append(
                {
                    "node_type": "Tool",
                    "owner_server": owner,
                    "name": tool,
                    "description": description or "Observed external MCP tool during execution validation.",
                    "inputs": inventory_meta_by_pair.get((norm(owner), norm(tool)), {}).get("inputs") if isinstance(inventory_meta_by_pair.get((norm(owner), norm(tool)), {}).get("inputs"), list) else [],
                    "outputs": inventory_meta_by_pair.get((norm(owner), norm(tool)), {}).get("outputs") if isinstance(inventory_meta_by_pair.get((norm(owner), norm(tool)), {}).get("outputs"), list) else [],
                }
            )
            continue
        add_candidates.append(
            {
                "component_type": "tool",
                "name": tool,
                "owner": owner,
                "description": description or "Observed during execution validation and validated by runtime trace.",
                "reason": "validated_new_tool",
            }
        )

    # Gather validated tool IO examples and summarize per-component attempt counts.
    validated_tool_pair_keys = {(row["owner"], row["tool"]) for row in identified_tools}
    validated_tool_pair_keys.update(
        {(str(row.get("owner") or "").strip(), str(row.get("tool") or "").strip()) for row in discovered_tools_for_add}
    )
    validated_tool_io_pairs = find_tool_io_pairs_for_validated(observed_tool_io_all, validated_tool_pair_keys)

    component_attempts = [
        {"component": [kind, owner, name], "attempts": count}
        for (kind, owner, name), count in sorted(attempts.items(), key=lambda item: item[0])
    ]

    # Apply the discovered additions to a new NodeSpec output file and capture the apply report.
    applied_nodes_out = Path(args.out).resolve() if args.out else (out_dir / "post_execution_validation_actions.py").resolve()
    apply_report_out = Path(args.apply_report_out).resolve() if args.apply_report_out else (run_dir / "execution_validation_apply_report.json").resolve()
    apply_report = apply_validation_additions(
        nodes_py=nodes_py,
        out_path=applied_nodes_out,
        add_candidates=add_candidates,
        external_mcp_tool_nodes=external_mcp_tool_nodes,
        report_out=apply_report_out,
    )

    # Assemble the final execution-validation report consumed by later stages and manual review.
    report = {
        "mode": "execution_validation_simple",
        "use_case": int(args.use_case or 0),
        "nodes_file": str(nodes_py),
        "applied_nodes_file": str(applied_nodes_out),
        "apply_report_file": str(apply_report_out),
        "inventory_query": INVENTORY_QUERY,
        "runs_executed": run_counter,
        "failed_attempts_count": len(failed_attempts),
        "failed_attempts": failed_attempts,
        "all_components_identified": {
            "agents": identified_agents,
            "tools": identified_tools,
        },
        "all_components_not_identified": {
            "agents": not_identified_agents,
            "tools": not_identified_tools,
        },
        "all_components_identified_not_in_nodespec": {
            "agents": discovered_agents_for_add,
            "tools_by_owner": [
                {"owner": str(row.get("owner") or "").strip(), "tool": str(row.get("tool") or "").strip()}
                for row in discovered_tools_for_add
            ],
            "tool_names_any": sorted({str(row.get("tool") or "").strip() for row in discovered_tools_for_add if str(row.get("tool") or "").strip()}),
        },
        "actions": {
            "rename": [],
            "rename_skipped_duplicate_guard": [],
            "add_remove_llm": {"remove": [], "add": add_candidates},
            "external_mcp_tool_nodes_to_add": external_mcp_tool_nodes,
        },
        "apply_report": apply_report,
        "validated_tool_input_output_pairs": validated_tool_io_pairs,
        "component_attempts": component_attempts,
        "rounds": rounds,
    }

    # Write the final report and print a short terminal summary.
    out_report = Path(args.report_out).resolve() if args.report_out else run_dir / "existence_validation_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    identified = report.get("all_components_identified") if isinstance(report, dict) else {}
    print(
        "[execution_validation] identified "
        f"agents={len((identified or {}).get('agents') or [])} "
        f"tools={len((identified or {}).get('tools') or [])}"
    )

    # Induce the deterministic runtime alias registry from the run artifacts just produced and persist it alongside the other execution-validation artifacts.
    runtime_mapping = induce_runtime_mapping(out_dir=out_dir, nodes_py=nodes_py)
    runtime_mapping_out = (out_dir / "validation" / "runtime_mapping.json").resolve()
    runtime_mapping_out.parent.mkdir(parents=True, exist_ok=True)
    runtime_mapping_out.write_text(json.dumps(runtime_mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[execution_validation] wrote {runtime_mapping_out}")


if __name__ == "__main__":
    main()
