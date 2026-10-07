#!/usr/bin/env python3
"""Live MCP tool-discovery stage: connect to real External_MCP_server constructors and replace tool_list."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    from ..execution_validation.utils import (
        command_template_from_exec_example,
        load_execution_command_example,
    )
    from ..global_utils import (
        call_with_cache_text,
        load_env_file,
        load_nodes_by_var,
        node_type_name,
        write_nodes_output,
    )
    from .prompts import build_mcp_connection_resolver_prompt
    from .utils import (
        build_probe_script,
        extract_resolver_body,
        find_constructor_context,
        input_ports_from_schema,
        load_project_root,
        output_ports_from_schema,
        parse_probe_output,
        replace_tool_list,
        run_probe_script_in_target_env,
    )
except ImportError:
    CURRENT_DIR = Path(__file__).resolve().parent
    CLEAN_CODE_ROOT = Path(__file__).resolve().parents[1]
    if str(CURRENT_DIR) not in sys.path:
        sys.path.insert(0, str(CURRENT_DIR))
    if str(CLEAN_CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CLEAN_CODE_ROOT))
    from execution_validation.utils import (
        command_template_from_exec_example,
        load_execution_command_example,
    )
    from global_utils import (
        call_with_cache_text,
        load_env_file,
        load_nodes_by_var,
        node_type_name,
        write_nodes_output,
    )
    from prompts import build_mcp_connection_resolver_prompt
    from utils import (
        build_probe_script,
        extract_resolver_body,
        find_constructor_context,
        input_ports_from_schema,
        load_project_root,
        output_ports_from_schema,
        parse_probe_output,
        replace_tool_list,
        run_probe_script_in_target_env,
    )


MAX_FILE_CHARS = 20000
STDERR_TAIL_CHARS = 2000


class MCPToolDiscovery:
    """Connect to each External_MCP_server node's live server and replace its tool_list with real tools."""

    def __init__(
        self,
        *,
        out_dir: str,
        nodes_py: str,
        out: str,
        report_out: str,
        raw_dir: str,
        entry: str,
        execution_command_example_json: str,
        model: str,
        env_file: str,
        max_probe_attempts: int = 3,
        probe_timeout_sec: int = 45,
        refresh_raw: bool = False,
        verbos: bool = False,
    ) -> None:
        """Resolve stage paths under out_dir and store the CLI-derived run configuration."""
        load_env_file(Path(env_file))
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.nodes_py = self.out_dir / Path(nodes_py).name
        self.out_py = self.out_dir / Path(out).name
        self.report_path = self.out_dir / Path(report_out).name
        self.raw_dir = self.out_dir / Path(raw_dir).name
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.entry = entry
        self.execution_command_example_json = execution_command_example_json
        self.model = model
        self.max_probe_attempts = max(1, int(max_probe_attempts))
        self.probe_timeout_sec = int(probe_timeout_sec)
        self.refresh_raw = bool(refresh_raw)
        self.verbos = verbos

    # ------------------------------------------------------------------
    # Target-repo execution context
    # ------------------------------------------------------------------

    def _resolve_target_context(self) -> Dict[str, Any]:
        """Resolve the target project root and the command template used to pick its interpreter."""
        project_root = load_project_root(self.out_dir)
        if not project_root:
            project_root = str(Path(self.entry).resolve().parent) if self.entry else str(Path.cwd())

        args_ns = argparse.Namespace(
            entry=self.entry,
            execution_command_example_json=self.execution_command_example_json,
        )
        exec_example = load_execution_command_example(args_ns)
        command_template = command_template_from_exec_example(exec_example)
        return {
            "project_root": Path(project_root),
            "command_template": command_template,
            "env_file": Path(project_root) / ".env",
        }

    # ------------------------------------------------------------------
    # Per-node probing
    # ------------------------------------------------------------------

    def _read_file_text(self, path: str) -> str:
        """Read a source file as text, truncating it if it exceeds MAX_FILE_CHARS."""
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
        if len(text) > MAX_FILE_CHARS:
            return text[:MAX_FILE_CHARS] + "\n# ... [truncated] ...\n"
        return text

    def _gather_file_context(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Read every distinct source file referenced by the constructor/assignment/definition refs."""
        files: Dict[str, str] = {}
        for key in ("constructor", "assignment", "definition"):
            ref = context.get(key)
            if not isinstance(ref, dict):
                continue
            file_path = str(ref.get("file") or "").strip()
            if file_path and file_path not in files:
                files[file_path] = self._read_file_text(file_path)
        return files

    def _probe_one_server(
        self,
        *,
        var: str,
        node: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the generate -> execute -> parse -> repair loop for one External_MCP_server node."""
        # Bail out immediately (no LLM call) if this node has no grounded constructor to probe.
        context = find_constructor_context(node)
        constructor_ref = context.get("constructor")
        if not isinstance(constructor_ref, dict):
            return {"node_var": var, "status": "skipped_no_constructor_ref"}

        # Gather everything the resolver-authoring prompt needs: the constructor/assignment
        # snippets plus the full text of every source file they reference.
        assignment_ref = context.get("assignment") or context.get("definition") or constructor_ref
        file_contents = self._gather_file_context(context)
        metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        component_handle = str(metadata.get("component_handle") or "").strip()

        node_raw_dir = self.raw_dir / var
        node_raw_dir.mkdir(parents=True, exist_ok=True)

        attempt_log: List[Dict[str, Any]] = []
        for attempt in range(1, self.max_probe_attempts + 1):
            # Ask the model for the `_resolve_connection()` body only; prior failures (if any)
            # are fed back in so each retry can target the specific error that occurred.
            payload = build_mcp_connection_resolver_prompt(
                node_name=str(node.get("name") or var),
                node_description=str(node.get("description") or ""),
                component_handle=component_handle,
                constructor_snippet=str(constructor_ref.get("snippet") or ""),
                constructor_file=str(constructor_ref.get("file") or ""),
                assignment_snippet=str(assignment_ref.get("snippet") or ""),
                assignment_file=str(assignment_ref.get("file") or ""),
                file_contents=file_contents,
                project_root=str(target["project_root"]),
                prior_attempts=attempt_log or None,
            )
            raw_text, _source = call_with_cache_text(
                payload,
                payload_path=node_raw_dir / f"resolver_prompt_attempt_{attempt:02d}.payload.json",
                raw_path=node_raw_dir / f"resolver_body_attempt_{attempt:02d}.txt",
                model=self.model,
                refresh_raw=self.refresh_raw,
            )

            # Splice the resolver body into the fixed probe template and run it inside the
            # target repo's own environment.
            resolver_body = extract_resolver_body(raw_text)
            script_text = build_probe_script(resolver_body)
            script_path = node_raw_dir / f"probe_attempt_{attempt:02d}.py"

            returncode, stdout, stderr = run_probe_script_in_target_env(
                script_text,
                project_root=target["project_root"],
                command_template=target["command_template"],
                env_file=target["env_file"],
                script_path=script_path,
                timeout_sec=self.probe_timeout_sec,
            )
            (node_raw_dir / f"probe_result_attempt_{attempt:02d}.json").write_text(
                json.dumps(
                    {"returncode": returncode, "stdout": stdout, "stderr": stderr},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            # A successful probe returns immediately; otherwise record the failure and retry.
            result = parse_probe_output(stdout)
            if str(result.get("status") or "") == "ok" and isinstance(result.get("tools"), list):
                tools_found = result["tools"]
                return {
                    "node_var": var,
                    "status": "ok",
                    "attempts": attempt,
                    "tools": tools_found,
                }

            attempt_log.append(
                {
                    "error_type": result.get("error_type") or "UnknownError",
                    "error": result.get("error") or "",
                    "stderr_tail": (stderr or "")[-STDERR_TAIL_CHARS:],
                }
            )
            if self.verbos:
                print(f"[mcp_tool_discovery] {var} attempt {attempt} failed: {attempt_log[-1]}")

        # Every attempt failed: leave the node untouched and report the last error.
        return {
            "node_var": var,
            "status": "error",
            "attempts": self.max_probe_attempts,
            "final_error": attempt_log[-1] if attempt_log else "unknown_error",
            "attempt_log": attempt_log,
        }

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Load the graph, probe every External_MCP_server node, and write the updated graph + report."""
        var_order, nodes_by_var = load_nodes_by_var(self.nodes_py)
        if not var_order:
            raise RuntimeError("mcp_tool_discovery: input nodes file has no ALL_NODES/MAIN_GRAPH_NODES.")

        server_vars = [v for v in var_order if node_type_name(nodes_by_var.get(v, {})) == "External_MCP_server"]

        # Cheap no-op path: nothing to probe, so skip resolving the target env / any LLM calls.
        per_server: List[Dict[str, Any]] = []
        if not server_vars:
            report = {
                "servers_total": 0,
                "servers_ok": 0,
                "servers_error": 0,
                "servers_skipped_no_constructor_ref": 0,
                "per_server": [],
            }
            write_nodes_output(self.out_py, var_order, nodes_by_var, report, "MCP_TOOL_DISCOVERY_REPORT")
            self.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print("[mcp_tool_discovery] no External_MCP_server nodes found, nothing to probe")
            return

        target = self._resolve_target_context()

        # Probe each server node in turn; on success, map its live tools into fresh Tool
        # NodeSpec dicts and replace the server's tool_list wholesale.
        for var in server_vars:
            node = nodes_by_var[var]
            outcome = self._probe_one_server(var=var, node=node, target=target)
            if outcome.get("status") == "ok":
                tools = outcome.pop("tools")
                new_tool_nodes: List[Dict[str, Any]] = []
                output_schema_present_any = False
                for tool in tools:
                    input_ports = input_ports_from_schema(tool.get("inputSchema"))
                    output_ports, output_schema_present = output_ports_from_schema(tool.get("outputSchema"))
                    output_schema_present_any = output_schema_present_any or output_schema_present
                    new_tool_nodes.append(
                        {
                            "name": str(tool.get("name") or "").strip() or "tool",
                            "node_type": {"type": "Tool"},
                            "description": str(tool.get("description") or ""),
                            "inputs": input_ports,
                            "outputs": output_ports,
                            "metadata": {"source": "mcp_tool_discovery", "mcp_server_var": var},
                        }
                    )
                replace_tool_list(
                    server_var=var,
                    var_order=var_order,
                    nodes_by_var=nodes_by_var,
                    new_tools=new_tool_nodes,
                )
                outcome["tools_found"] = len(new_tool_nodes)
                outcome["output_schema_present"] = output_schema_present_any
            per_server.append(outcome)
            print(f"[mcp_tool_discovery] {var}: {outcome.get('status')}")

        # Tally per-server outcomes into the final report and write both output artifacts.
        report = {
            "servers_total": len(server_vars),
            "servers_ok": sum(1 for r in per_server if r.get("status") == "ok"),
            "servers_error": sum(1 for r in per_server if r.get("status") == "error"),
            "servers_skipped_no_constructor_ref": sum(
                1 for r in per_server if r.get("status") == "skipped_no_constructor_ref"
            ),
            "per_server": per_server,
        }
        write_nodes_output(self.out_py, var_order, nodes_by_var, report, "MCP_TOOL_DISCOVERY_REPORT")
        self.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            "[mcp_tool_discovery] "
            f"ok={report['servers_ok']} error={report['servers_error']} "
            f"skipped={report['servers_skipped_no_constructor_ref']}"
        )


def main() -> None:
    """Parse CLI args and run the MCP tool-discovery stage."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_file", default=".env", help="Optional .env file to load (pipeline's own).")
    ap.add_argument("--out_dir", required=True, help="Shared pipeline outputs directory.")
    ap.add_argument("--nodes_py", default="final_nodespec.py", help="Input nodes file under --out_dir.")
    ap.add_argument("--out", default="mcp_tool_discovery_nodes.py", help="Output nodes file under --out_dir.")
    ap.add_argument("--report_out", default="mcp_tool_discovery_report.json", help="Report path under --out_dir.")
    ap.add_argument("--raw_dir", default="mcp_tool_discovery_raw", help="Raw probe artifacts folder under --out_dir.")
    ap.add_argument("--entry", required=True, help="Target entry source file path.")
    ap.add_argument("--execution_command_example_json", default="", help="Optional explicit execution command example JSON.")
    ap.add_argument("--model", default="gpt-5.2-codex", help="Model used to author the connection resolver.")
    ap.add_argument("--max_probe_attempts", type=int, default=3, help="Max generate/execute/repair attempts per server.")
    ap.add_argument("--probe_timeout_sec", type=int, default=45, help="Hard subprocess timeout per probe attempt.")
    ap.add_argument("--refresh_raw", action="store_true", help="Bypass the resolver-script cache and call the model again.")
    ap.add_argument("--verbos", action="store_true", help="Verbose per-attempt logging.")
    args = ap.parse_args()

    stage = MCPToolDiscovery(
        out_dir=args.out_dir,
        nodes_py=args.nodes_py,
        out=args.out,
        report_out=args.report_out,
        raw_dir=args.raw_dir,
        entry=args.entry,
        execution_command_example_json=args.execution_command_example_json,
        model=args.model,
        env_file=args.env_file,
        max_probe_attempts=args.max_probe_attempts,
        probe_timeout_sec=args.probe_timeout_sec,
        refresh_raw=args.refresh_raw,
        verbos=args.verbos,
    )
    stage.run()


if __name__ == "__main__":
    main()
