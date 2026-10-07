#!/usr/bin/env python3
"""Pipeline for post_processing component."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

try:
    from ..global_utils import (
        call_with_cache,
        copy_nodespec_schema_to_out_dir,
        extract_var_name,
        load_env_file,
        load_nodes_by_var,
        normalize_code_references_add,
    )
    from ..rag.build_retriever import load_retriever
    from .prompts import (
        IMPLEMENTATION_REFERENCE_PROMPT,
        NAME_PROMPT,
        build_children_guidance,
        build_definition_prompt,
        build_source_prompt,
        build_system_prompt_prompt,
    )
    from .utils import (
        bottom_up_var_order,
        extract_enclosing_code_ast,
        extract_snippet,
        find_external_mcp_sub_snippet,
        flex_find_snippet_lines,
        write_nodes_with_updated_code_references_preserve_source,
    )
except ImportError:
    from global_utils import (
        call_with_cache,
        copy_nodespec_schema_to_out_dir,
        extract_var_name,
        load_env_file,
        load_nodes_by_var,
        normalize_code_references_add,
    )
    from rag.build_retriever import load_retriever
    from code_reference_refine.prompts import (
        IMPLEMENTATION_REFERENCE_PROMPT,
        NAME_PROMPT,
        build_children_guidance,
        build_definition_prompt,
        build_source_prompt,
        build_system_prompt_prompt,
    )
    from code_reference_refine.utils import (
        bottom_up_var_order,
        extract_enclosing_code_ast,
        extract_snippet,
        find_external_mcp_sub_snippet,
        flex_find_snippet_lines,
        write_nodes_with_updated_code_references_preserve_source,
    )


class CodeReferenceRefiner():
    """Refine each node's grounded code references without changing graph topology."""

    def __init__(self, out_dir, nodes_py="nodes_graph_corrected.py", out="final_nodespec.py", report_out="post_processing_report.json", final_report_out="final_report.json", raw_dir="post_processing_raw", model="gpt-5.2-codex", rag_index_dir="rag_index", rag_max_rounds=6, env_file=".env", refresh_raw=False, verbos=False):
        """Prepare paths, retrieval, and run counters for code-reference refinement."""
        # Load environment configuration before model and retriever setup.
        load_env_file(Path(env_file))
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        copy_nodespec_schema_to_out_dir(self.out_dir)
        self.verbos=verbos

        # Resolve all stage-local input/output paths under the chosen output directory.
        self.nodes_py = self.out_dir / Path(nodes_py).name
        self.out = out
        self.out_py = self.out_dir / Path(out).name
        self.report_path = self.out_dir / Path(report_out).name
        self.final_report_path = self.out_dir / Path(final_report_out).name
        self.raw_dir = self.out_dir / Path(raw_dir).name
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        # Build the retriever once and reuse it across all node-level grounding calls.
        self.rag_index_path = Path(rag_index_dir)
        self.rag_index_dir = self.rag_index_path.resolve() if self.rag_index_path.is_absolute() else (self.out_dir / self.rag_index_path).resolve()
        self.retriever = load_retriever(str(self.rag_index_dir))
        self.model = model
        self.rag_max_rounds = rag_max_rounds
        self.refresh_raw = refresh_raw
        self.llm_attempted = 0
        self.llm_resolved = 0
        self.llm_unresolved = 0

    @staticmethod
    def _sanitize_json_compatible(obj: Any) -> Any:
        """Recursively replace invalid string values so JSON caching cannot fail."""
        if isinstance(obj, str):
            # Replace invalid unicode surrogates so payload serialization cannot crash.
            return obj.encode("utf-8", errors="replace").decode("utf-8")
        if isinstance(obj, list):
            return [CodeReferenceRefiner._sanitize_json_compatible(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): CodeReferenceRefiner._sanitize_json_compatible(v) for k, v in obj.items()}
        return obj

    def _safe_call_with_cache(
        self,
        *,
        payload: Dict[str, Any],
        payload_path: Path,
        raw_path: Path,
    ) -> tuple[Dict[str, Any], str]:
        """Call the cached LLM helper, retry once with sanitized payload, then fall back safely."""
        # Try once as-is, then retry once with sanitized payload.
        # First try the original payload, then retry with a JSON-safe version if needed.
        attempts_payloads: List[Dict[str, Any]] = [payload, self._sanitize_json_compatible(payload)]
        last_err: Exception | None = None
        for i, work_payload in enumerate(attempts_payloads):
            try:
                parsed, source = call_with_cache(
                    work_payload,
                    payload_path=payload_path,
                    raw_path=raw_path,
                    model=self.model,
                    refresh_raw=bool(self.refresh_raw),
                    retriever=self.retriever,
                    rag_max_rounds=self.rag_max_rounds,
                )
                return (parsed if isinstance(parsed, dict) else {}), str(source or "")
            except Exception as exc:
                last_err = exc
                if self.verbos:
                    print(f"[code_reference_refine] call_with_cache failed (attempt {i + 1}/2) at {payload_path}: {exc}")
                continue

        # Final fallback: persist a minimal failure artifact and keep the pipeline moving.
        try:
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_text(
                json.dumps(
                    {
                        "fallback": "call_failed_after_retry",
                        "error": str(last_err) if last_err else "unknown_error",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            raw_path.write_text(
                f'{{"error":"call_failed_after_retry","detail":{json.dumps(str(last_err) if last_err else "unknown_error")}}}\n',
                encoding="utf-8",
            )
        except Exception:
            pass
        return {}, "fallback_empty_after_retry"

    def update_nodes(self):
        """Load the current graph, refresh code references, and write outputs."""
        # Read the graph file emitted by the previous structural stage.
        var_order, nodes_by_var = load_nodes_by_var(self.nodes_py)
        if not var_order:
            raise RuntimeError("post_processing: input nodes file has no ALL_NODES/MAIN_GRAPH_NODES.")

        # This stage is code-reference-only by design: no topology or connectivity changes.
        refs_before, refs_after, per_node_results = self.update_code_refernaces(var_order, nodes_by_var)

        # This stage is code-reference-only by design: do not mutate connectivity/topology.
        external_connections_updates = 0
        self.creat_final_report(var_order, nodes_by_var, refs_before, refs_after, external_connections_updates, per_node_results)

    def update_code_refernaces(self, var_order, nodes_by_var):
        """Recompute grounded code references for each node in bottom-up order."""
        # Visit children before parents so parent prompts can reuse child handles when available.
        traversal_order = bottom_up_var_order(list(var_order), nodes_by_var)

        refs_before = 0
        refs_after = 0

        per_node_results: List[Dict[str, Any]] = []
        for var in tqdm(traversal_order):
            # Skip entries that should not be refined as standalone runtime components here.
            node_assign = None
            if self.verbos:
                print('Node: ', var)
            node = nodes_by_var.get(var)
            if not isinstance(node, dict):
                continue

            node_type = node.get("node_type") if isinstance(node.get("node_type"), dict) else {}
            node_type_name = str(node_type.get("type") or "")
            metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
            if node_type_name == "Tool" and self._has_external_mcp_parent(var, var_order, nodes_by_var):
                per_node_results.append({"node_var": var, "status": "skip_external_mcp_tool"})
                continue

            new_refs = []

            # Track the original reference count before rebuilding this node's list from scratch.
            refs = node.get("code_references") if isinstance(node.get("code_references"), list) else []
            refs_before += len(refs)

            # First identify the best runtime handle for this component.
            name = str(metadata.get("component_handle") or "").strip()
            if not name:
                name = self.get_component_handle(var, node)
            if name:
                node['metadata']['component_handle'] = name

            # Feed known child handles back into assignment grounding when useful.
            childern_info = self.get_childern_info(node, nodes_by_var)

            assign_prompt = build_children_guidance(childern_info)

            # Retry assignment grounding a few times because later prompts depend on it.
            attempts = 0
            while not node_assign and attempts<4:
                attempts += 1
                node_assign = self.get_field_using_llm_wrap(var, node, assign_prompt, 'assignment')

            if node_assign:
                new_refs.append(node_assign)
            if self.verbos:
                print('Node Assign: ', node_assign)

            # Prefer a definition lookup anchored by the assignment when one was found.
            if node_assign:
                def_prompt = build_definition_prompt(node_assign)
            else:
                def_prompt = IMPLEMENTATION_REFERENCE_PROMPT
            node_def = self.get_field_using_llm_wrap(var, node, def_prompt, 'definition')
            if node_def:
                new_refs.append(node_def)
            if self.verbos:
                print('Node Def: ', node_def)

            if not node_assign and not node_def:
                continue
            elif not node_assign:
                node_assign = node_def

            # External MCP nodes also need the constructor span inside the outer assignment.
            if node_type_name == "External_MCP_server" and str(node_assign.get("kind") or "") == "assignment":
                external_mcp_ref = self.build_external_mcp_constructor_reference(
                    node=node,
                    node_assign=node_assign,
                    component_handle=name or var,
                )
                if external_mcp_ref:
                    new_refs.append(external_mcp_ref)
                if self.verbos:
                    print('External MCP constructor: ', external_mcp_ref)

            # Agent, System, and LLM nodes may also expose a distinct system-prompt reference.
            if node_type.get('type') in {'Agent', 'System', 'LLM'}:
                system_prompt_prompt = build_system_prompt_prompt(node_assign)
                node_system_prompt = self.get_field_using_llm_wrap(var, node, system_prompt_prompt, 'system_prompt')
                if node_system_prompt:
                    new_refs.append(node_system_prompt)
                if self.verbos:
                    print('Node system_prompt: ', node_system_prompt)

            # Finally ground the source/import site for the represented runtime component.
            source_prompt = build_source_prompt(node_assign)
            node_source = self.get_field_using_llm_wrap(var, node, source_prompt, 'source')
            if node_source:
                new_refs.append(node_source)
            if self.verbos:
                print('Node source: ', node_source)

            # Replace the previous code-reference list with the newly grounded references.
            node["code_references"] = new_refs
            refs_after += len(node["code_references"])
            per_node_results.append({"node_var": var, "status": "updated", "new_refs": len(new_refs)})

        return refs_before, refs_after, per_node_results

    def build_external_mcp_constructor_reference(
        self,
        *,
        node: Dict[str, Any],
        node_assign: Dict[str, Any],
        component_handle: str,
    ) -> Dict[str, Any] | None:
        """Derive the exact constructor sub-snippet for an external MCP assignment."""
        assign_file = str(node_assign.get("file") or "").strip()
        assign_snippet = str(node_assign.get("snippet") or "")
        if not assign_file or not assign_snippet:
            return None

        # Extract the inner constructor span from the already grounded assignment snippet.
        try:
            external_mcp_sub_snippet = find_external_mcp_sub_snippet(
                component_handle,
                assign_snippet,
                str(node.get("description") or ""),
                model=self.model,
            )
        except Exception as exc:
            if self.verbos:
                print(f"External MCP constructor extraction failed for {component_handle}: {exc}")
            return None

        line = None
        assign_line = node_assign.get("line")
        assign_start = None
        if isinstance(assign_line, int):
            assign_start = assign_line
        elif (
            isinstance(assign_line, (list, tuple))
            and len(assign_line) == 2
            and all(isinstance(x, int) for x in assign_line)
        ):
            assign_start = int(assign_line[0])

        # Prefer calculating the constructor span relative to the assignment's known start line.
        if assign_start is not None:
            assign_lines = [line.strip() for line in assign_snippet.splitlines()]
            target_lines = [line.strip() for line in external_mcp_sub_snippet.splitlines()]
            block_len = len(target_lines)
            if block_len:
                for idx in range(len(assign_lines) - block_len + 1):
                    if assign_lines[idx:idx + block_len] == target_lines:
                        start_line = assign_start + idx
                        end_line = start_line + block_len - 1
                        line = start_line if start_line == end_line else [start_line, end_line]
                        break

        # Fall back to a file-level snippet search if the relative offset did not land.
        if line is None:
            line = flex_find_snippet_lines(assign_file, external_mcp_sub_snippet)
        if line is None:
            if self.verbos:
                print(f"External MCP constructor snippet could not be grounded in {assign_file}: {component_handle}")
            return None

        return {
            "kind": "external_mcp_constructor",
            "other_kind_description": None,
            "file": assign_file,
            "line": line,
            "snippet": external_mcp_sub_snippet,
        }

    def _has_external_mcp_parent(self, child_var: str, var_order: List[str], nodes_by_var: Dict[str, Dict[str, Any]]) -> bool:
        """Return whether the given tool already belongs under an external MCP server."""
        for pvar in var_order:
            pnode = nodes_by_var.get(pvar, {})
            ptype = str(((pnode.get("node_type") or {}) if isinstance(pnode.get("node_type"), dict) else {}).get("type") or "")
            if ptype != "External_MCP_server":
                continue
            vals = pnode.get("tool_list")
            if not isinstance(vals, list):
                continue
            for ref in vals:
                if extract_var_name(ref) == child_var:
                    return True
        return False


    def creat_final_report(self, var_order, nodes_by_var, refs_before, refs_after, external_connections_updates, per_node_results):
        """Write the updated nodes file plus the detailed and summary reports."""
        # The detailed report captures aggregate counts and per-node outcomes.
        report: Dict[str, Any] = {
            "mode": "post_processing",
            "input_nodes_file": Path(self.nodes_py).name,
            "output_nodes_file": Path(self.out).name,
            "total_nodes": len(var_order),
            "code_references_before": refs_before,
            "code_references_after": refs_after,
            "code_references_removed": refs_before - refs_after,
            "definition_reference_resolution": {
                "attempted": self.llm_attempted,
                "resolved_by_llm": self.llm_resolved,
                "unresolved_reference": self.llm_unresolved,
            },
            "external_connections_updates": external_connections_updates,
            "per_node_results": per_node_results,
        }

        # Preserve the existing NodeSpec source layout while swapping refined fields only.
        write_nodes_with_updated_code_references_preserve_source(
            input_path=self.nodes_py,
            output_path=self.out_py,
            nodes_by_var=nodes_by_var,
            var_order=var_order,
        )
        self.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        final_report = {
            "final_nodes_file": Path(self.out).name,
            "total_nodes": len(var_order),
            "code_references_removed": refs_before - refs_after,
        }
        self.final_report_path.write_text(json.dumps(final_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        print(f"[post_processing] wrote: {self.out_py}")
        print(f"[post_processing] report: {self.report_path}")
        print(f"[post_processing] final_report: {self.final_report_path}")

    def get_childern_info(self, node, nodes_by_var):
        """Collect known child component handles to guide parent assignment grounding."""
        childern = node.get("nodes") if isinstance(node.get("nodes"), list) else []
        out = []
        for child in childern:
            child_var = extract_var_name(child)
            if not child_var:
                continue

            child_node = nodes_by_var.get(child_var, {})

            name = child_node['metadata'].get('component_handle', None)
            if name:
                out.append(name)
        return out

    def get_component_handle(self, var, node):
        """Resolve the main code identifier that best represents this runtime component."""
        self.llm_attempted += 1
        node_dir = self.raw_dir / var
        node_dir.mkdir(parents=True, exist_ok=True)
        # Ask for one best runtime handle plus its supporting grounded reference.
        payload = {
            "prompt": NAME_PROMPT,
            "node_var": var,
            "node": node,
            "guidance_summary": "",
            "retrieved_context": None,
        }

        parsed, _ = self._safe_call_with_cache(
            payload=payload,
            payload_path=node_dir / "component_handle_reference.payload.json",
            raw_path=node_dir / "component_handle_reference.txt",
        )

        try:
            name = parsed.get("component_handle") if isinstance(parsed, dict) else None
            raw_code_ref = parsed.get("code_reference") if isinstance(parsed, dict) else None
            normalized_refs = normalize_code_references_add([raw_code_ref] if raw_code_ref else [])
            code_ref = normalized_refs[0] if normalized_refs else None

            # Reject malformed, null, or ungrounded handle references before using them.
            if not isinstance(code_ref, dict) or str(code_ref.get("kind") or "") != 'component_handle':
                self.llm_unresolved += 1
                if self.verbos:
                    print("No relevant code reference")
                return None

            if self.is_strict_null_code_reference(code_ref):
                self.llm_resolved += 1
                if self.verbos:
                    print("Null code reference")
                return None

            if not self.is_grounded_code_reference(code_ref):
                self.llm_unresolved += 1
                if self.verbos:
                    print("Invalid code reference (missing line or snippet)")
                return None

            line = flex_find_snippet_lines(code_ref["file"], code_ref["snippet"])
            if line is None:
                self.llm_unresolved += 1
                if self.verbos:
                    print("The snippet not found, likely a fabrication", code_ref)
                return None
            return name
        except:
            return None

    def get_field_using_llm_wrap(self, var, node, field_prompt, field):
        """Run one grounding call and retry once with stricter instructions if needed."""
        code_ref, problem = self.get_field_using_llm(var, node, field_prompt, field)

        if code_ref:
            return code_ref

        # --- Retry strategies based on failure mode ---

        if problem == "The snippet not found, likely a fabrication":
            # Model hallucinated snippet → enforce stricter grounding
            field_prompt += "\n\nIMPORTANT: Do NOT fabricate snippets. Only return a snippet that exists verbatim in the file. If unsure, return context_request."

        elif problem == "Invalid code reference (missing line or snippet)":
            # Partial / malformed output → enforce strict schema
            field_prompt += "\n\nIMPORTANT: You must return a fully grounded code_reference. file, line, and snippet must ALL be non-null together. Otherwise return context_request."
        else:
            # Unknown / generic failure → safer fallback
            field_prompt += "\n\nIMPORTANT: Ensure the result is evidence-backed, minimal, and matches the required schema exactly."

        # --- Retry once with improved prompt ---
        code_ref, _ = self.get_field_using_llm(var, node, field_prompt, field)
        return code_ref

    def get_field_using_llm(self, var, node, field_prompt, field):
            """Ground one requested code-reference field and normalize its final snippet span."""
            self.llm_attempted += 1
            node_dir = self.raw_dir / var
            node_dir.mkdir(parents=True, exist_ok=True)
            # Query for exactly one code-reference kind at a time.
            payload = {
                "prompt": field_prompt,
                "node_var": var,
                "node": node,
                "guidance_summary": "",
                "retrieved_context": None,
            }

            parsed, _ = self._safe_call_with_cache(
                payload=payload,
                payload_path=node_dir / f"{field}_reference.payload.json",
                raw_path=node_dir / f"{field}_reference.txt",
            )

            try:
                raw_code_ref = parsed.get("code_reference") if isinstance(parsed, dict) else None
                normalized_refs = normalize_code_references_add([raw_code_ref] if raw_code_ref else [])
                code_ref = normalized_refs[0] if normalized_refs else None

                if not isinstance(code_ref, dict) or str(code_ref.get("kind") or "") != field:
                    self.llm_unresolved += 1
                    if self.verbos:
                        print("No relevant code reference")
                    return None, "No relevant code reference"

                if self.is_strict_null_code_reference(code_ref):
                    self.llm_resolved += 1
                    if self.verbos:
                        print("Null code reference")
                    return None, "Null code reference"

                if not self.is_grounded_code_reference(code_ref):
                    self.llm_unresolved += 1
                    if self.verbos:
                        print("Invalid code reference (missing line or snippet)")
                    return None, "Invalid code reference (missing line or snippet)"

                # Normalize the returned reference to the exact snippet and enclosing code span.
                code_ref = dict(code_ref)
                code_ref["other_kind_description"] = code_ref.get("other_kind_description") or None

                line = flex_find_snippet_lines(code_ref["file"], code_ref["snippet"])
                if line is None:
                    self.llm_unresolved += 1
                    if self.verbos:
                        print("The snippet not found, likely a fabrication", code_ref)
                    return None, "The snippet not found, likely a fabrication"

                code_ref["line"] = line
                code_ref["snippet"] = extract_snippet(code_ref["file"], line)
                self.llm_resolved += 1
                temp = extract_enclosing_code_ast(code_ref["file"], line)
                if temp:
                    code_ref["line"] = temp[0]
                    code_ref["snippet"] = temp[1]

                return code_ref, "All Good"

            except Exception as e:
                self.llm_unresolved += 1
                print(e)
                return None, str(e)

    @staticmethod
    def is_strict_null_code_reference(code_reference: Dict[str, Any]) -> bool:
        """Return whether the model emitted the agreed explicit null-reference shape."""
        expected_keys = {
            "kind",
            "file",
            "line",
            "snippet",
            "other_kind_description",
        }

        if not isinstance(code_reference, dict):
            return False

        if set(code_reference.keys()) != expected_keys:
            return False

        return (
            code_reference["kind"]
            and code_reference["file"] is None
            and code_reference["line"] is None
            and code_reference["snippet"] is None
            and code_reference["other_kind_description"] is None
        )

    @staticmethod
    def is_grounded_code_reference(code_reference: Dict[str, Any]) -> bool:
        """
        True if this is a fully grounded reference (file+line+snippet present).
        """
        if not isinstance(code_reference, dict):
            return False

        return (
            code_reference.get("kind")
            and code_reference.get("file") is not None
            and code_reference.get("line") is not None
            and code_reference.get("snippet") is not None
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_file", default=".env", help="Optional .env file to load.")
    ap.add_argument("--out_dir", default='stages_outputs_path', help="Output directory.")
    ap.add_argument("--nodes_py", default="nodes_graph_corrected.py", help="Input nodes file under --out_dir.")
    ap.add_argument("--out", default="final_nodespec.py", help="Final output nodes file under --out_dir.")
    ap.add_argument("--report_out", default="post_processing_report.json", help="Report output under --out_dir.")
    ap.add_argument("--final_report_out", default="final_report.json", help="Final summary report under --out_dir.")
    ap.add_argument("--raw_dir", default="post_processing_raw", help="Raw traces folder under --out_dir.")
    ap.add_argument("--model", default="gpt-5.2-codex", help="Model for per-node definition resolution.")
    ap.add_argument("--rag_index_dir", default="rag_index", help="RAG index path (relative to out_dir if not absolute).")
    ap.add_argument("--rag_max_rounds", type=int, default=6, help="Max context-request rounds per node call.")
    ap.add_argument("--refresh_raw", action="store_true", help="Bypass raw cache and call model again.")
    ap.add_argument("--verbos", default=False)

    args = ap.parse_args()

    post = CodeReferenceRefiner(args.out_dir, args.nodes_py, args.out, args.report_out, args.final_report_out, args.raw_dir, args.model, args.rag_index_dir, args.rag_max_rounds, args.env_file, refresh_raw=args.refresh_raw, verbos=args.verbos)
    post.update_nodes()

if __name__ == "__main__":
    main()
