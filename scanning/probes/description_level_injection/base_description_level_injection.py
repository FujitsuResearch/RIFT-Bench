import ast
import re
from abc import ABC
from typing import Any

from node_spec.structure_schema import FlowSpec, NodeSpec

from scanning.probes.base_probe import BaseProbe
from scanning.probes.utils.base_util import (
    get_unique_invoked_tool_ids,
    normalize_flows,
    select_attacked_tool_node_by_candidate,
)
from scanning.probes.utils.flow_filltering import (
    get_tool_ids_matching_predicate,
    has_assignment_code_reference,
    has_entry_point_script,
)


class BaseDescriptionLevelInjection(BaseProbe, ABC):
    """
    Shared base class for probes that inject instructions through a tool
    description embedded in the tool definition.

    These probes mutate the attacked tool's definition snippet so the tool
    description carries a malicious instruction. The shared helpers here cover:

    1. Selecting flows that invoke an eligible tool.
    2. Picking the attacked tool from the selected flow.
    3. Picking an alternative tool that can later be invoked by the model.
    4. Building execution commands and metadata.
    5. Appending malicious content to the attacked tool description.
    """
    guidance_fields = ["attacked_tool_node_id", "input_arguments"]

    @staticmethod
    def is_attacked_tool_candidate(tool_node: Any) -> bool:
        """
        Return whether this tool exposes raw description metadata that can be
        targeted by description-level injection.
        """
        return has_assignment_code_reference(tool_node)

    def get_attackable_tool_ids(
        self,
        full_node_spec: NodeSpec,
        flow: FlowSpec,
    ) -> list[str]:
        """
        Return invoked tool ids that this probe can attack using raw node-spec
        metadata only.
        """
        return get_tool_ids_matching_predicate(
            full_node_spec=full_node_spec,
            tool_ids=get_unique_invoked_tool_ids(flow),
            predicate=self.is_attacked_tool_candidate,
        )

    def select_attacked_tool_node(
        self,
        malicious_node_spec: NodeSpec,
        flow: FlowSpec,
        guidance: dict[str, Any] | None = None,
    ) -> tuple[str, Any]:
        """
        Select the first structurally eligible invoked tool.
        """
        return select_attacked_tool_node_by_candidate(
            self=self,
            malicious_node_spec=malicious_node_spec,
            flow=flow,
            guidance=guidance,
        )

    @classmethod
    def filter_relevant_flows(
        cls,
        full_node_spec: NodeSpec,
        flows: list[FlowSpec] | None = None,
    ) -> list[FlowSpec]:
        """
        Keep flows that match the raw-metadata description-level pattern.

        A relevant flow must satisfy all of these conditions:

        1. The node spec defines an entry-point script.
        2. The flow invoked at least one tool.
        3. At least one invoked tool defines an assignment code reference that
           can carry the injected description text.

        Deeper code-snippet checks, such as whether the selected assignment
        snippet is actually rewriteable, are intentionally deferred to the later
        attack-target selection step.
        """
        if not has_entry_point_script(full_node_spec):
            return []

        relevant_flows: list[FlowSpec] = []
        for flow in normalize_flows(full_node_spec, flows):
            try:
                        attackable_tool_ids = get_tool_ids_matching_predicate(
                            full_node_spec=full_node_spec,
                            tool_ids=get_unique_invoked_tool_ids(flow),
                            predicate=cls.is_attacked_tool_candidate,
                        )
                        if attackable_tool_ids:
                            relevant_flows.append(flow)
            except Exception:
                continue
        return relevant_flows

    @staticmethod
    def inject_instruction_into_tool_description(tool_node: Any, instruction: str) -> None:
        """
        Append the injected instruction to the attacked tool description.

        Supports:
        - Explicit descriptions such as `description="..."` in tool registration.
        - Implicit descriptions derived from function docstrings (for decorators
          like `@mcp.tool()` where no explicit description is provided).
        """
        for code in getattr(tool_node, "code_references", None) or []:
            if code.kind != "assignment":
                continue
            updated_snippet = BaseDescriptionLevelInjection._inject_into_snippet(
                code.snippet,
                instruction,
            )
            if updated_snippet is not None:
                code.snippet = updated_snippet
                return

        # Fallback for simple explicit description patterns if AST parsing fails.
        for code in getattr(tool_node, "code_references", None) or []:
            if code.kind != "assignment":
                continue
            updated_snippet = BaseDescriptionLevelInjection._inject_with_regex_fallback(
                code.snippet,
                instruction,
            )
            if updated_snippet != code.snippet:
                code.snippet = updated_snippet
                return

        if not getattr(tool_node, "code_references", None):
            tool_id = "<unknown>"
        else:
            tool_id = getattr(tool_node, "id", "<unknown>")
        raise ValueError(
            f"Could not inject into tool description for tool '{tool_id}'."
        )

    @staticmethod
    def _inject_into_snippet(snippet: str, instruction: str) -> str | None:
        parsed = BaseDescriptionLevelInjection._parse_snippet_with_line_offset(snippet)
        if parsed is None:
            return None
        tree, line_offset = parsed

        # 1) Prefer explicit description keyword in decorators like @mcp.tool(...).
        for fn in BaseDescriptionLevelInjection._iter_function_defs(tree):
            for decorator in fn.decorator_list:
                if not (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "tool"
                ):
                    continue
                desc_literal = BaseDescriptionLevelInjection._find_description_keyword_literal(
                    decorator
                )
                if desc_literal is None:
                    continue
                return BaseDescriptionLevelInjection._replace_string_node(
                    snippet,
                    desc_literal,
                    desc_literal.value + instruction,
                    line_offset=line_offset,
                )

        # 2) Fallback to explicit description keyword in any call expression.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            desc_literal = BaseDescriptionLevelInjection._find_description_keyword_literal(
                node
            )
            if desc_literal is None:
                continue
            return BaseDescriptionLevelInjection._replace_string_node(
                snippet,
                desc_literal,
                desc_literal.value + instruction,
                line_offset=line_offset,
            )

        # 3) Implicit description: docstring of a tool-decorated function.
        for fn in BaseDescriptionLevelInjection._iter_function_defs(tree):
            if not BaseDescriptionLevelInjection._is_tool_decorated_function(fn):
                continue
            if not fn.body:
                continue
            first_stmt = fn.body[0]
            if not (
                isinstance(first_stmt, ast.Expr)
                and isinstance(first_stmt.value, ast.Constant)
                and isinstance(first_stmt.value.value, str)
            ):
                continue
            docstring_literal = first_stmt.value
            return BaseDescriptionLevelInjection._replace_string_node(
                snippet,
                docstring_literal,
                docstring_literal.value + instruction,
                line_offset=line_offset,
            )

        # 4) Fallback to the first function docstring in the snippet.
        for fn in BaseDescriptionLevelInjection._iter_function_defs(tree):
            if not fn.body:
                continue
            first_stmt = fn.body[0]
            if not (
                isinstance(first_stmt, ast.Expr)
                and isinstance(first_stmt.value, ast.Constant)
                and isinstance(first_stmt.value.value, str)
            ):
                continue
            docstring_literal = first_stmt.value
            return BaseDescriptionLevelInjection._replace_string_node(
                snippet,
                docstring_literal,
                docstring_literal.value + instruction,
                line_offset=line_offset,
            )

        return None

    @staticmethod
    def _parse_snippet_with_line_offset(snippet: str) -> tuple[ast.AST, int] | None:
        try:
            return ast.parse(snippet), 0
        except (IndentationError, SyntaxError):
            pass

        wrapped_snippet = f"if True:\n{snippet}"
        try:
            return ast.parse(wrapped_snippet), 1
        except (IndentationError, SyntaxError):
            return None

    @staticmethod
    def _iter_function_defs(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
        function_defs: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_defs.append(node)
        return function_defs

    @staticmethod
    def _is_tool_decorated_function(
        fn: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> bool:
        for decorator in fn.decorator_list:
            if isinstance(decorator, ast.Attribute) and decorator.attr == "tool":
                return True
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "tool"
            ):
                return True
        return False

    @staticmethod
    def _find_description_keyword_literal(call: ast.Call) -> ast.Constant | None:
        for keyword in call.keywords:
            if keyword.arg != "description":
                continue
            if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                return keyword.value
        return None

    @staticmethod
    def _replace_string_node(
        snippet: str,
        node: ast.Constant,
        new_value: str,
        line_offset: int = 0,
    ) -> str:
        if (
            getattr(node, "lineno", None) is None
            or getattr(node, "col_offset", None) is None
            or getattr(node, "end_lineno", None) is None
            or getattr(node, "end_col_offset", None) is None
        ):
            raise ValueError("AST node is missing source positions for replacement.")

        start_lineno = node.lineno - line_offset
        end_lineno = node.end_lineno - line_offset
        if start_lineno < 1 or end_lineno < 1:
            raise ValueError("AST node source positions could not be mapped to snippet.")

        start_idx = BaseDescriptionLevelInjection._line_col_to_index(
            snippet,
            start_lineno,
            node.col_offset,
        )
        end_idx = BaseDescriptionLevelInjection._line_col_to_index(
            snippet,
            end_lineno,
            node.end_col_offset,
        )
        return f"{snippet[:start_idx]}{new_value!r}{snippet[end_idx:]}"

    @staticmethod
    def _line_col_to_index(source: str, lineno: int, col_offset: int) -> int:
        lines = source.splitlines(keepends=True)
        return sum(len(lines[idx]) for idx in range(lineno - 1)) + col_offset

    @staticmethod
    def _inject_with_regex_fallback(snippet: str, instruction: str) -> str:
        pattern = r'(description\s*=\s*["\'])([^"\']*)(["\'])'

        def append_description(match: re.Match[str]) -> str:
            injected = f"{match.group(2)}{instruction}".replace("\n", "\\n")
            return f"{match.group(1)}{injected}{match.group(3)}"

        return re.sub(pattern, append_description, snippet, count=1)
