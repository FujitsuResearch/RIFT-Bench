import ast
import re

from node_spec.structure_schema import NodeSpec

from scanning.defenses.base_defense import BaseDefense
from scanning.probes.description_level_injection.base_description_level_injection import (
    BaseDescriptionLevelInjection,
)


class DescriptionRemoving(BaseDefense):
    # Keep the registration syntax valid after the defense: description-bearing
    # string literals are replaced with a neutral fallback, never removed.
    REDACTED_DESCRIPTION = "Tool description intentionally withheld."

    @staticmethod
    def remove_tool_description(tool_node: NodeSpec):
        for code in getattr(tool_node, "code_references", None) or []:
            if code.kind != "assignment":
                continue
            updated_snippet = DescriptionRemoving._remove_from_snippet(code.snippet)
            if updated_snippet is not None:
                code.snippet = updated_snippet
                return

        for code in getattr(tool_node, "code_references", None) or []:
            if code.kind != "assignment":
                continue
            updated_snippet = DescriptionRemoving._remove_with_regex_fallback(
                code.snippet
            )
            if updated_snippet != code.snippet:
                code.snippet = updated_snippet
                return

        tool_id = getattr(tool_node, "id", "<unknown>")
        raise ValueError(f"Could not remove tool description for tool '{tool_id}'.")

    @staticmethod
    def _remove_from_snippet(snippet: str) -> str | None:
        parsed = BaseDescriptionLevelInjection._parse_snippet_with_line_offset(snippet)
        if parsed is None:
            return None
        tree, line_offset = parsed

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
                    DescriptionRemoving.REDACTED_DESCRIPTION,
                    line_offset=line_offset,
                )

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
                DescriptionRemoving.REDACTED_DESCRIPTION,
                line_offset=line_offset,
            )

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
                DescriptionRemoving.REDACTED_DESCRIPTION,
                line_offset=line_offset,
            )

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
                DescriptionRemoving.REDACTED_DESCRIPTION,
                line_offset=line_offset,
            )

        return None

    @staticmethod
    def _remove_with_regex_fallback(snippet: str) -> str:
        pattern = r'(description\s*=\s*["\'])([^"\']*)(["\'])'

        def clear_description(match: re.Match[str]) -> str:
            return (
                f"{match.group(1)}{DescriptionRemoving.REDACTED_DESCRIPTION}"
                f"{match.group(3)}"
            )

        return re.sub(pattern, clear_description, snippet, count=1)

    def apply_defense_tool(self, tool_node: NodeSpec, user_instruction: str = ""):
        try:
            self.remove_tool_description(tool_node)
        except ValueError:
            return
