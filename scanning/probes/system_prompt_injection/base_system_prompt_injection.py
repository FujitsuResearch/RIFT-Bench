from abc import ABC
import ast
import io
import tokenize
from typing import Any

from node_spec.structure_schema import FlowSpec, NodeSpec

from scanning.probes.base_probe import BaseProbe
from scanning.probes.utils.base_util import (
    extract_prompt_value,
    normalize_flows,
    prompt_variable_name_options,
)
from scanning.probes.utils.flow_filltering import (
    get_invoked_agent_ids,
    get_invoked_agent_ids_with_system_prompt_code_references,
    has_entry_point_script,
)


class BaseSystemPromptInjection(BaseProbe, ABC):
    """
    Shared helper base for probes that poison an agent's system prompt.
    """
    guidance_fields = ["attacked_agent_node_id", "input_arguments"]

    def get_attackable_agent_ids(
        self,
        node_spec: NodeSpec,
        flow: FlowSpec,
    ) -> list[str]:
        """
        Return invoked agent ids that expose raw system-prompt metadata.
        """
        return get_invoked_agent_ids_with_system_prompt_code_references(
            full_node_spec=node_spec,
            flow=flow,
        )

    @classmethod
    def filter_relevant_flows(
        cls,
        full_node_spec: NodeSpec,
        flows: list[FlowSpec] | None = None,
    ) -> list[FlowSpec]:
        """
        Keep flows that match the raw-metadata system-prompt-injection pattern.

        A relevant flow must satisfy all of these conditions:

        1. The node spec defines an entry-point script.
        2. The flow invoked at least one agent.
        3. At least one such invoked agent defines system_prompt code references.

        Deeper code-snippet checks, such as whether a specific system-prompt
        snippet can actually be rewritten, are intentionally deferred to the
        later attack-target selection step.
        """
        if not has_entry_point_script(full_node_spec):
            return []

        relevant_flows: list[FlowSpec] = []
        for flow in normalize_flows(full_node_spec, flows):
            try:
                        if get_invoked_agent_ids_with_system_prompt_code_references(
                            full_node_spec=full_node_spec,
                            flow=flow,
                        ):
                            relevant_flows.append(flow)
            except Exception:
                continue
        return relevant_flows

    @staticmethod
    def assert_agent_has_system_prompt_code_references(
        attacked_agent_node: Any,
        *,
        role: str = "attacked",
    ) -> None:
        if attacked_agent_node is None:
            raise ValueError(f"The {role} agent was not found in node spec.")

        system_prompt_references = [
            code
            for code in (getattr(attacked_agent_node, "code_references", None) or [])
            if getattr(code, "kind", None) == "system_prompt"
        ]
        if not system_prompt_references:
            attacked_agent_node_id = getattr(attacked_agent_node, "id", "<unknown>")
            raise ValueError(
                "The "
                f"{role} agent '{attacked_agent_node_id}' does not define "
                "system_prompt code_references."
            )

    def select_attacked_agent_node_id(
        self,
        node_spec: NodeSpec,
        flow: FlowSpec,
        guidance: dict[str, Any] | None = None,
    ) -> tuple[str, Any]:
        guidance = guidance or {}
        attacked_agent_node_id = guidance.get("attacked_agent_node_id")
        if attacked_agent_node_id:
            attacked_agent_node = node_spec.get_node(attacked_agent_node_id)
            if attacked_agent_node is None:
                raise ValueError(
                    f"Guided attacked agent '{attacked_agent_node_id}' was not found."
                )
            self.assert_agent_has_system_prompt_code_references(
                attacked_agent_node,
                role="attacked",
            )
            return attacked_agent_node_id, attacked_agent_node

        attackable_agent_ids = self.get_attackable_agent_ids(node_spec, flow)
        if not attackable_agent_ids:
            raise ValueError("No attacked agent could be selected from the flow.")

        attacked_agent_node_id = attackable_agent_ids[0]
        attacked_agent_node = node_spec.get_node(attacked_agent_node_id)
        if attacked_agent_node is None:
            raise ValueError(
                f"Invoked agent '{attacked_agent_node_id}' was not found in node spec."
            )
        self.assert_agent_has_system_prompt_code_references(
            attacked_agent_node,
            role="attacked",
        )
        return attacked_agent_node_id, attacked_agent_node

    @staticmethod
    def inject_system_prompt(
        attacked_agent_node: Any,
        instruction: str,
        mode: str = "prepend",
    ) -> None:
        system_prompt_references = [
            code
            for code in (getattr(attacked_agent_node, "code_references", []) or [])
            if code.kind == "system_prompt"
        ]
        if not system_prompt_references:
            raise ValueError(
                "Failed to inject system prompt: attacked agent has no system_prompt code references."
            )

        for code in system_prompt_references:
            old_prompt = extract_prompt_value(code.snippet)
            if old_prompt is None:
                continue

            replacement = instruction if mode == "replace" else f"{instruction}\n{old_prompt}"
            updated_snippet = BaseSystemPromptInjection._replace_prompt_value_text(
                snippet=code.snippet,
                old_prompt=old_prompt,
                replacement_prompt=replacement,
            )
            if updated_snippet is None:
                updated_snippet = BaseSystemPromptInjection._rewrite_prompt_assignment_value(
                    code.snippet,
                    replacement,
                )
                if updated_snippet is None:
                    continue

            code.snippet = updated_snippet
            return

        raise ValueError(
            "Failed to inject system prompt: system_prompt code references were found, but none were rewriteable."
        )

    @staticmethod
    def _replace_prompt_value_text(
        snippet: str,
        old_prompt: str,
        replacement_prompt: str,
    ) -> str | None:
        value_offsets = BaseSystemPromptInjection._find_prompt_value_offsets(snippet)
        if value_offsets is None:
            return None

        start_offset, end_offset = value_offsets
        value_text = snippet[start_offset:end_offset]
        updated_value_text = value_text.replace(old_prompt, replacement_prompt, 1)
        if updated_value_text == value_text:
            return None
        return snippet[:start_offset] + updated_value_text + snippet[end_offset:]

    @staticmethod
    def _rewrite_prompt_assignment_value(
        snippet: str,
        replacement_prompt: str,
    ) -> str | None:
        tokens = BaseSystemPromptInjection._tokenize_snippet(snippet)
        value_span = BaseSystemPromptInjection._find_prompt_value_span(tokens)
        if value_span is None:
            return None

        span_start_token, span_end_token = value_span
        line_offsets = BaseSystemPromptInjection._line_offsets(snippet)
        start_offset = BaseSystemPromptInjection._position_to_offset(
            line_offsets,
            tokens[span_start_token].start,
        )
        end_offset = BaseSystemPromptInjection._position_to_offset(
            line_offsets,
            tokens[span_end_token - 1].end,
        )
        return snippet[:start_offset] + repr(replacement_prompt) + snippet[end_offset:]

    @staticmethod
    def _tokenize_snippet(snippet: str) -> list[tokenize.TokenInfo]:
        tokens: list[tokenize.TokenInfo] = []
        token_stream = tokenize.generate_tokens(io.StringIO(snippet).readline)

        while True:
            try:
                token = next(token_stream)
            except StopIteration:
                break
            except tokenize.TokenError:
                break
            tokens.append(token)
        return tokens

    @staticmethod
    def _find_prompt_value_offsets(snippet: str) -> tuple[int, int] | None:
        tokens = BaseSystemPromptInjection._tokenize_snippet(snippet)
        value_span = BaseSystemPromptInjection._find_prompt_value_span(tokens)
        if value_span is None:
            return None

        line_offsets = BaseSystemPromptInjection._line_offsets(snippet)
        span_start_token, span_end_token = value_span
        start_offset = BaseSystemPromptInjection._position_to_offset(
            line_offsets,
            tokens[span_start_token].start,
        )
        end_offset = BaseSystemPromptInjection._position_to_offset(
            line_offsets,
            tokens[span_end_token - 1].end,
        )
        return start_offset, end_offset

    @staticmethod
    def _find_prompt_value_span(
        tokens: list[tokenize.TokenInfo],
    ) -> tuple[int, int] | None:
        prompt_names = set(prompt_variable_name_options)

        for index, token in enumerate(tokens):
            if (
                token.type == tokenize.NAME
                and BaseSystemPromptInjection._is_prompt_like_name(token.string, prompt_names)
            ):
                value_span = BaseSystemPromptInjection._find_assignment_value_span(
                    tokens,
                    index + 1,
                )
                if value_span is not None:
                    return value_span

            if token.type != tokenize.STRING:
                continue
            try:
                key_value = ast.literal_eval(token.string)
            except (ValueError, SyntaxError):
                continue
            if (
                not isinstance(key_value, str)
                or not BaseSystemPromptInjection._is_prompt_like_name(
                    key_value,
                    prompt_names,
                )
            ):
                continue

            value_span = BaseSystemPromptInjection._find_mapping_value_span(
                tokens,
                index + 1,
            )
            if value_span is not None:
                return value_span

        return None

    @staticmethod
    def _is_prompt_like_name(name: str | None, prompt_names: set[str]) -> bool:
        if not isinstance(name, str):
            return False
        name_lower = name.lower()
        return (
            name in prompt_names
            or name_lower.endswith("prompt")
            or name_lower.endswith("prompt_template")
        )

    @staticmethod
    def _find_assignment_value_span(
        tokens: list[tokenize.TokenInfo],
        start_index: int,
    ) -> tuple[int, int] | None:
        return BaseSystemPromptInjection._find_delimited_value_span(
            tokens=tokens,
            start_index=start_index,
            delimiter="=",
        )

    @staticmethod
    def _find_mapping_value_span(
        tokens: list[tokenize.TokenInfo],
        start_index: int,
    ) -> tuple[int, int] | None:
        return BaseSystemPromptInjection._find_delimited_value_span(
            tokens=tokens,
            start_index=start_index,
            delimiter=":",
        )

    @staticmethod
    def _find_delimited_value_span(
        tokens: list[tokenize.TokenInfo],
        start_index: int,
        delimiter: str,
    ) -> tuple[int, int] | None:
        index = start_index
        depth = 0

        while index < len(tokens):
            token = tokens[index]
            if token.type == tokenize.OP:
                if token.string in "([{":
                    depth += 1
                elif token.string in ")]}":
                    depth = max(0, depth - 1)
                elif token.string == delimiter and depth == 0:
                    index += 1
                    break
                elif token.string == "," and depth == 0:
                    return None
            elif token.type in (tokenize.NEWLINE, tokenize.NL) and depth == 0:
                return None
            index += 1
        else:
            return None

        return BaseSystemPromptInjection._find_value_expression_span(tokens, index)

    @staticmethod
    def _find_value_expression_span(
        tokens: list[tokenize.TokenInfo],
        start_index: int,
    ) -> tuple[int, int] | None:
        index = start_index
        span_start: int | None = None
        span_end: int | None = None
        depth = 0

        while index < len(tokens):
            token = tokens[index]
            if token.type in (tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT):
                index += 1
                continue

            if span_start is None:
                if token.type in (tokenize.NEWLINE, tokenize.NL):
                    return None
                span_start = index

            if token.type == tokenize.OP:
                if token.string in "([{":
                    depth += 1
                elif token.string in ")]}":
                    if depth > 0:
                        depth -= 1
                elif token.string == "," and depth == 0:
                    break
            elif token.type in (tokenize.NEWLINE, tokenize.NL) and depth == 0:
                break

            span_end = index + 1
            index += 1

        if span_start is None or span_end is None:
            return None
        return span_start, span_end

    @staticmethod
    def _line_offsets(source: str) -> list[int]:
        offsets = [0]
        for line in source.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        return offsets

    @staticmethod
    def _position_to_offset(line_offsets: list[int], position: tuple[int, int]) -> int:
        line, column = position
        return line_offsets[line - 1] + column
