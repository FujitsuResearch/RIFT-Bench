import os
from langchain_openai import AzureChatOpenAI
import json
from typing import Any, Union, List, Optional
from node_spec.structure_schema import *
import ast
import re
import keyword
import textwrap

LineSelector = Union[int, tuple[int, int], list[int], list[tuple[int, int]]]


def unwrap_single_key_dict(value):
    if isinstance(value, dict) and len(value) == 1:
        return next(iter(value.values()))
    return value


def sanitize_function_name(name: str) -> str:
    """
    Convert arbitrary text into a safe Python function identifier.
    """
    sanitized = re.sub(r"[^0-9a-zA-Z_]+", "_", (name or "").strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")

    if not sanitized:
        sanitized = "tool"
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    if keyword.iskeyword(sanitized):
        sanitized = f"{sanitized}_"

    return sanitized



def build_args_schema_class(
    inputs: List["InputPort"], class_name: str = "ArgsSchema"
) -> str:
    """
    Generate a Pydantic args_schema class definition from InputPort objects.
    """

    def map_dtype(dtype: str) -> str:
        """Map InputPort dtype to Python type annotation."""
        if not dtype:
            return "Any"

        dtype = dtype.strip()

        # Handle array syntax, e.g. string[], Message[]
        if dtype.endswith("[]"):
            inner = map_dtype(dtype[:-2])
            return f"List[{inner}]"

        mapping = {
            "string": "str",
            "boolean": "bool",
            "bool": "bool",
            "int": "int",
            "integer": "int",
            "float": "float",
            "number": "float",
            "object": "dict",
            "dict": "dict",
            "any": "Any",
        }

        return mapping.get(dtype.lower(), "Any")

    def safe_field_name(name: str) -> str:
        """Ensure generated field name is a valid Python identifier."""
        if not name or not name.isidentifier() or keyword.iskeyword(name):
            return f"{name}_"
        return name

    lines = [
        "from typing import Any, List, Optional",
        "from pydantic import BaseModel, Field",
        "",
        f"class {class_name}(BaseModel):",
    ]

    if not inputs:
        lines.append("    pass")
        return "\n".join(lines)

    for inp in inputs:
        field_name = safe_field_name(inp.name)
        py_type = map_dtype(inp.dtype)

        if not inp.required:
            py_type = f"Optional[{py_type}]"

        if inp.required and inp.default is None:
            default_repr = "..."
        elif not inp.required and inp.default is None:
            default_repr = "None"
        else:
            default_repr = repr(inp.default)

        field_args = [default_repr]

        if inp.description:
            field_args.append(f"description={inp.description!r}")

        field_str = f"Field({', '.join(field_args)})"

        lines.append(f"    {field_name}: {py_type} = {field_str}")

    return "\n".join(lines)


def normalize_line_numbers(line_numbers: LineSelector) -> list[int]:
    """
    Normalize line selectors into a sorted list of line numbers.

    Args:
        line_numbers (LineSelector): Single line number or mixed line/range selectors.

    Return:
        list[int]: Sorted list of concrete line numbers.
    """
    if isinstance(line_numbers, int):
        return [line_numbers]

    # single range
    if (
        isinstance(line_numbers, (tuple, list))
        and len(line_numbers) == 2
        and all(isinstance(x, int) for x in line_numbers)
    ):
        start, end = line_numbers
        return [start, end]


def insert_line_in_content(
    content: str,
    line_number: int,
    code_line: str,
) -> str:
    """
    Insert code into content at the specified 1-based line_number.

    Rules:
    - line_number is clamped to valid content bounds
    - indentation is inferred primarily from upward context
    - if previous non-blank line ends with ':', indent one level deeper
    - preserves relative indentation inside multi-line code_line
    - preserves existing newline style when possible

    Args:
        content: Original file content.
        line_number: 1-based line index where code is inserted.
        code_line: Code to insert.

    Returns:
        Updated file content after insertion.
    """

    INDENT = "    "

    def is_non_blank(line: str) -> bool:
        return bool(line.strip())

    def leading_ws(s: str) -> str:
        return s[: len(s) - len(s.lstrip())]

    def common_prefix_whitespace(values: list[str]) -> str:
        if not values:
            return ""
        prefix = values[0]
        for value in values[1:]:
            while not value.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        return prefix

    lines = content.splitlines(keepends=True)

    # Preserve file newline style if possible
    newline = "\n"
    for line in lines:
        if line.endswith("\r\n"):
            newline = "\r\n"
            break
        if line.endswith("\n"):
            newline = "\n"
            break

    # Normalize input code
    code_line = code_line.rstrip("\r\n")

    # Clamp insertion index to valid bounds
    # 1-based line_number -> 0-based insertion index
    index = max(0, min(line_number - 1, len(lines)))

    # Find previous non-blank line
    prev_line = None
    prev_indent = ""
    for i in range(index - 1, -1, -1):
        if is_non_blank(lines[i]):
            prev_line = lines[i]
            prev_indent = leading_ws(lines[i])
            break

    # Find next non-blank line
    next_line = None
    next_indent = ""
    for i in range(index, len(lines)):
        if is_non_blank(lines[i]):
            next_line = lines[i]
            next_indent = leading_ws(lines[i])
            break

    # Infer target indentation
    if prev_line is not None:
        target_indent = prev_indent
        if prev_line.rstrip().endswith(":"):
            target_indent += INDENT

        # Bound against next line to avoid over-indentation
        if next_line is not None:
            target_indent = min(target_indent, next_indent, key=len)

    elif next_line is not None:
        target_indent = next_indent
    else:
        target_indent = ""

    # Split inserted block into lines
    raw_insert_lines = code_line.split("\n")
    if not raw_insert_lines:
        raw_insert_lines = [""]

    # Preserve relative indentation inside inserted block
    non_blank_insert_lines = [l for l in raw_insert_lines if l.strip()]
    base_insert_indent = common_prefix_whitespace(
        [leading_ws(l) for l in non_blank_insert_lines]
    )

    insert_lines = []
    for raw in raw_insert_lines:
        stripped = raw[len(base_insert_indent):] if raw.startswith(base_insert_indent) else raw
        insert_lines.append(target_indent + stripped + newline)

    # If appending to content whose last line has no newline, add one first
    if index == len(lines) and lines and not lines[-1].endswith(("\n", "\r\n")):
        lines[-1] += newline

    # Insert at bounded position
    lines[index:index] = insert_lines

    return "".join(lines)

def find_first_decorator_line(source: str) -> Optional[int]:
    """
    Return the 1-based line number of the first decorator that looks like a
    tool decorator, e.g. @tool(...) or @x.tool(...).

    Returns None if no matching decorator is found.
    """
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                return dec.lineno - 1
    return None

def parse_snippet_with_indent_support(snippet: str) -> ast.AST | None:
    """
    Parse Python snippet, tolerating leading indentation from nested scopes.

    Return:
        ast.AST | None: Parsed tree, or None when snippet is not parseable.
    """
    try:
        return ast.parse(snippet)
    except (IndentationError, SyntaxError):
        normalized_snippet = textwrap.dedent(snippet)
        if not normalized_snippet.strip():
            return None
        try:
            return ast.parse(normalized_snippet)
        except (IndentationError, SyntaxError):
            return None


def get_first_executable_line(code: str):
    tree = parse_snippet_with_indent_support(code)
    if tree is None:
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "__init__":
                continue

            body = node.body

            if not body:
                return None

            first_stmt = body[0]

            # Check if first statement is a docstring
            if (
                isinstance(first_stmt, ast.Expr)
                and isinstance(first_stmt.value, ast.Constant)
                and isinstance(first_stmt.value.value, str)
            ):
                # Skip docstring if there is another statement
                if len(body) > 1:
                    return body[1].lineno
                else:
                    return None

            return first_stmt.lineno

    return None

def indent_snippet(snippet: str, spaces: int = 4) -> str:
    """
    Indent each non-empty line in a code snippet by a given number of spaces.
    """
    indent = " " * spaces
    return "\n".join(
        (indent + line if line.strip() else line)
        for line in snippet.splitlines()
    )

def replace_lines_with_snippet(
        lines: List[str],
        snippet: str,
        line_numbers: int | list[int] | tuple[int, int],
    ) -> List[str]:
        """
        Replace specific line(s) in file content with a given code snippet,
        preserving indentation from the original content.

        Special cases:
        - line_numbers == 0  → insert at beginning of file
        - line_numbers == -1 → insert at end of file

        Args:
            lines (List[str]): Original file content lines.
            snippet (str): Replacement snippet.
            line_numbers (int | list[int] | tuple[int, int]): Target line selector.

        Return:
            List[str]: Updated file content lines.
        """
        # --- Handle special insertion modes ---
        if isinstance(line_numbers, int):
            if line_numbers == 0:
                start_idx = end_idx = 0
                base_line = lines[0] if lines else ""
            elif line_numbers == -1:
                start_idx = end_idx = len(lines)
                base_line = lines[-1] if lines else ""
            else:
                if line_numbers < 1:
                    raise ValueError("line_numbers must be >= 1, 0, or -1")
                start = end = line_numbers
                start_idx = start - 1
                end_idx = start
                temp_idx = start_idx
                base_line = lines[start_idx] if start_idx < len(lines) else ""
                while base_line == "\n" and temp_idx > 0:
                    temp_idx -= 1
                    base_line = lines[temp_idx]

        else:
            if len(line_numbers) != 2:
                raise ValueError("line_numbers must be an int or [start, end]")
            start, end = line_numbers
            if start < 1:  # or end < start:
                raise ValueError("Invalid line range")
            start_idx = start - 1
            end_idx = end
            temp_idx = start_idx
            base_line = lines[start_idx] if start_idx < len(lines) else ""
            while base_line == "\n" and temp_idx > 0:
                temp_idx -= 1
                base_line = lines[temp_idx]

        # --- Infer indentation ---
        indent_match = re.match(r"[ \t]*", base_line)
        base_indent = indent_match.group(0) if indent_match else ""

        # --- Normalize snippet ---
        snippet_lines = snippet.splitlines()
        if snippet and not snippet_lines[-1].endswith("\n"):
            snippet_lines[-1] += "\n"

        # --- Apply indentation ---
        indented_snippet_lines = []
        for line in snippet_lines:
            if line.strip() == "":
                indented_snippet_lines.append("\n")
            else:
                indented_snippet_lines.append(base_indent + line)

        return lines[:start_idx] + indented_snippet_lines + lines[end_idx:]


def find_snippet_lines(file_content: str, snippet: str) -> int | tuple[int, int] | None:
    """
    Find start and end line numbers of a code snippet in a Python file.

    Matching rules:
    - Lines are compared using strip()
    - If snippet has <= 10 lines, match entire snippet
    - If snippet has > 10 lines:
        * First 5 lines determine start_line
        * Last 5 lines determine end_line

    Args:
        file_content (str): Full source file content.
        snippet (str): Snippet to locate.

    Return:
        int | tuple[int, int] | None: Single line, line range tuple, or `None` when not found.
    """

    # Normalize snippet
    snippet_lines = [line.strip() for line in snippet.splitlines()]
    if not snippet_lines:
        return None

    file_lines = [line.strip() for line in file_content.splitlines()]

    n_snippet = len(snippet_lines)
    n_file = len(file_lines)

    # Helper to find first occurrence of a block
    def find_block(block: list[str]) -> int | None:
        block_len = len(block)
        for i in range(n_file - block_len + 1):
            if file_lines[i : i + block_len] == block:
                return i + 1  # 1-based
        return None

    # Short snippet → full match
    if "<<<Rest of snippet>>>" not in snippet_lines:
        start = find_block(snippet_lines)
        if start is None:
            return None
        end = start + n_snippet - 1
        return start if start == end else (start, end)

    mid_index = snippet_lines.index("<<<Rest of snippet>>>")

    # Long snippet → head/tail strategy
    head = snippet_lines[:mid_index]
    tail = snippet_lines[mid_index + 1 :]

    start_line = find_block(head)
    if start_line is None:
        return None

    end_line = find_block(tail)
    if end_line is None:
        return None

    end_line = end_line + len(tail) - 1

    return start_line if start_line == end_line else (start_line, end_line)


def _build_emulation_llm() -> AzureChatOpenAI:
    """
    Build the LLM client used for MCP-snippet localization and rewriting.
    """
    api_key = os.getenv("AZURE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing Azure OpenAI API key. Expected AZURE_API_KEY or AZURE_OPENAI_API_KEY."
        )

    return AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_TE"),
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_TE"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION_TE"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_TE"),
        api_key=api_key,
        temperature=0,
    )


def _invoke_emulation_llm(prompt: str) -> str:
    """
    Invoke the MCP-emulation LLM and return normalized text content.
    """
    response = _build_emulation_llm().invoke(prompt)
    content = getattr(response, "content", "") or ""

    if isinstance(content, list):
        normalized_parts = []
        for item in content:
            if isinstance(item, dict):
                normalized_parts.append(item.get("text", ""))
            else:
                normalized_parts.append(str(item))
        content = "".join(normalized_parts)

    return str(content).strip()


def _strip_code_fences(text: str) -> str:
    """
    Remove surrounding markdown code fences when the model returns them.
    """
    text = (text or "").strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1]).strip()
    return text


def _extract_tagged_payload(text: str, tag: str) -> str:
    """
    Extract a tagged payload such as <snippet>...</snippet> from model output.
    """
    text = _strip_code_fences(text)
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
    if match:
        return match.group(1).strip("\n")
    return text.strip("\n")


def _line_span_from_selector(
    selector: int | tuple[int, int] | list[int] | None,
) -> tuple[int, int] | None:
    """
    Convert a line selector into an inclusive `(start, end)` span.
    """
    if selector is None:
        return None
    if isinstance(selector, int):
        return selector, selector
    if isinstance(selector, (tuple, list)) and len(selector) == 2:
        return int(selector[0]), int(selector[1])
    return None


def _slice_source_by_lines(source: str, start_line: int, end_line: int) -> str:
    """
    Return the exact source slice for the provided inclusive line span.
    """
    lines = source.splitlines(keepends=True)
    return "".join(lines[start_line - 1 : end_line]).rstrip("\r\n")


def _realign_candidate_to_source(source: str, candidate: str) -> str | None:
    """
    Realign an LLM-produced candidate to the exact substring present in `source`.
    """
    candidate = _strip_code_fences(candidate).strip("\n")
    if not candidate:
        return None

    if candidate in source:
        return candidate

    selector = find_snippet_lines(source, candidate)
    span = _line_span_from_selector(selector)
    if span is None:
        return None

    return _slice_source_by_lines(source, *span)


def format_expected_io_for_prompt(exp_io: list[dict]) -> str:
    """
    Convert expected tool inputs/outputs into a clean system-prompt string.

    Args:
        exp_io (list[dict]): Expected inputs or outputs extracted from node metadata.

    Return:
        str: Prompt-friendly formatted text block.
    """

    lines = []
    if exp_io:
        for item in exp_io:
            lines.extend([
                f"- name: {item.name}",
                f"  type: {item.dtype}",
                f"  specification: {item.description}",
            ])

    return "\n".join(lines)


def _extend_substring_to_full_lines(source: str, candidate: str) -> str | None:
    """
    Extend a matched substring to its enclosing full source lines, preserving indentation.
    """
    if not candidate:
        return None

    start = source.find(candidate)
    if start == -1:
        return None

    end = start + len(candidate)
    line_start = source.rfind("\n", 0, start) + 1
    line_end = source.find("\n", end)
    if line_end == -1:
        line_end = len(source)

    return source[line_start:line_end].rstrip("\r\n")

def find_external_mcp_sub_snippet(
    mcp_node_assignment_name: str,
    snippet: str,
    description: str = "",
) -> str:
    """
    Use the LLM to extract the exact sub-snippet that constructs one external MCP.
    """

    prompt = f"""
You are locating one external MCP constructor inside a Python snippet.

Target MCP name: {mcp_node_assignment_name}
Target MCP description: {description or "<unknown>"}

Important bias:
- In most codebases, an external MCP is defined through an HTTP-based client/protocol.
- Prefer constructors that use HTTP concepts such as StreamableHttpServerParams, HttpServerParams, URL arguments, headers, bearer/API-key auth, or other network endpoint configuration.
- Only choose a non-HTTP constructor if the snippet clearly shows that the target external MCP is not the HTTP-based one.

Return ONLY the smallest exact contiguous substring from the source snippet that
constructs the target external MCP instance.

Rules:
- Copy the substring verbatim from the source snippet.
- Do not rewrite, normalize, or explain anything.
- Prefer the outer constructor call that a caller would replace.
- Wrap the final answer in <snippet>...</snippet>.

Source snippet:
```python
{snippet}
```
""".strip()

    raw_response = _invoke_emulation_llm(prompt)
    candidate = _extract_tagged_payload(raw_response, "snippet")
    realigned_candidate = _realign_candidate_to_source(snippet, candidate)
    if realigned_candidate is not None:
        expanded_candidate = _extend_substring_to_full_lines(snippet, realigned_candidate)
        if expanded_candidate is not None:
            return expanded_candidate
        return realigned_candidate
    raise ValueError(
        f"Could not locate external MCP constructor inside snippet for '{mcp_node_assignment_name}'."
    )


def find_external_mcp_sub_snippet_block(
    external_mcp_sub_snippet: str,
    snippet: str,
) -> str:
    """
    Use AST line spans to extend a localized constructor to its enclosing statement block.
    """
    selector = find_snippet_lines(snippet, external_mcp_sub_snippet)
    span = _line_span_from_selector(selector)
    if span is None:
        raise ValueError("Could not locate the external MCP sub-snippet inside its parent snippet.")

    target_start, target_end = span
    tree = parse_snippet_with_indent_support(snippet)
    if tree is None:
        return external_mcp_sub_snippet

    candidate_nodes: list[ast.stmt] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt):
            continue
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            continue
        if node.lineno <= target_start and node.end_lineno >= target_end:
            candidate_nodes.append(node)

    if not candidate_nodes:
        return external_mcp_sub_snippet

    best_node = min(
        candidate_nodes,
        key=lambda node: (
            node.end_lineno - node.lineno,
            node.lineno,
            getattr(node, "col_offset", 0),
        ),
    )
    return _slice_source_by_lines(snippet, best_node.lineno, best_node.end_lineno)


def create_external_mcp_sub_snippet_block(
    external_mcp_sub_snippet_block: str,
    external_mcp_sub_snippet: str,
    mcp_node_assignment_name: str,
    save_path: str,
    description: str = "",
) -> str:
    """
    Use the LLM to rewrite the external MCP block into a local stdio MCP block.
    """
    prompt = f"""
You are rewriting one Python block that currently constructs an external MCP client.

Target MCP name: {mcp_node_assignment_name}
Target MCP description: {description or "<unknown>"}
Replacement server file path: {save_path}

Rewrite the block so that ONLY the target external MCP constructor becomes a local
stdio MCP that points to {save_path!r}. Keep all unrelated code unchanged.

Rules:
- Return ONLY the rewritten Python block.
- Preserve the original surrounding structure.
- Keep the result valid Python in the same context.
- Prefer existing local patterns already present in the block.
- If the block already uses `PYTHON_EXE` and `base_dir`, reuse them.
- Wrap the final answer in <updated_block>...</updated_block>.

Original target constructor:
```python
{external_mcp_sub_snippet}
```

Original block:
```python
{external_mcp_sub_snippet_block}
```
""".strip()

    raw_response = _invoke_emulation_llm(prompt)
    candidate = _extract_tagged_payload(raw_response, "updated_block")
    realigned_candidate = _strip_code_fences(candidate).strip("\n")
    if save_path in realigned_candidate and parse_snippet_with_indent_support(realigned_candidate) is not None:
        return realigned_candidate

    raise ValueError(
        f"Could not rewrite external MCP block for '{mcp_node_assignment_name}'."
    )


def extract_signature(function_name: str, snippet: str) -> str:
    """
    Extract the function signature from a code snippet.

    Args:
        function_name (str): Name of the function to locate.
        snippet (str): Python source code.

    Return:
        str: Signature string (for example, `def foo(a, b=1) -> int`).

    Raises:
        ValueError: If function not found.
    """

    tree = ast.parse(snippet)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return _build_signature(node)

    raise ValueError(f"Function '{function_name}' not found in snippet.")


def _build_signature(func_node: ast.FunctionDef) -> str:
    """
    Build a function signature string from an AST function node.

    Args:
        func_node (ast.FunctionDef): Function node to convert.

    Return:
        str: Reconstructed function signature.
    """
    args = []

    # Positional + keyword args
    total = func_node.args.args
    defaults = func_node.args.defaults
    default_offset = len(total) - len(defaults)

    for i, arg in enumerate(total):
        name = arg.arg

        # Type annotation
        if arg.annotation:
            name += f": {ast.unparse(arg.annotation)}"

        # Default value
        if i >= default_offset:
            default_val = defaults[i - default_offset]
            name += f"={ast.unparse(default_val)}"

        args.append(name)

    # *args
    if func_node.args.vararg:
        name = f"*{func_node.args.vararg.arg}"
        if func_node.args.vararg.annotation:
            name += f": {ast.unparse(func_node.args.vararg.annotation)}"
        args.append(name)

    # Keyword-only args
    for arg, default in zip(func_node.args.kwonlyargs, func_node.args.kw_defaults):
        name = arg.arg
        if arg.annotation:
            name += f": {ast.unparse(arg.annotation)}"
        if default:
            name += f"={ast.unparse(default)}"
        args.append(name)

    # **kwargs
    if func_node.args.kwarg:
        name = f"**{func_node.args.kwarg.arg}"
        if func_node.args.kwarg.annotation:
            name += f": {ast.unparse(func_node.args.kwarg.annotation)}"
        args.append(name)

    signature = f"def {func_node.name}({', '.join(args)})"

    # Return annotation
    if func_node.returns:
        signature += f" -> {ast.unparse(func_node.returns)}"

    return signature


def extract_variable_names_from_signature_ast(signature: str) -> list[str]:
    """
    Extract argument variable names from a function signature string.

    Args:
        signature (str): Function signature text without trailing colon/body.

    Return:
        list[str]: Ordered argument names from positional, vararg, kw-only, and kwargs.
    """
    tree = ast.parse(signature + ":\n    pass")

    func = tree.body[0]
    args = []

    for a in func.args.args:
        args.append(a.arg)

    if func.args.vararg:
        args.append(func.args.vararg.arg)

    for a in func.args.kwonlyargs:
        args.append(a.arg)

    if func.args.kwarg:
        args.append(func.args.kwarg.arg)

    return args


def schema_to_signature(func_name: str, schema: dict[str, Any]) -> str:
    """
    Convert a JSON-schema-like object into a Python function signature.

    Args:
        func_name (str): Function name.
        schema (dict[str, Any]): JSON schema with `properties` and optional `required`.

    Return:
        str: Generated signature string.
    """
    props: dict = (schema or {}).get("properties", {}) or {}
    required = set((schema or {}).get("required", []) or [])

    def py_type(spec: dict) -> str:
        spec = spec or {}

        # enum -> Literal[...] (string form; caller can import Literal if desired)
        if "enum" in spec and isinstance(spec["enum"], list) and spec["enum"]:
            # Use repr() to ensure strings are quoted correctly
            items = ", ".join(repr(x) for x in spec["enum"])
            return f"Literal[{items}]"

        t = spec.get("type")

        # JSON Schema can use type as list: ["string","null"]
        if isinstance(t, list):
            non_null = [x for x in t if x != "null"]
            t = non_null[0] if non_null else "any"

        return {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "list",
            "object": "dict",
        }.get(t, "Any")

    def default_repr(spec: dict) -> str:
        spec = spec or {}
        if "default" not in spec:
            return "None"

        dv = spec.get("default")
        if dv is None:
            return "None"
        if isinstance(dv, str):
            return repr(dv)  # ensures quotes + escaping
        if isinstance(dv, bool):
            return "True" if dv else "False"
        if isinstance(dv, (int, float)):
            return str(dv)
        # lists/dicts/etc.
        return repr(dv)

    required_args: list[str] = []
    optional_args: list[str] = []

    # Keep deterministic ordering: required first (in schema order), then optional (schema order)
    for name, spec in props.items():
        ann = py_type(spec)
        if name in required:
            required_args.append(f"{name}: {ann}")
        else:
            optional_args.append(f"{name}: {ann} = {default_repr(spec)}")

    args = required_args + optional_args
    return f"{func_name}(" + ", ".join(args) + ")"


def build_function_signature(func_name: str, inputs: list[InputPort]) -> str:
    """
    Build a Python function signature from InputPort definitions.

    Args:
        func_name (str): Name of the function.
        inputs (list[InputPort]): List of input port objects.

    Return:
        str: Function signature string.
    """

    # Mapping InputPort dtypes to Python types
    type_map = {
        "string": "str",
        "int": "int",
        "integer": "int",
        "float": "float",
        "boolean": "bool",
        "bool": "bool",
        "number": "float",
        "object": "dict",
        "array": "list",
    }

    required = []
    optional = []

    for p in inputs:
        py_type = type_map.get(p.dtype.lower(), "Any")

        if p.required:
            required.append(f"{p.name}: {py_type}")
        else:
            default_val = repr(p.default)
            optional.append(f"{p.name}: {py_type} = {default_val}")

    params = required + optional
    params_str = ", ".join(params)

    return f"def {func_name}({params_str})"
