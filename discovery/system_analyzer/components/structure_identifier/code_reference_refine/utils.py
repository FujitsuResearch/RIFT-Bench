import ast
import os
import pprint
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Literal

try:
    from ..model_client import call_model
    from ..global_utils import build_parent_child_maps, extract_var_name, parse_json_loose
except ImportError:
    from model_client import call_model
    from global_utils import build_parent_child_maps, extract_var_name, parse_json_loose

def find_snippet_lines(file_path: str, snippet: str) -> int | tuple[int, int] | None:
    """
    Find start and end line numbers of a code snippet in a Python file.

    Matching rules:
    - Lines are compared using strip()
    - The full normalized snippet must match contiguously

    Args:
        file_path (str): path of the file.
        snippet (str): Snippet to locate.

    Return:
        int | tuple[int, int] | None: Single line, line range tuple, or `None` when not found.
    """
    file_content = Path(file_path).read_text(encoding="utf-8")

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
            if file_lines[i:i + block_len] == block:
                return i + 1  # 1-based
        return None

    start = find_block(snippet_lines)
    if start is None:
        return None
    end = start + n_snippet - 1
    return start if start == end else [start, end]


def flex_find_snippet_lines(file_path: str, snippet: str) -> int | list[int] | None:
    """Locate a snippet by exact match first, then by a conservative line-based fuzzy match."""

    exact_match = find_snippet_lines(file_path, snippet)
    if exact_match is not None:
        return exact_match

    file_lines_raw = Path(file_path).read_text(encoding="utf-8").splitlines()
    snippet_lines_raw = snippet.splitlines()

    def normalize_lines(lines: list[str]) -> tuple[list[str], list[int]]:
        """Drop blank lines and keep the original 1-based line numbers for the remaining content."""
        normalized: list[str] = []
        line_numbers: list[int] = []
        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            normalized.append(stripped)
            line_numbers.append(idx)
        return normalized, line_numbers

    # Compare normalized non-empty lines so minor whitespace differences do not break grounding.
    file_lines, file_line_numbers = normalize_lines(file_lines_raw)
    snippet_lines, _ = normalize_lines(snippet_lines_raw)

    if len(file_lines) < 2 or len(snippet_lines) < 2:
        return None

    matcher = SequenceMatcher(a=file_lines, b=snippet_lines, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size > 0]
    if not blocks:
        return None

    # Score how much of the requested snippet was recovered, and whether the match
    # stays anchored at the start and/or end of the snippet.
    matched_lines = sum(block.size for block in blocks)
    longest_block = max(block.size for block in blocks)
    coverage = matched_lines / len(snippet_lines)
    start_anchor = any(block.b == 0 for block in blocks)
    end_anchor = any(block.b + block.size == len(snippet_lines) for block in blocks)

    # Keep the flexible matcher conservative: require meaningful contiguous
    # overlap and enough overall coverage before returning a span.
    is_high_confidence = (
        (coverage >= 0.8 and longest_block >= 2)
        or (coverage >= 0.65 and longest_block >= 3 and (start_anchor or end_anchor))
        or (coverage >= 0.5 and longest_block >= 4 and start_anchor and end_anchor)
    )
    if not is_high_confidence:
        return None

    span_start_idx = blocks[0].a
    span_end_idx = blocks[-1].a + blocks[-1].size - 1
    span_len = span_end_idx - span_start_idx + 1
    max_reasonable_span = max(len(snippet_lines) * 2, longest_block + 12)
    if span_len > max_reasonable_span:
        return None

    # Convert the normalized match span back to original file line numbers.
    start_line = file_line_numbers[span_start_idx]
    end_line = file_line_numbers[span_end_idx]
    return start_line if start_line == end_line else [start_line, end_line]


def extract_snippet(file_path: str, line: int | list[int] | tuple[int, int] | None) -> str | None:
    """
    Extract a snippet from `file_path` using the line value returned by
    `flex_find_snippet_lines`.

    `line` can be:
    - int: a single 1-based line number
    - [start, end] or (start, end): inclusive 1-based line span
    - None: returns None
    """

    if line is None:
        return None

    file_lines = Path(file_path).read_text(encoding="utf-8").splitlines()

    if isinstance(line, int):
        if line < 1 or line > len(file_lines):
            return None
        return file_lines[line - 1]

    if isinstance(line, (list, tuple)) and len(line) == 2:
        start_line, end_line = line

        if not isinstance(start_line, int) or not isinstance(end_line, int):
            return None
        if start_line < 1 or end_line < start_line or end_line > len(file_lines):
            return None

        return "\n".join(file_lines[start_line - 1:end_line])

    return None

LineRef = int | list[int]

def extract_enclosing_code_ast(
    file_path: str,
    snippet_line: int | list[int] | tuple[int, int] | None,
    mode: Literal["statement", "function", "auto"] = "auto",
) -> tuple[LineRef, str] | None:
    """
    Expand a snippet line/span to an enclosing AST node.

    mode:
        - "statement": return smallest enclosing statement
        - "function": return innermost enclosing function, including decorators
        - "auto": return function if snippet is part of a def/async def header/body,
                  otherwise return smallest enclosing statement

    Returns:
        (line_or_span, source)
    """

    if snippet_line is None:
        return None

    source = Path(file_path).read_text(encoding="utf-8")
    lines = source.splitlines()

    if isinstance(snippet_line, int):
        snippet_start = snippet_end = snippet_line
    elif isinstance(snippet_line, (list, tuple)) and len(snippet_line) == 2:
        snippet_start, snippet_end = snippet_line
    else:
        return None

    if (
        not isinstance(snippet_start, int)
        or not isinstance(snippet_end, int)
        or snippet_start < 1
        or snippet_end < snippet_start
        or snippet_end > len(lines)
    ):
        return None

    def is_ignorable_boundary_line(line_text: str) -> bool:
        stripped = line_text.strip()
        return not stripped or stripped.startswith("#")

    # Normalize the requested span by trimming boundary-only blank/comment lines.
    # This lets ranges like [108, 114] still resolve when 114 is just whitespace.
    while snippet_start <= snippet_end and is_ignorable_boundary_line(lines[snippet_start - 1]):
        snippet_start += 1
    while snippet_start <= snippet_end and is_ignorable_boundary_line(lines[snippet_end - 1]):
        snippet_end -= 1

    if snippet_start > snippet_end:
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    def line_repr(start: int, end: int) -> LineRef:
        return start if start == end else [start, end]

    def source_for(start: int, end: int) -> str:
        return "\n".join(lines[start - 1:end])

    def decorator_start(func: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        if func.decorator_list:
            return min(dec.lineno for dec in func.decorator_list)
        return func.lineno

    functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    statements: list[ast.stmt] = []

    for node in ast.walk(tree):
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            continue

        node_start = node.lineno
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node_start = decorator_start(node)

        if node_start <= snippet_start and snippet_end <= node.end_lineno:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node)
            if isinstance(node, ast.stmt):
                statements.append(node)

    def extract_function():
        if not functions:
            return None

        func = max(
            functions,
            key=lambda n: (n.lineno, -n.end_lineno),
        )

        start = decorator_start(func)
        end = func.end_lineno
        return line_repr(start, end), source_for(start, end)

    def extract_statement():
        if not statements:
            return None

        stmt = min(
            statements,
            key=lambda n: (n.end_lineno - n.lineno, n.lineno),
        )

        start = stmt.lineno
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = decorator_start(stmt)
        end = stmt.end_lineno
        return line_repr(start, end), source_for(start, end)

    if mode == "function":
        return extract_function()

    if mode == "statement":
        return extract_statement()

    if mode == "auto":
        # If snippet falls within a function (header/body/decorators), return
        # the innermost enclosing function.
        function_result = extract_function()
        if function_result is not None:
            return function_result
        return extract_statement()

    raise ValueError("mode must be 'statement', 'function', or 'auto'")

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


def _find_snippet_lines_in_text(source: str, snippet: str) -> int | tuple[int, int] | None:
    """
    Find the 1-based line span of `snippet` within raw source text.
    """
    snippet_lines = [line.strip() for line in snippet.splitlines()]
    if not snippet_lines:
        return None

    source_lines = [line.strip() for line in source.splitlines()]
    n_snippet = len(snippet_lines)
    n_source = len(source_lines)

    def find_block(block: list[str]) -> int | None:
        block_len = len(block)
        for idx in range(n_source - block_len + 1):
            if source_lines[idx:idx + block_len] == block:
                return idx + 1
        return None

    start = find_block(snippet_lines)
    if start is None:
        return None

    end = start + n_snippet - 1
    return start if start == end else (start, end)


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
    return "".join(lines[start_line - 1:end_line]).rstrip("\r\n")


def _realign_candidate_to_source(source: str, candidate: str) -> str | None:
    """
    Realign an LLM-produced candidate to the exact substring present in `source`.
    """
    candidate = _strip_code_fences(candidate).strip("\n")
    if not candidate:
        return None

    if candidate in source:
        return candidate

    selector = _find_snippet_lines_in_text(source, candidate)
    span = _line_span_from_selector(selector)
    if span is None:
        return None

    return _slice_source_by_lines(source, *span)


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
    model: str = "gpt-5.1-codex",
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

    payload = {
        "prompt": prompt,
        "temperature": 0,
    }
    raw_response = str(call_model(payload, model)).strip()
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


def _line_offsets(text: str) -> List[int]:
    starts = [0]
    for idx, ch in enumerate(text):
        if ch == "\n":
            starts.append(idx + 1)
    return starts


def _abs_pos(line_starts: List[int], lineno: int, col: int) -> int:
    if lineno <= 0:
        return 0
    base = line_starts[min(lineno - 1, len(line_starts) - 1)]
    return base + max(0, col)


def _render_code_reference_item(ref: Dict[str, Any], indent: int = 8) -> str:
    pad = " " * indent
    fields = [
        "kind",
        "other_kind_description",
        "file",
        "line",
        "snippet",
    ]
    args: List[str] = []
    for key in fields:
        if key not in ref:
            continue
        val = ref.get(key)
        if val is None:
            continue
        args.append(f"{key}={repr(val)}")
    if not args:
        return f"{pad}CodeReference()"
    return f"{pad}CodeReference(" + ", ".join(args) + ")"


def _render_code_references_value(code_refs: List[Dict[str, Any]], indent: int = 4) -> str:
    pad = " " * indent
    if not code_refs:
        return "[]"
    rendered_items = "\n".join(_render_code_reference_item(r, indent=indent + 4) + "," for r in code_refs)
    return "[\n" + rendered_items + f"\n{pad}]"


def _render_metadata_value(metadata: Dict[str, Any], indent: int = 4) -> str:
    if not metadata:
        return "{}"
    # Keep output as valid Python literal for NodeSpec source files.
    return pprint.pformat(metadata, width=100, sort_dicts=True)


def _metadata_additions_only(existing: Dict[str, Any], additions: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(existing)
    for k, v in additions.items():
        if k not in out:
            out[k] = v
            continue
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = _metadata_additions_only(
                out.get(k) if isinstance(out.get(k), dict) else {},
                v,
            )
    return out


def write_nodes_with_updated_code_references_preserve_source(
    *,
    input_path: Path,
    output_path: Path,
    nodes_by_var: Dict[str, Dict[str, Any]],
    var_order: List[str],
) -> None:
    """
    Preserve the original NodeSpec file structure and mutate:
    - `code_references` for top-level `var = NodeSpec(...)` assignments in var_order.
    - `metadata` by ADDITION ONLY (existing metadata keys/values are preserved).
    """
    # Read the original emitted nodes file and parse it so we can patch only the
    # relevant literal sections instead of regenerating the whole file.
    src = input_path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(src, filename=str(input_path))
    line_starts = _line_offsets(src)

    # Keep only vars actually present in output order.
    allowed_vars = set(var_order)
    assign_calls: Dict[str, ast.Call] = {}
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            continue
        var = stmt.targets[0].id
        if var not in allowed_vars:
            continue
        if isinstance(stmt.value, ast.Call):
            fn = stmt.value.func
            fname = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
            if fname == "NodeSpec":
                assign_calls[var] = stmt.value

    # Collect source-text replacements as (start, end, replacement_text) triples.
    replacements: List[Tuple[int, int, str]] = []
    for var in var_order:
        call = assign_calls.get(var)
        if call is None:
            continue
        node = nodes_by_var.get(var, {})
        refs = node.get("code_references") if isinstance(node.get("code_references"), list) else []
        normalized: List[Dict[str, Any]] = [r for r in refs if isinstance(r, dict)]
        rendered_refs = _render_code_references_value(normalized, indent=4)
        metadata_add = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}

        kw_ref = None
        for kw in call.keywords:
            if kw.arg == "code_references":
                kw_ref = kw
                break

        # Replace the existing code_references literal in-place when the field already exists.
        if kw_ref is not None and hasattr(kw_ref.value, "lineno") and hasattr(kw_ref.value, "end_lineno"):
            start = _abs_pos(line_starts, kw_ref.value.lineno, kw_ref.value.col_offset)
            end = _abs_pos(line_starts, kw_ref.value.end_lineno, kw_ref.value.end_col_offset)
            replacements.append((start, end, rendered_refs))

        # If missing, insert before closing ')' of the NodeSpec call.
        elif hasattr(call, "end_lineno") and hasattr(call, "end_col_offset"):
            insert_pos = _abs_pos(line_starts, call.end_lineno, call.end_col_offset) - 1
            if insert_pos < 0:
                continue
            insertion = f",\n    code_references={rendered_refs}\n"
            replacements.append((insert_pos, insert_pos, insertion))

        # Metadata additions-only merge:
        # - existing keys are never overwritten/removed
        # - new keys from nodes_by_var[var]["metadata"] are inserted
        if metadata_add:
            kw_meta = None
            for kw in call.keywords:
                if kw.arg == "metadata":
                    kw_meta = kw
                    break

            existing_meta: Dict[str, Any] = {}
            can_update_meta = True
            if kw_meta is not None:
                try:
                    parsed_meta = ast.literal_eval(kw_meta.value)
                    if isinstance(parsed_meta, dict):
                        existing_meta = parsed_meta
                    else:
                        can_update_meta = False
                except Exception:
                    can_update_meta = False

            # Merge only new metadata keys so previously emitted metadata is preserved.
            if can_update_meta:
                merged_meta = _metadata_additions_only(existing_meta, metadata_add)
                if merged_meta != existing_meta:
                    rendered_meta = _render_metadata_value(merged_meta, indent=4)
                    if kw_meta is not None and hasattr(kw_meta.value, "lineno") and hasattr(kw_meta.value, "end_lineno"):
                        m_start = _abs_pos(line_starts, kw_meta.value.lineno, kw_meta.value.col_offset)
                        m_end = _abs_pos(line_starts, kw_meta.value.end_lineno, kw_meta.value.end_col_offset)
                        replacements.append((m_start, m_end, rendered_meta))
                    elif hasattr(call, "end_lineno") and hasattr(call, "end_col_offset"):
                        insert_pos = _abs_pos(line_starts, call.end_lineno, call.end_col_offset) - 1
                        if insert_pos >= 0:
                            insertion = f",\n    metadata={rendered_meta}\n"
                            replacements.append((insert_pos, insert_pos, insertion))

    # Apply replacements from the end of the file backward so earlier offsets stay valid.
    out = src
    for start, end, val in sorted(replacements, key=lambda x: x[0], reverse=True):
        out = out[:start] + val + out[end:]

    # Write the patched source into the target output file.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(out, encoding="utf-8")

def bottom_up_var_order(var_order: List[str], nodes_by_var: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Return a leaves-to-root processing order based on node["nodes"].

    Nodes with no children are processed first. A parent becomes eligible only
    after all of its children were already emitted. If cycles exist, the
    remaining nodes are appended deterministically at the end.
    """
    ordered_vars = [v for v in var_order if v in nodes_by_var]
    valid = set(ordered_vars)
    rank = {v: i for i, v in enumerate(var_order)}

    # Build parent/child maps from both `nodes` and `tool_list` so all direct
    # children are processed before their parents.
    children_by_parent, parents_by_child = build_parent_child_maps(ordered_vars, nodes_by_var)

    # Nodes with zero remaining children are ready to be processed first.
    remaining_children = {v: len(children_by_parent[v]) for v in ordered_vars}
    ready: List[str] = sorted([v for v in ordered_vars if remaining_children[v] == 0], key=lambda v: (rank.get(v, 10**9), v))
    emitted: Set[str] = set()
    out: List[str] = []

    # Each emitted child reduces the wait-count of its parents; once that count reaches zero,
    # the parent becomes ready too.
    while ready:
        cur = ready.pop(0)
        if cur in emitted:
            continue
        emitted.add(cur)
        out.append(cur)
        newly_ready: List[str] = []
        for parent in parents_by_child.get(cur, []):
            if parent in emitted:
                continue
            remaining_children[parent] -= 1
            if remaining_children[parent] == 0:
                newly_ready.append(parent)
        if newly_ready:
            ready.extend(sorted(newly_ready, key=lambda v: (rank.get(v, 10**9), v)))

    # If cycles or malformed links prevented a full traversal, append the unresolved nodes
    # deterministically instead of failing.
    if len(out) < len(ordered_vars):
        unresolved = [v for v in ordered_vars if v not in emitted]
        unresolved.sort(key=lambda v: (remaining_children.get(v, 0), rank.get(v, 10**9), v))
        out.extend(unresolved)

    # Preserve any original vars that were filtered out of the graph lookup, at the end.
    for v in var_order:
        if v not in out:
            out.append(v)
    return out
