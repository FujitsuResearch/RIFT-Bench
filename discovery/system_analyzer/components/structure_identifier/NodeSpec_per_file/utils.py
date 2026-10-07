#!/usr/bin/env python3
"""NodeSpec_per_file: per-file LLM extraction + deterministic combine."""

import ast
import builtins
import difflib
import io
import json
import keyword
import re
import textwrap
import tokenize
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from ..global_utils import identity_key, read_snippet, render_multiline_string, slugify
    from ..model_client import call_model
except ImportError:
    import sys

    PACKAGE_ROOT = Path(__file__).resolve().parents[1]
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from global_utils import identity_key, read_snippet, render_multiline_string, slugify
    from model_client import call_model


def _read_file(path: str, max_bytes: int) -> Optional[str]:
    """Read a text file with a byte cap and return its truncated content when available."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    if max_bytes > 0 and len(data.encode("utf-8", errors="replace")) > max_bytes:
        data = data.encode("utf-8", errors="replace")[:max_bytes].decode("utf-8", errors="replace")
    return data

def _validate_nodespec_per_file_output_text(text: str) -> tuple[bool, str]:
    """Validate that model output is parseable Python and defines `NEW_NODES`."""
    if not isinstance(text, str) or not text.strip():
        return False, "empty_output"
    try:
        ast.parse(text)
    except Exception as exc:
        return False, f"python_parse_error: {exc}"
    if "NEW_NODES" not in text:
        return False, "missing_NEW_NODES_assignment"
    return True, ""


def _fallback_empty_nodes_file() -> str:
    """Return a minimal valid Python file with an empty `NEW_NODES` list."""
    return (
        "from NodeSpec_schema import NodeSpec, NodeType, CodeReference, InputPort, OutputPort\n\n"
        "NEW_NODES = []\n"
    )


def _line_range_from_value(line_val: Any) -> Optional[Tuple[int, int]]:
    """Normalize a stored line scalar or range into a `(start, end)` tuple."""
    if isinstance(line_val, int):
        return int(line_val), int(line_val)
    if isinstance(line_val, (list, tuple)) and len(line_val) == 2:
        try:
            start, end = int(line_val[0]), int(line_val[1])
        except Exception:
            return None
        if end < start:
            start, end = end, start
        return start, end
    return None

def line_value_from_range(start: int, end: int) -> Any:
    """Convert a normalized line span into the stored scalar-or-range form."""
    if start == end:
        return start
    return [start, end]

def normalize_ref_file_path(file_path: str, project_root: Path) -> str:
    """Resolve a code-reference file path against the project root."""
    p = Path(file_path)
    if not p.is_absolute():
        p = (project_root / p).resolve()
    else:
        p = p.resolve()
    return str(p)

def _render_value(value: Any, indent: int = 0) -> str:
    """Render a Python value into deterministic source code text."""
    if isinstance(value, str):
        return render_multiline_string(value, indent)
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if len(value) == 2 and all(isinstance(v, int) for v in value):
            return f"[{value[0]}, {value[1]}]"
        if not value:
            return "[]"
        pad = " " * indent
        inner = ",\n".join(f"{pad}    {_render_value(v, indent + 4)}" for v in value)
        return "[\n" + inner + f"\n{pad}]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        pad = " " * indent
        items = [f"{pad}    {repr(k)}: {_render_value(v, indent + 4)}" for k, v in value.items()]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    return repr(value)

def render_nodespec(node: Dict[str, Any]) -> str:
    """Render a node dictionary into a `NodeSpec(...)` constructor string."""
    lines: List[str] = []
    lines.append("    NodeSpec(")

    def add_field(name: str, value: Any, rendered: Optional[str] = None) -> None:
        """Append a rendered field to the current `NodeSpec` output when it has a value."""
        if value is None:
            return
        if rendered is None:
            rendered = _render_value(value, indent=8)
        lines.append(f"        {name}={rendered},")

    add_field("name", node.get("name"))
    add_field("id", node.get("id"))

    node_type = node.get("node_type")
    if isinstance(node_type, dict):
        nt_parts = []
        if "type" in node_type:
            nt_parts.append(f"type={_render_value(node_type.get('type'))}")
        if "other_description" in node_type:
            nt_parts.append(f"other_description={_render_value(node_type.get('other_description'))}")
        add_field("node_type", node_type, rendered=f"NodeType({', '.join(nt_parts)})")

    add_field("description", node.get("description"))

    code_refs = node.get("code_references")
    if isinstance(code_refs, list):
        rendered_refs: List[str] = []
        for ref in code_refs:
            if not isinstance(ref, dict):
                continue
            ref_lines = ["CodeReference("]
            for key in ["kind", "other_kind_description", "file", "line", "snippet"]:
                if key in ref and ref[key] is not None:
                    ref_lines.append(f"    {key}={_render_value(ref[key], indent=4)},")
            ref_lines.append(")")
            rendered_refs.append("\n".join(ref_lines))
        if rendered_refs:
            rendered_items = []
            for r in rendered_refs:
                rendered_items.append(" " * 12 + r.replace("\n", "\n" + " " * 12) + ",")
            rendered = "[\n" + "\n".join(rendered_items) + "\n        ]"
        else:
            rendered = "[]"
        add_field("code_references", code_refs, rendered=rendered)

    inputs = node.get("inputs")
    if isinstance(inputs, list):
        rendered_inputs: List[str] = []
        for inp in inputs:
            if not isinstance(inp, dict):
                continue
            parts = []
            for key in ["name", "dtype", "description", "required", "default"]:
                if key in inp and inp[key] is not None:
                    parts.append(f"{key}={_render_value(inp[key])}")
            rendered_inputs.append(f"InputPort({', '.join(parts)})")
        rendered = "[\n" + "\n".join(f"            {r}," for r in rendered_inputs) + "\n        ]" if rendered_inputs else "[]"
        add_field("inputs", inputs, rendered=rendered)

    outputs = node.get("outputs")
    if isinstance(outputs, list):
        rendered_outputs: List[str] = []
        for out in outputs:
            if not isinstance(out, dict):
                continue
            parts = []
            for key in ["name", "dtype", "description", "output_kind"]:
                if key in out and out[key] is not None:
                    parts.append(f"{key}={_render_value(out[key])}")
            rendered_outputs.append(f"OutputPort({', '.join(parts)})")
        rendered = "[\n" + "\n".join(f"            {r}," for r in rendered_outputs) + "\n        ]" if rendered_outputs else "[]"
        add_field("outputs", outputs, rendered=rendered)

    add_field("metadata", node.get("metadata"))

    lines.append("    )")
    rendered = "\n".join(lines)
    if rendered.startswith("    "):
        rendered = "\n".join(line[4:] if line.startswith("    ") else line for line in rendered.splitlines())
    return rendered

def fmt_seconds(total_seconds: float) -> str:
    """Format a duration in seconds as `HH:MM:SS`."""
    secs = max(0, int(total_seconds))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def prefer_file_scoped_var_name(
    node_name: Optional[str],
    file_path: str,
    used_names: set[str],
    identity_map: Dict[str, str],
) -> None:
    """
    Deterministic collision handling:
    if a base var name already exists in this run, pin this node's mapping to
    a file-scoped var name (`<base>_<filename>`), before final resolution.
    """
    base = slugify(node_name or "node")
    if base not in used_names:
        return
    key = identity_key(node_name, file_path)
    file_tag = slugify(Path(file_path).name)
    if not file_tag:
        return
    preferred = f"{base}_{file_tag}"
    identity_map[key] = preferred

# -----------------------------
# Unified code-reference enrichment (deterministic)
# -----------------------------

_CR_AST_CACHE: Dict[str, Optional[ast.AST]] = {}
_CR_IMPORTS_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_CR_GLOBAL_INDEX_CACHE: Dict[Tuple[str, ...], Dict[str, Any]] = {}


def _cr_parse_ast(file_path: str) -> Optional[ast.AST]:
    """Parse and cache an AST for a source file used in code-reference enrichment."""
    if file_path in _CR_AST_CACHE:
        return _CR_AST_CACHE[file_path]
    try:
        src = Path(file_path).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src, filename=file_path)
    except Exception:
        tree = None
    _CR_AST_CACHE[file_path] = tree
    return tree


def _cr_find_enclosing_object_span(file_path: str, ref_span: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Find the smallest enclosing object-like AST span around a reference span."""
    tree = _cr_parse_ast(file_path)
    if tree is None:
        return None
    rs, re_ = ref_span
    candidates: List[Tuple[int, int]] = []
    object_types = (ast.List, ast.Tuple, ast.Set, ast.Dict, ast.Call)
    for node in ast.walk(tree):
        if not isinstance(node, object_types):
            continue
        ln = getattr(node, "lineno", None)
        en = getattr(node, "end_lineno", None)
        if not isinstance(ln, int) or not isinstance(en, int):
            continue
        if ln <= rs and en >= re_:
            candidates.append((ln, en))
    if not candidates:
        return None
    candidates.sort(key=lambda p: ((p[1] - p[0]), p[0], p[1]))
    best = candidates[0]
    if best == ref_span:
        return None
    return best


def _cr_parse_imports(file_path: str) -> List[Dict[str, Any]]:
    """Parse and cache import statements from a source file."""
    if file_path in _CR_IMPORTS_CACHE:
        return _CR_IMPORTS_CACHE[file_path]
    tree = _cr_parse_ast(file_path)
    if tree is None:
        _CR_IMPORTS_CACHE[file_path] = []
        return []
    imports: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "module": alias.name,
                    "name": None,
                    "lineno": getattr(node, "lineno", None),
                    "end_lineno": getattr(node, "end_lineno", getattr(node, "lineno", None)),
                    "asname": alias.asname,
                })
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                imports.append({
                    "module": mod,
                    "name": alias.name,
                    "lineno": getattr(node, "lineno", None),
                    "end_lineno": getattr(node, "end_lineno", getattr(node, "lineno", None)),
                    "asname": alias.asname,
                })
    _CR_IMPORTS_CACHE[file_path] = imports
    return imports


def _cr_symbol_imports(imports: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Map imported symbols to their originating import records."""
    out: Dict[str, Dict[str, Any]] = {}
    for imp in imports:
        mod = imp.get("module") or ""
        name = imp.get("name")
        asname = imp.get("asname")
        if name is None:
            sym = asname or mod.split(".")[0]
            if sym:
                out[str(sym)] = imp
        else:
            sym = asname or name
            if sym:
                out[str(sym)] = imp
    return out


def _cr_module_name_from_file(file_path: str, project_root: str) -> Optional[str]:
    """Convert a source file path into its dotted module name under the project root."""
    try:
        root = Path(project_root).resolve()
        fp = Path(file_path).resolve()
        rel = fp.relative_to(root)
    except Exception:
        return None
    if rel.suffix != ".py":
        return None
    rel_no_ext = rel.with_suffix("")
    parts = list(rel_no_ext.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def _cr_normalize_and_filter_refs(refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize code-reference spans and discard malformed entries."""
    out: List[Dict[str, Any]] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        file_path = ref.get("file")
        if not isinstance(file_path, str) or not file_path:
            continue
        span = _line_range_from_value(ref.get("line"))
        if span is not None:
            ref["line"] = line_value_from_range(span[0], span[1])
        if not isinstance(ref.get("snippet"), str) or not ref.get("snippet"):
            ref["snippet"] = read_snippet(file_path, ref.get("line"))
        out.append(ref)
    return out


def _cr_expand_refs_to_full_objects(refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand reference spans to enclosing object spans when available."""
    out: List[Dict[str, Any]] = []
    for ref in refs:
        file_path = ref.get("file")
        if not isinstance(file_path, str) or not file_path:
            out.append(ref)
            continue
        span = _line_range_from_value(ref.get("line"))
        if span is None:
            out.append(ref)
            continue
        full_span = _cr_find_enclosing_object_span(file_path, span)
        if full_span is None:
            out.append(ref)
            continue
        ref["line"] = line_value_from_range(full_span[0], full_span[1])
        snippet = read_snippet(file_path, ref["line"])
        if snippet:
            ref["snippet"] = snippet
        out.append(ref)
    return out


def _cr_extract_identifiers(snippet: str | List[str] | Tuple[str, ...]) -> List[str]:
    """Extract identifier candidates from a code snippet."""
    if not snippet:
        return []
    if isinstance(snippet, (list, tuple)):
        snippet = "".join(str(s) for s in snippet)
    cleaned = []
    for line in snippet.splitlines():
        if line.strip() == "..." or line.strip().startswith("..."):
            continue
        cleaned.append(line)
    text = textwrap.dedent("\n".join(cleaned))
    ids: List[str] = []
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                ids.append(node.id)
    except Exception:
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
            for i, tok in enumerate(toks):
                if tok.type != tokenize.NAME:
                    continue
                nm = tok.string
                if keyword.iskeyword(nm) or nm in dir(builtins):
                    continue
                prev = toks[i - 1] if i > 0 else None
                if prev and prev.type == tokenize.OP and prev.string == ".":
                    continue
                nxt = toks[i + 1] if i + 1 < len(toks) else None
                if nxt and nxt.type == tokenize.OP and nxt.string == "=":
                    continue
                ids.append(nm)
        except Exception:
            ids = []
    return [x for x in ids if not keyword.iskeyword(x) and x not in dir(builtins)]


def _cr_extract_bound_identifiers(snippet: str | List[str] | Tuple[str, ...]) -> set[str]:
    """Extract locally bound identifier names from a code snippet."""
    if not snippet:
        return set()
    if isinstance(snippet, (list, tuple)):
        snippet = "".join(str(s) for s in snippet)
    cleaned = []
    for line in snippet.splitlines():
        if line.strip() == "..." or line.strip().startswith("..."):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned)
    try:
        tree = ast.parse(text)
    except Exception:
        out: set[str] = set()
        for m in re.finditer(r"\bfor\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\b", text):
            out.add(m.group(1))
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*:=", text):
            out.add(m.group(1))
        return out

    out: set[str] = set()

    def _collect(node: ast.AST) -> None:
        """Collect bound names from nested assignment-like AST targets."""
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for e in node.elts:
                _collect(e)
        elif isinstance(node, ast.Starred):
            _collect(node.value)

    for node in ast.walk(tree):
        if isinstance(node, ast.comprehension):
            _collect(node.target)
        elif isinstance(node, ast.NamedExpr):
            _collect(node.target)
    return out


def _cr_extract_assign_target_names(target: ast.AST) -> List[str]:
    """Collect assigned identifier names from an assignment target AST node."""
    out: List[str] = []
    if isinstance(target, ast.Name):
        out.append(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for e in target.elts:
            out.extend(_cr_extract_assign_target_names(e))
    elif isinstance(target, ast.Starred):
        out.extend(_cr_extract_assign_target_names(target.value))
    return out


def _cr_build_global_index(reachable_py: List[str]) -> Dict[str, Any]:
    """Build and cache cross-file definition and assignment indexes."""
    key = tuple(sorted(set(str(Path(p).resolve()) for p in reachable_py if p)))
    cached = _CR_GLOBAL_INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    defs_by_name: Dict[str, List[Dict[str, Any]]] = {}
    assigns_by_file: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    scope_spans_by_file: Dict[str, List[Dict[str, Any]]] = {}

    class _Collector(ast.NodeVisitor):
        def __init__(self, file_path: str) -> None:
            """Initialize the collector with the current file path and empty scope state."""
            self.file_path = file_path
            self.scope_stack: List[str] = ["module"]
            self.scope_spans: List[Dict[str, Any]] = []

        def _scope(self) -> str:
            """Return the current lexical scope tag for the collector."""
            return self.scope_stack[-1] if self.scope_stack else "module"

        def _add_def(self, name: str, node: ast.AST) -> None:
            """Record a discovered definition span for the current file and scope."""
            ln = getattr(node, "lineno", None)
            en = getattr(node, "end_lineno", ln)
            if not isinstance(name, str) or not name or not isinstance(ln, int):
                return
            defs_by_name.setdefault(name, []).append({"file": self.file_path, "start": int(ln), "end": int(en) if isinstance(en, int) else int(ln), "scope": self._scope()})

        def _add_assign(self, name: str, node: ast.AST) -> None:
            """Record a discovered assignment span for the current file and scope."""
            ln = getattr(node, "lineno", None)
            en = getattr(node, "end_lineno", ln)
            if not isinstance(name, str) or not name or not isinstance(ln, int):
                return
            pf = assigns_by_file.setdefault(self.file_path, {})
            pf.setdefault(name, []).append({"start": int(ln), "end": int(en) if isinstance(en, int) else int(ln), "scope": self._scope()})

        def _enter_scope(self, tag: str, node: ast.AST) -> None:
            """Push a new lexical scope tag and record its span."""
            ln = getattr(node, "lineno", None)
            en = getattr(node, "end_lineno", ln)
            if isinstance(ln, int):
                self.scope_spans.append({"scope": tag, "start": int(ln), "end": int(en) if isinstance(en, int) else int(ln)})
            self.scope_stack.append(tag)

        def _exit_scope(self) -> None:
            """Pop the current lexical scope tag after visiting a scoped node."""
            if self.scope_stack:
                self.scope_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            """Record a function definition and traverse its nested body."""
            self._add_def(node.name, node)
            self._enter_scope(f"func:{node.name}@{getattr(node, 'lineno', 0)}", node)
            self.generic_visit(node)
            self._exit_scope()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            """Record an async function definition and traverse its nested body."""
            self._add_def(node.name, node)
            self._enter_scope(f"func:{node.name}@{getattr(node, 'lineno', 0)}", node)
            self.generic_visit(node)
            self._exit_scope()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            """Record a class definition and traverse its nested body."""
            self._add_def(node.name, node)
            self._enter_scope(f"class:{node.name}@{getattr(node, 'lineno', 0)}", node)
            self.generic_visit(node)
            self._exit_scope()

        def visit_Assign(self, node: ast.Assign) -> None:
            """Record assignment targets and keep traversing nested expressions."""
            for t in node.targets:
                for name in _cr_extract_assign_target_names(t):
                    self._add_assign(name, node)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            """Record annotated assignment targets and keep traversing nested expressions."""
            for name in _cr_extract_assign_target_names(node.target):
                self._add_assign(name, node)
            self.generic_visit(node)

        def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
            """Record walrus-assignment targets and keep traversing nested expressions."""
            for name in _cr_extract_assign_target_names(node.target):
                self._add_assign(name, node)
            self.generic_visit(node)

    for fp in key:
        tree = _cr_parse_ast(fp)
        if tree is None:
            continue
        c = _Collector(fp)
        c.visit(tree)
        scope_spans_by_file[fp] = c.scope_spans

    out = {"defs_by_name": defs_by_name, "assigns_by_file": assigns_by_file, "scope_spans_by_file": scope_spans_by_file}
    _CR_GLOBAL_INDEX_CACHE[key] = out
    return out


def _cr_scope_at_line(scope_spans: List[Dict[str, Any]], line_anchor: Optional[int]) -> Optional[str]:
    """Return the innermost recorded scope that contains a given line number."""
    if not isinstance(line_anchor, int):
        return None
    containing = [s for s in scope_spans if isinstance(s.get("start"), int) and isinstance(s.get("end"), int) and s["start"] <= line_anchor <= s["end"]]
    if not containing:
        return None
    containing.sort(key=lambda s: ((s["end"] - s["start"]), s["start"]))
    return str(containing[0].get("scope") or "")


def _cr_pick_assignment(spans: List[Dict[str, Any]], line_anchor: Optional[int], preferred_scope: Optional[str]) -> Optional[Dict[str, Any]]:
    """Pick the assignment span that best matches the current scope and line anchor."""
    if not spans:
        return None
    in_scope = [s for s in spans if preferred_scope and str(s.get("scope") or "") == preferred_scope]
    cands = in_scope if in_scope else spans
    if isinstance(line_anchor, int):
        before = [s for s in cands if isinstance(s.get("start"), int) and s["start"] <= line_anchor]
        if before:
            before.sort(key=lambda s: (line_anchor - int(s.get("start") or line_anchor), int(s.get("start") or 0)))
            return before[0]
        cands.sort(key=lambda s: abs(int(s.get("start") or line_anchor) - line_anchor))
        return cands[0]
    cands.sort(key=lambda s: int(s.get("start") or 10**9))
    return cands[0]


def _cr_ref_key(ref: Dict[str, Any]) -> Tuple[str, str, str]:
    """Build a dedupe key for a code-reference record."""
    return (str(ref.get("kind") or ""), str(ref.get("file") or ""), str(ref.get("line") or ""))


def _cr_append_ref_unique(refs: List[Dict[str, Any]], seen: set[Tuple[str, str, str]], ref: Dict[str, Any]) -> bool:
    """Append a reference only when its dedupe key has not been seen yet."""
    key = _cr_ref_key(ref)
    if key in seen:
        return False
    seen.add(key)
    refs.append(ref)
    return True


def _cr_enrich_identifier_links(refs: List[Dict[str, Any]], reachable_py: List[str]) -> List[Dict[str, Any]]:
    """Add definition, assignment, and import references implied by snippet identifiers."""
    idx = _cr_build_global_index(reachable_py)
    defs_by_name = idx.get("defs_by_name", {})
    assigns_by_file = idx.get("assigns_by_file", {})
    scope_spans_by_file = idx.get("scope_spans_by_file", {})

    out = list(refs)
    seen = {_cr_ref_key(r) for r in out if isinstance(r, dict)}
    pending = list(out)
    processed: set[Tuple[str, str, str]] = set()
    ignore = {"self", "cls"}

    i = 0
    while i < len(pending):
        ref = pending[i]
        i += 1
        if not isinstance(ref, dict):
            continue
        rk = _cr_ref_key(ref)
        if rk in processed:
            continue
        processed.add(rk)

        file_path = ref.get("file")
        if not isinstance(file_path, str) or not file_path:
            continue
        imports = _cr_parse_imports(file_path)
        import_map = _cr_symbol_imports(imports)

        span = _line_range_from_value(ref.get("line"))
        line_anchor = span[0] if span else None
        current_scope = _cr_scope_at_line(scope_spans_by_file.get(file_path, []), line_anchor)

        snippet = ref.get("snippet") if isinstance(ref.get("snippet"), str) else ""
        ids = _cr_extract_identifiers(snippet)
        bound = _cr_extract_bound_identifiers(snippet)

        for ident in ids:
            if not ident or ident in ignore or ident in bound:
                continue
            if keyword.iskeyword(ident) or ident in dir(builtins):
                continue

            imp = import_map.get(ident)
            if isinstance(imp, dict):
                ln = imp.get("lineno")
                en = imp.get("end_lineno")
                if isinstance(ln, int):
                    line_val: Any = [ln, en] if isinstance(en, int) and en != ln else ln
                    nr = {"kind": "source", "file": file_path, "line": line_val, "snippet": read_snippet(file_path, line_val)}
                    if _cr_append_ref_unique(out, seen, nr):
                        pending.append(nr)
                continue

            by_name = assigns_by_file.get(file_path, {}).get(ident, [])
            asg = _cr_pick_assignment(by_name, line_anchor, current_scope)
            if isinstance(asg, dict):
                s = int(asg.get("start") or 0)
                e = int(asg.get("end") or s)
                if s > 0:
                    line_val = line_value_from_range(s, e)
                    nr = {"kind": "usage", "file": file_path, "line": line_val, "snippet": read_snippet(file_path, line_val)}
                    if _cr_append_ref_unique(out, seen, nr):
                        pending.append(nr)
                    continue

            defs = defs_by_name.get(ident, [])
            same = [d for d in defs if str(d.get("file") or "") == file_path]
            chosen = None
            if same:
                if isinstance(line_anchor, int):
                    chosen = min(same, key=lambda d: abs(int(d.get("start") or line_anchor) - line_anchor))
                else:
                    chosen = same[0]
            elif defs:
                chosen = defs[0]

            if isinstance(chosen, dict):
                dfile = str(chosen.get("file") or "")
                ds = int(chosen.get("start") or 0)
                de = int(chosen.get("end") or ds)
                if dfile and ds > 0:
                    line_val = line_value_from_range(ds, de)
                    nr = {"kind": "definition", "file": dfile, "line": line_val, "snippet": read_snippet(dfile, line_val)}
                    if _cr_append_ref_unique(out, seen, nr):
                        pending.append(nr)

    return out


def _cr_add_importer_source_refs(*, node_name: str, refs: List[Dict[str, Any]], project_root: str, reachable_py: List[str]) -> List[Dict[str, Any]]:
    """Add importer-side source references for definitions used across files."""
    symbol = str(node_name or "").strip()
    if not symbol or not project_root:
        return refs

    defining_modules: set[str] = set()
    defining_files: set[str] = set()
    for ref in refs:
        if ref.get("kind") not in {"definition", "implementation", "initialization"}:
            continue
        f = ref.get("file")
        if not isinstance(f, str):
            continue
        mod = _cr_module_name_from_file(f, project_root)
        if mod:
            defining_modules.add(mod)
            defining_files.add(str(Path(f).resolve()))
    if not defining_modules:
        return refs

    out = list(refs)
    for importer in sorted(set(str(Path(p).resolve()) for p in reachable_py if p)):
        if importer in defining_files:
            continue
        for imp in _cr_parse_imports(importer):
            if imp.get("module") not in defining_modules:
                continue
            if imp.get("name") != symbol:
                continue
            ln = imp.get("lineno")
            en = imp.get("end_lineno")
            if not isinstance(ln, int):
                continue
            line_val: Any = [ln, en] if isinstance(en, int) and en != ln else ln
            out.append({"kind": "source", "file": importer, "line": line_val, "snippet": read_snippet(importer, line_val)})
    return out


def _cr_dedupe_merge_refs(refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge overlapping references per file and remove duplicates."""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for ref in refs:
        file_path = ref.get("file")
        if isinstance(file_path, str) and file_path:
            buckets.setdefault(file_path, []).append(ref)

    pri = {
        "definition": 0,
        "implementation": 1,
        "LLM_config": 2,
        "system_prompt": 3,
        "user_prompt_template": 4,
        "input_scheme": 5,
        "output_scheme": 6,
        "source": 7,
        "initialization": 8,
        "usage": 9,
        "logic": 10,
        "other": 11,
    }

    merged: List[Dict[str, Any]] = []
    for file_path, items in buckets.items():
        with_lines: List[Tuple[int, int, Dict[str, Any]]] = []
        without_lines: List[Dict[str, Any]] = []
        for ref in items:
            span = _line_range_from_value(ref.get("line"))
            if span is None:
                without_lines.append(ref)
            else:
                with_lines.append((span[0], span[1], ref))
        with_lines.sort(key=lambda x: (x[0], x[1]))

        cur: Optional[List[Any]] = None
        for s, e, ref in with_lines:
            if cur is None:
                cur = [s, e, [ref]]
                continue
            cs, ce, crefs = cur
            if s <= ce + 1:
                cur[1] = max(ce, e)
                crefs.append(ref)
                continue
            chosen = min(crefs, key=lambda r: pri.get(str(r.get("kind") or ""), 99))
            lv = line_value_from_range(cs, ce)
            merged.append({"kind": chosen.get("kind"), "file": file_path, "line": lv, "snippet": read_snippet(file_path, lv)})
            cur = [s, e, [ref]]

        if cur is not None:
            cs, ce, crefs = cur
            chosen = min(crefs, key=lambda r: pri.get(str(r.get("kind") or ""), 99))
            lv = line_value_from_range(cs, ce)
            merged.append({"kind": chosen.get("kind"), "file": file_path, "line": lv, "snippet": read_snippet(file_path, lv)})

        merged.extend(without_lines)

    seen = set()
    out: List[Dict[str, Any]] = []
    for ref in merged:
        key = (ref.get("kind"), ref.get("file"), str(ref.get("line")))
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def enrich_node_code_references(node: Dict[str, Any], *, project_root: str, reachable_py: List[str]) -> List[Dict[str, Any]]:
    """Expand a node’s code references using deterministic cross-file evidence."""
    refs = list(node.get("code_references") or [])
    refs = _cr_normalize_and_filter_refs(refs)
    refs = _cr_expand_refs_to_full_objects(refs)
    refs = _cr_enrich_identifier_links(refs, reachable_py)
    refs = _cr_add_importer_source_refs(
        node_name=str(node.get("name") or ""),
        refs=refs,
        project_root=project_root,
        reachable_py=reachable_py,
    )
    refs = _cr_normalize_and_filter_refs(refs)
    refs = _cr_dedupe_merge_refs(refs)
    return refs


_ME_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ME_RESERVED = set(keyword.kwlist) | {"True", "False", "None", "self", "cls"}


def _me_ast_from_snippet(snippet: str) -> Optional[ast.AST]:
    """Parse a snippet for missing-evidence analysis when possible."""
    if not isinstance(snippet, str) or not snippet.strip():
        return None
    try:
        return ast.parse(snippet)
    except Exception:
        return None


def _me_collect_call_symbol_candidates(refs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Collect unresolved call-site symbol candidates from code references."""
    out: List[Dict[str, str]] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        if str(ref.get("kind") or "") not in {"initialization", "usage", "implementation"}:
            continue
        tree = _me_ast_from_snippet(str(ref.get("snippet") or ""))
        if tree is None:
            continue
        file_path = str(ref.get("file") or "")
        line_val = ref.get("line")
        line_hint = ""
        span = _line_range_from_value(line_val)
        if span is not None:
            line_hint = f"{file_path}:{span[0]}"
        elif file_path:
            line_hint = file_path
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    out.append({
                        "field": arg.id,
                        "symbol": arg.id,
                        "hint": line_hint,
                    })
            for kw in node.keywords:
                if not isinstance(kw, ast.keyword) or not kw.arg:
                    continue
                if isinstance(kw.value, ast.Name):
                    out.append({
                        "field": kw.arg,
                        "symbol": kw.value.id,
                        "hint": line_hint,
                    })
    return out


def _me_symbol_has_local_resolution(symbol: str, node: Dict[str, Any], refs: List[Dict[str, Any]]) -> bool:
    """Check whether a symbol already has local evidence in the node or refs."""
    if not symbol:
        return True
    node_name = str(node.get("name") or "")
    if symbol == node_name:
        return True
    inputs = node.get("inputs") if isinstance(node.get("inputs"), list) else []
    for inp in inputs:
        if isinstance(inp, dict) and str(inp.get("name") or "") == symbol:
            return True
    patt_assign = re.compile(rf"\b{re.escape(symbol)}\b\s*=")
    patt_import = re.compile(rf"\b(import|from)\b[^\n]*\b{re.escape(symbol)}\b")
    patt_def = re.compile(rf"\b(def|class)\s+{re.escape(symbol)}\b")
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        snip = str(ref.get("snippet") or "")
        if not snip:
            continue
        if patt_assign.search(snip) or patt_import.search(snip) or patt_def.search(snip):
            return True
    return False


def ensure_missing_evidence_from_code_refs(node: Dict[str, Any]) -> Dict[str, Any]:
    """Add missing-evidence metadata inferred from unresolved call-site symbols."""
    out = dict(node)
    refs = out.get("code_references") if isinstance(out.get("code_references"), list) else []
    refs = [x for x in refs if isinstance(x, dict)]
    if not refs:
        return out

    md = out.get("metadata") if isinstance(out.get("metadata"), dict) else {}
    missing = md.get("missing_evidence") if isinstance(md.get("missing_evidence"), list) else []
    missing = [x for x in missing if isinstance(x, dict)]
    existing_fields = {str(x.get("field") or "").strip() for x in missing if str(x.get("field") or "").strip()}

    for cand in _me_collect_call_symbol_candidates(refs):
        field = str(cand.get("field") or "").strip()
        symbol = str(cand.get("symbol") or "").strip()
        hint = str(cand.get("hint") or "").strip()
        if not field or not symbol:
            continue
        if field in existing_fields:
            continue
        if not _ME_IDENT_RE.match(symbol):
            continue
        if symbol in _ME_RESERVED:
            continue
        if symbol.startswith("_"):
            continue
        if _me_symbol_has_local_resolution(symbol, out, refs):
            continue
        reason = f"Referenced as `{symbol}` in call-site evidence, but origin/value is not resolved in current node evidence."
        item = {
            "field": field,
            "reason": reason,
            "evidence_code_refs_hint": hint or "call-site code reference",
        }
        missing.append(item)
        existing_fields.add(field)

    if missing:
        md = dict(md)
        md["missing_evidence"] = missing
        if not isinstance(md.get("open_question"), str) or not str(md.get("open_question") or "").strip():
            md["open_question"] = "Some initialization or runtime parameters are referenced but not yet evidenced."
        out["metadata"] = md
    return out
