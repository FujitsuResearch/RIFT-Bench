#!/usr/bin/env python3
"""Helpers for discovering Python and JSON files reachable from an entrypoint."""

import ast
import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# -----------------------------
# Project root helpers
# -----------------------------

ROOT_MARKERS = [
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "requirements.txt",
    ".git",
]


def build_root_record(entry_py: str) -> Dict[str, str | None]:
    """Resolve the project root and normalized entry paths for a Python entrypoint."""
    entry = Path(entry_py).resolve()
    if not entry.exists():
        raise FileNotFoundError(f"Entrypoint not found: {entry}")
    if entry.suffix != ".py":
        raise ValueError(f"Entrypoint must be a .py file, got: {entry}")

    cur = entry.parent
    best_root: Optional[Path] = None
    while True:
        for marker in ROOT_MARKERS:
            if (cur / marker).exists():
                best_root = cur
                break
        if best_root is not None:
            break
        if cur.parent == cur:
            break
        cur = cur.parent
    if best_root is None:
        best_root = entry.parent

    root_record: Dict[str, str | None] = {
        "project_root": str(best_root),
        "entry_abs": str(entry),
    }
    try:
        root_record["entry_rel"] = str(entry.relative_to(best_root))
    except Exception:
        root_record["entry_rel"] = None
    return root_record

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".tox", "site-packages",
    "build", "dist"
}

def build_py_file_index(project_root: str) -> dict[str, list[str]]:
    """Build a filename-to-path index for Python files under the project root."""
    index: dict[str, list[str]] = {}
    for dirpath, dirnames, filenames in os.walk(project_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                index.setdefault(fn, []).append(os.path.join(dirpath, fn))
    return index

def resolve_import_to_local_py(
    module_name: str,
    current_file_abs: str,
    project_root: str,
    py_index: dict[str, list[str]],
    level: int = 0,
) -> Optional[str]:
    """
    Same permissive resolver as component_analysis.py.
    - absolute: pkg.mod -> <root>/pkg/mod.py or <root>/pkg/mod/__init__.py
    - relative: from .sub import x -> relative to current file dir
    """
    current_dir = os.path.dirname(os.path.abspath(current_file_abs))

    # relative base for import-from levels
    base_dir = current_dir
    for _ in range(max(level - 1, 0)):
        base_dir = os.path.dirname(base_dir)

    parts = module_name.split(".") if module_name else []
    rel_mod_path = os.path.sep.join(parts) if parts else ""

    candidates: List[str] = []

    if level == 0:
        if rel_mod_path:
            candidates.append(os.path.join(project_root, rel_mod_path + ".py"))
            candidates.append(os.path.join(project_root, rel_mod_path, "__init__.py"))
    else:
        if rel_mod_path:
            candidates.append(os.path.join(base_dir, rel_mod_path + ".py"))
            candidates.append(os.path.join(base_dir, rel_mod_path, "__init__.py"))
        else:
            candidates.append(os.path.join(base_dir, "__init__.py"))

    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)

    target_py = (parts[-1] + ".py") if parts else "__init__.py"
    for full in py_index.get(target_py, []):
        if rel_mod_path and (
            full.endswith(os.path.sep + rel_mod_path + ".py")
            or full.endswith(os.path.sep + rel_mod_path + os.path.sep + "__init__.py")
        ):
            return os.path.abspath(full)

    matches = py_index.get(target_py, [])
    if len(matches) == 1:
        return os.path.abspath(matches[0])

    return None

def extract_paths_from_code(file_path: str, project_root: str) -> tuple[set[str], set[str]]:
    """
    Permissive path finder.
    Finds "*.json" and "*.py" literal strings in the code and resolves them
    relative to BOTH:
      (a) the current file directory, and
      (b) the project root.

    Returns absolute normalized paths (only those that exist AND are inside project_root).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set(), set()

    base_dir = os.path.dirname(os.path.abspath(file_path))
    project_root = os.path.abspath(project_root)
    root_prefix = project_root + os.sep

    json_paths: set[str] = set()
    py_paths: set[str] = set()

    def candidates(p: str) -> list[str]:
        p = p.strip()
        if os.path.isabs(p):
            return [os.path.normpath(p)]
        return [
            os.path.normpath(os.path.join(base_dir, p)),
            os.path.normpath(os.path.join(project_root, p)),
        ]

    def collect_str(val: str) -> None:
        if not (val.endswith(".json") or val.endswith(".py")):
            return
        for c in candidates(val):
            c_abs = os.path.abspath(c)
            if not c_abs.startswith(root_prefix):
                continue
            if os.path.isfile(c_abs):
                if c_abs.endswith(".json"):
                    json_paths.add(c_abs)
                elif c_abs.endswith(".py"):
                    py_paths.add(c_abs)

    def collect_embedded_paths(text: str) -> None:
        # Extract path-like substrings from longer prompt/code strings.
        for tok in re.findall(r"[\w./\\-]+\.(?:py|json)\b", text):
            clean = tok.strip().strip("\"'`()[]{}<>,;:")
            if clean:
                collect_str(clean)

    def collect_node(node: ast.AST) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            collect_str(node.value)
            collect_embedded_paths(node.value)

    for node in ast.walk(tree):
        collect_node(node)

        if isinstance(node, ast.Call):
            for arg in node.args:
                collect_node(arg)
            for kw in node.keywords or []:
                collect_node(kw.value)

        if isinstance(node, ast.Dict):
            for v in node.values:
                collect_node(v)

    return json_paths, py_paths


def _extract_import_modules_from_text(text: str) -> List[str]:
    """Extract module names from plain-text import statements embedded in a string."""
    modules: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m_from = re.match(r"^from\s+([A-Za-z_][A-Za-z0-9_\.]*)\s+import\s+(.+)$", s)
        if m_from:
            base = m_from.group(1).strip()
            names = m_from.group(2).strip()
            modules.append(base)
            for part in names.split(","):
                name_tok = part.strip().split(" as ")[0].strip()
                if not name_tok or name_tok == "*":
                    continue
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name_tok):
                    modules.append(f"{base}.{name_tok}")
            continue

        m_imp = re.match(r"^import\s+(.+)$", s)
        if m_imp:
            rhs = m_imp.group(1).strip()
            for part in rhs.split(","):
                name_tok = part.strip().split(" as ")[0].strip()
                if re.match(r"^[A-Za-z_][A-Za-z0-9_\.]*$", name_tok):
                    modules.append(name_tok)
    return modules


def extract_import_like_modules_from_string_literals(
    file_path: str, project_root: str, py_index: dict[str, list[str]]
) -> Set[str]:
    """Resolve import-like statements found inside string literals to local Python files."""
    out: Set[str] = set()
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except Exception:
        return out

    root_abs_prefix = str(Path(project_root).resolve()) + os.sep
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        text = node.value
        if "import " not in text:
            continue
        for mod in _extract_import_modules_from_text(text):
            resolved = resolve_import_to_local_py(mod, file_path, project_root, py_index, level=0)
            if not resolved:
                continue
            rp = str(Path(resolved).resolve())
            if rp.startswith(root_abs_prefix) and rp.endswith(".py") and os.path.isfile(rp):
                out.add(rp)
    return out


def _extract_iterdir_name_needles(source: str) -> Set[str]:
    """Extract filename tokens used to filter sibling-module scans driven by `iterdir()`."""
    needles: Set[str] = set()
    try:
        tree = ast.parse(source)
    except Exception:
        return needles

    # Look for patterns like: p.name for p in current_dir.iterdir() if "server" in p.name
    def _collect_needles_from_expr(expr: ast.AST, iter_var: str) -> Set[str]:
        out: Set[str] = set()
        if isinstance(expr, ast.BoolOp):
            for v in expr.values:
                out.update(_collect_needles_from_expr(v, iter_var))
            return out
        if isinstance(expr, ast.UnaryOp):
            out.update(_collect_needles_from_expr(expr.operand, iter_var))
            return out
        if (
            isinstance(expr, ast.Compare)
            and len(expr.ops) == 1
            and isinstance(expr.ops[0], ast.In)
            and isinstance(expr.left, ast.Constant)
            and isinstance(expr.left.value, str)
            and len(expr.comparators) == 1
        ):
            rhs = expr.comparators[0]
            if (
                isinstance(rhs, ast.Attribute)
                and rhs.attr == "name"
                and isinstance(rhs.value, ast.Name)
                and rhs.value.id == iter_var
            ):
                token = expr.left.value.strip()
                if token:
                    out.add(token)
        return out

    for node in ast.walk(tree):
        if not isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            continue
        for comp in getattr(node, "generators", []) or []:
            if not isinstance(comp, ast.comprehension):
                continue
            if not isinstance(comp.target, ast.Name):
                continue
            iter_var = comp.target.id

            iter_call = comp.iter
            if not (isinstance(iter_call, ast.Call) and isinstance(iter_call.func, ast.Attribute)):
                continue
            if iter_call.func.attr != "iterdir":
                continue

            for if_expr in comp.ifs or []:
                needles.update(_collect_needles_from_expr(if_expr, iter_var))
    return needles


def extract_iterdir_dynamic_sibling_py_files(file_path: str, project_root: str) -> Set[str]:
    """Include sibling Python files when code dynamically scans its directory with `iterdir()`."""
    out: Set[str] = set()
    try:
        source = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return out
    if "__file__" not in source or "iterdir(" not in source:
        return out

    needles = _extract_iterdir_name_needles(source)

    # Dynamic plugin/module loading pattern:
    # when code scans sibling files via iterdir, include sibling .py files,
    # optionally filtered by extracted name tokens from the real condition.
    base = Path(file_path).resolve().parent
    root_abs_prefix = str(Path(project_root).resolve()) + os.sep
    for p in base.iterdir():
        try:
            if p.is_file() and p.suffix == ".py":
                if p.resolve() == Path(file_path).resolve():
                    continue
                if needles and not any(token in p.name for token in needles):
                    continue
                rp = str(p.resolve())
                if rp.startswith(root_abs_prefix):
                    out.add(rp)
        except Exception:
            continue
    return out

def extract_py_files_from_json(json_path: str, project_root: str) -> Set[str]:
    """
    Recursively search inside JSON for string fields ending with '.py'.

    Resolve relative paths against BOTH:
      (a) the JSON file directory, and
      (b) the project root.

    Returns absolute paths (not filtered to exist here; caller can decide).
    """
    result: Set[str] = set()
    if not os.path.isfile(json_path):
        return result

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return result

    base_dir = os.path.dirname(os.path.abspath(json_path))
    project_root = os.path.abspath(project_root)

    def _candidates(p: str) -> list[str]:
        p = os.path.expandvars(os.path.expanduser(p.strip()))
        if os.path.isabs(p):
            return [os.path.abspath(os.path.normpath(p))]
        return [
            os.path.abspath(os.path.normpath(os.path.join(base_dir, p))),
            os.path.abspath(os.path.normpath(os.path.join(project_root, p))),
        ]

    def recurse(obj: Any) -> None:
        if isinstance(obj, str):
            s = obj.strip()
            if s.endswith(".py"):
                for cand in _candidates(s):
                    result.add(cand)
        elif isinstance(obj, list):
            for x in obj:
                recurse(x)
        elif isinstance(obj, dict):
            for v in obj.values():
                recurse(v)

    recurse(data)
    return result

# -----------------------------
# Reachability logic
# -----------------------------

def looks_like_local_import(module: str, level: int, project_root: str) -> bool:
    """Return whether an unresolved import still looks like an in-repo module worth reporting."""
    if level and level > 0:
        return True
    # absolute import: if it contains '.' it might still be external, but could be local package
    # We'll log unresolved only if it resembles an in-repo module path by checking prefix existence.
    # This is conservative: better to under-log than to spam stdlib.
    parts = module.split(".")
    if not parts:
        return False
    # if top-level package dir exists in repo root, it might be local
    top = os.path.join(project_root, parts[0])
    return os.path.isdir(top) or os.path.isfile(top + ".py")

def _expand_hint_paths(project_root: Path, values: Any) -> Set[str]:
    """Expand hint file entries and globs into absolute in-repo file paths."""
    out: Set[str] = set()
    if not isinstance(values, list):
        return out
    root = project_root.resolve()
    cwd_root = Path.cwd().resolve()
    root_prefix = str(root) + os.sep
    for raw in values:
        if not isinstance(raw, str) or not raw.strip():
            continue
        token = raw.strip()
        candidates: List[str] = []
        bases: List[str] = []
        if any(ch in token for ch in ("*", "?", "[")):
            if os.path.isabs(token):
                bases = [token]
            else:
                bases = [str(root / token), str(cwd_root / token)]
            seen_bases: Set[str] = set()
            for base in bases:
                if base in seen_bases:
                    continue
                seen_bases.add(base)
                candidates.extend(glob.glob(base, recursive=True))
        else:
            if os.path.isabs(token):
                candidates = [token]
            else:
                candidates = [str(root / token), str(cwd_root / token)]
        for c in candidates:
            abs_c = str(Path(c).resolve())
            if not abs_c.startswith(root_prefix):
                continue
            if os.path.isfile(abs_c):
                out.add(abs_c)
    return out


def _resolve_hint_file(project_root: Path, hint_path: Path) -> Path:
    """Resolve a hints file path relative to the project root before falling back to CWD."""
    if hint_path.is_absolute():
        return hint_path.resolve()
    p1 = (project_root / hint_path).resolve()
    if p1.exists():
        return p1
    return hint_path.resolve()


def load_reachable_hints(project_root: Path, hints_paths: List[Path] | None) -> Dict[str, Any]:
    """Load optional reachable-file hints and normalize them into absolute file lists."""
    if not hints_paths:
        return {
            "extra_py_abs": [],
            "extra_json_abs": [],
            "hints_files_tried": [],
            "hints_files_used": [],
        }

    extra_py: Set[str] = set()
    extra_json: Set[str] = set()
    tried: List[str] = []
    used: List[str] = []

    for hp_in in hints_paths:
        hp = _resolve_hint_file(project_root, hp_in)
        hp_s = str(hp)
        tried.append(hp_s)
        if not hp.exists():
            continue
        try:
            obj = json.loads(hp.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        py_vals: List[Any] = []
        json_vals: List[Any] = []
        py_vals.extend(obj.get("extra_py_files") or [])
        py_vals.extend(obj.get("extra_py_globs") or [])
        json_vals.extend(obj.get("extra_json_files") or [])
        json_vals.extend(obj.get("extra_json_globs") or [])
        extra_py.update(_expand_hint_paths(project_root, py_vals))
        extra_json.update(_expand_hint_paths(project_root, json_vals))
        used.append(hp_s)

    return {
        "extra_py_abs": sorted(extra_py),
        "extra_json_abs": sorted(extra_json),
        "hints_files_tried": tried,
        "hints_files_used": used,
    }
