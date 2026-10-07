import os
import json
from collections import defaultdict
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import PythonCodeTextSplitter
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", "site-packages", ".mypy_cache", ".pytest_cache", "dist", "build"}
DEFAULT_INCLUDE_EXTS = {".py", ".json"}

def _parse_include_exts() -> set[str]:
    """Return the set of file extensions to index: the defaults plus any from the RAG_EXTRA_EXTS env var."""
    extra = os.getenv("RAG_EXTRA_EXTS", "").strip()
    out = set(DEFAULT_INCLUDE_EXTS)
    if not extra:
        return out
    for raw in extra.split(","):
        ext = raw.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        out.add(ext)
    return out


def _safe_int_env(name: str, default: int, minimum: int = 1) -> int:
    """Read an int from env var `name`, clamped to at least `minimum`; return `default` if unset/invalid."""
    try:
        val = int(os.getenv(name, "").strip())
        return max(minimum, val)
    except Exception:
        return default


def load_repo_py(repo_root: str):
    """Walk repo_root (skipping excluded dirs) and load every included-extension file as a Document."""
    include_exts = _parse_include_exts()
    docs = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            suffix = Path(f).suffix.lower()
            if suffix not in include_exts:
                continue
            p = os.path.join(root, f)
            docs.extend(TextLoader(p, encoding="utf-8").load())
    return docs


def _load_docs_from_reachable_record(reachable_record: str):
    """Load only the files listed in a reachable-files JSON record (reachable py/json + entry) as Documents."""
    p = Path(reachable_record).resolve()
    if not p.exists():
        raise FileNotFoundError(f"reachable_record not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("reachable_record must be a JSON object")

    files = []
    for key in ("reachable_py_abs", "reachable_json_abs"):
        vals = obj.get(key)
        if isinstance(vals, list):
            for v in vals:
                if isinstance(v, str) and v:
                    files.append(str(Path(v).resolve()))

    entry_abs = obj.get("entry_abs")
    if isinstance(entry_abs, str) and entry_abs:
        files.append(str(Path(entry_abs).resolve()))

    docs = []
    seen = set()
    for f in files:
        if f in seen:
            continue
        seen.add(f)
        fp = Path(f)
        if not fp.exists() or not fp.is_file():
            continue
        if fp.suffix.lower() not in _parse_include_exts():
            continue
        docs.extend(TextLoader(str(fp), encoding="utf-8").load())
    return docs


def _annotate_chunk_line_ranges(docs, chunks):
    """Populate line_start/line_end metadata on each chunk when possible."""
    source_to_text = {}
    for d in docs:
        src = d.metadata.get("source")
        if isinstance(src, str) and src and src not in source_to_text:
            source_to_text[src] = d.page_content or ""

    # Per-source cursor used when start_index metadata is absent.
    source_cursor = defaultdict(int)

    for c in chunks:
        meta = c.metadata or {}
        src = meta.get("source")
        if not isinstance(src, str) or src not in source_to_text:
            continue

        full_text = source_to_text[src]
        chunk_text = c.page_content or ""
        if not chunk_text:
            continue

        start_idx = meta.get("start_index")
        if not isinstance(start_idx, int):
            # Fallback: best-effort locate chunk text from the current cursor.
            cursor = source_cursor[src]
            found = full_text.find(chunk_text, cursor)
            if found < 0:
                found = full_text.find(chunk_text)
            if found < 0:
                continue
            start_idx = found

        end_idx = start_idx + len(chunk_text)
        line_start = full_text.count("\n", 0, start_idx) + 1
        line_end = line_start + chunk_text.count("\n")

        c.metadata["line_start"] = int(line_start)
        c.metadata["line_end"] = int(line_end)
        source_cursor[src] = max(source_cursor[src], end_idx - 1)


def _split_docs(docs):
    """Split docs into chunks, using a Python-aware splitter for .py files and a generic text splitter otherwise."""
    py_chunk_size = _safe_int_env("RAG_PY_CHUNK_SIZE", 1200)
    py_chunk_overlap = _safe_int_env("RAG_PY_CHUNK_OVERLAP", 150, minimum=0)
    txt_chunk_size = _safe_int_env("RAG_TEXT_CHUNK_SIZE", 900)
    txt_chunk_overlap = _safe_int_env("RAG_TEXT_CHUNK_OVERLAP", 120, minimum=0)

    py_docs = []
    other_docs = []
    for d in docs:
        src = str((d.metadata or {}).get("source") or "")
        if src.lower().endswith(".py"):
            py_docs.append(d)
        else:
            other_docs.append(d)

    chunks = []
    if py_docs:
        py_splitter = PythonCodeTextSplitter(
            chunk_size=py_chunk_size,
            chunk_overlap=py_chunk_overlap,
            add_start_index=True,
        )
        chunks.extend(py_splitter.split_documents(py_docs))
    if other_docs:
        txt_splitter = RecursiveCharacterTextSplitter(
            chunk_size=txt_chunk_size,
            chunk_overlap=txt_chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks.extend(txt_splitter.split_documents(other_docs))

    return chunks


def build_faiss(repo_root: str, persist_dir: str, reachable_record: str | None = None):
    """Load docs (from reachable record or full repo), chunk + annotate them, embed, and save a FAISS index to disk."""
    docs = _load_docs_from_reachable_record(reachable_record) if reachable_record else load_repo_py(repo_root)
    docs = sorted(
        docs,
        key=lambda d: str((d.metadata or {}).get("source") or ""),
    )

    chunks = _split_docs(docs)
    _annotate_chunk_line_ranges(docs, chunks)

    model_name = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()
    if not model_name:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(persist_dir)
