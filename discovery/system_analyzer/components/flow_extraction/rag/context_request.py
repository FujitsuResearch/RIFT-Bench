import json
import os
import time
from datetime import datetime
from pathlib import Path
try:
    from ..model_client import call_model
except ImportError:
    from discovery.system_analyzer.components.flow_extraction.model_client import call_model

REPO_ROOT = Path(__file__).resolve().parents[2]
DEBUG_LOG_PATH = REPO_ROOT / "modular_imp" / "stages_outputs" / "rag_context_debug.log"
_EXEC_EXAMPLE_ENV = "MODULAR_IMP_EXECUTION_COMMAND_EXAMPLE_JSON"


def _task_arg_names(exec_example: dict | None) -> set[str]:
    if not isinstance(exec_example, dict):
        return {"--task", "--query", "--prompt"}
    raw = exec_example.get("task_arg_names")
    names: set[str] = set()
    if isinstance(raw, list):
        for x in raw:
            s = str(x or "").strip()
            if s:
                names.add(s)
    if not names:
        names = {"--task", "--query", "--prompt"}
    return names


def _load_execution_command_example_from_env() -> dict | None:
    p = str(os.environ.get(_EXEC_EXAMPLE_ENV) or "").strip()
    if not p:
        return None
    try:
        obj = json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _format_execution_command_context(exec_example: dict | None) -> str:
    if not isinstance(exec_example, dict):
        return ""
    runner = str(exec_example.get("runner") or "").strip()
    ep = exec_example.get("entrypoint") if isinstance(exec_example.get("entrypoint"), dict) else {}
    ep_kind = str(ep.get("kind") or "").strip()
    ep_val = str(ep.get("value") or "").strip()
    task_names = _task_arg_names(exec_example)
    args = exec_example.get("args") if isinstance(exec_example.get("args"), list) else []

    arg_lines: list[str] = []
    for item in args:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if name in task_names:
            continue
        val = item.get("value")
        arg_lines.append(f"- {name}: {val!r}")

    lines = [
        "Execution command example (non-task args):",
        f"- runner: {runner or '<unknown>'}",
        f"- entrypoint: {ep_kind or '<unknown>'}::{ep_val or '<unknown>'}",
    ]
    if arg_lines:
        lines.extend(arg_lines)
    else:
        lines.append("- args: <none>")
    return "\n".join(lines)


def _resolve_debug_log_path(debug_log_path: str | Path | None = None) -> Path:
    if debug_log_path is None:
        return DEBUG_LOG_PATH
    try:
        return Path(debug_log_path).resolve()
    except Exception:
        return DEBUG_LOG_PATH


def _append_rag_event(event: str, data: dict | None = None, *, debug_log_path: str | Path | None = None) -> None:
    log_path = _resolve_debug_log_path(debug_log_path)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write("\n" + "-" * 80 + "\n")
            f.write(f"timestamp: {datetime.utcnow().isoformat()}Z\n")
            f.write(f"event: {event}\n")
            if isinstance(data, dict) and data:
                f.write("data:\n")
                f.write(json.dumps(data, indent=2, ensure_ascii=False, default=str))
                f.write("\n")
    except Exception:
        pass


def _append_rag_debug_log(
    request_obj: dict,
    retrieved_context: str,
    next_payload: dict,
    retrieval_duration_ms: float | None = None,
    debug_log_path: str | Path | None = None,
) -> None:
    log_path = _resolve_debug_log_path(debug_log_path)
    # Quick-and-dirty debug logging for temporary inspection.
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"timestamp: {datetime.utcnow().isoformat()}Z\n")
            f.write("context_request:\n")
            f.write(json.dumps(request_obj, indent=2, ensure_ascii=False))
            if retrieval_duration_ms is not None:
                f.write(f"\nretrieval_duration_ms: {retrieval_duration_ms:.2f}")
            f.write("\n\nretrieved_context:\n")
            f.write(retrieved_context or "")
            f.write("\n\nnext_payload:\n")
            f.write(json.dumps(next_payload, indent=2, ensure_ascii=False, default=str))
            f.write("\n")
    except Exception:
        with log_path.open("a", encoding="utf-8") as f:
            f.write("failed to log context request\n")


def touch_rag_debug_log(debug_log_path: str | Path | None = None) -> None:
    log_path = _resolve_debug_log_path(debug_log_path)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")  # always clears
    except Exception:
        pass


def try_parse_context_request(response):
    # 1) JSON has non-empty "context_request" or "context_quest" -> treat as context request.
    # 2) Otherwise -> treat as final response (no more context needed).
    if isinstance(response, dict):
        obj = response
    elif isinstance(response, str):
        try:
            obj = json.loads(response)
        except Exception:
            return None
    else:
        return None

    if not isinstance(obj, dict):
        return None

    context_req_obj = obj.get("context_request")
    context_quest = obj.get("context_quest")
    context_obj = context_req_obj if context_req_obj else context_quest
    if not context_obj:
        return None

    # Support structured needs format under context_request/context_quest.
    if isinstance(context_obj, dict):
        needs = context_obj.get("needs")
        req_level_k = context_obj.get("k", 8)
        parsed_needs = []
        if isinstance(needs, list):
            for need in needs:
                if not isinstance(need, dict):
                    continue
                q = str(need.get("query") or "").strip()
                if not q:
                    continue
                try:
                    k_val = int(need.get("k", req_level_k))
                except Exception:
                    k_val = 8
                filters = need.get("filters") if isinstance(need.get("filters"), dict) else {}
                raw_ext = filters.get("file_ext")
                raw_path_contains = filters.get("path_contains")
                file_ext = []
                if isinstance(raw_ext, list):
                    for ext in raw_ext:
                        txt = str(ext or "").strip().lower()
                        if not txt:
                            continue
                        if not txt.startswith("."):
                            txt = "." + txt
                        file_ext.append(txt)
                path_contains = []
                if isinstance(raw_path_contains, list):
                    for part in raw_path_contains:
                        txt = str(part or "").strip()
                        if txt:
                            path_contains.append(txt)
                parsed_needs.append(
                    {
                        "kind": str(need.get("kind") or "").strip() or None,
                        "query": q,
                        "k": k_val,
                        "filters": {
                            "file_ext": file_ext,
                            "path_contains": path_contains,
                        },
                    }
                )

        if not parsed_needs:
            # Accept direct query shape too: {"context_request":{"query":"..."}}.
            direct_query = str(context_obj.get("query") or "").strip()
            if not direct_query:
                return None
            try:
                k_val = int(req_level_k)
            except Exception:
                k_val = 8
            parsed_needs = [
                {
                    "kind": None,
                    "query": direct_query,
                    "k": k_val,
                    "filters": {"file_ext": [], "path_contains": []},
                }
            ]

        return {
            "type": "context_request",
            "query": " ; ".join(str(n.get("query") or "") for n in parsed_needs),
            "request_id": context_obj.get("request_id"),
            "k": int(parsed_needs[0].get("k", 8)),
            "needs": parsed_needs,
            "source": "context_request" if context_req_obj else "context_quest",
        }

    # Accept shorthand: {"context_request":"..."} or {"context_quest":"..."}
    if isinstance(context_obj, str) and context_obj.strip():
        return {
            "type": "context_request",
            "query": context_obj.strip(),
            "k": 8,
            "needs": [
                {
                    "kind": None,
                    "query": context_obj.strip(),
                    "k": 8,
                    "filters": {"file_ext": [], "path_contains": []},
                }
            ],
            "source": "context_request" if context_req_obj else "context_quest",
        }

    return None

def format_context(docs, max_chars=12000):
    out = []
    total = 0
    for d in docs:
        path = d.metadata.get("source", "unknown")
        line_start = d.metadata.get("line_start")
        line_end = d.metadata.get("line_end")
        snippet = d.page_content
        if isinstance(line_start, int) and isinstance(line_end, int):
            line_info = f"{line_start}-{line_end}" if line_start != line_end else f"{line_start}"
            block = f"\n---\nFILE: {path}\nLINES: {line_info}\n{snippet}\n"
        else:
            block = f"\n---\nFILE: {path}\n{snippet}\n"
        if total + len(block) > max_chars:
            break
        out.append(block)
        total += len(block)
    return "".join(out)


def _parse_line_range(line_info):
    if not isinstance(line_info, str):
        return None, None
    txt = line_info.strip()
    if not txt:
        return None, None
    if "-" in txt:
        a, b = txt.split("-", 1)
        try:
            sa = int(a.strip())
            sb = int(b.strip())
            return min(sa, sb), max(sa, sb)
        except Exception:
            return None, None
    try:
        val = int(txt)
        return val, val
    except Exception:
        return None, None


def _parse_formatted_context_blocks(text):
    if not isinstance(text, str) or not text.strip():
        return []
    s = text.strip()
    if s.startswith("---\nFILE: "):
        s = "\n" + s
    parts = s.split("\n---\nFILE: ")
    out = []
    order = 0
    for part in parts:
        p = part.strip("\n")
        if not p:
            continue
        lines = p.splitlines()
        if not lines:
            continue
        file_path = str(lines[0]).strip()
        if not file_path:
            continue
        body_start = 1
        line_start = None
        line_end = None
        if len(lines) > 1 and lines[1].startswith("LINES: "):
            ls, le = _parse_line_range(lines[1][7:])
            line_start, line_end = ls, le
            body_start = 2
        body = "\n".join(lines[body_start:]).strip("\n")
        out.append(
            {
                "file": file_path,
                "line_start": line_start,
                "line_end": line_end,
                "body": body,
                "_order": order,
            }
        )
        order += 1
    return out


def _merge_snippet_lines(a, b):
    a_lines = a.splitlines() if isinstance(a, str) and a else []
    b_lines = b.splitlines() if isinstance(b, str) and b else []
    if not a_lines:
        return "\n".join(b_lines)
    if not b_lines:
        return "\n".join(a_lines)
    max_k = min(len(a_lines), len(b_lines))
    overlap = 0
    for k in range(max_k, 0, -1):
        if a_lines[-k:] == b_lines[:k]:
            overlap = k
            break
    merged = a_lines + b_lines[overlap:]
    deduped = []
    for ln in merged:
        if deduped and deduped[-1] == ln:
            continue
        deduped.append(ln)
    return "\n".join(deduped)


def _render_formatted_context_blocks(blocks):
    if not isinstance(blocks, list) or not blocks:
        return ""
    rendered = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        file_path = str(b.get("file") or "").strip()
        if not file_path:
            continue
        body = str(b.get("body") or "").strip("\n")
        ls = b.get("line_start")
        le = b.get("line_end")
        lines = ["---", f"FILE: {file_path}"]
        if isinstance(ls, int) and isinstance(le, int):
            line_txt = f"{ls}-{le}" if ls != le else str(ls)
            lines.append(f"LINES: {line_txt}")
        if body:
            lines.append(body)
        rendered.append("\n".join(lines))
    if not rendered:
        return ""
    return "\n" + "\n\n".join(rendered) + "\n"


def merge_aggregated_context(existing_context, new_context):
    existing_blocks = _parse_formatted_context_blocks(existing_context)
    new_blocks = _parse_formatted_context_blocks(new_context)
    if not existing_blocks and not new_blocks:
        old = str(existing_context or "").strip()
        new = str(new_context or "").strip()
        if not old:
            return new
        if not new:
            return old
        return old + "\n\n" + new

    all_blocks = [*existing_blocks, *new_blocks]
    seen = set()
    unique = []
    for b in all_blocks:
        key = (
            str(b.get("file") or ""),
            b.get("line_start"),
            b.get("line_end"),
            str(b.get("body") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(b)

    ranged_by_file = {}
    passthrough = []
    for b in unique:
        ls = b.get("line_start")
        le = b.get("line_end")
        if isinstance(ls, int) and isinstance(le, int):
            ranged_by_file.setdefault(str(b.get("file") or ""), []).append(b)
        else:
            passthrough.append(b)

    merged = []
    for file_path, items in ranged_by_file.items():
        seq = sorted(items, key=lambda x: (int(x.get("line_start") or 0), int(x.get("line_end") or 0), int(x.get("_order") or 0)))
        cur = dict(seq[0])
        for nxt in seq[1:]:
            nls = int(nxt.get("line_start"))
            nle = int(nxt.get("line_end"))
            cls = int(cur.get("line_start"))
            cle = int(cur.get("line_end"))
            if nls <= cle + 1:
                cur["line_start"] = min(cls, nls)
                cur["line_end"] = max(cle, nle)
                cur["body"] = _merge_snippet_lines(str(cur.get("body") or ""), str(nxt.get("body") or ""))
                cur["_order"] = min(int(cur.get("_order") or 0), int(nxt.get("_order") or 0))
            else:
                merged.append(cur)
                cur = dict(nxt)
        merged.append(cur)

    final_blocks = sorted([*passthrough, *merged], key=lambda x: int(x.get("_order") or 0))
    for b in final_blocks:
        b.pop("_order", None)
    return _render_formatted_context_blocks(final_blocks)


def _set_retriever_k(retriever, k):
    try:
        kv = int(k)
    except Exception:
        kv = 8
    kv = max(1, min(50, kv))
    try:
        if hasattr(retriever, "search_kwargs"):
            retriever.search_kwargs = {**getattr(retriever, "search_kwargs", {}), "k": kv}
    except Exception:
        pass
    try:
        if hasattr(retriever, "k"):
            setattr(retriever, "k", kv)
    except Exception:
        pass
    try:
        if hasattr(retriever, "bm25") and hasattr(retriever.bm25, "k"):
            retriever.bm25.k = kv
    except Exception:
        pass
    try:
        if hasattr(retriever, "vec") and hasattr(retriever.vec, "search_kwargs"):
            retriever.vec.search_kwargs = {**getattr(retriever.vec, "search_kwargs", {}), "k": kv}
    except Exception:
        pass
    return kv


def _apply_need_filters(docs, filters):
    if not isinstance(docs, list):
        return []
    if not isinstance(filters, dict):
        return docs
    raw_ext = filters.get("file_ext")
    raw_path_contains = filters.get("path_contains")
    ext_set = set()
    if isinstance(raw_ext, list):
        for ext in raw_ext:
            txt = str(ext or "").strip().lower()
            if not txt:
                continue
            if not txt.startswith("."):
                txt = "." + txt
            ext_set.add(txt)
    contains = []
    if isinstance(raw_path_contains, list):
        for part in raw_path_contains:
            txt = str(part or "").strip().lower()
            if txt:
                contains.append(txt)

    if not ext_set and not contains:
        return docs

    out = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        source = str(meta.get("source") or "")
        src_l = source.lower()
        if ext_set:
            suffix = Path(source).suffix.lower()
            if suffix not in ext_set:
                continue
        if contains:
            if not any(token in src_l for token in contains):
                continue
        out.append(d)
    return out


def run_task_with_rag(
    payload,
    model,
    retriever,
    max_rounds=6,
    debug_log_path: str | Path | None = None,
    use_first_round_context_request_hint: bool = False,
):
    _append_rag_event(
        "run_task_with_rag_invoked",
        {
            "model": model,
            "max_rounds": max_rounds,
            "payload_keys": sorted(list(payload.keys())) if isinstance(payload, dict) else [],
        },
        debug_log_path=debug_log_path,
    )
    working_payload = dict(payload)  # to avoid mutating the caller's payload
    working_payload.setdefault("additional_context", None)
    exec_example = (
        working_payload.get("execution_command_example")
        if isinstance(working_payload.get("execution_command_example"), dict)
        else _load_execution_command_example_from_env()
    )
    exec_ctx = _format_execution_command_context(exec_example)
    if exec_ctx:
        prev_ctx = str(working_payload.get("additional_context") or "").strip()
        working_payload["additional_context"] = merge_aggregated_context(prev_ctx, exec_ctx) if prev_ctx else exec_ctx
        working_payload["execution_command_example"] = exec_example
    history: list[dict] = []
    base_prompt = payload["prompt"]
    first_round_prompt = (
        base_prompt
        + "\n\nFIRST CONTEXT ROUND: you must return a context request."
        + ' Respond with the task response shape using "context_request" and at least one need.'
        + " Do not return a final answer in this round."
    )
    for i in range(max_rounds):
        round_no = i + 1
        _append_rag_event(
            "round_start",
            {
                "round": round_no,
                "max_rounds": max_rounds,
                "history_len": len(history),
                "has_additional_context": bool(working_payload.get("additional_context")),
            },
            debug_log_path=debug_log_path,
        )
        working_payload["context_request_history"] = history
        working_payload["context_request_queries"] = [
            str(item.get("query") or "") for item in history if isinstance(item, dict)
        ]
        if i == 0 and bool(use_first_round_context_request_hint):
            working_payload["prompt"] = first_round_prompt
        if i == max_rounds - 1:
            working_payload["prompt"] = (
                base_prompt
                + "\n\nFINAL CONTEXT ROUND: this is the last allowed context request."
                + " You must reach a final decision based on all currently provided context."
                + " Do not request additional context."
            )
            _append_rag_event("final_round_prompt_injected", {"round": round_no}, debug_log_path=debug_log_path)
        t_model_start = time.perf_counter()
        resp = call_model(working_payload, model)
        model_duration_ms = (time.perf_counter() - t_model_start) * 1000.0
        _append_rag_event(
            "model_response_received",
            {
                "round": round_no,
                "model_duration_ms": round(model_duration_ms, 2),
                "response_type": type(resp).__name__,
            },
            debug_log_path=debug_log_path,
        )
        req = try_parse_context_request(resp)
        if not req:
            _append_rag_event(
                "final_response_returned",
                {
                    "round": round_no,
                    "reason": "no_context_request_in_response",
                    "history_len": len(history),
                },
                debug_log_path=debug_log_path,
            )
            return resp  # final

        parsed_needs = req.get("needs") if isinstance(req.get("needs"), list) else []
        if not parsed_needs:
            parsed_needs = [
                {
                    "kind": None,
                    "query": str(req.get("query") or "").strip(),
                    "k": int(req.get("k", 8)),
                    "filters": {"file_ext": [], "path_contains": []},
                }
            ]
        context = ""
        retrieval_duration_ms = 0.0
        total_retrieved_docs = 0
        need_stats = []
        for need in parsed_needs:
            if not isinstance(need, dict):
                continue
            need_query = str(need.get("query") or "").strip()
            if not need_query:
                continue
            need_k = _set_retriever_k(retriever, need.get("k", 8))
            filters = need.get("filters") if isinstance(need.get("filters"), dict) else {}
            t_retrieval_start = time.perf_counter()
            docs_raw = retriever.get_relevant_documents(need_query)
            docs_filtered = _apply_need_filters(docs_raw, filters)
            docs = docs_filtered[:need_k]
            retrieval_duration_ms += (time.perf_counter() - t_retrieval_start) * 1000.0
            total_retrieved_docs += len(docs)
            ctx_piece = format_context(docs)
            if ctx_piece:
                context = merge_aggregated_context(context, ctx_piece) if context else ctx_piece
            need_stats.append(
                {
                    "query": need_query,
                    "k": need_k,
                    "filters": filters,
                    "raw_docs": len(docs_raw) if isinstance(docs_raw, list) else None,
                    "filtered_docs": len(docs_filtered) if isinstance(docs_filtered, list) else None,
                    "kept_docs": len(docs),
                }
            )
        _append_rag_event(
            "context_retrieved",
            {
                "round": round_no,
                "request_id": req.get("request_id"),
                "k": req.get("k", 8),
                "retrieved_docs": total_retrieved_docs,
                "needs_count": len(parsed_needs),
                "need_stats": need_stats,
                "retrieval_duration_ms": round(retrieval_duration_ms, 2),
                "retrieved_chars": len(context or ""),
            },
            debug_log_path=debug_log_path,
        )
        history_entry = {
            "round": round_no,
            "request_id": req.get("request_id"),
            "query": req.get("query"),
            "k": req.get("k", 8),
            "needs": parsed_needs,
            "source": req.get("source"),
            "retrieved_context": context,
        }
        history = [*history, history_entry]
        prev_context = working_payload.get("additional_context") or ""
        if context:
            working_payload["additional_context"] = merge_aggregated_context(prev_context, context)
        working_payload["context_request_history"] = history
        working_payload["context_request_queries"] = [
            str(item.get("query") or "") for item in history if isinstance(item, dict)
        ]
        working_payload["prompt"] = (
            base_prompt
            + "\n\nIf more evidence is still needed, return the same task response shape with "
              '"context_request": {"request_id":"...","needs":[{"kind":"...","query":"...","k":8,"filters":{"file_ext":[".py"],"path_contains":["..."]}}]}.'
        )
        _append_rag_debug_log(
            req,
            context,
            working_payload,
            retrieval_duration_ms=retrieval_duration_ms,
            debug_log_path=debug_log_path,
        )

    _append_rag_event(
        "max_rounds_reached_returning_last_response",
        {
            "max_rounds": max_rounds,
            "history_len": len(history),
        },
        debug_log_path=debug_log_path,
    )
    return resp  # last response if it keeps requesting
