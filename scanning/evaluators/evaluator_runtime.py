from __future__ import annotations

import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterator

try:
    import tiktoken
except ImportError:
    tiktoken = None


_TOKEN_FALLBACK_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class EvaluatorRuntimeMetrics:
    llm_calls: int = 0
    total_tokens_used: int = 0


_CURRENT_METRICS: ContextVar[EvaluatorRuntimeMetrics | None] = ContextVar(
    "evaluator_runtime_metrics",
    default=None,
)


@contextmanager
def collect_evaluator_runtime_metrics() -> Iterator[EvaluatorRuntimeMetrics]:
    metrics = EvaluatorRuntimeMetrics()
    token = _CURRENT_METRICS.set(metrics)
    try:
        yield metrics
    finally:
        _CURRENT_METRICS.reset(token)


def _active_metrics() -> EvaluatorRuntimeMetrics | None:
    return _CURRENT_METRICS.get()


def _to_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value if value >= 0 else None

    if isinstance(value, float):
        if value < 0:
            return None
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        try:
            parsed = int(normalized)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None

    return None


def record_llm_usage(
    *,
    llm_calls: int = 1,
    prompt_tokens: Any = None,
    completion_tokens: Any = None,
    total_tokens: Any = None,
) -> None:
    metrics = _active_metrics()
    if metrics is None:
        return

    normalized_llm_calls = _to_non_negative_int(llm_calls)
    metrics.llm_calls += normalized_llm_calls if normalized_llm_calls is not None else 0

    resolved_total_tokens = _to_non_negative_int(total_tokens)
    if resolved_total_tokens is None:
        resolved_total_tokens = (
            (_to_non_negative_int(prompt_tokens) or 0)
            + (_to_non_negative_int(completion_tokens) or 0)
        )

    metrics.total_tokens_used += resolved_total_tokens


def _iter_usage_candidates(value: Any):
    queue = [value]
    seen_ids: set[int] = set()

    while queue:
        current = queue.pop(0)
        if current is None:
            continue

        current_id = id(current)
        if current_id in seen_ids:
            continue
        seen_ids.add(current_id)
        yield current

        if isinstance(current, dict):
            for key in (
                "usage",
                "token_usage",
                "usage_metadata",
                "response_metadata",
                "llm_output",
                "metadata",
            ):
                nested = current.get(key)
                if nested is not None:
                    queue.append(nested)
            continue

        for attr in (
            "usage",
            "token_usage",
            "usage_metadata",
            "response_metadata",
            "llm_output",
            "metadata",
        ):
            nested = getattr(current, attr, None)
            if nested is not None:
                queue.append(nested)


def _extract_usage_fields(value: Any) -> tuple[int | None, int | None, int | None]:
    if isinstance(value, dict):
        getter = value.get
    else:
        getter = lambda key: getattr(value, key, None)

    prompt_tokens = _to_non_negative_int(getter("prompt_tokens"))
    if prompt_tokens is None:
        prompt_tokens = _to_non_negative_int(getter("input_tokens"))

    completion_tokens = _to_non_negative_int(getter("completion_tokens"))
    if completion_tokens is None:
        completion_tokens = _to_non_negative_int(getter("output_tokens"))

    total_tokens = _to_non_negative_int(getter("total_tokens"))
    return prompt_tokens, completion_tokens, total_tokens


def record_response_usage(response: Any, *, default_llm_calls: int = 1) -> None:
    for candidate in _iter_usage_candidates(response):
        prompt_tokens, completion_tokens, total_tokens = _extract_usage_fields(candidate)
        if (
            prompt_tokens is not None
            or completion_tokens is not None
            or total_tokens is not None
        ):
            record_llm_usage(
                llm_calls=default_llm_calls,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            return

    record_llm_usage(llm_calls=default_llm_calls, total_tokens=0)


def _serialize_for_token_estimate(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    model_dump_json = getattr(value, "model_dump_json", None)
    if callable(model_dump_json):
        try:
            serialized = model_dump_json()
        except TypeError:
            serialized = None
        if isinstance(serialized, str):
            return serialized

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped_value = model_dump(mode="json")
        except TypeError:
            dumped_value = None
        if dumped_value is not None:
            value = dumped_value

    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _normalize_model_name(model_name: Any) -> str | None:
    if not isinstance(model_name, str):
        return None
    normalized = model_name.strip()
    return normalized or None


@lru_cache(maxsize=32)
def _get_encoder_for_model(model_name: str | None):
    if tiktoken is None:
        return None

    normalized_model_name = _normalize_model_name(model_name)
    if normalized_model_name:
        try:
            return tiktoken.encoding_for_model(normalized_model_name)
        except Exception:
            pass
        try:
            return tiktoken.get_encoding(normalized_model_name)
        except Exception:
            pass

    for fallback_name in ("o200k_base", "cl100k_base"):
        try:
            return tiktoken.get_encoding(fallback_name)
        except Exception:
            continue

    return None


def estimate_text_token_count(text: str, model_name: str | None = None) -> int:
    if not text:
        return 0

    encoder = _get_encoder_for_model(model_name)
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass

    return len(_TOKEN_FALLBACK_PATTERN.findall(text))


def record_estimated_llm_usage(
    *,
    prompt: Any,
    response: Any = None,
    model_name: str | None = None,
    llm_calls: int = 1,
) -> None:
    prompt_text = _serialize_for_token_estimate(prompt)
    response_text = _serialize_for_token_estimate(response)
    total_tokens = estimate_text_token_count(prompt_text, model_name) + estimate_text_token_count(
        response_text,
        model_name,
    )
    record_llm_usage(llm_calls=llm_calls, total_tokens=total_tokens)
