import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI

_ENV_BOOTSTRAPPED = False
_SI_ENV_VALUES: Dict[str, str] = {}


# These are the only env vars that must exist for the SI Azure client to run.
_ENV_KEYS_TO_CHECK = (
    "AZURE_OPENAI_ENDPOINT_SI",
    "AZURE_OPENAI_DEPLOYMENT_SI",
    "AZURE_OPENAI_API_KEY_SI",
    "AZURE_OPENAI_API_VERSION_SI"
)


def _strip_outer_quotes(v: str) -> str:
    """Remove a single pair of matching surrounding quotes (' or ") from a string, if present."""
    s = v.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s


def _load_env_file_if_exists(path: Path) -> Dict[str, str]:
    """Read SI settings from a `.env`-style file without exporting them to child processes."""
    if not path.exists() or not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    values: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        if key not in _ENV_KEYS_TO_CHECK:
            continue
        values[key] = _strip_outer_quotes(v)
    return values


def _has_model_env_ready() -> bool:
    """Return whether all required SI Azure settings are available to this client."""
    return all(
        str(os.getenv(key) or _SI_ENV_VALUES.get(key) or "").strip()
        for key in _ENV_KEYS_TO_CHECK
    )


def _ensure_env_bootstrap() -> None:
    """Load SI Azure settings privately, preferring process values and SI-specific env files."""
    global _ENV_BOOTSTRAPPED, _SI_ENV_VALUES
    if _ENV_BOOTSTRAPPED:
        return

    # Read only SI-specific env files, and keep their credentials inside this
    # module so target-system subprocesses cannot inherit them.
    si_env_candidates = []
    explicit_env_file = str(os.getenv("STRUCTURE_SI_ENV_FILE") or "").strip()
    if explicit_env_file:
        si_env_candidates.append(Path(explicit_env_file).expanduser())
    si_env_candidates.extend([
        Path.cwd() / ".env.si",
        Path(__file__).resolve().parents[1] / ".env.si",
        Path("/sandbox/.env.si"),
    ])

    for path in si_env_candidates:
        for key, value in _load_env_file_if_exists(path).items():
            _SI_ENV_VALUES.setdefault(key, value)
        if _has_model_env_ready():
            _ENV_BOOTSTRAPPED = True
            return

    _ENV_BOOTSTRAPPED = True


def _max_prompt_chars() -> int:
    """Return the max allowed prompt length in characters (from env, default 10M, floored at 1024)."""
    try:
        return max(1024, int(os.getenv("STRUCTURE_LLM_MAX_PROMPT_CHARS", "10000000")))
    except Exception:
        return 10000000


def _openai_client_settings(model: str) -> Dict[str, Any]:
    """Resolve the SI Azure OpenAI settings after env bootstrapping has populated any missing values."""
    _ensure_env_bootstrap()
    requested_model = str(model or "").strip()
    api_base = (_SI_ENV_VALUES.get("AZURE_OPENAI_ENDPOINT_SI") or os.getenv("AZURE_OPENAI_ENDPOINT_SI", "")).strip()
    api_version = (_SI_ENV_VALUES.get("AZURE_OPENAI_API_VERSION_SI") or os.getenv("AZURE_OPENAI_API_VERSION_SI", "")).strip()
    env_deployment = (_SI_ENV_VALUES.get("AZURE_OPENAI_DEPLOYMENT_SI") or os.getenv("AZURE_OPENAI_DEPLOYMENT_SI", "")).strip()
    api_key = (_SI_ENV_VALUES.get("AZURE_OPENAI_API_KEY_SI") or os.getenv("AZURE_OPENAI_API_KEY_SI", "")).strip()

    # An explicit model argument overrides the env deployment name.
    deployment = requested_model or env_deployment
    if not deployment:
        raise RuntimeError("Azure is configured but no deployment name was provided.")
    if not api_base:
        raise RuntimeError("Azure OpenAI endpoint is not configured.")
    if not api_key:
        raise RuntimeError("Azure OpenAI API key is not configured.")

    return {
        "provider": "azure_openai",
        "model": deployment,
        "api_base": api_base,
        "api_version": api_version,
        "api_key": api_key,
    }


def _extract_responses_text(resp: Any) -> str:
    """Pull the text out of an OpenAI Responses object (output_text, else concatenated output[].content[].text); '' if none."""
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    try:
        pieces = []
        for item in list(getattr(resp, "output", []) or []):
            for c in list(getattr(item, "content", []) or []):
                t = getattr(c, "text", None)
                if isinstance(t, str) and t:
                    pieces.append(t)
        if pieces:
            return "\n".join(pieces).strip()
    except Exception:
        pass
    return ""


def _as_int(value: Any) -> int:
    """Best-effort convert a usage counter to a non-negative int (0 when unavailable)."""
    try:
        n = int(value)
    except Exception:
        return 0
    return n if n >= 0 else 0


def _extract_usage(resp: Any) -> Dict[str, int]:
    """Pull token counts from an OpenAI Responses object, tolerating attribute or dict shapes."""
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    if usage is None:
        return {}

    def _get(obj: Any, key: str) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    in_details = _get(usage, "input_tokens_details")
    out_details = _get(usage, "output_tokens_details")
    return {
        "input_tokens": _as_int(_get(usage, "input_tokens")),
        "output_tokens": _as_int(_get(usage, "output_tokens")),
        "total_tokens": _as_int(_get(usage, "total_tokens")),
        # Detail breakdowns matter for real cost (cached input is cheaper; reasoning adds output).
        "cached_input_tokens": _as_int(_get(in_details, "cached_tokens")),
        "reasoning_tokens": _as_int(_get(out_details, "reasoning_tokens")),
    }


def _process_stage_label() -> str:
    """Infer the pipeline stage from this subprocess's own argv (e.g. `-m pkg.graph_correctness.main`)."""
    argv = list(sys.argv or [])
    script = argv[0] if argv else ""
    label = "unknown"
    try:
        p = Path(script)
        label = p.parent.name if p.name in ("main.py", "__main__.py") else (p.stem or "unknown")
    except Exception:
        label = "unknown"
    # Disambiguate stages that run multiple times (e.g. graph_correctness) by their --phase.
    if "--phase" in argv:
        try:
            label = f"{label}:{argv[argv.index('--phase') + 1]}"
        except Exception:
            pass
    return label or "unknown"


def _usage_log_path() -> Path | None:
    """Resolve where to append the per-call usage log, preferring an explicit env override."""
    env = os.getenv("STRUCTURE_LLM_USAGE_LOG", "").strip()
    if env:
        return Path(env)
    argv = list(sys.argv or [])
    if "--out_dir" in argv:
        try:
            return Path(argv[argv.index("--out_dir") + 1]) / "llm_usage.jsonl"
        except Exception:
            pass
    return Path.cwd() / "llm_usage.jsonl"


def _log_usage(model: str, usage: Dict[str, int], latency_sec: float) -> None:
    """Append one best-effort usage record per model call; never raise into the caller."""
    if os.getenv("STRUCTURE_LLM_USAGE_LOG_DISABLE", "0").strip() in ("1", "true", "True"):
        return
    try:
        record = {
            "ts": round(time.time(), 3),
            "pid": os.getpid(),
            "stage": _process_stage_label(),
            "model": str(model or ""),
            "latency_sec": round(float(latency_sec), 3),
            **usage,
        }
        path = _usage_log_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Telemetry must never interfere with the actual model call.
        pass


def _call_openai_response(prompt: str, settings: Dict[str, Any], timeout: int) -> str:
    """Send the prompt through the API-key-based Azure OpenAI client and return the response text."""
    client = OpenAI(
        api_key=str(settings["api_key"]),
        base_url=str(settings["api_base"]).rstrip("/") + "/openai/v1/",
        timeout=timeout,
    )
    start = time.perf_counter()
    resp = client.responses.create(model=str(settings["model"]), input=prompt, reasoning={"effort":"low"})
    _log_usage(str(settings["model"]), _extract_usage(resp), time.perf_counter() - start)
    return _extract_responses_text(resp)


def call_model_openai(payload: Dict[str, Any], model: str, timeout: int = 1000) -> str:
    """Serialize the payload, call the SI Azure model, and return either model text or a JSON fallback error payload."""
    # The pipeline always sends one JSON payload as the prompt body.
    prompt = json.dumps(payload, ensure_ascii=False)
    max_chars = _max_prompt_chars()
    if len(prompt) > max_chars:
        return json.dumps(
            {
                "error": "payload_too_large",
                "reason": "prompt exceeds max allowed length",
                "prompt_chars": len(prompt),
                "max_prompt_chars": max_chars,
            },
            ensure_ascii=False,
        )
    settings = _openai_client_settings(model)

    # If the request ultimately fails, some stages prefer a strict JSON error object
    # over raising, so downstream logic can continue conservatively.
    def _fallback_enabled() -> bool:
        return os.getenv("STRUCTURE_LLM_FALLBACK_ON_ERROR", "1").strip() not in ("0", "false", "False")

    def _fallback_response(exc: Exception) -> str:
        return json.dumps(
            {
                "_llm_error": exc.__class__.__name__,
                "_llm_error_message": str(exc)[:1200],
            },
            ensure_ascii=False,
        )

    def _is_retryable(exc: Exception) -> bool:
        msg = str(exc).lower()
        exc_name = exc.__class__.__name__.lower()
        retry_markers = (
            "connection error",
            "server disconnected",
            "connection reset",
            "connection aborted",
            "connection refused",
            "network is unreachable",
            "temporary failure in name resolution",
            "name resolution",
            "failed to resolve",
            "nodename nor servname provided",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
            "internal server error",
            "rate limit",
            "too many requests",
            "429",
            "502",
            "503",
            "504",
            "apierror",
        )
        non_retry_markers = (
            "authentication",
            "unauthorized",
            "forbidden",
            "invalid api key",
            "invalid_request_error",
            "not found",
            "deploymentnotfound",
            "context_length_exceeded",
            "content_filter",
        )
        if any(x in msg for x in non_retry_markers):
            return False
        if any(x in msg for x in retry_markers):
            return True
        return ("apiconnectionerror" in exc_name) or ("connectionerror" in exc_name)

    max_attempts = max(1, int(os.getenv("STRUCTURE_LLM_MAX_ATTEMPTS", "6")))
    base_delay = float(os.getenv("STRUCTURE_LLM_RETRY_BASE_SEC", "1.5"))
    max_delay = float(os.getenv("STRUCTURE_LLM_RETRY_MAX_SEC", "20"))
    retry_log = os.getenv("STRUCTURE_LLM_RETRY_LOG", "1").strip() not in ("0", "false", "False")

    # Retry only for transient transport/service failures; auth and request-shape
    # failures should surface immediately or through the JSON fallback.
    for attempt in range(1, max_attempts + 1):
        try:
            return _call_openai_response(prompt, settings, timeout)
        except Exception as exc:
            if attempt >= max_attempts or not _is_retryable(exc):
                if _fallback_enabled():
                    return _fallback_response(exc)
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay = delay + random.uniform(0, min(0.5, delay * 0.15))
            if retry_log:
                print(
                    f"[llm-retry] attempt {attempt}/{max_attempts} failed "
                    f"({exc.__class__.__name__}); sleeping {delay:.2f}s",
                    file=sys.stderr,
                )
            time.sleep(delay)


def call_model(payload: Dict[str, Any], model: str) -> str:
    """Public entry point: call the model with the given payload and return its text response."""
    return call_model_openai(payload, model)
