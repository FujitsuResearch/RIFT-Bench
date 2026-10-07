from __future__ import annotations

import re
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict
 
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

 
_ENV_BOOTSTRAPPED = False
 
 
def _strip_outer_quotes(v: str) -> str:
    s = v.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s
 
 
def _load_env_file_if_exists(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return
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
        if not key:
            continue
        if key in os.environ:
            continue
        os.environ[key] = _strip_outer_quotes(v)
 
 
def _ensure_env_bootstrap() -> None:
    global _ENV_BOOTSTRAPPED
    if _ENV_BOOTSTRAPPED:
        return
    candidates = []
    si_env_file = os.getenv("STRUCTURE_SI_ENV_FILE", "").strip()
    if si_env_file:
        candidates.append(Path(si_env_file).expanduser())
    candidates.extend([
        Path("discovery/.env"),
        Path(".env")
        # Path.cwd() / ".env",
        # Path(__file__).resolve().parents[1] / ".env",
    ])
    for p in candidates:
        _load_env_file_if_exists(p)
    _ENV_BOOTSTRAPPED = True
 
 
def _max_prompt_chars() -> int:
    try:
        return max(1024, int(os.getenv("STRUCTURE_LLM_MAX_PROMPT_CHARS", "10000000")))
    except Exception:
        return 10000000
 
 
def _openai_client_settings(model: str) -> Dict[str, Any]:
    _ensure_env_bootstrap()
    api_base = (
        os.getenv("AZURE_OPENAI_ENDPOINT_SI", "").strip()
        or os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        or os.getenv("AZURE_AI_ENDPOINT", "").strip()
    )
    api_version = os.getenv("AZURE_OPENAI_API_VERSION_SI", "").strip()
    if not api_version:
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
    deployment = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT_SI", "").strip()
        or os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
        or os.getenv("AZURE_DEPLOYMENT_NAME", "").strip()
    )
    api_key = (
        os.getenv("AZURE_OPENAI_API_KEY_SI", "").strip()
        or os.getenv("AZURE_OPENAI_API_KEY", "").strip()
        or os.getenv("AZURE_API_KEY", "").strip()
    )
    tenant_id = os.getenv("AZURE_TENANT_ID", "").strip()
 
    # Non-Azure fallback (standard OpenAI): only if Azure endpoint/version are not set.
    if not (api_base and api_version):
        return {"provider": "openai", "model": model}
 
    if not deployment:
        raise RuntimeError("Azure is configured but no deployment name was provided.")
 
    return {
        "provider": "azure_openai",
        "model": deployment,
        "api_base": api_base,
        "api_version": api_version,
        "api_key": api_key,
        "tenant_id": tenant_id,
        "auth_mode": "api_key" if api_key else "azure_cli",
    }
 
   

def _extract_responses_text_from_chat_completion(resp: ChatCompletion) -> str:
    return resp.choices[0].message.content

def _extract_responses_text(resp: Any) -> str:
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    if isinstance(resp, ChatCompletion):
        return _extract_responses_text_from_chat_completion(resp)
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
 
 
def _call_openai_response(prompt: str, settings: Dict[str, Any], timeout: int) -> str:
    provider = str(settings.get("provider") or "")
    if provider == "azure_openai":
        api_key = str(settings.get("api_key") or "").strip()
        if api_key:
            client = AzureOpenAI(
                api_key=api_key,
                api_version=str(settings["api_version"]),
                azure_endpoint=str(settings["api_base"]),
                timeout=timeout,
            )
        else:
            from azure.identity import AzureCliCredential, get_bearer_token_provider
 
            tenant_id = str(settings.get("tenant_id") or "").strip()
            if tenant_id:
                cred = AzureCliCredential(tenant_id=tenant_id)
            else:
                cred = AzureCliCredential()
            token_provider = get_bearer_token_provider(
                cred, "https://cognitiveservices.azure.com/.default"
            )
            client = AzureOpenAI(
                api_version=str(settings["api_version"]),
                azure_endpoint=str(settings["api_base"]),
                azure_ad_token_provider=token_provider,
                timeout=timeout,
            )
    else:
        client = OpenAI(timeout=timeout)
 
    resp = client.chat.completions.create(model=str(settings["model"]), messages=[{"role": "user", "content": prompt}])
    return _extract_responses_text(resp)
 
 
def call_model_openai(payload: Dict[str, Any], model: str, timeout: int = 300) -> str:
    prompt = json.dumps(payload, ensure_ascii=False)
    max_chars = _max_prompt_chars()
    if len(prompt) > max_chars:
        # Non-crashing guard for oversized requests.
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
 
    def _fallback_enabled() -> bool:
        return os.getenv("STRUCTURE_LLM_FALLBACK_ON_ERROR", "1").strip() not in ("0", "false", "False")
 
    def _fallback_response(exc: Exception) -> str:
        # Keep pipeline alive with a strict JSON object; downstream code already
        # handles missing/empty keys conservatively.
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
    last_exc: Exception | None = None
 
    for attempt in range(1, max_attempts + 1):
        try:
            return _call_openai_response(prompt, settings, timeout)
        except Exception as exc:
            last_exc = exc
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
 
    if last_exc is not None:
        if _fallback_enabled():
            return _fallback_response(last_exc)
        raise last_exc
    raise RuntimeError("LLM call failed without exception details.")
 
 
def call_model(payload: Dict[str, Any], model: str) -> str:
    return call_model_openai(payload, model)
