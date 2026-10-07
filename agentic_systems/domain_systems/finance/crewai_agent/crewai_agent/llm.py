import os
from urllib.parse import urlparse

from crewai import LLM


def _azure_root(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}"


def build_llm() -> LLM:
    model_name = os.getenv("AZURE_MODEL_NAME")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    root = _azure_root(os.getenv("AZURE_OPENAI_ENDPOINT"))
    endpoint = f"{root}/openai/deployments/{model_name}" if root else None
    return LLM(
        model=f"azure/{model_name}",
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=api_key,
        endpoint=endpoint,
        temperature=0.0,
    )
