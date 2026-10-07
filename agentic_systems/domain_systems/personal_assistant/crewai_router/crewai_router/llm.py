import os
from urllib.parse import urlparse

from crewai import LLM


def _azure_root(endpoint: str | None) -> str | None:
    if not endpoint:
        return endpoint
    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        return endpoint
    return f"{parsed.scheme}://{parsed.netloc}"


def _is_openai_v1_endpoint(endpoint: str | None) -> bool:
    if not endpoint:
        return False
    path = urlparse(endpoint).path.rstrip("/")
    return path.endswith("/openai/v1")


def build_llm() -> LLM:
    model_name = os.getenv("AZURE_MODEL_NAME")
    api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION")

    if not model_name or not api_key or not endpoint:
        raise ValueError(
            "Missing Azure LLM configuration. Set AZURE_MODEL_NAME, "
            "AZURE_OPENAI_API_KEY, and AZURE_OPENAI_ENDPOINT."
        )

    if _is_openai_v1_endpoint(endpoint):
        return LLM(
            model=model_name,
            api_key=api_key,
            base_url=endpoint.rstrip("/") + "/",
            temperature=0.0,
        )

    if not api_version:
        raise ValueError(
            "Missing AZURE_API_VERSION for non-v1 Azure endpoints. "
            "Set AZURE_API_VERSION or use an /openai/v1/ endpoint."
        )

    return LLM(
        model=f"azure/{model_name}",
        api_version=api_version,
        api_key=api_key,
        endpoint=_azure_root(endpoint),
        temperature=0.0,
    )
