import os
from urllib.parse import urlparse

from langchain_openai import AzureChatOpenAI


def _azure_root(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}"


class LLM:
    def __init__(self):
        endpoint = _azure_root(os.getenv("AZURE_OPENAI_ENDPOINT"))
        self.client = AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=endpoint,
            azure_deployment=os.getenv("AZURE_MODEL_NAME"),
            temperature=0.0,
        )
