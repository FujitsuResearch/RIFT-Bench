import os

from autogen_core.models import ModelFamily
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

CUSTOM_MODEL_INFO = {
    "claude-fable-5": {
        "vision": True,
        "function_calling": True,
        "json_output": True,
        "family": ModelFamily.UNKNOWN,
        "structured_output": False,
        "multiple_system_messages": False,
    },
}


class LLM:
    def __init__(self) -> None:
        model_name = os.getenv("MODEL_NAME")
        client_kwargs = {
            "base_url": os.getenv("BASE_URL"),
            "api_key": os.getenv("API_KEY"),
            "model": model_name,
        }
        model_info = CUSTOM_MODEL_INFO.get(model_name)
        if model_info is not None:
            client_kwargs["model_info"] = model_info
        self.client = AnthropicChatCompletionClient(**client_kwargs)
