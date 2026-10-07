import os

from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.models import ModelFamily


class LLM:
    def __init__(self) -> None:
        self.client = AzureOpenAIChatCompletionClient(
            model=os.getenv("AZURE_MODEL_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_MODEL_NAME"),
            api_version=os.getenv("AZURE_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            model_info={
                "vision": True,
                "function_calling": True,
                "json_output": True,
                "structured_output": True,
                "family": ModelFamily.GPT_5,
            },
        )
