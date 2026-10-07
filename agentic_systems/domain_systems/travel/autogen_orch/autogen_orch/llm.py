import os

from autogen_ext.models.openai import AzureOpenAIChatCompletionClient


class LLM:
    def __init__(self) -> None:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        model_name = os.getenv("AZURE_MODEL_NAME")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_API_VERSION")
        self.client = AzureOpenAIChatCompletionClient(
            model=model_name,
            azure_endpoint=azure_endpoint,
            azure_deployment=model_name,
            api_version=api_version,
            api_key=api_key,
            temperature=0.0,
        )
