import os

from langchain_openai import AzureChatOpenAI


class LLM:
    def __init__(self) -> None:
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        model_name = os.getenv("AZURE_MODEL_NAME")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_API_VERSION")
        self.client = AzureChatOpenAI(
            model=model_name,
            azure_deployment=model_name,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            temperature=0.0,
        )
