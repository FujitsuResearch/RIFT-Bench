import os

from langchain_openai import AzureChatOpenAI


class LLM:
    def __init__(self) -> None:
        self.client = AzureChatOpenAI(
            model=os.getenv("AZURE_MODEL_NAME"),
            azure_deployment=os.getenv("AZURE_MODEL_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            temperature=0.0,
        )
