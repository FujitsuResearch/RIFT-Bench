import os

from langchain_anthropic import ChatAnthropic


class LLM:
    def __init__(self) -> None:
        self.client = ChatAnthropic(
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("API_KEY"),
            model=os.getenv("MODEL_NAME"),
        )