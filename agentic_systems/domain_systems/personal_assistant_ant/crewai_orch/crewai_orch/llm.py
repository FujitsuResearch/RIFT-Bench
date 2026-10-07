import os

from crewai import LLM


def build_llm() -> LLM:
    model_name = os.getenv("MODEL_NAME")
    return LLM(
        model=f"anthropic/{model_name}",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )
