import os
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
import litellm
from litellm import completion
from pydantic import BaseModel
from scanning.evaluators.evaluator_runtime import record_response_usage
litellm.enable_json_schema_validation = True


StructuredResponseT = TypeVar("StructuredResponseT", bound=BaseModel)


class LLM:
    """Small Azure OpenAI wrapper for attack-generation flows via LiteLLM."""

    def __init__(
        self,
        model_name: str | None = None,
        deployment_name: str | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        api_version: str | None = None,
        temperature: float = 1,
        env_path: str | os.PathLike[str] | None = None,
    ) -> None:
        load_dotenv(
            dotenv_path=Path(env_path)
            if env_path
            else Path("rift.env"),
            override=True,
        )

        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version or os.getenv("AZURE_API_VERSION")
        self.deployment_name = deployment_name or os.getenv("AZURE_DEPLOYMENT_NAME")
        self.model_name = model_name or os.getenv("AZURE_MODEL_NAME") or self.deployment_name
        self.temperature = temperature

        missing = [
            name
            for name, value in (
                ("AZURE_OPENAI_API_KEY", self.api_key),
                ("AZURE_OPENAI_ENDPOINT", self.endpoint),
                ("AZURE_API_VERSION", self.api_version),
                ("AZURE_DEPLOYMENT_NAME", self.deployment_name),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "Missing required Azure OpenAI configuration: " + ", ".join(missing)
            )

        self.model = f"azure/{self.deployment_name}"

    def get_response(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = completion(
            model=self.model,
            messages=messages,
            api_key=self.api_key,
            api_base=self.endpoint,
            api_version=self.api_version,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        record_response_usage(response)
        return self._extract_text(response)

    def get_structured_response(
        self,
        prompt: str,
        response_model: type[StructuredResponseT],
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> StructuredResponseT:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = completion(
            model=self.model,
            messages=messages,
            api_key=self.api_key,
            api_base=self.endpoint,
            api_version=self.api_version,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens,
            response_format=response_model,
            **kwargs,
        )
        record_response_usage(response)

        message = response.choices[0].message
        parsed = getattr(message, "parsed", None)
        if isinstance(parsed, response_model):
            return parsed

        return response_model.model_validate_json(self._extract_text(response))

    @staticmethod
    def _extract_text(response: Any) -> str:
        message = response.choices[0].message
        content = message.content

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            return "\n".join(part for part in text_parts if part)

        return str(content)
