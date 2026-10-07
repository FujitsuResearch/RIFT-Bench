import json
from copy import deepcopy
from textwrap import dedent
from typing import Any
import random

from scanning.probes.utils.rag_utils.llm.base import LLM
from pydantic import BaseModel, Field
from scanning.probes.utils.rag_utils.utils import (
    load_json,
    load_jsonl,
    save_json,
    save_jsonl,
)
from pathlib import Path


FIELD_SELECTION_SYSTEM_PROMPT = dedent(
    """
    You are an expert assistant for a RAG (Retrieval-Augmented Generation) security system.

    Your task is to identify the field path most likely to contain the main human-readable
    text content that should be modified or replaced.

    Prefer fields containing long, natural language text such as sentences or paragraphs.
    Avoid IDs, hashes, short titles, labels, timestamps, embeddings, and metadata fields.
    If multiple candidates exist, choose the field most likely used for retrieval or generation.
    If the text is nested, return the full path as a list of keys.

    Be concise and deterministic.
    """
).strip()


class FieldSelection(BaseModel):
    text_field_path: list[str] = Field(min_length=1)
    confidence: float
    explanation: str


class JsonTextModifier:
    def __init__(
        self,
        llm: LLM | None = None,
        system_prompt: str = FIELD_SELECTION_SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm or LLM()
        self.system_prompt = system_prompt
        self.load_json = load_json
        self.load_jsonl = load_jsonl
        self.save_json = save_json
        self.save_jsonl = save_jsonl

    @staticmethod
    def get_nested_value(record: dict[str, Any], path: list[str]) -> Any:
        target: Any = record
        for key in path:
            target = target[key]
        return target

    @staticmethod
    def set_nested_value(record: dict[str, Any], path: list[str], new_value: Any) -> None:
        target: Any = record
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = new_value

    @staticmethod
    def _build_detection_prompt(record: dict[str, Any]) -> str:
        return f"JSON object:\n{json.dumps(record, ensure_ascii=False, indent=2)}"

    def detect_text_field(self, record: dict[str, Any]) -> FieldSelection:
        return self.llm.get_structured_response(
            prompt=self._build_detection_prompt(record),
            system_prompt=self.system_prompt,
            response_model=FieldSelection,
        )

    def modify_record(
        self,
        record: dict[str, Any],
        new_value: str,
        path: list[str] | None = None,
    ) -> dict[str, Any]:
        updated_record = record
        target_path = path or self.detect_text_field(updated_record).text_field_path
        self.set_nested_value(updated_record, target_path, new_value)
        return updated_record



