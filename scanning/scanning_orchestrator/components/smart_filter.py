import json
import importlib
import inspect
import os
from pathlib import Path
from typing import Dict, List, Type
from dotenv import load_dotenv
from litellm import completion

from node_spec.structure_schema import FlowSpec, NodeSpec
from scanning.probes.base_probe import BaseProbe


load_dotenv(dotenv_path="rift.env")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

PROBE_REQUIREMENTS_SYSTEM_PROMPT = """
You are helping decide whether a security probe is applicable to an agentic system.

You will receive:
- system_description
- probe_requirements

Decide whether the system description satisfies the probe requirements well enough
to continue to the deeper flow-level filtering stage.

Output contract:
- Return STRICT JSON only.
- Use exactly this schema: {"requirements_met": true} or {"requirements_met": false}
- If the requirements are not clearly satisfied, return false.
"""


class SmartFilter:
    """
    Smart filter that dynamically discovers probes and maps each selected flow
    to the probes that consider it relevant.

    The returned mapping is keyed by ``FlowSpec.cluster_id`` so it remains
    JSON-safe and easy to pass through the existing pipeline boundaries.
    """

    def __init__(self):
        pass

    @staticmethod
    def _probes_directory() -> Path:
        return Path(__file__).resolve().parents[2] / "probes"

    @classmethod
    def discover_probe_classes(cls) -> List[Type[BaseProbe]]:
        probe_classes: List[Type[BaseProbe]] = []
        probes_directory = cls._probes_directory()

        for module_path in sorted(probes_directory.rglob("*.py")):
            if module_path.name == "__init__.py":
                continue
            if module_path.stem == "base_probe" or module_path.stem.startswith("base_"):
                continue
            if module_path.stem.startswith("_"):
                continue
            if "__pycache__" in module_path.parts:
                continue

            relative_module = module_path.relative_to(probes_directory).with_suffix("")
            module_name = ".".join(relative_module.parts)
            module = importlib.import_module(f"scanning.probes.{module_name}")
            for _, member in inspect.getmembers(module, inspect.isclass):
                if member is BaseProbe:
                    continue
                if not issubclass(member, BaseProbe):
                    continue
                if inspect.isabstract(member):
                    continue
                if member.__module__ != module.__name__:
                    continue
                if "__init__" not in member.__dict__:
                    continue
                probe_classes.append(member)

        probe_classes.sort(
            key=lambda probe_class: (probe_class.__module__, probe_class.__name__)
        )
        return probe_classes

    @staticmethod
    def _get_probe_submodule_path(probe_class: Type[BaseProbe]) -> str:
        module_name = probe_class.__module__
        if module_name.startswith("scanning.probes."):
            module_name = module_name.removeprefix("scanning.probes.")
        return f"{module_name}.{probe_class.__name__}"

    @staticmethod
    def _get_init_default(
        probe_class: Type[BaseProbe],
        parameter_name: str,
        fallback: str = "",
    ) -> str:
        try:
            parameters = inspect.signature(probe_class.__init__).parameters
        except (TypeError, ValueError):
            return fallback

        parameter = parameters.get(parameter_name)
        if parameter is None or parameter.default is inspect.Signature.empty:
            return fallback
        return parameter.default

    @classmethod
    def normalize_probe_class(cls, probe_class: Type[BaseProbe]) -> dict:
        submodule_path = cls._get_probe_submodule_path(probe_class)
        description = cls._get_init_default(
            probe_class,
            "description",
            fallback=(inspect.getdoc(probe_class) or "").strip(),
        )

        return {
            "probe_name": cls._get_init_default(
                probe_class,
                "name",
                fallback=submodule_path,
            ),
            "probe_submodule_path": submodule_path,
            "description": description,
            "requirements": cls._get_init_default(probe_class, "requirements", fallback=""),
        }

    @staticmethod
    def _stringify_requirements(requirements: list[str] | None) -> str:
        if not requirements:
            return ""

        normalized_requirements = [
            requirement.strip()
            for requirement in requirements
            if isinstance(requirement, str) and requirement.strip()
        ]
        return "\n".join(f"- {requirement}" for requirement in normalized_requirements)

    @staticmethod
    def _get_system_description(full_analysis_node_spec: NodeSpec) -> str:
        system_summary = (full_analysis_node_spec.system_summary or "").strip()
        if system_summary:
            return system_summary
        return full_analysis_node_spec.name

    def _call_llm_json(self, system_prompt: str, user_content: str) -> dict:
        if completion is None:
            raise RuntimeError("litellm is not installed")

        response = (
            completion(
                model=f"{AZURE_DEPLOYMENT}",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                api_key=AZURE_API_KEY,
                api_base=AZURE_ENDPOINT,
                api_version=AZURE_API_VERSION,
                custom_llm_provider="azure",
            )
            .choices[0]
            .message.content
        )
        return json.loads(response)

    def _requirements_apply_to_system(
        self,
        full_analysis_node_spec: NodeSpec,
        requirements: str | list[str] | None,
    ) -> bool:
        requirements_text = self._stringify_requirements(requirements)
        if not requirements_text:
            return True

        user_content = f"""
        System description:
        {self._get_system_description(full_analysis_node_spec)}

        Probe requirements:
        {requirements_text}
        """

        try:
            parsed = self._call_llm_json(
                system_prompt=PROBE_REQUIREMENTS_SYSTEM_PROMPT,
                user_content=user_content,
            )
            value = parsed.get("requirements_met", True)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"true", "yes"}
        except Exception:
            return True

        return True

    @staticmethod
    def _probe_has_relevant_selected_flow(
        probe_class: Type[BaseProbe],
        full_analysis_node_spec: NodeSpec,
        selected_flows: List[FlowSpec],
    ) -> bool:
        relevant_flows = probe_class.filter_relevant_flows(
            full_node_spec=full_analysis_node_spec,
            flows=selected_flows,
        )
        return bool(relevant_flows)

    def run_smart_filtering(
        self,
        selected_flows: List[FlowSpec],
        full_analysis_node_spec: NodeSpec,
    ) -> List[dict]:
        selected_probes: List[dict] = []

        for probe_class in self.discover_probe_classes():
            normalized_probe = self.normalize_probe_class(probe_class)
            if not self._probe_has_relevant_selected_flow(
                probe_class,
                full_analysis_node_spec,
                selected_flows,
            ):
                continue

            selected_probes.append(dict(normalized_probe))

        return selected_probes
