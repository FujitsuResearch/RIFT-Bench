from abc import ABC, abstractmethod
from typing import Any

from node_spec.structure_schema import FlowSpec, NodeSpec

from scanning.probes.base_probe import BaseProbe
from scanning.probes.utils.base_util import (
    get_main_user_task_argument_name,
    normalize_flows,
)
from scanning.probes.utils.flow_filltering import (
    flow_has_first_agent_with_system_prompt_code_references,
    get_agent_event_node_ids,
    has_entry_point_script,
    has_user_task_entry_point_argument,
)


class BaseDirectPromptInjection(BaseProbe, ABC):
    """
    Shared base class for direct prompt-injection probes.

    These probes do not mutate the system graph itself. Instead, they append a
    malicious instruction to the entry-point argument that represents the
    user's task. Concrete subclasses are expected to provide the
    attack-specific payload, while this base class centralizes the repetitive
    mechanics needed to:

    1. Resolve the entry-point argument marked as ``user_task``.
    2. Build the baseline input arguments for the selected flow.
    3. Inject the malicious instruction into the user prompt.
    4. Rebuild the command used to execute the malicious twin.

    Keeping these steps here makes the concrete prompt-injection probes mostly
    about the attack string and any extra metadata they want to attach.
    """
    guidance_fields = ["input_arguments"]

    @abstractmethod
    def get_attack_instruction(
        self,
        malicious_node_spec: NodeSpec,
        flow: FlowSpec,
        guidance: dict[str, Any] | None = None,
    ) -> str:
        """
        Return the attack payload that should be appended to the user task.

        Subclasses should implement only the scenario-specific instruction
        generation. The rest of the direct prompt-injection flow can then reuse
        the helpers defined on this base class.
        """

    def get_attack_goal(
        self,
        malicious_node_spec: NodeSpec,
        flow: FlowSpec,
        guidance: dict[str, Any] | None = None,
    ) -> str:
        """
        Return a short textual description of the probe's objective.
        """
        return "Manipulate the agent through a direct prompt injection."

    @classmethod
    def filter_relevant_flows(
        cls,
        full_node_spec: NodeSpec,
        flows: list[FlowSpec] | None = None,
    ) -> list[FlowSpec]:
        """
        Keep flows that match the raw-metadata direct-prompt pattern.

        A relevant flow must satisfy all of these conditions:

        1. The node spec defines an entry-point script.
        2. The node spec defines an entry-point argument marked as the main
           user task.
        3. The flow contains at least one agent event.

        Deeper attack-specific checks that require extracting content from
        code snippets are intentionally deferred to the later
        attack-construction step.
        """
        if not has_entry_point_script(full_node_spec):
            return []
        if not has_user_task_entry_point_argument(full_node_spec):
            return []

        relevant_flows: list[FlowSpec] = []
        for flow in normalize_flows(full_node_spec, flows):
            try:
                        if get_agent_event_node_ids(flow):
                            relevant_flows.append(flow)
            except Exception:
                continue
        return relevant_flows

    @classmethod
    def filter_flows_with_first_agent_system_prompt(
        cls,
        full_node_spec: NodeSpec,
        flows: list[FlowSpec] | None = None,
    ) -> list[FlowSpec]:
        """
        Keep only direct-prompt flows whose first attacked agent exposes raw
        system-prompt metadata in the node spec.
        """
        relevant_flows = []
        for flow in BaseDirectPromptInjection.filter_relevant_flows(full_node_spec, flows):
            try:
                if flow_has_first_agent_with_system_prompt_code_references(full_node_spec, flow):
                    relevant_flows.append(flow)
            except Exception:
                continue
        return relevant_flows

    @staticmethod
    def get_first_agent_node_id_from_flow(flow: FlowSpec) -> str:
        """
        Return the first Agent node id observed in the flow trace.

        The existing prompt-injection probes target the final LLM event seen in
        the trace, so this helper preserves that behavior.
        """
        attacked_agent_node_ids = get_agent_event_node_ids(flow)
        if not attacked_agent_node_ids:
            raise ValueError(
                "Direct prompt injection requires a flow trace with at least "
                "one agent event."
            )
        return attacked_agent_node_ids[0]

    @staticmethod
    def build_affected_nodes(attacked_agent_node_id: str, node_spec: NodeSpec) -> list[dict[str, list[str]]]:
        llm_ids: set[str] = set()
        agent_node = node_spec.get_node(attacked_agent_node_id)
        if agent_node is None:
            raise ValueError(
                f"Attacked agent '{attacked_agent_node_id}' was not found in node spec."
            )
        for node in agent_node.iter_descendants(include_self=False):
            if getattr(node.node_type, "type", None) == "LLM" and node.id:
                llm_ids.add(node.id)
                break
        return [{llm: []} for llm in llm_ids]

    def inject_via_user_prompt(
        self,
        malicious_node_spec: NodeSpec,
        input_arguments: dict[str, Any],
        attack_instruction: str,
    ) -> dict[str, Any]:
        """
        Return input arguments with the attack appended to the user task field.

        The original ``input_arguments`` mapping is copied before mutation so
        callers can safely reuse the unmodified payload elsewhere.

        Raises:
            KeyError: If the resolved user-task argument is missing from the
                provided input arguments.
        """
        main_input_argument_name = get_main_user_task_argument_name(
            malicious_node_spec
        )
        if main_input_argument_name not in input_arguments:
            raise KeyError(
                "Direct prompt injection could not find the user-task input "
                f"argument '{main_input_argument_name}' in the provided "
                "input_arguments."
            )

        injected_arguments = dict(input_arguments)
        original_prompt = injected_arguments[main_input_argument_name]
        separator = " " if str(original_prompt) and attack_instruction else ""
        injected_arguments[main_input_argument_name] = (
            f"{original_prompt}{separator}{attack_instruction}"
        )
        return injected_arguments
