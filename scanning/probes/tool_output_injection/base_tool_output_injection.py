from abc import ABC, abstractmethod
from typing import Any

from node_spec.structure_schema import FlowSpec, NodeSpec

from scanning.probes.base_probe import BaseProbe
from scanning.probes.utils.base_util import (
    get_unique_invoked_tool_ids,
    normalize_flows,
)
from scanning.probes.utils.flow_filltering import (
    get_tool_ids_matching_predicate,
    get_tool_ids_with_code_references,
    has_entry_point_script,
)


class BaseToolOutputInjection(BaseProbe, ABC):
    """
    Shared base class for probes that inject malicious content through a tool's
    returned output.

    The common flow across these probes is:

    1. Pick a relevant flow.
    2. Select the attacked tool from the tools used in that flow.
    3. Build the malicious instruction that will be appended to that tool's
       returned value.
    4. Patch the tool implementation so its output carries the injected text.
    5. Build execution commands and probe metadata.

    Family-specific subclasses only need to define how attacked tools are
    selected. Attack-style subclasses can then override the instruction and
    goal hooks without reimplementing the mutation mechanics.
    """
    guidance_fields = ["attacked_tool_node_id", "input_arguments"]

    def get_attackable_tool_ids(
        self,
        full_node_spec: NodeSpec,
        flow: FlowSpec,
    ) -> list[str]:
        """
        Return invoked tool ids that this probe can attack using raw node-spec
        metadata only.
        """
        family_candidate_tool_ids = get_tool_ids_matching_predicate(
            full_node_spec=full_node_spec,
            tool_ids=get_unique_invoked_tool_ids(flow),
            predicate=self.is_attacked_tool_candidate,
        )
        return get_tool_ids_with_code_references(
            full_node_spec=full_node_spec,
            tool_ids=family_candidate_tool_ids,
        )

    @staticmethod
    @abstractmethod
    def is_attacked_tool_candidate(tool_node: Any) -> bool:
        """
        Return whether the given tool node is eligible to be attacked.
        """

    def get_attack_instruction(
        self,
        malicious_node_spec: NodeSpec,
        flow: FlowSpec,
        attacked_tool_node: Any,
        guidance: dict[str, Any] | None = None,
    ) -> str:
        """
        Return the malicious instruction that should be appended to the tool
        output.

        Subclasses are expected to override this.
        """
        raise NotImplementedError

    def get_attack_goal(
        self,
        malicious_node_spec: NodeSpec,
        flow: FlowSpec,
        attacked_tool_node: Any,
        guidance: dict[str, Any] | None = None,
    ) -> str:
        """
        Return the human-readable goal for the attack.
        """
        return "Manipulate the agent through malicious tool output."

    def build_additional_instance_info(
        self,
        malicious_node_spec: NodeSpec,
        flow: FlowSpec,
        attacked_tool_node: Any,
        guidance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Return extra metadata to merge into the standard instance info payload.
        """
        return {}

    @classmethod
    def filter_relevant_flows(
        cls,
        full_node_spec: NodeSpec,
        flows: list[FlowSpec] | None = None,
    ) -> list[FlowSpec]:
        """
        Keep flows that match the raw-metadata tool-output-injection pattern.

        A relevant flow must satisfy all of these conditions:

        1. The node spec defines an entry-point script.
        2. The flow invoked at least one tool matching the family-specific
           attacked-tool predicate.
        3. At least one such invoked tool defines code references.

        Deeper code-snippet checks, such as locating a rewriteable return path,
        are intentionally deferred to the later attack-target selection step.
        """
        if not has_entry_point_script(full_node_spec):
            return []

        relevant_flows: list[FlowSpec] = []
        for flow in normalize_flows(full_node_spec, flows):
            try:
                        family_candidate_tool_ids = get_tool_ids_matching_predicate(
                            full_node_spec=full_node_spec,
                            tool_ids=get_unique_invoked_tool_ids(flow),
                            predicate=cls.is_attacked_tool_candidate,
                        )
                        attackable_tool_ids = get_tool_ids_with_code_references(
                            full_node_spec=full_node_spec,
                            tool_ids=family_candidate_tool_ids,
                        )
                        if attackable_tool_ids:
                            relevant_flows.append(flow)
            except Exception:
                continue
        return relevant_flows
