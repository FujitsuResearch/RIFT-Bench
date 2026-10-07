from abc import ABC, abstractmethod

from node_spec.structure_schema import FlowSpec, NodeSpec

from scanning.probes.base_probe import BaseProbe
from scanning.probes.utils.base_util import (
    get_unique_invoked_tool_ids,
    normalize_flows,
)
from scanning.probes.utils.flow_filltering import (
    get_tool_ids_with_definition_code_references,
    get_tool_ids_with_inputs,
    has_entry_point_script,
)


class BaseToolImplementationInjection(BaseProbe, ABC):
    """
    Shared base class for probes that inject malicious behavior into a tool's
    implementation.

    These probes mutate the attacked tool's implementation snippet rather than
    its description or returned output. The helpers here cover:

    1. Normalizing and filtering candidate flows.
    2. Selecting the attacked tool from the selected flow.
    3. Resolving representative tool inputs and entry-point arguments.
    4. Building execution commands and metadata.
    5. Applying common implementation-level mutations to a tool snippet.
    """
    guidance_fields = [
        "attacked_tool_node_id",
        "invoked_tool_node_input",
        "input_arguments",
    ]

    def get_attackable_tool_ids(
        self,
        full_node_spec: NodeSpec,
        flow: FlowSpec,
    ) -> list[str]:
        """
        Return invoked tools that this probe can attack using raw node-spec
        metadata only.
        """
        invoked_tool_ids_with_inputs = get_tool_ids_with_inputs(
            full_node_spec=full_node_spec,
            tool_ids=get_unique_invoked_tool_ids(flow),
        )
        return get_tool_ids_with_definition_code_references(
            full_node_spec=full_node_spec,
            tool_ids=invoked_tool_ids_with_inputs,
        )

    @classmethod
    @abstractmethod
    def attack_family(cls) -> str:
        """
        Return the tool-implementation attack family identifier.

        This keeps intermediate helper classes abstract so discovery only
        returns concrete family-specific probes.
        """

    @classmethod
    def filter_flows_with_invoked_tools(
        cls,
        full_node_spec: NodeSpec,
        flows: list[FlowSpec] | None = None,
    ) -> list[FlowSpec]:
        """
        Keep flows that invoke at least one tool with inputs and a definition
        code reference.
        """
        if not has_entry_point_script(full_node_spec):
            return []

        relevant_flows: list[FlowSpec] = []
        for flow in normalize_flows(full_node_spec, flows):
            try:
                        invoked_tool_ids_with_inputs = get_tool_ids_with_inputs(
                            full_node_spec=full_node_spec,
                            tool_ids=get_unique_invoked_tool_ids(flow),
                        )
                        attackable_tool_ids = get_tool_ids_with_definition_code_references(
                            full_node_spec=full_node_spec,
                            tool_ids=invoked_tool_ids_with_inputs,
                        )
                        if attackable_tool_ids:
                            relevant_flows.append(flow)
            except Exception:
                continue
        return relevant_flows

    @classmethod
    def filter_relevant_flows(
        cls,
        full_node_spec: NodeSpec,
        flows: list[FlowSpec] | None = None,
    ) -> list[FlowSpec]:
        """
        Default relevant-flow filter for tool-implementation-injection probes.

        By default, these probes require a flow that invokes at least one tool
        with inputs and a definition code reference. Deeper code-snippet checks,
        such as locating an executable line or a rewriteable return statement,
        are intentionally deferred to the later attack-target selection step.
        """
        return cls.filter_flows_with_invoked_tools(full_node_spec, flows)
