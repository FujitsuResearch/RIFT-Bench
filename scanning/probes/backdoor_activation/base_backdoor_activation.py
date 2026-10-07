from abc import ABC

from node_spec.structure_schema import FlowSpec, NodeSpec

from scanning.probes.base_probe import BaseProbe
from scanning.probes.utils.base_util import (
    get_required_tool_arguments_from_flow,
    get_unique_invoked_tool_ids,
    normalize_flows,
)
from scanning.probes.utils.flow_filltering import (
    get_tool_ids_with_definition_code_references,
    get_tool_ids_with_inputs,
    has_entry_point_script,
)


class BaseBackdoorActivation(BaseProbe, ABC):
    """
    Shared base class for probes that inject conditional backdoor behavior into
    a tool implementation.

    These probes mutate a tool so that a hidden trigger on its input arguments
    appends a malicious instruction to the tool output. The shared helpers here
    cover:

    1. Selecting flows that invoke a tool whose implementation can be patched.
    2. Selecting the attacked tool and the alternative tool to be invoked.
    3. Resolving representative tool arguments from the original flow.
    4. Building execution commands and probe metadata.
    5. Locating the return expression that will be wrapped by the backdoor.
    """
    guidance_fields = [
        "attacked_tool_node_id",
        "attacked_tool_arguments",
        "input_arguments",
    ]

    def get_attackable_tool_ids(
        self,
        full_node_spec: NodeSpec,
        flow: FlowSpec,
    ) -> list[str]:
        """
        Return invoked tool ids that this probe can inject a trigger into.
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
    def filter_relevant_flows(
        cls,
        full_node_spec: NodeSpec,
        flows: list[FlowSpec] | None = None,
    ) -> list[FlowSpec]:
        """
        Keep flows that match the raw-metadata backdoor pattern.

        A relevant flow must satisfy all of these conditions:

        1. The node spec defines an entry-point script.
        2. The flow invoked at least one tool.
        3. At least one invoked tool has observed, non-empty arguments in the
           flow.
        4. At least one such tool defines a definition code reference.

        Deeper code-snippet checks, such as locating an executable line or a
        simple return statement inside the selected definition snippet, are
        intentionally deferred to the later attack-construction step.
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
                tool_ids_with_arguments = [
                    tool_id
                    for tool_id in invoked_tool_ids_with_inputs
                    if get_required_tool_arguments_from_flow(
                        flow=flow,
                        tool_id=tool_id,
                    )
                ]
                attackable_tool_ids = get_tool_ids_with_definition_code_references(
                    full_node_spec=full_node_spec,
                    tool_ids=tool_ids_with_arguments,
                )
                if attackable_tool_ids:
                    relevant_flows.append(flow)
            except Exception:
                continue
        return relevant_flows
