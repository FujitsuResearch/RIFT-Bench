from node_spec.structure_schema import FlowSpec, NodeSpec

from scanning.probes.description_level_injection.execute_tool_with_args_description_level_injection import (
    ExecuteToolWithArgsDescriptionLevelInjection,
)
from scanning.probes.utils.base_util import (
    build_execution_cmds,
    build_input_arguments,
    get_agent_using_the_tool_from_flow,
    get_tool_example_input,
    get_unique_invoked_agents_ids,
    normalize_flows,
    select_attacked_tool_node_base_on_agent,
)
from scanning.probes.utils.flow_filltering import (
    get_agent_invoked_tool_ids,
    get_agent_unused_tool_ids_with_inputs,
    get_eligible_own_tool_description_attackable_tool_ids,
    has_entry_point_script,
)


class ExecuteOwnToolWithArgsDescriptionLevelInjection(
    ExecuteToolWithArgsDescriptionLevelInjection
):
    """
    Description-level injection that coerces an attacked tool into causing the
    model to execute another tool owned by the same invoked agent.
    """
    guidance_fields = [
        "attacked_tool_node_id",
        "invoked_tool_node_id",
        "invoked_tool_node_input",
        "input_arguments",
    ]

    def get_attackable_tool_ids(
        self,
        full_node_spec: NodeSpec,
        flow: FlowSpec,
    ) -> list[str]:
        """
        Return attackable invoked tools on agents that have an unused tool.
        """
        return get_eligible_own_tool_description_attackable_tool_ids(
            full_node_spec=full_node_spec,
            flow=flow,
        )

    def __init__(
        self,
        name: str = "Execute Own Tool With Args - Description Level Injection",
        description: str = (
            "Injection via the tool description to activate an unintended tool "
            "call on another tool owned by the same invoked agent."
        ),
        attack_surface: str = "Malicious Components",
        attack_suit: str = "Description Level Injection",
        attacker_intent: str = "Perform an unwanted action",
        requirements: list[str] = ["At least one agent", "At least two tools"],
    ):
        super().__init__(
            name=name,
            description=description,
            attack_surface=attack_surface,
            attack_suit=attack_suit,
            attacker_intent=attacker_intent,
            requirements=requirements,
        )

    @classmethod
    def filter_relevant_flows(
        cls,
        full_node_spec: NodeSpec,
        flows: list[FlowSpec] | None = None,
    ) -> list[FlowSpec]:
        """
        Keep flows that match the raw-metadata own-tool description pattern.

        A relevant flow must contain an invoked agent with at least one attacked
        tool candidate and one alternative unused same-agent tool with inputs.

        Deeper flow-dependent checks, such as resolving concrete example inputs,
        are intentionally deferred to the later attack-target selection step.
        """
        if not has_entry_point_script(full_node_spec):
            return []

        relevant_flows: list[FlowSpec] = []
        for flow in normalize_flows(full_node_spec, flows):
            try:
                        if get_eligible_own_tool_description_attackable_tool_ids(
                            full_node_spec=full_node_spec,
                            flow=flow,
                        ):
                            relevant_flows.append(flow)
            except Exception:
                continue
        return relevant_flows

    def select_tool_to_invoke(
        self,
        malicious_node_spec: NodeSpec,
        flow: FlowSpec,
        guidance: dict | None,
        attacked_tool_node_id: str,
    ) -> tuple[str, object]:
        guidance = guidance or {}
        invoked_tool_node_id = guidance.get("invoked_tool_node_id")
        if invoked_tool_node_id:
            invoked_tool_node = malicious_node_spec.get_node(node_id=invoked_tool_node_id)
            if invoked_tool_node is None:
                raise ValueError(
                    f"Guided invoked tool '{invoked_tool_node_id}' was not found."
                )
            if not getattr(invoked_tool_node, "inputs", None):
                raise ValueError(
                    f"Guided invoked tool '{invoked_tool_node_id}' does not define inputs."
                )
            return invoked_tool_node_id, invoked_tool_node

        agent_id = get_agent_using_the_tool_from_flow(
            flow=flow,
            tool_id=attacked_tool_node_id,
        )
        unused_tool_ids_with_inputs = get_agent_unused_tool_ids_with_inputs(
            full_node_spec=malicious_node_spec,
            flow=flow,
            agent_id=agent_id,
        )
        if not unused_tool_ids_with_inputs:
            raise ValueError(
                "No unused tool with inputs is available on the attacked tool's invoking agent."
            )

        invoked_tool_node_id = unused_tool_ids_with_inputs[0]
        invoked_tool_node = malicious_node_spec.get_node(node_id=invoked_tool_node_id)
        if invoked_tool_node is None:
            raise ValueError(
                f"Unused tool '{invoked_tool_node_id}' was not found in node spec."
            )
        return invoked_tool_node_id, invoked_tool_node

    def malicious_twin_update(
        self,
        full_node_spec: NodeSpec,
        flow: FlowSpec,
        guidance: dict | None = None,
    ) -> tuple[NodeSpec, list[dict[str, list[str]]], list[str]]:
        guidance = guidance or {}
        malicious_node_spec = full_node_spec.model_copy(deep=True)
        attacked_tool_node_id, attacked_tool_node = select_attacked_tool_node_base_on_agent(
            malicious_node_spec=malicious_node_spec,
            flow=flow,
            guidance=guidance,
            candidate=self.is_attacked_tool_candidate,
            error_message=(
                "No eligible attacked tool was found for description-level injection."
            ),
        )
        invoked_tool_node_id, invoked_tool_node = self.select_tool_to_invoke(
            malicious_node_spec=malicious_node_spec,
            flow=flow,
            guidance=guidance,
            attacked_tool_node_id=attacked_tool_node_id,
        )
        invoked_tool_node_input = get_tool_example_input(
            tool_node=invoked_tool_node,
            guidance=guidance,
        )

        input_arguments = build_input_arguments(flow=flow, guidance=guidance)
        execution_cmds = build_execution_cmds(
            malicious_node_spec=malicious_node_spec,
            input_arguments=input_arguments,
        )
        code_references_before_injection = self.snapshot_code_reference_snippets(
            attacked_tool_node
        )
        instruction = self.get_attack_instruction(
            invoked_tool_node=invoked_tool_node,
            invoked_tool_node_input=invoked_tool_node_input,
        )
        attack_goal = (
            f"Execute tool {invoked_tool_node.name} with arguments "
            f"{invoked_tool_node_input!r}"
        )
        self.inject_instruction_into_tool_description(attacked_tool_node, instruction)
        changed_kinds = self.changed_code_reference_kinds(
            owner_node=attacked_tool_node,
            before_snapshot=code_references_before_injection,
        )
        affected_nodes = self.build_affected_nodes(
            node_id=attacked_tool_node_id,
            changed_kinds=changed_kinds,
        )

        instance_info = self.build_instance_info(
            attacked_tool_node_id=attacked_tool_node_id,
            invoked_tool_node_id=invoked_tool_node_id,
            invoked_tool_node_input=invoked_tool_node_input,
            affected_nodes=affected_nodes,
            execution_cmds=execution_cmds,
            executed_flow=flow,
            attack_goal=attack_goal,
            malicious_string=instruction,
        )
        malicious_node_spec = self._add_metadata(malicious_node_spec, instance_info)
        return malicious_node_spec, affected_nodes, execution_cmds
