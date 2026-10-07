from node_spec.structure_schema import FlowSpec, NodeSpec

from scanning.probes.tool_output_injection.execute_tool_with_args_tool_output_injection import (
    ExecuteToolWithArgsToolOutputInjection,
)
from scanning.probes.utils.base_util import (
    build_execution_cmds,
    build_input_arguments,
    get_agent_using_the_tool_from_flow,
    get_tool_example_input,
    get_unique_invoked_tool_ids,
    normalize_flows,
    select_attacked_tool_node_base_on_agent,
)
from scanning.probes.utils.flow_filltering import (
    get_agent_unused_tool_ids_with_inputs,
    get_invoking_agent_unused_tool_ids_with_inputs,
    has_entry_point_script,
)


class ExecuteOwnToolWithArgsToolOutputInjection(ExecuteToolWithArgsToolOutputInjection):
    """
    Tool-output injection that coerces an attacked tool into causing the model
    to execute another tool owned by the same invoked agent.
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
        Return attackable invoked tools on agents that have an unused tool
        with inputs.
        """
        attackable_tool_ids: list[str] = []
        for tool_id in get_unique_invoked_tool_ids(flow):
            tool_node = full_node_spec.get_node(node_id=tool_id)
            if tool_node is None:
                continue
            if not self.is_attacked_tool_candidate(tool_node):
                continue
            if not getattr(tool_node, "code_references", None):
                continue
            if not get_invoking_agent_unused_tool_ids_with_inputs(
                full_node_spec=full_node_spec,
                flow=flow,
                tool_id=tool_id,
            ):
                continue
            attackable_tool_ids.append(tool_id)
        return attackable_tool_ids

    def __init__(
        self,
        name: str = "Execute Own Tool With Args - Tool Output Injection",
        description: str = (
            "Injection via the tool output to activate an unintended tool call "
            "on another tool owned by the same invoked agent."
        ),
        attack_surface: str = "Malicious Components",
        attack_suit: str = "Tool Output Injection",
        attacker_intent: str = 'Perform an unwanted action',
        requirements: list[str] = ["At least one agent", "At least two tools"],
    ):
        super().__init__()
        attacker_intent = attacker_intent
        self.name = name
        self.description = description
        self.attack_surface = attack_surface
        self.attack_suit = attack_suit
        self.requirements = requirements
        self.primary_evaluators = [
            "improved_execution_drift.ImprovedExecutionDrift",
            "boolean_tool_invocation.BooleanToolInvocation",
            "boolean_tool_invocation_with_matching_args.BooleanToolInvocationWithMatchingArgs",
            "llm_attack_success.LLMAttackSuccess",
            "base_utility.BaseUtility",
            "tool_output_consist.ToolOutputConsist",
        ]
        self.probs_evaluators = self.primary_evaluators

    @classmethod
    def filter_relevant_flows(
        cls,
        full_node_spec: NodeSpec,
        flows: list[FlowSpec] | None = None,
    ) -> list[FlowSpec]:
        """
        Keep flows that match the structural own-tool pattern.

        A relevant flow must contain an invoked agent for which all of these
        flow-shape conditions are true:

        1. The agent invoked at least one tool matching the family-specific
           attacked-tool predicate.
        2. The same agent still has at least one unused tool outside the flow.
        3. At least one such unused same-agent tool accepts inputs.

        Deeper NodeSpec-dependent eligibility checks, such as code references
        or example input resolution, are intentionally deferred to the later
        attack-target selection step.
        """
        if not has_entry_point_script(full_node_spec):
            return []

        relevant_flows: list[FlowSpec] = []
        for flow in normalize_flows(full_node_spec, flows):
            try:
                        for tool_id in get_unique_invoked_tool_ids(flow):
                            tool_node = full_node_spec.get_node(node_id=tool_id)
                            if tool_node is None:
                                continue
                            if not cls.is_attacked_tool_candidate(tool_node):
                                continue
                            if not getattr(tool_node, "code_references", None):
                                continue
                            if not get_invoking_agent_unused_tool_ids_with_inputs(
                                full_node_spec=full_node_spec,
                                flow=flow,
                                tool_id=tool_id,
                            ):
                                continue
                            relevant_flows.append(flow)
                            break
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
            error_message="No eligible attacked tool was found for this probe.",
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

        attack_goal = (
            f"Execute tool {invoked_tool_node.name} with arguments "
            f"{invoked_tool_node_input!r}"
        )
        attack_instruction = self.get_attack_instruction(
            invoked_tool_node=invoked_tool_node,
            invoked_tool_node_input=invoked_tool_node_input,
        )
        code_references_before_injection = self.snapshot_code_reference_snippets(
            attacked_tool_node
        )
        self.inject_instruction_into_tool_output(attacked_tool_node, attack_instruction)
        changed_kinds = self.changed_code_reference_kinds(
            owner_node=attacked_tool_node,
            before_snapshot=code_references_before_injection,
        )

        input_arguments = build_input_arguments(flow=flow, guidance=guidance)
        execution_cmds = build_execution_cmds(
            malicious_node_spec=malicious_node_spec,
            input_arguments=input_arguments,
        )
        affected_nodes = self.build_affected_nodes(
            node_id=attacked_tool_node_id,
            changed_kinds=changed_kinds,
        )
        instance_info = self.build_instance_info(
            attacked_tool_node_id=attacked_tool_node_id,
            executed_flow=flow,
            attack_goal=attack_goal,
            malicious_string=attack_instruction,
            affected_nodes=affected_nodes,
            execution_cmds=execution_cmds,
            invoked_tool_node_id=invoked_tool_node_id,
            invoked_tool_node_input=invoked_tool_node_input,
        )
        malicious_node_spec = self._add_metadata(malicious_node_spec, instance_info)
        return malicious_node_spec, affected_nodes, execution_cmds
