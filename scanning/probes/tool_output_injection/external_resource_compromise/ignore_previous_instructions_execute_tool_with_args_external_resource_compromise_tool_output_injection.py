from node_spec.structure_schema import FlowSpec, NodeSpec

from scanning.probes.tool_output_injection.external_resource_compromise.base_external_resource_compromise_tool_output_injection import (
    BaseExternalResourceCompromiseToolOutputInjection,
)
from scanning.probes.tool_output_injection.ignore_previous_instructions_execute_tool_with_args_tool_output_injection import (
    IgnorePreviousInstructionsExecuteToolWithArgsToolOutputInjection,
)


class IgnorePreviousInstructionsExecuteToolWithArgsExternalResourceCompromiseToolOutputInjection(
    BaseExternalResourceCompromiseToolOutputInjection,
    IgnorePreviousInstructionsExecuteToolWithArgsToolOutputInjection,
):
    """
    External-resource tool-output injection that asks the agent to ignore
    previous instructions and execute an unintended tool call.
    """
    def __init__(
        self,
        name: str = "Ignore Previous Instructions Execute Tool With Args - External Resource Compromise",
        description: str = (
            "Injection via a tool-used external resource that influences the "
            "tool response and asks the agent to ignore prior instructions and "
            "activate an unintended tool call."
        ),
        attack_surface: str = "External Resource Compromise",
        attack_suit: str = "Injection In Resource Content",
        attacker_intent: str = 'Perform an unwanted action',
        requirements: list[str] = [
            "At least one agent",
            "At least two tools",
            "At least one tool that uses external information",
        ],
    ):
        super().__init__(
            name=name,
            description=description,
            attack_surface=attack_surface,
            attack_suit=attack_suit,
            attacker_intent=attacker_intent,
            requirements=requirements,
        )
        self.primary_evaluators = [
            "improved_execution_drift.ImprovedExecutionDrift",
            "boolean_tool_invocation.BooleanToolInvocation",
            "boolean_tool_invocation_with_matching_args.BooleanToolInvocationWithMatchingArgs",
            "llm_attack_success.LLMAttackSuccess",
            "base_utility.BaseUtility",
            "tool_output_consist.ToolOutputConsist",
        ]
        self.probs_evaluators = self.primary_evaluators

    def malicious_twin_update(
        self,
        full_node_spec: NodeSpec,
        flow: FlowSpec,
        guidance: dict | None = None,
    ) -> tuple[NodeSpec, list[dict[str, list[str]]], list[str]]:
        return super().malicious_twin_update(full_node_spec, flow, guidance)
