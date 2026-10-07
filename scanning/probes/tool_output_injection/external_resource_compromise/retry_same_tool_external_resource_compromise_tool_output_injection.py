from node_spec.structure_schema import FlowSpec, NodeSpec

from scanning.probes.tool_output_injection.external_resource_compromise.base_external_resource_compromise_tool_output_injection import (
    BaseExternalResourceCompromiseToolOutputInjection,
)
from scanning.probes.tool_output_injection.retry_same_tool_tool_output_injection import (
    RetrySameToolToolOutputInjection,
)


class RetrySameToolExternalResourceCompromiseToolOutputInjection(
    BaseExternalResourceCompromiseToolOutputInjection,
    RetrySameToolToolOutputInjection,
):
    """
    External-resource tool-output injection that tries to make the agent repeat
    the same tool invocation.
    """
    def __init__(
        self,
        name: str = "Retry Same Tool - External Resource Compromise",
        description: str = (
            "Injection via a tool-used external resource that influences the "
            "tool response and makes the tool invoke more than it should."
        ),
        attack_surface: str = "External Resource Compromise",
        attack_suit: str = "Injection In Resource Content",
        attacker_intent: str = 'Resource overload',
        requirements: list[str] = [
            "At least one agent",
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
            "repeated_invocations.RepeatedInvocations",
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
