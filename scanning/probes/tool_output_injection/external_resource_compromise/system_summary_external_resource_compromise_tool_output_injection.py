from node_spec.structure_schema import FlowSpec, NodeSpec

from scanning.probes.tool_output_injection.external_resource_compromise.base_external_resource_compromise_tool_output_injection import (
    BaseExternalResourceCompromiseToolOutputInjection,
)
from scanning.probes.tool_output_injection.system_summary_tool_output_injection import (
    SystemSummaryToolOutputInjection,
)


class SystemSummaryExternalResourceCompromiseToolOutputInjection(
    BaseExternalResourceCompromiseToolOutputInjection,
    SystemSummaryToolOutputInjection,
):
    """
    External-resource tool-output injection that steers the run toward exposing
    the system summary.
    """
    def __init__(
        self,
        name: str = "System Summary - External Resource Compromise",
        description: str = (
            "Injection via a tool-used external resource that influences the "
            "tool response and pushes the agent to expose internal system "
            "information."
        ),
        attack_surface: str = "External Resource Compromise",
        attack_suit: str = "Injection In Resource Content",
        attacker_intent: str = 'Output internal information',
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
            "llm_attack_success.LLMAttackSuccess",
            "agnet_output_consist.LLMAgnetOutputConsist",
            "agnet_output_consist.SystemInfoStringAgnetOutputConsist",
            "system_output_consist.LLMSystemOutputConsist",
            "system_output_consist.SystemInfoStringSystemOutputConsist",
            "tool_output_consist.ToolOutputConsist",
            "base_utility.BaseUtility",
        ]
        self.probs_evaluators = self.primary_evaluators

    def malicious_twin_update(
        self,
        full_node_spec: NodeSpec,
        flow: FlowSpec,
        guidance: dict | None = None,
    ) -> tuple[NodeSpec, list[dict[str, list[str]]], list[str]]:
        return super().malicious_twin_update(full_node_spec, flow, guidance)
