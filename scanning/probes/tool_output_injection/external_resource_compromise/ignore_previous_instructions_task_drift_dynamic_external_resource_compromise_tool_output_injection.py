from node_spec.structure_schema import FlowSpec, NodeSpec

from scanning.probes.tool_output_injection.external_resource_compromise.base_external_resource_compromise_tool_output_injection import (
    BaseExternalResourceCompromiseToolOutputInjection,
)
from scanning.probes.tool_output_injection.ignore_previous_instructions_task_drift_dynamic_tool_output_injection import (
    IgnorePreviousInstructionsTaskDriftDynamicToolOutputInjection,
)


class IgnorePreviousInstructionsTaskDriftDynamicExternalResourceCompromiseToolOutputInjection(
    BaseExternalResourceCompromiseToolOutputInjection,
    IgnorePreviousInstructionsTaskDriftDynamicToolOutputInjection,
):
    """
    External-resource tool-output injection that causes dynamic task drift
    toward a different flow's task.
    """
    def __init__(
        self,
        name: str = "Ignore Previous Instructions Task Drift Dynamic - External Resource Compromise",
        description: str = (
            "Injection via a tool-used external resource that influences the "
            "tool response and causes the agent to ignore the original task and "
            "answer a different flow-derived task."
        ),
        attack_surface: str = "External Resource Compromise",
        attack_suit: str = "Injection In Resource Content",
        attacker_intent: str = 'Not perform the user task',
        requirements: list[str] = [
            "At least one agent",
            "At least two tools",
            "At least one tool that uses external information",
            "At least two relevant flows",
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
            "base_utility.BaseUtility",
            "tool_output_consist.ToolOutputConsist",
            "agnet_output_consist.StringAgnetOutputConsist",
        ]
        self.probs_evaluators = self.primary_evaluators

    def malicious_twin_update(
        self,
        full_node_spec: NodeSpec,
        flow: FlowSpec,
        guidance: dict | None = None,
    ) -> tuple[NodeSpec, list[dict[str, list[str]]], list[str]]:
        return super().malicious_twin_update(full_node_spec, flow, guidance)
