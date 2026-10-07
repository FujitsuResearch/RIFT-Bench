from node_spec.structure_schema import FlowSpec, NodeSpec

from scanning.probes.tool_output_injection.internal_state_poisoning.base_internal_state_poisoning_tool_output_injection import (
    BaseInternalStatePoisoningToolOutputInjection,
)
from scanning.probes.tool_output_injection.llm_retry_same_tool_tool_output_injection import (
    LLMRetrySameToolToolOutputInjection,
)


class LLMRetrySameToolInternalStatePoisoningToolOutputInjection(
    BaseInternalStatePoisoningToolOutputInjection,
    LLMRetrySameToolToolOutputInjection,
):
    """
    Internal-state tool-output injection that uses LLM-generated retry text to
    make the agent repeatedly invoke the same tool.
    """

    def __init__(
        self,
        name: str = "LLM Retry Same Tool - Internal State Poisoning",
        description: str = (
            "Injection via a tool-used internal resource that influences the "
            "tool response with LLM-generated retry instructions and makes "
            "the tool invoke more than it should."
        ),
        attack_surface: str = "Internal State Poisoning",
        attack_suit: str = "Local Resource Poisoning",
        attacker_intent: str = "Resource overload",
        requirements: list[str] = [
            "At least one agent",
            "At least one tool that uses internal information",
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
