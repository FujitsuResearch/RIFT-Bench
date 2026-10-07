from scanning.probes.tool_output_injection.attention_execute_own_tool_with_args_tool_output_injection import (
    AttentionExecuteOwnToolWithArgsToolOutputInjection,
)
from scanning.probes.tool_output_injection.external_resource_compromise.base_external_resource_compromise_tool_output_injection import (
    BaseExternalResourceCompromiseToolOutputInjection,
)


class AttentionExecuteOwnToolWithArgsExternalResourceCompromiseToolOutputInjection(
    BaseExternalResourceCompromiseToolOutputInjection,
    AttentionExecuteOwnToolWithArgsToolOutputInjection,
):
    """
    External-resource ATTENTION-style tool-output injection that pushes the
    agent to execute another tool owned by the same invoked agent.
    """

    def __init__(
        self,
        name: str = (
            "Attention Execute Own Tool With Args - External Resource Compromise"
        ),
        description: str = (
            "Injection via a tool-used external resource that influences the "
            "tool response with urgent ATTENTION-style wording and activates an "
            "unintended tool call on another tool owned by the same invoked agent."
        ),
        attack_surface: str = "External Resource Compromise",
        attack_suit: str = "Injection In Resource Content",
        attacker_intent: str = "Perform an unwanted action",
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

