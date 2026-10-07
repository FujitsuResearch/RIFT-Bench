from scanning.probes.tool_output_injection.attention_execute_own_tool_with_args_tool_output_injection import (
    AttentionExecuteOwnToolWithArgsToolOutputInjection,
)
from scanning.probes.tool_output_injection.internal_state_poisoning.base_internal_state_poisoning_tool_output_injection import (
    BaseInternalStatePoisoningToolOutputInjection,
)


class AttentionExecuteOwnToolWithArgsInternalStatePoisoningToolOutputInjection(
    BaseInternalStatePoisoningToolOutputInjection,
    AttentionExecuteOwnToolWithArgsToolOutputInjection,
):
    """
    Internal-state ATTENTION-style tool-output injection that pushes the agent
    to execute another tool owned by the same invoked agent.
    """

    def __init__(
        self,
        name: str = "Attention Execute Own Tool With Args - Internal State Poisoning",
        description: str = (
            "Injection via an internal resource that influences a tool response "
            "with urgent ATTENTION-style wording and activates an unintended "
            "tool call on another tool owned by the same invoked agent."
        ),
        attack_surface: str = "Internal State Poisoning",
        attack_suit: str = "Local Resource Poisoning",
        attacker_intent: str = "Perform an unwanted action",
        requirements: list[str] = [
            "At least one agent",
            "At least two tools",
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
            "boolean_tool_invocation.BooleanToolInvocation",
            "boolean_tool_invocation_with_matching_args.BooleanToolInvocationWithMatchingArgs",
            "llm_attack_success.LLMAttackSuccess",
            "base_utility.BaseUtility",
            "tool_output_consist.ToolOutputConsist",
        ]
        self.probs_evaluators = self.primary_evaluators

