from scanning.probes.tool_output_injection.attention_execute_tool_with_args_tool_output_injection import (
    AttentionExecuteToolWithArgsToolOutputInjection,
)
from scanning.probes.tool_output_injection.execute_own_tool_with_args_tool_output_injection import (
    ExecuteOwnToolWithArgsToolOutputInjection,
)


class AttentionExecuteOwnToolWithArgsToolOutputInjection(
    ExecuteOwnToolWithArgsToolOutputInjection,
    AttentionExecuteToolWithArgsToolOutputInjection,
):
    """
    Agent-scoped ATTENTION-style tool-output injection variant.
    """

    def __init__(
        self,
        name: str = "Attention Execute Own Tool With Args - Tool Output Injection",
        description: str = (
            "Injection via tool output that uses an urgent ATTENTION-style "
            "message to activate an unintended tool call on another tool owned "
            "by the same invoked agent."
        ),
        attack_surface: str = "Malicious Components",
        attack_suit: str = "Tool Output Injection",
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
        self.primary_evaluators = [
            "improved_execution_drift.ImprovedExecutionDrift",
            "boolean_tool_invocation.BooleanToolInvocation",
            "boolean_tool_invocation_with_matching_args.BooleanToolInvocationWithMatchingArgs",
            "llm_attack_success.LLMAttackSuccess",
            "base_utility.BaseUtility",
            "tool_output_consist.ToolOutputConsist",
        ]
        self.probs_evaluators = self.primary_evaluators
