from scanning.probes.tool_output_injection.execute_own_tool_with_args_tool_output_injection import (
    ExecuteOwnToolWithArgsToolOutputInjection,
)
from scanning.probes.tool_output_injection.moved_execute_tool_with_args_tool_output_injection import (
    MovedExecuteToolWithArgsToolOutputInjection,
)


class MovedExecuteOwnToolWithArgsToolOutputInjection(
    ExecuteOwnToolWithArgsToolOutputInjection,
    MovedExecuteToolWithArgsToolOutputInjection,
):
    """
    Agent-scoped moved-functionality ATTENTION tool-output execute variant.
    """

    def __init__(
        self,
        name: str = "Moved Execute Own Tool With Args - Tool Output Injection",
        description: str = (
            "Injection via tool output that uses moved-functionality ATTENTION "
            "wording to activate an unintended tool call on another tool owned "
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
