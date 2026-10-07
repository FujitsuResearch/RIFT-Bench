from scanning.probes.backdoor_activation.attention_execute_tool_on_trigger_match_backdoor_activation import (
    AttentionExecuteToolOnTriggerMatchBackdoorActivation,
)
from scanning.probes.backdoor_activation.execute_own_tool_on_trigger_match_backdoor_activation import (
    ExecuteOwnToolOnTriggerMatchBackdoorActivation,
)


class AttentionExecuteOwnToolOnTriggerMatchBackdoorActivation(
    ExecuteOwnToolOnTriggerMatchBackdoorActivation,
    AttentionExecuteToolOnTriggerMatchBackdoorActivation,
):
    """
    Agent-scoped backdoor activation probe.

    This probe reuses the trigger-injection workflow from
    :class:`ExecuteToolOnTriggerMatchBackdoorActivation`, but changes tool
    selection to be agent-aware:

    1. The attacked tool is selected through an agent-level helper.
    2. The tool that should be secretly invoked is also selected through an
       agent-level helper, relative to the attacked tool.

    The intent is to model a compromised flow where an agent is coerced into
    executing one of its own additional tools when trigger-matching inputs are
    observed.
    """

    def __init__(
        self,
        name: str = "Attention Execute Own Tool On Trigger Match - Backdoor Activation",
        description: str = (
            "Injection via a hidden trigger in the tool implementation that "
            "activates an unintended tool call."
        ),
        attack_surface: str = "Adversarial User",
        attack_suit: str = "Backdoor Activation",
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
