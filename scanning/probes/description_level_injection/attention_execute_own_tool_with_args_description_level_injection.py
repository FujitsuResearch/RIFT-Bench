from scanning.probes.description_level_injection.attention_execute_tool_with_args_description_level_injection import (
    AttentionExecuteToolWithArgsDescriptionLevelInjection,
)
from scanning.probes.description_level_injection.execute_own_tool_with_args_description_level_injection import (
    ExecuteOwnToolWithArgsDescriptionLevelInjection,
)


class AttentionExecuteOwnToolWithArgsDescriptionLevelInjection(
    ExecuteOwnToolWithArgsDescriptionLevelInjection,
    AttentionExecuteToolWithArgsDescriptionLevelInjection,
):
    """
    Agent-scoped ATTENTION-style description-level injection variant.
    """

    def __init__(
        self,
        name: str = "Attention Execute Own Tool With Args - Description Level Injection",
        description: str = (
            "Injection via tool description that uses an urgent ATTENTION-style "
            "message to activate an unintended tool call on another tool owned "
            "by the same invoked agent."
        ),
        attack_surface: str = "Malicious Components",
        attack_suit: str = "Description Level Injection",
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
