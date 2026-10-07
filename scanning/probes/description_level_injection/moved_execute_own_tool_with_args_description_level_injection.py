from scanning.probes.description_level_injection.execute_own_tool_with_args_description_level_injection import (
    ExecuteOwnToolWithArgsDescriptionLevelInjection,
)
from scanning.probes.description_level_injection.moved_execute_tool_with_args_description_level_injection import (
    MovedExecuteToolWithArgsDescriptionLevelInjection,
)


class MovedExecuteOwnToolWithArgsDescriptionLevelInjection(
    ExecuteOwnToolWithArgsDescriptionLevelInjection,
    MovedExecuteToolWithArgsDescriptionLevelInjection,
):
    """
    Agent-scoped moved-functionality ATTENTION description-level injection.
    """

    def __init__(
        self,
        name: str = "Moved Execute Own Tool With Args - Description Level Injection",
        description: str = (
            "Injection via the tool description that uses a moved-functionality "
            "ATTENTION payload to activate an unintended tool call on another "
            "tool owned by the same invoked agent."
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
