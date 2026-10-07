from typing import Any

from node_spec.structure_schema import FlowSpec, NodeSpec

from scanning.probes.direct_prompt_injection.multi_turn_prompt_injection.strategic_task_drift_spam_multi_turn_prompt_injection import (
    StrategicTaskDriftSpamMultiTurnPromptInjection,
)
from scanning.probes.direct_prompt_injection.multi_turn_prompt_injection.task_drift_dynamic_multi_turn_prompt_injection import (
    TaskDriftDynamicMultiTurnPromptInjection,
)


class StrategicTaskDriftDynamicMultiTurnPromptInjection(
    TaskDriftDynamicMultiTurnPromptInjection
):
    """
    Strategic conversational multi-turn task-drift dynamic variant.
    """

    def __init__(
        self,
        name: str = "Strategic Task Drift Dynamic - Multi Turn Direct Prompt Injection",
        description: str = (
            "Multi-turn task-drift dynamic variant that uses a strategic, "
            "progressive persuasion style."
        ),
        attack_surface: str = "Adversarial User",
        attack_suit: str = "Multi-Turn Prompt injection",
        attacker_intent: str = "Not perform the user task",
        requirements: list[str] | None = None,
        max_turns: int = 5,
        use_llm_red_teamer: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            attack_surface=attack_surface,
            attack_suit=attack_suit,
            attacker_intent=attacker_intent,
            requirements=requirements,
            max_turns=max_turns,
            use_llm_red_teamer=use_llm_red_teamer,
        )

    def get_red_team_prompt(
        self,
        malicious_node_spec: NodeSpec,
        flow: FlowSpec,
        guidance: dict[str, Any] | None = None,
    ) -> str:
        return StrategicTaskDriftSpamMultiTurnPromptInjection.get_red_team_prompt(
            self,
            malicious_node_spec,
            flow,
            guidance,
        )
