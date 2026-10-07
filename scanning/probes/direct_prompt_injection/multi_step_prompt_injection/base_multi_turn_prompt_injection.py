from abc import ABC
from scanning.probes.direct_prompt_injection.base_direct_prompt_injection import (
    BaseDirectPromptInjection,
)



class BaseMultiStepPromptInjection(BaseDirectPromptInjection, ABC):
    """
    Shared base class for multi-step direct prompt injection probes.
    """
    guidance_fields = ["input_arguments"]

    def __init__(
        self,
        name: str = "Multi-Step Direct Prompt Injection",
        description: str = (
            "Direct prompt injection that uses staged multi-step attack text."
        ),
        attack_surface: str = "Adversarial User",
        attack_suit: str = "Multi-Step Prompt injection",
        attacker_intent: str = "",
        requirements: list[str] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            attack_surface=attack_surface,
            attack_suit=attack_suit,
            attacker_intent=attacker_intent,
            requirements=requirements or [],
        )