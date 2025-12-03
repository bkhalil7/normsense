from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from .scenarios import Scenario


class PromptVariant(str, Enum):
    NEUTRAL = "neutral"
    ROLE_PRIMED = "role_primed"
    EMPATHY_PRIMED = "empathy_primed"


@dataclass
class PromptTemplateConfig:
    """
    High-level text that controls how we talk to the model.
    """
    neutral_system: str = (
        "You are a helpful AI assistant."
    )

    role_primed_system: str = (
        "You are a considerate, culturally aware assistant. "
        "Respond respectfully, clearly, and in a way appropriate for the social context."
    )

    empathy_primed_system: str = (
        "You are a considerate, empathetic assistant. "
        "Acknowledge the other person's feelings, validate their concerns, "
        "and respond in a warm but concise and practical way."
    )

    user_prefix: str = "Here is a social situation:\n\n"
    user_suffix: str = (
        "\n\nHow would you respond in a socially appropriate way?\n"
        "Keep the answer under about 6–8 sentences."
    )


def build_system_prompt(
    variant: PromptVariant,
    config: PromptTemplateConfig | None = None,
) -> str:
    if config is None:
        config = PromptTemplateConfig()

    if variant == PromptVariant.NEUTRAL:
        return config.neutral_system
    if variant == PromptVariant.ROLE_PRIMED:
        return config.role_primed_system
    if variant == PromptVariant.EMPATHY_PRIMED:
        return config.empathy_primed_system

    raise ValueError(f"Unknown variant: {variant}")


def build_user_prompt(
    scenario: Scenario,
    config: PromptTemplateConfig | None = None,
) -> str:
    if config is None:
        config = PromptTemplateConfig()

    return f"{config.user_prefix}{scenario.text}{config.user_suffix}"
