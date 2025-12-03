from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass
class ModelResponse:
    """
    Standard container for a single model generation.
    """
    model_name: str
    prompt_variant: str
    scenario_id: str
    system_prompt: str
    user_prompt: str
    response_text: str
    raw: Dict[str, Any] | None = None


class LLMModel(Protocol):
    """
    Minimal interface all model wrappers must implement.
    """

    name: str  # human-readable identifier, e.g. "gpt-4o" or "claude-3.5-sonnet"

    def generate(self, *, system_prompt: str, user_prompt: str,
                 scenario_id: str, prompt_variant: str) -> ModelResponse:
        ...
