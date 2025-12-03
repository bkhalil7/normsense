from __future__ import annotations
import os
from typing import Any, Dict

from openai import OpenAI

from .base import LLMModel, ModelResponse


class OpenAIChatModel:
    """
    Wrapper for OpenAI chat models like GPT-4(o) and GPT-3.5-Turbo.

    Expects OPENAI_API_KEY to be set in the environment (or .env loaded).
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 256,
    ) -> None:
        self.model_name = model_name
        self.name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Please add it to your .env or environment."
            )

        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        scenario_id: str,
        prompt_variant: str,
    ) -> ModelResponse:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        choice = resp.choices[0]
        text = choice.message.content

        # raw is stored as dict for easier JSON logging
        raw: Dict[str, Any] = {
            "id": resp.id,
            "usage": resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage),
        }

        return ModelResponse(
            model_name=self.model_name,
            prompt_variant=prompt_variant,
            scenario_id=scenario_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=text,
            raw=raw,
        )


# Type alias for type checkers
OpenAIChatModelType = LLMModel
