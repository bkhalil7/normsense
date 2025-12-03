from __future__ import annotations
import os
from typing import Any, Dict

import anthropic

from .base import LLMModel, ModelResponse


class AnthropicChatModel:
    """
    Wrapper for Anthropic Claude models, e.g. Claude 3.5 Sonnet.

    Expects ANTHROPIC_API_KEY to be set.
    """

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20240620",
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 256,
    ) -> None:
        self.model_name = model_name
        self.name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Please add it to your .env or environment."
            )

        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        scenario_id: str,
        prompt_variant: str,
    ) -> ModelResponse:
        # Claude uses "system" + "messages"
        resp = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}],
                }
            ],
        )

        # Claude response is a list of content blocks
        text_parts = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
        text = "\n".join(text_parts).strip()

        raw: Dict[str, Any] = {
            "id": resp.id,
            "usage": {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
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


AnthropicChatModelType = LLMModel
