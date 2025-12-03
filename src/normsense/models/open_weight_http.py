from __future__ import annotations
import os
from typing import Any, Dict

import requests

from .base import LLMModel, ModelResponse


class HTTPJSONGenerationModel:
    """
    Generic HTTP JSON generation wrapper for open-weight models (e.g., Llama-3, Mistral-7B)
    served via a custom endpoint or HF Inference.

    Expects:
      - URL in MODEL_ENDPOINT_URL (or a passed url)
      - Optional API token in MODEL_API_TOKEN (or passed token)
    The endpoint is assumed to accept:
      POST { "inputs": prompt, "parameters": {...} } and return JSON with "generated_text".
    """

    def __init__(
        self,
        name: str,
        endpoint_url_env: str,
        api_token_env: str | None = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 256,
    ) -> None:
        self.name = name
        self.endpoint_url = os.getenv(endpoint_url_env)
        if not self.endpoint_url:
            raise RuntimeError(
                f"{endpoint_url_env} not set. Please provide the HTTP endpoint URL "
                f"for model {name} in your environment or .env."
            )

        self.api_token = os.getenv(api_token_env) if api_token_env else None
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        scenario_id: str,
        prompt_variant: str,
    ) -> ModelResponse:
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"

        payload: Dict[str, Any] = {
            "inputs": combined_prompt,
            "parameters": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_new_tokens": self.max_tokens,
            },
        }

        resp = requests.post(
            self.endpoint_url,
            headers=self._build_headers(),
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        # HF-style endpoints often return a list of dicts, each with 'generated_text'
        if isinstance(data, list) and data and "generated_text" in data[0]:
            text = data[0]["generated_text"]
        elif isinstance(data, dict) and "generated_text" in data:
            text = data["generated_text"]
        else:
            # Fallback to string
            text = str(data)

        return ModelResponse(
            model_name=self.name,
            prompt_variant=prompt_variant,
            scenario_id=scenario_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=text,
            raw={"endpoint_url": self.endpoint_url, "raw": data},
        )


def make_llama3_70b_instruct_http() -> HTTPJSONGenerationModel:
    """
    Llama-3 70B Instruct wrapper.
    Expects:
      - LLAMA3_70B_ENDPOINT_URL
      - LLAMA3_70B_API_TOKEN (optional)
    """
    return HTTPJSONGenerationModel(
        name="llama-3-70b-instruct",
        endpoint_url_env="LLAMA3_70B_ENDPOINT_URL",
        api_token_env="LLAMA3_70B_API_TOKEN",
    )


def make_mistral_7b_instruct_http() -> HTTPJSONGenerationModel:
    """
    Mistral-7B Instruct wrapper.
    Expects:
      - MISTRAL_7B_ENDPOINT_URL
      - MISTRAL_7B_API_TOKEN (optional)
    """
    return HTTPJSONGenerationModel(
        name="mistral-7b-instruct",
        endpoint_url_env="MISTRAL_7B_ENDPOINT_URL",
        api_token_env="MISTRAL_7B_API_TOKEN",
    )
