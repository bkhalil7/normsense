from __future__ import annotations
import os
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .base import ModelResponse


class HFLocalCausalLM:
    """
    Local Hugging Face causal LM wrapper.
    Uses a chat/instruct model from Hugging Face and runs it locally on CPU/GPU.

    Example models (free, open-weight):
      - TinyLlama/TinyLlama-1.1B-Chat-v1.0   (small, best for laptops)
      - mistralai/Mistral-7B-Instruct-v0.2   (larger, may require more RAM/GPU)
    """

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> None:
        self.model_id = model_id
        self.name = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HUGGINGFACE_API_TOKEN not set. Add it to your .env or environment."
            )

        # Choose device
        device = 0 if torch.cuda.is_available() else -1  # -1 = CPU

        print(f"[HFLocalCausalLM] Loading model {model_id} on device={device} ...")
        self.generator = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device=device,
            token=hf_token,
        )
        print(f"[HFLocalCausalLM] Loaded model {model_id}.")

    def _build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        # Simple chat-style formatting
        return (
            f"{system_prompt}\n\n"
            f"User: {user_prompt}\n\n"
            f"Assistant:"
        )

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        scenario_id: str,
        prompt_variant: str,
    ) -> ModelResponse:
        full_prompt = self._build_prompt(system_prompt, user_prompt)

        outputs = self.generator(
            full_prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            num_return_sequences=1,
        )

        # HF pipeline usually returns a list of dicts with 'generated_text'
        if isinstance(outputs, list) and outputs and "generated_text" in outputs[0]:
            generated = outputs[0]["generated_text"]
        else:
            generated = str(outputs)

        # Strip the prompt part if the model echoes it
        if generated.startswith(full_prompt):
            response_text = generated[len(full_prompt):].strip()
        else:
            response_text = generated.strip()

        return ModelResponse(
            model_name=self.name,
            prompt_variant=prompt_variant,
            scenario_id=scenario_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=response_text,
            raw={"model_id": self.model_id},
        )
