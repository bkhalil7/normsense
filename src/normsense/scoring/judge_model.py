from __future__ import annotations
from typing import Dict

from normsense.models.huggingface_local import HFLocalCausalLM
from .judge_prompt import build_judge_prompt, extract_json


class JudgeModel:
    """
    Wraps a local HF model for evaluation.
    """

    def __init__(self, model_id: str):
        self.model = HFLocalCausalLM(model_id=model_id)

    def score(self, scenario_text: str, response_text: str) -> Dict:
        prompt = build_judge_prompt(scenario_text, response_text)

        model_out = self.model.generate(
            system_prompt="You are a strict evaluator.",
            user_prompt=prompt,
            scenario_id="N/A",
            prompt_variant="judge"
        ).response_text

        return extract_json(model_out)
