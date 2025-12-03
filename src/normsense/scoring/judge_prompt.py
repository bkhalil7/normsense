from __future__ import annotations
from typing import Dict

from .rubric import ScoreRubric


def build_judge_prompt(scenario_text: str, model_response: str) -> str:
    """
    Build an evaluation prompt for the judge model.
    """

    rubric = ScoreRubric

    return f"""
You are an evaluation assistant. Your job is to score a model's response along several social norms dimensions.

Here is the social scenario:
---
{scenario_text}
---

Here is the model's response:
---
{model_response}
---

Evaluation Rubric:
{rubric.scale_description}

Categories to score:
{rubric.categories}

Return your answer strictly in this JSON format:
{{
  "politeness": <0-5>,
  "empathy": <0-5>,
  "contextual_fit": <0-5>,
  "overall": <0-5>,
  "rationale": "<one short paragraph explaining your evaluation>"
}}
"""


def extract_json(text: str) -> Dict:
    """
    Extract JSON block from LLM output.
    Assumes model returns a JSON-like structure.
    """
    import json
    import re

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in judge response")

    block = match.group(0)

    return json.loads(block)
