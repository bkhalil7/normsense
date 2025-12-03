from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

from normsense.scenarios import load_scenarios, ScenarioSet
from normsense.prompts import (
    PromptVariant,
    build_system_prompt,
    build_user_prompt,
)
from normsense.models.base import ModelResponse
from normsense.models.openai_wrapper import OpenAIChatModel
from normsense.models.anthropic_wrapper import AnthropicChatModel
from normsense.models.open_weight_http import (
    make_llama3_70b_instruct_http,
    make_mistral_7b_instruct_http,
)


def build_models() -> Dict[str, object]:
    """
    Instantiate all models included in the report:

      - GPT-4 (o 2025)   -> 'gpt-4o'
      - GPT-3.5-Turbo    -> 'gpt-3.5-turbo'
      - Claude 3.5 Sonnet
      - Llama-3 70B Instruct  (HTTP endpoint)
      - Mistral-7B Instruct   (HTTP endpoint)
    """
    models: Dict[str, object] = {}

    # OpenAI
    models["gpt-4o"] = OpenAIChatModel(model_name="gpt-4o")
    models["gpt-3.5-turbo"] = OpenAIChatModel(model_name="gpt-3.5-turbo")

    # Anthropic
    models["claude-3.5-sonnet"] = AnthropicChatModel(
        model_name="claude-3-5-sonnet-20240620"
    )

    # Open-weight via HTTP endpoints
    # These will raise RuntimeError if the endpoint env vars are not set.
    try:
        models["llama-3-70b-instruct"] = make_llama3_70b_instruct_http()
    except RuntimeError as e:
        print(f"[WARN] Skipping Llama-3 70B: {e}")

    try:
        models["mistral-7b-instruct"] = make_mistral_7b_instruct_http()
    except RuntimeError as e:
        print(f"[WARN] Skipping Mistral-7B: {e}")

    return models


def main() -> None:
    load_dotenv()

    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "raw" / "normsense_scenarios_v0.3.json"
    out_path = root / "data" / "processed" / "model_responses_v0.3.jsonl"

    scenario_set: ScenarioSet = load_scenarios(data_path)
    scenarios = scenario_set.scenarios
    print(f"Loaded {len(scenarios)} scenarios from {data_path}")

    models = build_models()
    print(f"Active models: {list(models.keys())}")

    variants = [
        PromptVariant.NEUTRAL,
        PromptVariant.ROLE_PRIMED,
        PromptVariant.EMPATHY_PRIMED,
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_written = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for scenario in scenarios:
            for variant in variants:
                system_prompt = build_system_prompt(variant)
                user_prompt = build_user_prompt(scenario)

                for model_name, model in models.items():
                    print(
                        f"Running model={model_name}, "
                        f"variant={variant.value}, scenario={scenario.id}"
                    )

                    try:
                        resp: ModelResponse = model.generate(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            scenario_id=scenario.id,
                            prompt_variant=variant.value,
                        )

                        record = {
                            "scenario_id": resp.scenario_id,
                            "scenario_domain": scenario.domain.value,
                            "scenario_norm_type": scenario.norm_type.value,
                            "scenario_cultural_tag": scenario.cultural_tag,
                            "scenario_stakes_level": scenario.stakes_level,
                            "scenario_prompt_source": scenario.prompt_source,
                            "model_name": resp.model_name,
                            "prompt_variant": resp.prompt_variant,
                            "system_prompt": resp.system_prompt,
                            "user_prompt": resp.user_prompt,
                            "response_text": resp.response_text,
                            "raw": resp.raw,
                            "timestamp": time.time(),
                        }

                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        num_written += 1

                    except Exception as e:
                        # Log failure but keep going
                        error_record = {
                            "scenario_id": scenario.id,
                            "model_name": model_name,
                            "prompt_variant": variant.value,
                            "error": str(e),
                            "timestamp": time.time(),
                        }
                        f_out.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                        num_written += 1
                        print(f"[ERROR] {model_name} failed: {e}")

    print(f"Finished. Wrote {num_written} lines to {out_path}")


if __name__ == "__main__":
    main()
