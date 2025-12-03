from __future__ import annotations
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from normsense.scenarios import load_scenarios, ScenarioSet
from normsense.prompts import (
    PromptVariant,
    build_system_prompt,
    build_user_prompt,
)
from normsense.models.base import ModelResponse
from normsense.models.huggingface_local import HFLocalCausalLM


def build_hf_models():
    """
    Build the local Hugging Face models we will actually run.
    """
    models = {}

    # Small, lightweight chat model 
    models["TinyLlama/TinyLlama-1.1B-Chat-v1.0"] = HFLocalCausalLM(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
    )

    # Optional: larger model (may be too heavy for local machine)
    try:
        models["mistralai/Mistral-7B-Instruct-v0.2"] = HFLocalCausalLM(
            model_id="mistralai/Mistral-7B-Instruct-v0.2",
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
        )
    except Exception as e:
        print(f"[WARN] Could not load Mistral-7B model: {e}")
        print("[WARN] Continuing with TinyLlama only.")

    return models


def main() -> None:
    load_dotenv()

    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "raw" / "normsense_scenarios_v0.3.json"
    out_path = root / "data" / "processed" / "model_responses_hf_local.jsonl"

    scenario_set: ScenarioSet = load_scenarios(data_path)
    scenarios = scenario_set.scenarios
    print(f"Loaded {len(scenarios)} scenarios from {data_path}")

    models = build_hf_models()
    print(f"Active HF models: {list(models.keys())}")

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
                        f"Running HF model={model_name}, "
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
