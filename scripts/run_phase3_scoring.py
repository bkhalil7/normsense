from __future__ import annotations
import json
from pathlib import Path
import time

from dotenv import load_dotenv

from normsense.scoring.judge_model import JudgeModel
from normsense.scenarios import load_scenarios, ScenarioSet


def main() -> None:
    load_dotenv()

    root = Path(__file__).resolve().parents[1]

    # Phase 2 output (model responses)
    responses_path = root / "data" / "processed" / "model_responses_hf_local.jsonl"

    # Output path for scoring
    out_path = root / "data" / "processed" / "model_scores_v0.3.jsonl"

    # Load scenarios so we can access the original scenario text by ID
    scenario_set: ScenarioSet = load_scenarios(
        root / "data" / "raw" / "normsense_scenarios_v0.3.json"
    )
    scenario_by_id = {s.id: s for s in scenario_set.scenarios}

    judge = JudgeModel(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    num_scored = 0

    with responses_path.open("r", encoding="utf-8") as f_in, \
         out_path.open("w", encoding="utf-8") as f_out:

        for line in f_in:
            record = json.loads(line)

            # Skip error records from Phase 2
            if "error" in record:
                continue

            scenario_id = record["scenario_id"]
            model_response = record["response_text"]

            scenario = scenario_by_id.get(scenario_id)
            if scenario is None:
                # Should not happen, but be safe
                scenario_text = record.get("user_prompt", "")
            else:
                scenario_text = scenario.text

            try:
                score = judge.score(scenario_text, model_response)
            except Exception as e:
                score = {"error": str(e)}

            out_record = {
                "scenario_id": scenario_id,
                "model_name": record["model_name"],
                "prompt_variant": record["prompt_variant"],
                "scores": score,
                "timestamp": time.time(),
            }

            f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            num_scored += 1

    print(f"Done scoring. Wrote {num_scored} records to {out_path}")


if __name__ == "__main__":
    main()
