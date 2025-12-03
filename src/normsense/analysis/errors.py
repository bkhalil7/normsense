from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import json

import pandas as pd


def load_scores_df(scores_path: str | Path) -> pd.DataFrame:
    """
    Load scores file (Phase 3 output) into a DataFrame.
    Expects each line to be a JSON object with a 'scores' dict.
    """
    scores_path = Path(scores_path)
    records: List[Dict[str, Any]] = []

    with scores_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            scores = rec.get("scores", {})
            if "error" in scores:
                # skip items where judge failed
                continue

            records.append(
                {
                    "scenario_id": rec["scenario_id"],
                    "model_name": rec["model_name"],
                    "prompt_variant": rec["prompt_variant"],
                    "politeness": scores.get("politeness"),
                    "empathy": scores.get("empathy"),
                    "contextual_fit": scores.get("contextual_fit"),
                    "overall": scores.get("overall"),
                    "rationale": scores.get("rationale"),
                }
            )

    return pd.DataFrame.from_records(records)


def extract_worst_examples(df: pd.DataFrame, k: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Extract bottom-k examples for each score category.
    """
    groups: Dict[str, pd.DataFrame] = {}
    if df.empty:
        return groups

    groups["politeness_low"] = df.nsmallest(k, "politeness")
    groups["empathy_low"] = df.nsmallest(k, "empathy")
    groups["contextual_fit_low"] = df.nsmallest(k, "contextual_fit")
    groups["overall_low"] = df.nsmallest(k, "overall")
    return groups


def extract_best_examples(df: pd.DataFrame, k: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Extract top-k examples for each score category.
    """
    groups: Dict[str, pd.DataFrame] = {}
    if df.empty:
        return groups

    groups["politeness_high"] = df.nlargest(k, "politeness")
    groups["empathy_high"] = df.nlargest(k, "empathy")
    groups["contextual_fit_high"] = df.nlargest(k, "contextual_fit")
    groups["overall_high"] = df.nlargest(k, "overall")
    return groups
