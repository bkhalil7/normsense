from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any

import json
import pandas as pd


def load_scores(jsonl_path: str | Path) -> pd.DataFrame:
    """
    Load model score records from a JSONL file into a tidy DataFrame.
    Assumes each line is:
      {
        "scenario_id": ...,
        "model_name": ...,
        "prompt_variant": ...,
        "scores": {
          "politeness": ...,
          "empathy": ...,
          "contextual_fit": ...,
          "overall": ...,
          "rationale": "...",
          ... or "error": "..."
        },
        ...
      }
    """
    jsonl_path = Path(jsonl_path)

    records: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            scores = rec.get("scores", {})
            if "error" in scores:
                # Keep error info but no numeric scores
                records.append(
                    {
                        "scenario_id": rec.get("scenario_id"),
                        "model_name": rec.get("model_name"),
                        "prompt_variant": rec.get("prompt_variant"),
                        "politeness": None,
                        "empathy": None,
                        "contextual_fit": None,
                        "overall": None,
                        "rationale": scores.get("error"),
                        "is_error": True,
                    }
                )
                continue

            records.append(
                {
                    "scenario_id": rec.get("scenario_id"),
                    "model_name": rec.get("model_name"),
                    "prompt_variant": rec.get("prompt_variant"),
                    "politeness": scores.get("politeness"),
                    "empathy": scores.get("empathy"),
                    "contextual_fit": scores.get("contextual_fit"),
                    "overall": scores.get("overall"),
                    "rationale": scores.get("rationale"),
                    "is_error": False,
                }
            )

    df = pd.DataFrame.from_records(records)
    return df


def summarize_by_model_variant(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean scores for each (model_name, prompt_variant).
    Excludes rows where is_error == True.
    """
    df_ok = df[~df["is_error"].astype(bool)].copy()

    group_cols = ["model_name", "prompt_variant"]
    agg = (
        df_ok
        .groupby(group_cols)
        .agg(
            n=("scenario_id", "count"),
            politeness_mean=("politeness", "mean"),
            empathy_mean=("empathy", "mean"),
            contextual_fit_mean=("contextual_fit", "mean"),
            overall_mean=("overall", "mean"),
        )
        .reset_index()
    )
    return agg
