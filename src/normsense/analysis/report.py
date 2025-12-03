from __future__ import annotations
from pathlib import Path
from typing import Optional

import pandas as pd


def load_summary_table(path: str | Path) -> Optional[pd.DataFrame]:
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def df_to_markdown(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame to a GitHub-flavored markdown table.
    """
    if df is None or df.empty:
        return "_No summary data available yet._"

    # Limit columns to the most relevant ones
    cols = [
        "model_name",
        "prompt_variant",
        "n",
        "politeness_mean",
        "empathy_mean",
        "contextual_fit_mean",
        "overall_mean",
    ]
    cols = [c for c in cols if c in df.columns]

    return df[cols].to_markdown(index=False, floatfmt=".2f")
