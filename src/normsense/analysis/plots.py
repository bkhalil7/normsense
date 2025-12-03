from __future__ import annotations
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_summary(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    return pd.read_csv(csv_path)


def plot_overall_by_model_variant(df: pd.DataFrame, out_path: str | Path) -> None:
    """
    Simple barplot: overall_mean score for each (model_name, prompt_variant).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="model_name",
        y="overall_mean",
        hue="prompt_variant",
    )
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Overall Mean Score")
    plt.title("Overall Mean Score by Model and Prompt Variant")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_dimension_by_model(df: pd.DataFrame, out_dir: str | Path) -> None:
    """
    Create one plot per dimension (politeness, empathy, contextual_fit).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dims = [
        ("politeness_mean", "Politeness"),
        ("empathy_mean", "Empathy"),
        ("contextual_fit_mean", "Contextual Fit"),
    ]

    for col, label in dims:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df,
            x="model_name",
            y=col,
            hue="prompt_variant",
        )
        plt.xticks(rotation=30, ha="right")
        plt.ylabel(f"{label} Mean Score")
        plt.title(f"{label} Mean Score by Model and Prompt Variant")
        plt.tight_layout()
        out_path = out_dir / f"{col}_by_model_variant.png"
        plt.savefig(out_path)
        plt.close()
