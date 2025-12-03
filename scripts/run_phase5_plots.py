from __future__ import annotations
from pathlib import Path

from dotenv import load_dotenv

from normsense.analysis.plots import (
    load_summary,
    plot_overall_by_model_variant,
    plot_dimension_by_model,
)


def main() -> None:
    load_dotenv()

    root = Path(__file__).resolve().parents[1]
    summary_csv = root / "data" / "processed" / "model_score_summary_by_model_variant.csv"
    plots_dir = root / "reports" / "figures"

    print(f"Loading summary from {summary_csv} ...")
    df = load_summary(summary_csv)
    print(df)

    plot_overall_by_model_variant(df, plots_dir / "overall_by_model_variant.png")
    plot_dimension_by_model(df, plots_dir)
    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
