from __future__ import annotations
from pathlib import Path
import json

from dotenv import load_dotenv

from normsense.analysis.errors import (
    load_scores_df,
    extract_worst_examples,
    extract_best_examples,
)


def main() -> None:
    load_dotenv()

    root = Path(__file__).resolve().parents[1]
    scores_path = root / "data" / "processed" / "model_scores_v0.3.jsonl"
    out_path = root / "reports" / "error_analysis" / "qualitative_examples.md"

    df = load_scores_df(scores_path)

    worst = extract_worst_examples(df, k=10)
    best = extract_best_examples(df, k=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Qualitative Error Analysis\n\n")

        f.write("This file contains representative examples of model performance.\n\n")

        f.write("## Worst-scoring examples\n\n")
        for key, subset in worst.items():
            f.write(f"### {key}\n\n")
            for idx, row in subset.iterrows():
                f.write(f"- **Model**: {row['model_name']} ({row['prompt_variant']})\n")
                f.write(f"  - Politeness: {row['politeness']}, "
                        f"Empathy: {row['empathy']}, "
                        f"Contextual Fit: {row['contextual_fit']}, "
                        f"Overall: {row['overall']}\n")
                f.write(f"  - Rationale: {row['rationale']}\n\n")

        f.write("\n## Best-scoring examples\n\n")
        for key, subset in best.items():
            f.write(f"### {key}\n\n")
            for idx, row in subset.iterrows():
                f.write(f"- **Model**: {row['model_name']} ({row['prompt_variant']})\n")
                f.write(f"  - Politeness: {row['politeness']}, "
                        f"Empathy: {row['empathy']}, "
                        f"Contextual Fit: {row['contextual_fit']}, "
                        f"Overall: {row['overall']}\n")
                f.write(f"  - Rationale: {row['rationale']}\n\n")

    print(f"Saved qualitative analysis to {out_path}")


if __name__ == "__main__":
    main()
