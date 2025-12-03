from __future__ import annotations
from pathlib import Path

from dotenv import load_dotenv

from normsense.analysis.aggregate import load_scores, summarize_by_model_variant


def main() -> None:
    load_dotenv()

    root = Path(__file__).resolve().parents[1]
    scores_path = root / "data" / "processed" / "model_scores_v0.3.jsonl"
    out_csv = root / "data" / "processed" / "model_score_summary_by_model_variant.csv"

    print(f"Loading scores from {scores_path} ...")
    df = load_scores(scores_path)
    print(f"Loaded {len(df)} scored rows.")

    summary = summarize_by_model_variant(df)
    print("Summary:")
    print(summary)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    print(f"Wrote summary to {out_csv}")


if __name__ == "__main__":
    main()
