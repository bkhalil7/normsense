from __future__ import annotations
from pathlib import Path
import json
from collections import Counter

from normsense.scenarios import load_scenarios, ScenarioSet
from normsense.prompts import (
    PromptVariant,
    build_system_prompt,
    build_user_prompt,
)


def pretty(d):
    """Pretty JSON for nicer notebook output."""
    return json.dumps(d, indent=4, ensure_ascii=False)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    json_path = root / "data" / "raw" / "normsense_scenarios_v0.3.json"

    scenario_set: ScenarioSet = load_scenarios(json_path)
    scenarios = scenario_set.scenarios

    print("=" * 90)
    print("PHASE 1 — DATASET INSPECTION")
    print("=" * 90)
    print(f"Total scenarios loaded: {len(scenarios)} (version={scenario_set.version})\n")

    # -----------------------------
    # Dataset statistics
    # -----------------------------
    domains = Counter([s.domain.value for s in scenarios])
    norm_types = Counter([s.norm_type.value for s in scenarios])
    stakes = Counter([s.stakes_level for s in scenarios])
    cultures = Counter([s.cultural_tag for s in scenarios])

    print("Dataset Breakdown:")
    print("- Domains:", dict(domains))
    print("- Norm Types:", dict(norm_types))
    print("- Stakes Levels:", dict(stakes))
    print("- Cultural Tags:", dict(cultures))
    print()

    # -----------------------------
    # Show first three scenarios
    # -----------------------------
    print("=" * 90)
    print("Sample of First 3 Scenarios")
    print("=" * 90)
    for s in scenarios[:3]:
        print(f"\nScenario ID: {s.id}")
        print(pretty(s.model_dump()))
        print("-" * 90)

    # -----------------------------
    # Prompt variant
    # -----------------------------

    example = scenarios[0]
    print("\n" + "=" * 90)
    print("Prompt Variants for Scenario 1")
    print("=" * 90)

    for variant in [
        PromptVariant.NEUTRAL,
        PromptVariant.ROLE_PRIMED,
        PromptVariant.EMPATHY_PRIMED,
    ]:
        print("\n" + "=" * 40)
        print(f"VARIANT: {variant.value.upper()}")
        print("=" * 40)

        system_msg = build_system_prompt(variant)
        user_msg = build_user_prompt(example)

        print("\n--- SYSTEM PROMPT ---")
        print(system_msg)

        print("\n--- USER PROMPT ---")
        print(user_msg)

    print("\n" + "=" * 90)
    print("Phase 1 Sanity Check Complete.")
    print("=" * 90)


if __name__ == "__main__":
    main()
