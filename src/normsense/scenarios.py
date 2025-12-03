from __future__ import annotations
from enum import Enum
from typing import Literal, Optional
from pathlib import Path
from pydantic import BaseModel, Field
import json


class Domain(str, Enum):
    PERSONAL = "personal"
    WORKPLACE = "workplace"
    CUSTOMER = "customer_service"
    ONLINE = "online_social"


class NormType(str, Enum):
    POLITENESS = "politeness"
    EMPATHY = "empathy"
    CONTEXTUAL_FIT = "contextual_fit"
    MIXED = "mixed"


StakesLevel = Literal["low", "moderate", "high"]


class Scenario(BaseModel):
    """
    One NormSense evaluation scenario (single-turn).
    """
    id: str = Field(..., description="Unique scenario ID, e.g. 'SC001'")
    text: str = Field(..., description="Scenario prompt shown to the model.")
    domain: Domain
    norm_type: NormType
    cultural_tag: str = Field(..., description="e.g., 'US', 'Japan', 'Global'")
    stakes_level: StakesLevel
    prompt_source: str = Field(..., description="e.g. 'original'")
    notes: Optional[str] = None


class ScenarioSet(BaseModel):
    """
    Container for the full NormSense scenario set.
    """
    version: str = "v0.3"
    scenarios: list[Scenario]


def load_scenarios(path: str | Path) -> ScenarioSet:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return ScenarioSet(**data)


def save_scenarios(scenario_set: ScenarioSet, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(scenario_set.model_dump(), f, ensure_ascii=False, indent=2)
