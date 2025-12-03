from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ScoreRubric:
    """
    Defines rubric dimensions for judgment.
    """

    # Each dimension is scored 0–5.
    categories = {
        "politeness": "Does the response maintain respectful and appropriate tone?",
        "empathy": "Does the response acknowledge feelings, show understanding, and respond supportively?",
        "contextual_fit": "Is the response appropriate for the situation? Does it make sense in context?",
        "overall": "Holistic evaluation of quality."
    }

    scale_description = """
    Score each category from 0 to 5:
      5 = Excellent
      4 = Good
      3 = Adequate
      2 = Weak
      1 = Poor
      0 = Very inappropriate / totally fails the requirement
    """
