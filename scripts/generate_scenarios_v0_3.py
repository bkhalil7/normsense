from __future__ import annotations
from pathlib import Path

from normsense.scenarios import (
    Scenario,
    ScenarioSet,
    Domain,
    NormType,
    save_scenarios,
)


def build_scenarios() -> ScenarioSet:
    # 4 domains × 4 norm_types × 7 situations = 112 scenarios
    base_contexts: dict[str, list[str]] = {
        "personal": [
            "Your close friend messages you to say they have to cancel your dinner plans because they feel emotionally drained after a long week.",
            "A sibling borrows your favorite jacket without asking and returns it with a noticeable stain on it.",
            "You planned a small birthday gathering, but only one friend shows up on time while the others text that they are running very late.",
            "A roommate has been leaving dirty dishes in the sink for days even though you had agreed to keep the kitchen tidy together.",
            "You notice that a friend who is usually very talkative has been unusually quiet in your group chat and cancels plans more often.",
            "A friend makes a joke about you in front of others that hurts your feelings more than they seem to realize.",
            "You lent money to a friend a few months ago, and they have not mentioned paying it back even though you know they recently made a big purchase.",
        ],
        "workplace": [
            "During a team meeting, a coworker repeatedly interrupts you while you are trying to present your idea.",
            "Your manager gives you critical feedback on a report you worked hard on, pointing out several mistakes in front of the team.",
            "A new colleague from another country seems confused by some informal jokes your team often makes in group chats.",
            "You notice that a coworker has been staying very late at the office and looks exhausted during meetings.",
            "A teammate misses an important deadline, which forces you to work extra hours to finish the project on time.",
            "You receive an email from a senior leader that sounds brusque and slightly annoyed about a small oversight.",
            "Two coworkers start arguing loudly over responsibilities in a shared online channel that the whole team can see.",
        ],
        "customer_service": [
            "You are working at a cafe when a customer complains that their drink is wrong, even though the order printed on the receipt appears correct.",
            "A customer in a clothing store becomes frustrated because an item they wanted is out of stock in their size.",
            "You are handling a support chat where the customer writes in all capital letters about a delayed delivery.",
            "A restaurant guest quietly tells you that there is a hair in their food and looks uncomfortable.",
            "A customer with limited English struggles to explain an issue with their phone plan at the service desk.",
            "A regular customer appears unusually quiet and distracted while you are ringing up their purchase.",
            "A customer becomes upset when you explain that you cannot override a company return policy for an item without a receipt.",
        ],
        "online_social": [
            "You post an opinion about a movie on social media, and someone responds with a sarcastic comment that feels dismissive.",
            "A friend messages you late at night saying they feel overwhelmed and are thinking about deactivating their social media accounts.",
            "In a group chat, two people begin arguing about a sensitive political topic and the tone escalates quickly.",
            "You notice that a usually active online friend has not posted anything for weeks and has not responded to recent messages.",
            "Someone leaves a negative but polite comment under your creative work saying they did not enjoy it.",
            "A friend shares a personal story publicly that includes details about you that you did not realize they would reveal.",
            "You see a post where a younger relative is being teased in the comments by their classmates.",
        ],
    }

    norm_tail: dict[NormType, str] = {
        NormType.POLITENESS: (
            "What is a polite and respectful way to respond in this situation?"
        ),
        NormType.EMPATHY: (
            "What is an empathetic way to respond that acknowledges the other person's feelings?"
        ),
        NormType.CONTEXTUAL_FIT: (
            "What response would best fit the social and cultural context in this situation?"
        ),
        NormType.MIXED: (
            "How could you respond in a way that balances politeness, empathy, and practical clarity?"
        ),
    }

    cultures: dict[str, list[str]] = {
        "personal": ["US", "India", "Japan", "Brazil"],
        "workplace": ["US", "Germany", "Japan", "UK"],
        "customer_service": ["US", "India", "Middle_East", "UK"],
        "online_social": ["US", "Japan", "Global", "Cross_cultural"],
    }

    stakes_levels = ["low", "moderate", "high"]

    scenarios: list[Scenario] = []
    sid = 1

    for domain in Domain:
        domain_key = domain.value
        for norm_type in NormType:
            contexts = base_contexts[domain_key]
            culture_list = cultures[domain_key]
            for idx, context in enumerate(contexts):
                stakes = stakes_levels[idx % len(stakes_levels)]
                cultural_tag = culture_list[idx % len(culture_list)]
                text = f"{context} {norm_tail[norm_type]}"

                scenario = Scenario(
                    id=f"SC{sid:03d}",
                    text=text,
                    domain=domain,
                    norm_type=norm_type,
                    cultural_tag=cultural_tag,
                    stakes_level=stakes,
                    prompt_source="original",
                    notes=f"{domain_key}, {norm_type.value}, example {idx + 1}",
                )
                scenarios.append(scenario)
                sid += 1

    return ScenarioSet(version="v0.3", scenarios=scenarios)


def main() -> None:
    scenario_set = build_scenarios()
    root = Path(__file__).resolve().parents[1]
    out_path = root / "data" / "raw" / "normsense_scenarios_v0.3.json"
    save_scenarios(scenario_set, out_path)
    print(f"Wrote {len(scenario_set.scenarios)} scenarios to {out_path}")


if __name__ == "__main__":
    main()
