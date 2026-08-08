"""Tests for the user-facing requirement evidence map."""

from __future__ import annotations

from dashboard_evidence_map import (
    build_evidence_cards,
    evidence_counts,
    evidence_status,
    filter_evidence_cards,
    missing_dimension,
)


def test_current_transparent_matches_map_to_three_user_states() -> None:
    matches = [
        {"accepted": True, "match_type": "Direct support"},
        {"accepted": True, "match_type": "Semantic support"},
        {"accepted": False, "match_type": "Insufficient evidence"},
    ]

    assert [evidence_status(match) for match in matches] == [
        "Direct",
        "Partial",
        "No Support",
    ]


def test_future_student_predictions_use_the_same_ui_contract() -> None:
    assert evidence_status({"prediction": "Direct"}) == "Direct"
    assert evidence_status({"prediction": "Partial"}) == "Partial"
    assert evidence_status({"prediction": "No Support", "accepted": True}) == "No Support"


def test_cards_prefer_student_strongest_evidence_and_count_states() -> None:
    analysis = {
        "semantic_evidence": {
            "matches": [
                {
                    "requirement": "Deploy ML systems",
                    "prediction": "Direct",
                    "strongest_evidence": "Deployed a local evidence-ranking service.",
                    "evidence": "Older baseline evidence.",
                },
                {
                    "requirement": "Lead cross-functional projects",
                    "accepted": True,
                    "match_type": "Semantic support",
                    "evidence": "Collaborated with engineering and operations.",
                },
                {
                    "requirement": "Five years of Kubernetes experience",
                    "accepted": False,
                    "match_type": "Insufficient evidence",
                    "evidence": "",
                },
            ]
        }
    }

    cards = build_evidence_cards(analysis)

    assert cards[0]["evidence"] == "Deployed a local evidence-ranking service."
    assert "scope, outcome, or ownership" in cards[1]["explanation"]
    assert cards[2]["evidence"] == ""
    assert evidence_counts(cards) == {"Direct": 1, "Partial": 1, "No Support": 1}


def test_partial_explanation_identifies_compound_and_numeric_boundaries() -> None:
    assert "duration or scale" in missing_dimension(
        {"numeric_constraint_supported": False}, "Partial"
    )
    assert "compound requirement" in missing_dimension(
        {"compound_requirement_supported": False}, "Partial"
    )


def test_filter_preserves_requirement_order() -> None:
    cards = build_evidence_cards(
        {
            "semantic_evidence": {
                "matches": [
                    {"requirement": "First", "prediction": "Direct"},
                    {"requirement": "Second", "prediction": "Partial"},
                    {"requirement": "Third", "prediction": "Direct"},
                ]
            }
        }
    )

    assert filter_evidence_cards(cards, "All") == cards
    assert [card["requirement"] for card in filter_evidence_cards(cards, "Direct")] == [
        "First",
        "Third",
    ]


def test_keyword_only_fallback_is_visible_but_never_promoted_to_direct() -> None:
    cards = build_evidence_cards(
        {
            "semantic_evidence": {"matches": []},
            "matched_keywords": ["machine learning", "intern"],
            "missing_keywords": ["Python"],
        }
    )

    assert [card["status"] for card in cards] == ["Partial", "Partial", "No Support"]
    assert cards[0]["context"] == "Keyword signal"
    assert "sentence-level evidence is unavailable" in cards[0]["evidence"]
    assert cards[2]["evidence"] == ""
