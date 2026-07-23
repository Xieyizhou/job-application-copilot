from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.barrier_contracts import BarrierContractError, validate_generated_case, validate_review
from ml.barrier_audit import audit_proposal_pool, blind_packet_leaks
from ml.review_consensus import (
    adjudicate_reviews,
    build_blind_queue,
    fleiss_kappa,
    merge_generated_cases,
    promote_consensus_gold,
    unblind_review,
)


def _proposal() -> dict[str, object]:
    return {
        "schema_version": 1,
        "producer_id": "producer-semantic",
        "generation_run_id": "batch-001",
        "semantic_case_group_id": "semantic-deployment-001",
        "barrier_id": "B2_SEMANTIC_DIRECT",
        "role_family": "ML",
        "requirement": "Deploy predictive models into a production service.",
        "candidates": [
            {
                "candidate_id": "candidate-1",
                "evidence": "Shipped a churn scorer behind a versioned service endpoint.",
            },
            {
                "candidate_id": "candidate-2",
                "evidence": "Compared classification metrics in an offline notebook.",
            },
            {
                "candidate_id": "candidate-3",
                "evidence": "Prepared weekly summaries for the analytics leadership team.",
            },
            {
                "candidate_id": "candidate-4",
                "evidence": "Maintained interface specifications for an internal reporting tool.",
            },
        ],
        "intended_label": "Direct",
        "intended_candidate_id": "candidate-1",
    }


def _review(
    case: dict[str, object],
    reviewer_id: str,
    *,
    label: str = "Direct",
    selected_id: str | None = "candidate-1",
    flags: list[str] | None = None,
) -> dict[str, object]:
    judgments = [
        {
            "candidate_id": candidate["candidate_id"],
            "support_label": (
                label
                if candidate["candidate_id"] == selected_id
                else "No Support"
            ),
        }
        for candidate in case["candidates"]  # type: ignore[union-attr]
    ]
    return {
        "schema_version": 1,
        "case_id": case["case_id"],
        "reviewer_id": reviewer_id,
        "support_label": label,
        "selected_candidate_id": selected_id,
        "candidate_judgments": judgments,
        "flags": flags or [],
    }


def test_generated_case_contract_normalizes_content_id() -> None:
    case = validate_generated_case(_proposal())

    assert case["record_type"] == "generation_proposal"
    assert str(case["case_id"]).startswith("case-")
    assert case["source_kind"] == "synthetic"


def test_blind_packets_are_allow_listed_and_rekeyed_per_reviewer() -> None:
    case = validate_generated_case(_proposal())

    first_queue, first_maps = build_blind_queue([case], reviewer_id="reviewer-1")
    second_queue, second_maps = build_blind_queue([case], reviewer_id="reviewer-2")

    assert set(first_queue[0]) == {
        "record_type",
        "schema_version",
        "review_task_id",
        "requirement",
        "candidates",
    }
    serialized = repr(first_queue[0])
    for hidden_value in (
        case["producer_id"],
        case["barrier_id"],
        case["intended_label"],
        case["semantic_case_group_id"],
        case["case_id"],
    ):
        assert str(hidden_value) not in serialized
    first_ids = {candidate["candidate_id"] for candidate in first_queue[0]["candidates"]}
    second_ids = {candidate["candidate_id"] for candidate in second_queue[0]["candidates"]}
    assert first_ids.isdisjoint(second_ids)
    assert first_maps[0]["candidate_id_map"] != second_maps[0]["candidate_id_map"]
    assert blind_packet_leaks(first_queue) == []

    leaked_task = {**first_queue[0], "barrier_id": "B2_SEMANTIC_DIRECT"}
    assert "disallowed fields" in blind_packet_leaks([leaked_task])[0]


def test_review_contract_requires_every_candidate_and_matching_overall_label() -> None:
    case = validate_generated_case(_proposal())
    valid = _review(case, "reviewer-1")

    assert validate_review(valid, case)["support_label"] == "Direct"

    incomplete = deepcopy(valid)
    incomplete["candidate_judgments"] = incomplete["candidate_judgments"][:-1]
    with pytest.raises(BarrierContractError, match="every candidate"):
        validate_review(incomplete, case)

    inconsistent = deepcopy(valid)
    inconsistent["support_label"] = "Partial"
    with pytest.raises(BarrierContractError, match="strongest candidate"):
        validate_review(inconsistent, case)


def test_unblind_review_restores_canonical_candidate_ids() -> None:
    case = validate_generated_case(_proposal())
    queue, mappings = build_blind_queue([case], reviewer_id="reviewer-1")
    task = queue[0]
    mapping = mappings[0]
    canonical_to_blind = {
        canonical: blind
        for blind, canonical in mapping["candidate_id_map"].items()
    }
    selected_blind = canonical_to_blind["candidate-1"]
    blind_review = {
        "review_task_id": task["review_task_id"],
        "support_label": "Direct",
        "selected_candidate_id": selected_blind,
        "candidate_judgments": [
            {
                "candidate_id": candidate["candidate_id"],
                "support_label": (
                    "Direct"
                    if candidate["candidate_id"] == selected_blind
                    else "No Support"
                ),
            }
            for candidate in task["candidates"]
        ],
        "flags": [],
    }

    review = unblind_review(blind_review, mapping, case)

    assert review["selected_candidate_id"] == "candidate-1"
    assert {item["candidate_id"] for item in review["candidate_judgments"]} == {
        "candidate-1",
        "candidate-2",
        "candidate-3",
        "candidate-4",
    }

    blind_review["candidate_judgments"] = {
        judgment["candidate_id"]: judgment["support_label"]
        for judgment in blind_review["candidate_judgments"]
    }
    assert unblind_review(blind_review, mapping, case) == review


def test_only_unanimous_blind_reviews_are_promoted_to_gold() -> None:
    case = validate_generated_case(_proposal())
    reviews = [_review(case, f"reviewer-{index}") for index in range(3)]

    decisions, report = adjudicate_reviews([case], reviews)
    gold = promote_consensus_gold([case], decisions, reviews)

    assert report["status_counts"] == {"accepted": 1}
    assert report["overall_fleiss_kappa"] == 1.0
    assert report["candidate_fleiss_kappa"] == 1.0
    assert decisions[0]["status"] == "accepted"
    assert gold[0]["decision_source"] == "blind_consensus"
    assert gold[0]["support_label"] == "Direct"
    assert gold[0]["best_candidate_id"] == "candidate-1"
    assert "producer_intended_label" not in gold[0]


def test_generator_intent_does_not_vote_in_consensus() -> None:
    case = validate_generated_case(_proposal())
    reviews = [
        _review(
            case,
            f"reviewer-{index}",
            label="Partial",
            selected_id="candidate-2",
        )
        for index in range(3)
    ]

    decisions, _ = adjudicate_reviews([case], reviews)
    gold = promote_consensus_gold([case], decisions, reviews)

    assert case["intended_label"] == "Direct"
    assert decisions[0]["consensus_label"] == "Partial"
    assert "producer_intended_label" not in decisions[0]
    assert gold[0]["support_label"] == "Partial"
    assert gold[0]["best_candidate_id"] == "candidate-2"


def test_disagreement_and_quality_flags_route_away_from_gold() -> None:
    case = validate_generated_case(_proposal())
    disagreeing = [
        _review(case, "reviewer-1"),
        _review(case, "reviewer-2"),
        _review(case, "reviewer-3", label="Partial", selected_id="candidate-2"),
    ]

    decisions, _ = adjudicate_reviews([case], disagreeing)
    assert decisions[0]["status"] == "human_review"
    assert promote_consensus_gold([case], decisions, disagreeing) == []

    privacy_reviews = [
        _review(case, f"reviewer-{index}", flags=["privacy_risk"])
        for index in range(3)
    ]
    privacy_decisions, _ = adjudicate_reviews([case], privacy_reviews)
    assert privacy_decisions[0]["status"] == "rejected"


def test_merge_rejects_duplicate_requirements_across_producers() -> None:
    duplicate = deepcopy(_proposal())
    duplicate["producer_id"] = "producer-other"

    merged, rejected = merge_generated_cases([_proposal(), duplicate])

    assert len(merged) == 1
    assert len(rejected) == 1
    assert "Duplicate generated case" in rejected[0]


def test_proposal_audit_keeps_generator_intent_diagnostic_only() -> None:
    case = validate_generated_case(_proposal())

    report = audit_proposal_pool([case])

    assert report["case_count"] == 1
    assert "Producer intent is diagnostic metadata only" in report["gold_policy"]
    assert report["near_duplicate_pairs"] == []


def test_fleiss_kappa_reports_perfect_and_imperfect_agreement() -> None:
    assert fleiss_kappa([["Direct", "Direct", "Direct"]]) == 1.0
    assert fleiss_kappa([["Direct", "Partial", "No Support"]]) < 0.0
