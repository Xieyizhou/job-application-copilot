"""Tests for offline evidence shadow comparisons."""

from __future__ import annotations

from ml.evidence_shadow import SHADOW_CANDIDATE_LIMIT, comparison_status


def test_comparison_status_covers_acceptance_and_ranking_disagreements() -> None:
    assert SHADOW_CANDIDATE_LIMIT == 4
    assert (
        comparison_status(
            baseline_accepted=True,
            shadow_accepted=True,
            baseline_evidence="same",
            shadow_evidence="same",
        )
        == "both_accept_same_evidence"
    )
    assert (
        comparison_status(
            baseline_accepted=True,
            shadow_accepted=True,
            baseline_evidence="old",
            shadow_evidence="new",
        )
        == "both_accept_different_evidence"
    )
    assert (
        comparison_status(
            baseline_accepted=False,
            shadow_accepted=True,
            baseline_evidence="",
            shadow_evidence="new",
        )
        == "shadow_only_accept"
    )
    assert (
        comparison_status(
            baseline_accepted=True,
            shadow_accepted=False,
            baseline_evidence="old",
            shadow_evidence="",
        )
        == "baseline_only_accept"
    )
