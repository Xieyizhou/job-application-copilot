"""Tests for role-balanced shadow disagreement review sampling."""

from __future__ import annotations

from scripts.ml.prepare_shadow_disagreement_review import _balanced_sample


def test_balanced_sample_prefers_non_shadow_only_disagreements() -> None:
    rows = [
        {
            "role_family": family,
            "comparison": status,
            "source_job_hash": f"{family}-{index}",
            "requirement": f"Requirement {index}",
        }
        for family in ("Data", "ML", "Software", "Business")
        for index, status in enumerate(
            [
                "shadow_only_accept",
                "shadow_only_accept",
                "both_accept_different_evidence",
            ]
        )
    ]

    sampled = _balanced_sample(
        rows,
        tasks_per_family=2,
        random_state=7,
    )

    assert len(sampled) == 8
    assert sum(
        row["comparison"] == "both_accept_different_evidence"
        for row in sampled
    ) == 4
