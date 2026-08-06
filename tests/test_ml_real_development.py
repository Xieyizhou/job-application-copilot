"""Tests for real-text development extension construction."""

from __future__ import annotations

import pytest

from ml.real_development import (
    CONTEXTUAL_EVIDENCE_EXTRACTION,
    DevelopmentDataError,
    build_development_task,
    evidence_overlap,
    evidence_extraction_manifest,
    extract_action_evidence,
    infer_role_family,
    source_hash,
)
from ml.real_development_isolation import (
    assert_development_isolated,
    build_development_isolation_index,
    development_task_isolated,
)
from ml.real_development_sampling import (
    PROFILE_SEARCH_LIMIT,
    affinity_stratum,
    balanced_reviewer_queue,
    construction_affinity,
)


def test_role_family_and_source_hash_are_deterministic() -> None:
    assert infer_role_family("Machine Learning Engineer", "PyTorch") == "ML"
    assert infer_role_family("Sales Manager", "Lead generation") == "Business"
    assert source_hash("djinni_profile", "123") == source_hash(
        "djinni_profile", "123"
    )


def test_action_evidence_rejects_contact_and_heading_lines() -> None:
    evidence = extract_action_evidence(
        """
        Example Company, London
        Developed reporting dashboards used by finance leaders.
        Contact me at candidate@example.com for more details.
        Automated daily SQL extracts and validation checks.
        Developed a reporting workflow using
        I worked at Example University on a research project.
        Created an internal Azure cloud
        Created the design concept for the book “Unclosed title.
        """
    )

    assert evidence == [
        "Developed reporting dashboards used by finance leaders.",
        "Automated daily SQL extracts and validation checks.",
    ]


def test_contextual_extraction_adds_complete_evidence_without_changing_legacy() -> None:
    source = """
    Responsible for developing forecasting pipelines used by analysts.
    Experience includes building classification models with Python.
    Customer churn project: Built validation reports for model reviewers.
    As a Data Analyst, created dashboards for operational leaders.
    Proficient in Python, SQL, and statistical analysis.
    Project: customer churn prediction.
    Contact me at candidate@example.com for project details.
    """

    assert extract_action_evidence(source) == []
    assert extract_action_evidence(
        source,
        policy=CONTEXTUAL_EVIDENCE_EXTRACTION,
    ) == [
        "Responsible for developing forecasting pipelines used by analysts.",
        "Experience includes building classification models with Python.",
        "Customer churn project: Built validation reports for model reviewers.",
        "As a Data Analyst, created dashboards for operational leaders.",
        "Proficient in Python, SQL, and statistical analysis.",
    ]


def test_contextual_extraction_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match="Unknown evidence extraction policy"):
        extract_action_evidence(
            "Built reliable reporting pipelines for finance leaders.",
            policy="future-policy",
        )


def test_extraction_manifest_separates_frozen_policies() -> None:
    legacy = evidence_extraction_manifest("action_prefix_v1")
    contextual = evidence_extraction_manifest(
        CONTEXTUAL_EVIDENCE_EXTRACTION
    )

    assert legacy["accepted_prefix_groups"] == ["action"]
    assert contextual["accepted_prefix_groups"] == [
        "action",
        "contextual_action",
        "labeled_action",
        "capability",
    ]
    assert legacy["policy_sha256"] != contextual["policy_sha256"]


def test_development_task_has_stable_candidates_and_no_label() -> None:
    statements = [
        "Built Python pipelines for daily analytics reporting.",
        "Created Tableau dashboards used by finance leaders.",
        "Documented data definitions and validation procedures.",
        "Collaborated with product teams to prioritize releases.",
        "Automated spreadsheet checks for monthly reports.",
    ]
    task = build_development_task(
        requirement="Build reliable Python data pipelines.",
        evidence=statements,
        role_family="Data",
        source_job_hash="job-new",
        source_resume_hash="resume-new",
        source_dataset="public_real_text",
        seed="stable",
    )

    assert len(task["candidates"]) == 4
    assert "support_label" not in task
    assert "selected_candidate_id" not in task
    assert evidence_overlap(
        task["requirement"],
        "Built Python pipelines for daily analytics reporting.",
    ) > 0


def test_isolation_rejects_consumed_sources_and_text() -> None:
    task = build_development_task(
        requirement="Build reliable Python data pipelines.",
        evidence=[
            "Built Python pipelines for daily analytics reporting.",
            "Created Tableau dashboards used by finance leaders.",
            "Documented data definitions and validation procedures.",
            "Collaborated with product teams to prioritize releases.",
        ],
        role_family="Data",
        source_job_hash="job-new",
        source_resume_hash="resume-used",
        source_dataset="public_real_text",
        seed="stable",
    )

    with pytest.raises(DevelopmentDataError, match="source overlaps"):
        assert_development_isolated(
            [task],
            [
                {
                    "source_job_hash": "job-old",
                    "source_resume_hash": "resume-used",
                    "requirement": "Use SQL",
                    "candidates": [],
                }
            ],
            [],
        )


def test_isolation_index_reuses_the_same_source_and_text_rules() -> None:
    task = build_development_task(
        requirement="Build reliable Python data pipelines.",
        evidence=[
            "Built Python pipelines for daily analytics reporting.",
            "Created Tableau dashboards used by finance leaders.",
            "Documented data definitions and validation procedures.",
            "Collaborated with product teams to prioritize releases.",
        ],
        role_family="Data",
        source_job_hash="job-new",
        source_resume_hash="resume-new",
        source_dataset="public_real_text",
        seed="stable",
    )
    index = build_development_isolation_index(
        [
            {
                "source_job_hash": "job-old",
                "source_resume_hash": "resume-old",
                "requirement": "Use SQL for reporting.",
                "candidates": [
                    {
                        "candidate_id": "old",
                        "evidence": "Built SQL reports for finance.",
                    }
                ],
            }
        ],
        [],
    )

    assert development_task_isolated(task, index)
    task["source_resume_hash"] = "resume-old"
    assert not development_task_isolated(task, index)


def test_reviewer_queue_balances_lexical_anchor_positions() -> None:
    tasks = []
    for index in range(8):
        tasks.append(
            build_development_task(
                requirement=f"Build Python pipeline {index}.",
                evidence=[
                    f"Built Python pipeline {index} for reporting.",
                    f"Created Tableau dashboard number {index}.",
                    f"Documented validation procedure number {index}.",
                    f"Collaborated with product team number {index}.",
                ],
                role_family="Data",
                source_job_hash=f"job-{index}",
                source_resume_hash=f"resume-{index}",
                source_dataset="public_real_text",
                seed=f"source-{index}",
            )
        )

    queue = balanced_reviewer_queue(
        tasks,
        reviewer_id="reviewer-a",
        random_state=42,
    )
    positions = []
    for task in queue:
        scores = [
            evidence_overlap(task["requirement"], candidate["evidence"])
            for candidate in task["candidates"]
        ]
        positions.append(scores.index(max(scores)))

    assert sorted(positions) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_construction_affinity_recognizes_semantic_skill_family() -> None:
    assert construction_affinity(
        "Experience with ML-based image processing",
        "Implemented computer vision research in production.",
    ) >= 0.18
    assert construction_affinity(
        "Experience with Spring Boot",
        "Created marketing presentations.",
    ) < 0.18


def test_sampling_search_window_covers_diverse_profile_pool() -> None:
    assert PROFILE_SEARCH_LIMIT >= 100


def test_construction_affinity_strata_are_contiguous_and_exclusive() -> None:
    assert affinity_stratum(0.22) == "supportive"
    assert affinity_stratum(0.20) == "supportive"
    assert affinity_stratum(0.18) == "adjacent"
    assert affinity_stratum(0.08) == "adjacent"
    assert affinity_stratum(0.079) == "low"
