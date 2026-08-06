"""Regression tests for successor source-text quality gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml import real_development_sources
from ml.real_development import CONTEXTUAL_EVIDENCE_EXTRACTION
from ml.source_text_quality import (
    LEGACY_SOURCE_TEXT_QUALITY,
    SUCCESSOR_V5_SOURCE_TEXT_QUALITY,
    SUCCESSOR_V6_SOURCE_TEXT_QUALITY,
    SUCCESSOR_V7_SOURCE_TEXT_QUALITY,
    SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
    audit_materialized_text_quality,
    canonical_public_text,
    diverse_quality_evidence,
    evidence_quality_reasons,
    evidence_quality_reasons_for_policy,
    requirement_quality_reasons,
    requirement_quality_reasons_for_policy,
    source_text_quality_manifest,
    texts_are_near_duplicates,
    validate_source_text_quality_policy,
)


@pytest.mark.parametrize(
    "text",
    (
        "As a QA you will be responsible for:**",
        "Experience with Azure data services, including",
        "You will participate in implementation of technical",
        "Hands on experience with data",
        "Experience with algorithms for classification, regression,",
    ),
)
def test_v5_requirement_gate_rejects_v4_failure_shapes(text: str) -> None:
    assert requirement_quality_reasons(text)


@pytest.mark.parametrize(
    "text",
    (
        "Automated data mining processes which significantly",
        "Developed an application based on a neural network model that",
        "Created clusters using k-means and logistic regression,",
        "Built funnels that simplified the interface and eliminated",
        "Developed an online store for a fitness company called",
        "Analyzed four years of records and built data",
    ),
)
def test_v5_evidence_gate_rejects_v4_failure_shapes(text: str) -> None:
    assert evidence_quality_reasons(text)


def test_v5_quality_gate_preserves_complete_reviewable_text() -> None:
    requirements = (
        "Experience creating dashboards and reports with Power BI.",
        "Develop reliable cloud infrastructure using AWS or GCP services.",
        "At least three years of software application development experience;",
    )
    evidence = (
        "Developed tools that extract information from large-scale data.",
        "Automated daily SQL extracts and validation checks for finance teams.",
        "Created dashboards that helped leaders monitor operational outcomes.",
    )

    assert all(not requirement_quality_reasons(text) for text in requirements)
    assert all(not evidence_quality_reasons(text) for text in evidence)


def test_v5_evidence_pool_removes_fragments_and_near_paraphrases() -> None:
    first = (
        "Collaborated with the frontend team to integrate backend "
        "functions into the user interface."
    )
    paraphrase = (
        "Collaborated with the frontend team to integrate the backend "
        "with the user interface effectively."
    )
    complete = "Automated daily deployment checks for the engineering team."
    fragment = "Developed an application based on a neural network model that"

    selected = diverse_quality_evidence(
        [first, paraphrase, complete, fragment]
    )

    assert texts_are_near_duplicates(first, paraphrase)
    assert selected == [first, complete]


def test_quality_policy_manifest_is_stable_and_versioned() -> None:
    first = source_text_quality_manifest(SUCCESSOR_V5_SOURCE_TEXT_QUALITY)
    second = source_text_quality_manifest(SUCCESSOR_V5_SOURCE_TEXT_QUALITY)

    assert first == second
    assert first["policy"] == SUCCESSOR_V5_SOURCE_TEXT_QUALITY
    assert first["policy_sha256"] == (
        "15f34aceb8aba334971ed87a73ff73ddeee9ccf3ba1018e14a6560154c505c34"
    )
    validate_source_text_quality_policy(LEGACY_SOURCE_TEXT_QUALITY)
    with pytest.raises(ValueError, match="Unknown source text quality policy"):
        validate_source_text_quality_policy("future-unfrozen-policy")


def test_v6_validates_the_same_canonical_text_shown_to_reviewers() -> None:
    prefixed = "(Full-time) Experience with REST API integration."
    complete = (
        "(Full-time) Experience integrating REST APIs for production systems."
    )

    assert not requirement_quality_reasons(prefixed)
    assert canonical_public_text(prefixed) == (
        "Experience with REST API integration."
    )
    assert requirement_quality_reasons_for_policy(
        prefixed,
        SUCCESSOR_V6_SOURCE_TEXT_QUALITY,
    ) == ("requirement_word_count",)
    assert not requirement_quality_reasons_for_policy(
        complete,
        SUCCESSOR_V6_SOURCE_TEXT_QUALITY,
    )
    manifest = source_text_quality_manifest(
        SUCCESSOR_V6_SOURCE_TEXT_QUALITY
    )
    assert manifest["canonicalize_before_validation"] is True
    assert manifest["canonicalization_contract"] == (
        "canonical_public_text_v1"
    )


def test_v6_requirement_loader_filters_after_canonical_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = real_development_sources.pd.DataFrame(
        [
            {
                "id": "job-1",
                "Position": "Data Analyst",
                "Primary Keyword": "analytics",
                "Long Description": "unused",
            }
        ]
    )
    monkeypatch.setattr(
        real_development_sources.pd,
        "read_parquet",
        lambda *_args, **_kwargs: frame,
    )
    monkeypatch.setattr(
        real_development_sources,
        "extract_requirement_records",
        lambda _text: [
            {
                "text": "(Full-time) Experience with REST API integration."
            },
            {
                "text": (
                    "(Full-time) Experience integrating REST APIs for "
                    "production systems."
                )
            },
        ],
    )

    pools = real_development_sources.load_djinni_requirements(
        Path("unused.parquet"),
        set(),
        random_state=3,
        text_quality_policy=SUCCESSOR_V6_SOURCE_TEXT_QUALITY,
    )

    assert pools["Data"][0]["requirement"] == (
        "Experience integrating REST APIs for production systems."
    )


def test_v6_known_manual_failure_is_not_hidden_by_automated_gate() -> None:
    truncated = (
        "Built and managed a sales stack with validation rules and dashboard man"
    )

    assert not evidence_quality_reasons(truncated)


def test_v7_rejects_v6_manual_failure_and_requires_sentence_boundary() -> None:
    truncated = (
        "Built and managed a sales stack with validation rules and dashboard man"
    )
    complete = (
        "Built and managed a sales stack with validation rules and dashboards."
    )

    assert evidence_quality_reasons_for_policy(
        truncated,
        SUCCESSOR_V7_SOURCE_TEXT_QUALITY,
    ) == ("evidence_missing_sentence_boundary",)
    assert not evidence_quality_reasons_for_policy(
        complete,
        SUCCESSOR_V7_SOURCE_TEXT_QUALITY,
    )


def test_v8_filters_boundary_failures_before_profile_eligibility() -> None:
    truncated = (
        "Built and managed a sales stack with validation rules and dashboard man"
    )
    complete = (
        "Built and managed a sales stack with validation rules and dashboards."
    )

    assert diverse_quality_evidence(
        [truncated, complete],
        policy=SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
    ) == [complete]
    assert not evidence_quality_reasons_for_policy(
        "Documented the deployment process for reviewers;",
        SUCCESSOR_V7_SOURCE_TEXT_QUALITY,
    )


def test_v7_policy_manifest_does_not_change_frozen_v6_checksum() -> None:
    v6 = source_text_quality_manifest(SUCCESSOR_V6_SOURCE_TEXT_QUALITY)
    v7 = source_text_quality_manifest(SUCCESSOR_V7_SOURCE_TEXT_QUALITY)

    assert v6["policy_sha256"] == (
        "39e2c05f45e3b648e7dc9ff1aacb291b38af30bd7618c723e6f47ea3774b6be6"
    )
    assert v7["require_terminal_sentence_boundary"] is True
    assert v7["accepted_terminal_boundaries"] == [".", "!", "?", ";"]
    assert v7["policy_sha256"] != v6["policy_sha256"]
    v8 = source_text_quality_manifest(SUCCESSOR_V8_SOURCE_TEXT_QUALITY)
    assert v8["policy_sha256"] != v7["policy_sha256"]


def test_v7_materialized_audit_applies_the_frozen_policy() -> None:
    task = {
        "task_id": "task-1",
        "requirement": (
            "Experience delivering reliable analytics dashboards for leaders."
        ),
        "source_resume_hash": "resume-1",
        "candidates": [
            {
                "candidate_id": "candidate-1",
                "evidence": (
                    "Built analytics dashboards for operational leaders and dashboard man"
                ),
            }
        ],
    }

    audit = audit_materialized_text_quality(
        [task],
        policy=SUCCESSOR_V7_SOURCE_TEXT_QUALITY,
    )

    assert audit["automated_quality_gate_passed"] is False
    assert audit["finding_counts"] == {
        "evidence_missing_sentence_boundary": 1
    }


def test_djinni_requirement_quality_policy_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = real_development_sources.pd.DataFrame(
        [
            {
                "id": "job-1",
                "Position": "Data Analyst",
                "Primary Keyword": "analytics",
                "Long Description": "unused",
            }
        ]
    )
    monkeypatch.setattr(
        real_development_sources.pd,
        "read_parquet",
        lambda *_args, **_kwargs: frame,
    )
    monkeypatch.setattr(
        real_development_sources,
        "extract_requirement_records",
        lambda _text: [
            {"text": "As a Data Analyst you will be responsible for:**"},
            {
                "text": (
                    "Develop reliable analytics dashboards using SQL "
                    "and Power BI."
                )
            },
        ],
    )

    legacy = real_development_sources.load_djinni_requirements(
        Path("unused.parquet"),
        set(),
        random_state=3,
    )
    strict = real_development_sources.load_djinni_requirements(
        Path("unused.parquet"),
        set(),
        random_state=3,
        text_quality_policy=SUCCESSOR_V5_SOURCE_TEXT_QUALITY,
    )
    excluded_by_frozen_selection = (
        real_development_sources.load_djinni_requirements(
            Path("unused.parquet"),
            set(),
            random_state=3,
            text_quality_policy=SUCCESSOR_V5_SOURCE_TEXT_QUALITY,
            included_job_hashes={"different-source-hash"},
        )
    )

    assert legacy["Data"][0]["requirement"].endswith("for:**")
    assert strict["Data"][0]["requirement"].endswith("Power BI.")
    assert not any(excluded_by_frozen_selection.values())


def test_djinni_profile_quality_policy_filters_before_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = real_development_sources.pd.DataFrame(
        [
            {
                "id": "profile-1",
                "Position": "Data Analyst",
                "Primary Keyword": "analytics",
                "CV": "unused",
            }
        ]
    )
    evidence = [
        "Automated data mining processes which significantly",
        "Built reliable SQL pipelines for daily finance reporting.",
        "Created Power BI dashboards for operational leaders.",
        "Documented data definitions and validation procedures.",
        "Collaborated with product teams to prioritize analytics work.",
    ]
    monkeypatch.setattr(
        real_development_sources.pd,
        "read_parquet",
        lambda *_args, **_kwargs: frame,
    )
    monkeypatch.setattr(
        real_development_sources,
        "extract_action_evidence",
        lambda _text: evidence,
    )

    legacy = real_development_sources.load_djinni_profiles(
        Path("unused.parquet"),
        set(),
        random_state=3,
        minimum_evidence=4,
    )
    strict = real_development_sources.load_djinni_profiles(
        Path("unused.parquet"),
        set(),
        random_state=3,
        minimum_evidence=4,
        text_quality_policy=SUCCESSOR_V5_SOURCE_TEXT_QUALITY,
    )
    excluded_by_frozen_selection = real_development_sources.load_djinni_profiles(
        Path("unused.parquet"),
        set(),
        random_state=3,
        minimum_evidence=4,
        text_quality_policy=SUCCESSOR_V5_SOURCE_TEXT_QUALITY,
        included_resume_hashes={"different-source-hash"},
    )

    assert legacy["Data"][0]["evidence"] == evidence
    assert len(strict["Data"]) == 1
    assert strict["Data"][0]["evidence"] == evidence[1:]
    assert not any(excluded_by_frozen_selection.values())


def test_djinni_profile_contextual_extraction_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = real_development_sources.pd.DataFrame(
        [
            {
                "id": "profile-contextual",
                "Position": "Data Analyst",
                "Primary Keyword": "analytics",
                "CV": (
                    "Responsible for developing forecasting pipelines used by analysts.\n"
                    "Experience includes building classification models with Python.\n"
                    "Customer churn project: Built validation reports for reviewers.\n"
                    "Proficient in Python, SQL, and statistical analysis."
                ),
            }
        ]
    )
    monkeypatch.setattr(
        real_development_sources.pd,
        "read_parquet",
        lambda *_args, **_kwargs: frame,
    )

    legacy = real_development_sources.load_djinni_profiles(
        Path("unused.parquet"),
        set(),
        random_state=3,
        minimum_evidence=4,
        text_quality_policy=SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
    )
    contextual = real_development_sources.load_djinni_profiles(
        Path("unused.parquet"),
        set(),
        random_state=3,
        minimum_evidence=4,
        text_quality_policy=SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
        evidence_extraction_policy=CONTEXTUAL_EVIDENCE_EXTRACTION,
    )

    assert not any(legacy.values())
    assert len(contextual["Data"]) == 1


def test_combined_source_revision_is_content_free_and_detects_drift(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"dataset": "djinni"}', encoding="utf-8")
    ats = tmp_path / "ats"
    ats.mkdir()
    (ats / "state.json").write_text('{"rows": 2}', encoding="utf-8")
    first = real_development_sources.development_source_revision(
        manifest,
        ats_dataset_dir=ats,
    )
    repeated = real_development_sources.development_source_revision(
        manifest,
        ats_dataset_dir=ats,
    )
    (ats / "state.json").write_text('{"rows": 3}', encoding="utf-8")
    changed = real_development_sources.development_source_revision(
        manifest,
        ats_dataset_dir=ats,
    )

    assert first == repeated
    assert first.startswith("djinni+ats-sha256:")
    assert first != changed
    assert "rows" not in first
