"""Tests for realistic grouped operational-development contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.operational_development import assert_operational_development_contract
from ml.real_development import DevelopmentDataError, build_development_task
from ml.real_development_sources import load_talentclef_profiles


def _tasks() -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    families = ("Data", "ML", "Software", "Business")
    for family in families:
        task = build_development_task(
            requirement=f"Build a new {family} workflow.",
            evidence=[
                f"Built a {family} workflow for reporting.",
                f"Documented {family} quality checks for delivery.",
                f"Collaborated on {family} project planning.",
                f"Supported a distinct {family} implementation.",
            ],
            role_family=family,
            source_job_hash=f"job-{family}",
            source_resume_hash=f"resume-{family}",
            source_dataset="operational_public_text",
            seed=family,
        )
        task["profile_role_family"] = family
        task["pairing_type"] = "aligned"
        tasks.append(task)
    return tasks


def test_contract_allows_grouped_resume_reuse_but_not_job_reuse() -> None:
    tasks = _tasks()
    report = assert_operational_development_contract(
        tasks,
        [],
        [],
        tasks_per_resume=1,
        resumes_per_profile_family=1,
    )

    assert report["resume_groups"] == 4
    tasks[1]["source_job_hash"] = tasks[0]["source_job_hash"]
    with pytest.raises(DevelopmentDataError, match="distinct job"):
        assert_operational_development_contract(
            tasks,
            [],
            [],
            tasks_per_resume=1,
            resumes_per_profile_family=1,
        )


def test_talentclef_loader_keeps_only_profiles_with_enough_evidence(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "development" / "en" / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "eligible").write_text(
        "\n".join(
            [
                "Data Analyst",
                "Built recurring SQL reports for operational teams.",
                "Developed Tableau dashboards for weekly performance reviews.",
                "Automated data quality checks across reporting datasets.",
                "Analyzed customer trends to support planning decisions.",
                "Presented measured findings to business stakeholders.",
            ]
        ),
        encoding="utf-8",
    )
    (corpus / "short").write_text(
        "Data Analyst\nBuilt one SQL report for a team.",
        encoding="utf-8",
    )

    profiles = load_talentclef_profiles(
        tmp_path,
        set(),
        random_state=4,
    )

    assert len(profiles["Data"]) == 1
    assert len(profiles["Data"][0]["evidence"]) == 5
    assert (
        profiles["Data"][0]["source_dataset"]
        == "talentclef_2026_development_synthetic_profile"
    )
