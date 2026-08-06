"""Contract tests for the MiniLM Web shadow integration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

from ml.web_candidate_shadow import (
    DEFAULT_BUNDLE_DIR,
    DEFAULT_REQUIREMENT_BUNDLE_DIR,
    build_web_candidate_shadow_report,
    enqueue_web_candidate_shadow,
    privacy_safe_shadow_report,
    record_web_candidate_shadow,
    summarize_web_candidate_shadow_reports,
)


JOB_TEXT = "Required: Python, SQL, model evaluation, and stakeholder communication."
RESUME_TEXT = """# Experience
- Built reliable Python analysis services.
- Queried large operational datasets with SQL.
- Evaluated several classification models systematically.
- Presented analytical results to senior stakeholders.
"""


class FakeCandidate:
    def predict_task_safe(self, requirement: str, evidence: list[str]) -> dict[str, object]:
        assert requirement
        assert len(evidence) == 4
        return {
            "available": True,
            "prediction": "Direct",
            "accepted": True,
            "strongest_evidence_index": 0,
            "strongest_evidence": evidence[0],
            "support_probability": 0.9,
            "direct_probability_given_support": 0.8,
            "rank_score": 1.2,
            "model_version": "fake",
        }


class FakeRequirementExtractor:
    def extract_safe(self, job_text: str) -> dict[str, object]:
        assert job_text
        return {
            "available": True,
            "model_version": "fake-extractor",
            "candidate_segments": 1,
            "requirements": [
                {
                    "text": "Python and SQL",
                    "classification": "Required",
                    "probability": 0.9,
                    "section": "requirements",
                }
            ],
        }


def test_default_bundle_is_e15_passed_v21_shadow_candidate() -> None:
    assert DEFAULT_BUNDLE_DIR.name == "web_candidate_minilm_v21_e15"
    assert DEFAULT_REQUIREMENT_BUNDLE_DIR.name == "minilm_requirement_extractor_e4_bundle"


def test_report_is_explicitly_diagnostic_and_compares_baseline() -> None:
    report = build_web_candidate_shadow_report(
        JOB_TEXT,
        RESUME_TEXT,
        candidate=FakeCandidate(),
    )
    assert report["mode"] == "web_shadow_only"
    assert report["comparisons"]
    assert report["product_state_modified"] is False
    assert report["user_visible_state_modified"] is False
    assert all("baseline" in row and "shadow" in row for row in report["comparisons"])


def test_two_stage_report_uses_semantic_requirements_missing_from_baseline() -> None:
    report = build_web_candidate_shadow_report(
        "You will apply Python and SQL to operational analysis.",
        RESUME_TEXT,
        candidate=FakeCandidate(),
        requirement_extractor=FakeRequirementExtractor(),
    )
    assert report["legacy_requirement_count"] == 0
    assert report["requirement_count"] == 1
    assert report["comparisons"][0]["comparison"] == "baseline_missing_requirement"
    assert report["requirement_extractor"]["model_version"] == "fake-extractor"
    assert report["shadow_prediction_counts"] == {"Direct": 1}
    assert report["shadow_accepted_count"] == 1


def test_personal_record_is_private_content_addressed_and_deduplicated(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    requirement_bundle = tmp_path / "requirement-bundle"
    requirement_bundle.mkdir()
    (requirement_bundle / "manifest.json").write_text("{}", encoding="utf-8")
    shadow_root = tmp_path / "shadow"
    loads = 0

    def load_candidate(bundle_path: str, device: str) -> FakeCandidate:
        nonlocal loads
        loads += 1
        assert bundle_path == str(bundle.resolve())
        assert device in {"cpu", "mps"}
        return FakeCandidate()

    first = record_web_candidate_shadow(
        JOB_TEXT,
        RESUME_TEXT,
        workspace_mode="personal",
        bundle_dir=bundle,
        requirement_bundle_dir=requirement_bundle,
        shadow_root=shadow_root,
        candidate_loader=load_candidate,
        requirement_extractor_loader=lambda path: FakeRequirementExtractor(),
    )
    second = record_web_candidate_shadow(
        JOB_TEXT,
        RESUME_TEXT,
        workspace_mode="personal",
        bundle_dir=bundle,
        requirement_bundle_dir=requirement_bundle,
        shadow_root=shadow_root,
        candidate_loader=load_candidate,
        requirement_extractor_loader=lambda path: FakeRequirementExtractor(),
    )
    assert first == second
    assert loads == 1
    assert first is not None
    assert first.stat().st_mode & 0o777 == 0o600
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["status"] == "recorded"
    assert payload["input_paths_recorded"] is False
    assert payload["raw_text_persisted"] is False
    assert payload["content_hashes_only"] is True
    assert "comparisons" not in payload
    assert JOB_TEXT not in first.read_text(encoding="utf-8")
    assert "Built reliable Python analysis services" not in first.read_text(
        encoding="utf-8"
    )
    assert payload["extractor_version"] == "minilm-two-stage-v3"


def test_background_shadow_is_disabled_by_default_and_requires_opt_in() -> None:
    calls: list[tuple[str, str, str]] = []

    def recorder(job_text: str, resume_text: str, *, workspace_mode: str) -> None:
        calls.append((job_text, resume_text, workspace_mode))

    with patch.dict(os.environ, {}, clear=True):
        assert not enqueue_web_candidate_shadow(
            JOB_TEXT,
            RESUME_TEXT,
            workspace_mode="personal",
            recorder=recorder,
            submitter=lambda callback: callback(),
        )
    with patch.dict(os.environ, {"JOB_COPILOT_WEB_SHADOW_ENABLED": "true"}):
        assert enqueue_web_candidate_shadow(
            JOB_TEXT,
            RESUME_TEXT,
            workspace_mode="personal",
            recorder=recorder,
            submitter=lambda callback: callback(),
        )
    assert calls == [(JOB_TEXT, RESUME_TEXT, "personal")]


def test_privacy_sanitizer_removes_raw_text_paths_and_exception_details() -> None:
    safe = privacy_safe_shadow_report(
        {
            "schema_version": 1,
            "mode": "web_shadow_only",
            "status": "unavailable",
            "reason": "local model path failed on resume text",
            "comparisons": [{"requirement": JOB_TEXT, "evidence": RESUME_TEXT}],
            "job_text_sha256": "job-hash",
            "resume_text_sha256": "resume-hash",
            "comparison_counts": {"both_reject": 1},
        }
    )

    assert safe["reason_code"] == "shadow_runtime_unavailable"
    assert safe["raw_text_persisted"] is False
    assert safe["job_text_sha256"] == "job-hash"
    assert "reason" not in safe
    assert "comparisons" not in safe


def test_demo_never_loads_or_writes_shadow(tmp_path: Path) -> None:
    def forbidden_loader(bundle_path: str, device: str) -> object:
        raise AssertionError((bundle_path, device))

    result = record_web_candidate_shadow(
        JOB_TEXT,
        RESUME_TEXT,
        workspace_mode="demo",
        bundle_dir=tmp_path / "missing",
        shadow_root=tmp_path / "shadow",
        candidate_loader=forbidden_loader,
    )
    assert result is None
    assert not (tmp_path / "shadow").exists()


def test_missing_candidate_records_failure_without_raising(tmp_path: Path) -> None:
    output = record_web_candidate_shadow(
        JOB_TEXT,
        RESUME_TEXT,
        workspace_mode="personal",
        bundle_dir=tmp_path / "missing",
        shadow_root=tmp_path / "shadow",
    )
    assert output is not None
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "unavailable"
    assert payload["product_state_modified"] is False
    assert payload["user_visible_state_modified"] is False


def test_unwritable_shadow_destination_cannot_break_product_path(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("occupied", encoding="utf-8")
    result = record_web_candidate_shadow(
        JOB_TEXT,
        RESUME_TEXT,
        workspace_mode="personal",
        bundle_dir=bundle,
        shadow_root=blocked,
        candidate_loader=lambda bundle_path, device: FakeCandidate(),
    )
    assert result is None


def test_legacy_unparseable_job_is_still_recorded_by_semantic_extractor(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    requirement_bundle = tmp_path / "requirement-bundle"
    requirement_bundle.mkdir()
    (requirement_bundle / "manifest.json").write_text("{}", encoding="utf-8")
    result = record_web_candidate_shadow(
        "An old job stub without requirements.",
        RESUME_TEXT,
        workspace_mode="personal",
        bundle_dir=bundle,
        requirement_bundle_dir=requirement_bundle,
        shadow_root=tmp_path / "shadow",
        candidate_loader=lambda bundle_path, device: FakeCandidate(),
        requirement_extractor_loader=lambda path: FakeRequirementExtractor(),
    )
    assert result is not None
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["status"] == "recorded"


def test_summary_reports_coverage_and_disagreement_without_accuracy(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(
            {
                "observation_id": "one",
                "status": "recorded",
                "requirement_count": 2,
                "legacy_requirement_count": 0,
                "shadow_accepted_count": 1,
                "shadow_prediction_counts": {"Direct": 1, "No Support": 1},
                "comparison_counts": {"both_reject": 1, "shadow_only_accept": 1},
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps({"observation_id": "two", "status": "unavailable"}),
        encoding="utf-8",
    )
    summary = summarize_web_candidate_shadow_reports(
        [first, second],
        discovered_jobs=10,
        parseable_jobs=2,
    )
    assert summary["unparseable_jobs"] == 8
    assert summary["usable_observations"] == 1
    assert summary["disagreement_count"] == 1
    assert summary["semantic_only_observations"] == 1
    assert summary["shadow_acceptance_rate"] == 0.5
    assert summary["shadow_prediction_counts"] == {"Direct": 1, "No Support": 1}
    assert summary["accuracy_claimed"] is False
    assert summary["teacher_or_human_labels_used"] is False
