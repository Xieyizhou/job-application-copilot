"""Reusable end-to-end QA cases for successor source extraction."""

from __future__ import annotations

import json
from pathlib import Path

from ml.real_development import extract_action_evidence
from ml.source_text_quality import (
    SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
    diverse_quality_evidence,
    evidence_quality_reasons_for_policy,
    requirement_quality_reasons_for_policy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "ml" / "source_pipeline_qa_v1.json"


def test_pipeline_qa_fixture_covers_known_failures_and_clean_controls() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["policy"] == SUCCESSOR_V8_SOURCE_TEXT_QUALITY
    assert len(payload["cases"]) == 8
    for case in payload["cases"]:
        reasons = (
            requirement_quality_reasons_for_policy(
                case["text"],
                SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
            )
            if case["kind"] == "requirement"
            else evidence_quality_reasons_for_policy(
                case["text"],
                SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
            )
        )
        expected = case["expected_reason"]
        assert (expected is None and not reasons) or expected in reasons


def test_raw_extraction_cannot_bypass_v8_policy_filter() -> None:
    truncated = (
        "Built and managed a sales stack with validation rules and dashboard man"
    )
    complete = (
        "Built and managed a sales stack with validation rules and dashboards."
    )
    extracted = extract_action_evidence(f"{truncated}\n{complete}")

    assert truncated in extracted
    assert complete in extracted
    assert diverse_quality_evidence(
        extracted,
        policy=SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
    ) == [complete]
