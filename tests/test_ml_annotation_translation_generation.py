"""Tests for locally stored annotation translation sidecars."""

from __future__ import annotations

from ml.annotation_translation_generation import build_translation_sidecar


def test_translation_sidecar_preserves_queue_sources() -> None:
    tasks = [
        {
            "schema_version": 1,
            "task_id": "task-1",
            "requirement": "Build SQL reports.",
            "candidates": [
                {"candidate_id": "a", "evidence": "Automated SQL reports."},
                {"candidate_id": "b", "evidence": "Prepared meeting notes."},
            ],
        }
    ]

    rows = build_translation_sidecar(
        tasks,
        translate=lambda source: f"zh::{source}",
    )

    assert rows[0]["requirement"] == {
        "source": "Build SQL reports.",
        "zh": "zh::Build SQL reports.",
    }
    assert rows[0]["candidates"][0]["candidate_id"] == "a"
