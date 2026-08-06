"""Tests for offline-only evidence distillation contracts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from ml.evidence_distillation import (
    DistillationError,
    LocalCrossEncoderTeacher,
    OllamaEmbeddingTeacher,
    TeacherSpec,
    assert_allowed_input_path,
    build_teacher_error_audit_queue,
    build_teacher_score_rows,
    cosine_scores,
    load_distillation_protocol,
    score_manifest,
    select_balanced_task_sample,
    training_ranking_errors,
    training_score_diagnostics,
    training_score_slices,
    validate_distillation_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "config" / "ml_distillation_v1.json"


def pairs() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for family in ("Business", "Data"):
        for index in range(3):
            task_id = f"{family}-{index}"
            for candidate in range(2):
                rows.append(
                    {
                        "pair_id": f"{task_id}-{candidate}",
                        "task_id": task_id,
                        "role_family": family,
                        "requirement": f"{family} requirement {index}",
                        "evidence": f"{family} evidence {candidate}",
                    }
                )
    return rows


def test_protocol_is_offline_and_holdout_safe() -> None:
    protocol = load_distillation_protocol(str(PROTOCOL_PATH))

    assert protocol["product_integration_allowed"] is False
    assert protocol["dataset_policy"]["allow_v6_sealed_holdout"] is False

    unsafe = deepcopy(protocol)
    unsafe["dataset_policy"]["allow_shadow_diagnostics"] = True
    with pytest.raises(DistillationError, match="unsafe"):
        validate_distillation_protocol(unsafe)


@pytest.mark.parametrize(
    "path",
    [
        "data/ml/annotations/real_holdout_v1/gold.jsonl",
        "data/ml/shadow/review.jsonl",
        "data/ml/processed/successor_v8/training_pairs.jsonl",
    ],
)
def test_teacher_scoring_rejects_restricted_input_paths(path: str) -> None:
    with pytest.raises(DistillationError, match="may not read"):
        assert_allowed_input_path(path, input_role="human_reviewed_training")


def test_balanced_sampling_keeps_complete_tasks_and_families() -> None:
    sample = select_balanced_task_sample(pairs(), max_tasks=4, random_state=7)

    assert len(sample) == 8
    assert {pair["role_family"] for pair in sample} == {"Business", "Data"}
    assert all(
        sum(pair["task_id"] == task_id for pair in sample) == 2
        for task_id in {pair["task_id"] for pair in sample}
    )


def test_balanced_sampling_can_require_both_binary_classes() -> None:
    source = pairs()
    for row in source:
        row["binary_label"] = int(row["task_id"] == "Data-2")
        row["support_label"] = (
            "Direct" if row["binary_label"] == 1 else "No Support"
        )

    sample = select_balanced_task_sample(
        source,
        max_tasks=2,
        random_state=7,
        require_binary_classes=True,
    )

    assert {int(pair["binary_label"]) for pair in sample} == {0, 1}
    assert len({pair["task_id"] for pair in sample}) == 2
    assert all(
        sum(pair["task_id"] == task_id for pair in sample) == 2
        for task_id in {pair["task_id"] for pair in sample}
    )


def test_binary_class_sampling_rejects_unlabeled_pairs() -> None:
    with pytest.raises(DistillationError, match="binary_label"):
        select_balanced_task_sample(
            pairs(),
            max_tasks=2,
            random_state=7,
            require_binary_classes=True,
        )


def test_cosine_rows_and_manifest_do_not_persist_source_text() -> None:
    source = pairs()[:2]
    scores = cosine_scores([[1.0, 0.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]])
    teacher = TeacherSpec(
        teacher_id="test",
        model="local",
        source="ollama",
        kind="embedding_baseline",
    )
    rows = build_teacher_score_rows(source, scores, teacher=teacher)
    manifest = score_manifest(
        source,
        rows,
        teacher=teacher,
        input_role="human_reviewed_training",
    )

    assert np.allclose(scores, [1.0, np.sqrt(0.5)])
    assert all("requirement" not in row and "evidence" not in row for row in rows)
    assert manifest["source_text_persisted"] is False
    assert manifest["model_selection_allowed"] is False
    assert manifest["product_integration_allowed"] is False


def test_ollama_client_requires_aligned_embedding_response() -> None:
    teacher = OllamaEmbeddingTeacher(
        "qwen3-embedding:0.6b",
        request=lambda _: b'{"embeddings": [[1.0, 0.0]]}',
    )

    with pytest.raises(DistillationError, match="misaligned"):
        teacher.embed(["one", "two"])


def test_local_reranker_is_local_and_requires_aligned_scores(tmp_path: Path) -> None:
    teacher = LocalCrossEncoderTeacher(
        str(tmp_path),
        instruction="Judge evidence support.",
        predict=lambda inputs: [0.8] * len(inputs),
    )

    assert teacher.score(pairs()[:2]) == [0.8, 0.8]

    misaligned = LocalCrossEncoderTeacher(
        str(tmp_path),
        instruction="Judge evidence support.",
        predict=lambda _: [0.5],
    )
    with pytest.raises(DistillationError, match="misaligned"):
        misaligned.score(pairs()[:2])

    efficiency = teacher.efficiency_manifest(pairs=2)
    assert efficiency["device"] == "injected"
    assert efficiency["pairs_per_second"] > 0
    assert "peak_process_rss_bytes" in efficiency


def test_training_diagnostics_report_ranking_without_threshold_selection() -> None:
    source = pairs()[:4]
    source[0]["binary_label"] = 1
    source[0]["support_label"] = "Direct"
    source[1]["binary_label"] = 0
    source[1]["support_label"] = "No Support"
    source[2]["binary_label"] = 1
    source[2]["support_label"] = "Partial"
    source[3]["binary_label"] = 0
    source[3]["support_label"] = "No Support"

    diagnostics = training_score_diagnostics(source, [0.9, 0.2, 0.8, 0.3])

    assert diagnostics["binary_auc"] == 1.0
    assert diagnostics["supported_task_top1"] == 1.0
    assert diagnostics["threshold_selected"] is False
    assert diagnostics["model_selection_allowed"] is False


def test_training_slices_and_errors_use_identifiers_only() -> None:
    source = pairs()[:4]
    for index, row in enumerate(source):
        row["binary_label"] = int(index % 2 == 0)
        row["support_label"] = "Direct" if index % 2 == 0 else "No Support"
    scores = [0.2, 0.9, 0.8, 0.1]

    slices = training_score_slices(source, scores, slice_field="role_family")
    errors = training_ranking_errors(source, scores)

    assert set(slices) == {"Business"}
    assert slices["Business"]["binary_auc"] == 0.5
    assert len(errors) == 1
    assert set(errors[0]) == {
        "task_id",
        "role_family",
        "top_pair_id",
        "top_score",
        "best_supported_pair_id",
        "best_supported_score",
    }
    assert "requirement" not in errors[0]
    assert "evidence" not in errors[0]


def test_teacher_error_audit_is_diagnostic_only() -> None:
    source = pairs()[:4]
    for index, row in enumerate(source):
        row["binary_label"] = int(index % 2 == 0)
        row["support_label"] = "Direct" if index % 2 == 0 else "No Support"
    queue = build_teacher_error_audit_queue(
        source,
        [0.9, 0.2, 0.8, 0.1],
        [0.2, 0.9, 0.8, 0.1],
        max_business_inversions=2,
    )

    assert len(queue) == 1
    assert queue[0]["audit_reasons"] == [
        "business_pair_inversion",
        "qwen_top1_error",
    ]
    assert queue[0]["diagnostic_only"] is True
    assert queue[0]["human_gold_changes_allowed"] is False
    assert queue[0]["holdout_accessed"] is False
