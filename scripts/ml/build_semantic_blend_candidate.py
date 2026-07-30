"""Build a development-tuned semantic candidate for a new untouched reserve."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
TRAINING_DIR = (
    PROJECT_ROOT / "data" / "ml" / "processed" / "reviewed_evidence_training_v3"
)
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "models"
    / "evidence_semantic_blend_candidate_v2.joblib"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "ml"
    / "generated"
    / "semantic_blend_candidate_v2_build.json"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.evidence_artifact import training_corpus_fingerprint  # noqa: E402
from ml.evidence_models import WordTfidfCosineScorer  # noqa: E402
from ml.evidence_semantic_blend import (  # noqa: E402
    SemanticConceptBlendScorer,
    evaluate_ranker_with_fixed_gate,
)
from ml.evidence_two_stage import reviewed_multiclass_rows  # noqa: E402
from ml.real_holdout_evaluation import (  # noqa: E402
    assert_holdout_isolated,
    evaluate_scored_holdout,
    score_holdout_tasks,
    select_validation_threshold,
)


CHARACTER_WEIGHT = 0.25
RANKING_FLOOR = 0.03


def _load_development(
    version: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = (
        PROJECT_ROOT
        / "data"
        / "ml"
        / "annotations"
        / f"real_development_{version}"
    )
    gold_path = base / f"real_development_gold_{version}.jsonl"
    manifest = json.loads(
        (base / f"real_development_frozen_manifest_{version}.json").read_text(
            encoding="utf-8"
        )
    )
    if hashlib.sha256(gold_path.read_bytes()).hexdigest() != manifest["gold_sha256"]:
        raise SystemExit(f"Frozen development {version} checksum mismatch.")
    return load_jsonl(gold_path), manifest


def _training_text(
    base_pairs: list[dict[str, Any]],
    development_tasks: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    reviewed = reviewed_multiclass_rows(development_tasks)
    return (
        [
            *[str(pair["requirement"]) for pair in base_pairs],
            *[row[0] for row in reviewed],
        ],
        [
            *[str(pair["evidence"]) for pair in base_pairs],
            *[row[1] for row in reviewed],
        ],
    )


def _evaluate(
    tasks: list[dict[str, Any]],
    gate: WordTfidfCosineScorer,
    ranker: SemanticConceptBlendScorer,
    *,
    threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gate_scores = score_holdout_tasks(tasks, score=gate.score)
    baseline = evaluate_scored_holdout(
        tasks,
        gate_scores,
        threshold=threshold,
    )
    candidate = evaluate_ranker_with_fixed_gate(
        tasks,
        score_holdout_tasks(tasks, score=ranker.score),
        gate_scores,
        gate_threshold=threshold,
        ranking_floor=RANKING_FLOOR,
    )
    return baseline, candidate


def _does_not_regress(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    reference = baseline["retrieval"]
    result = candidate["retrieval"]
    return all(
        float(result[metric]) >= float(reference[metric])
        for metric in (
            "task_balanced_accuracy",
            "task_decision_accuracy",
            "recall_at_1",
            "supported_task_success_rate",
            "no_support_rejection_rate",
        )
    )


def _source_isolated(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> bool:
    return all(
        not (
            {str(task.get(field, "")) for task in left}
            & {str(task.get(field, "")) for task in right}
        )
        for field in ("source_job_hash", "source_resume_hash")
    )


def main() -> None:
    if ARTIFACT_PATH.exists():
        raise SystemExit("Candidate v2 artifact exists; refusing overwrite.")
    base_tasks = load_jsonl(TRAINING_DIR / "annotated_tasks.jsonl")
    base_pairs = load_jsonl(TRAINING_DIR / "training_pairs.jsonl")
    development_v3, manifest_v3 = _load_development("v3")
    confirmation_v2, manifest_v2 = _load_development("v2")
    assert_holdout_isolated(development_v3, base_tasks, base_pairs)
    assert_holdout_isolated(confirmation_v2, base_tasks, base_pairs)
    if not _source_isolated(development_v3, confirmation_v2):
        raise SystemExit("Development v2/v3 source overlap detected.")

    requirements, evidence = _training_text(base_pairs, development_v3)
    calibration_gate = WordTfidfCosineScorer().fit(requirements, evidence)
    calibration_ranker = SemanticConceptBlendScorer(
        character_weight=CHARACTER_WEIGHT
    ).fit(requirements, evidence)
    v3_gate_scores = score_holdout_tasks(
        development_v3, score=calibration_gate.score
    )
    threshold, baseline_v3 = select_validation_threshold(
        development_v3, v3_gate_scores
    )
    candidate_v3 = evaluate_ranker_with_fixed_gate(
        development_v3,
        score_holdout_tasks(development_v3, score=calibration_ranker.score),
        v3_gate_scores,
        gate_threshold=threshold,
        ranking_floor=RANKING_FLOOR,
    )
    baseline_v2, candidate_v2 = _evaluate(
        confirmation_v2,
        calibration_gate,
        calibration_ranker,
        threshold=threshold,
    )
    calibration_improves = (
        float(candidate_v3["retrieval"]["task_balanced_accuracy"])
        > float(baseline_v3["retrieval"]["task_balanced_accuracy"])
        and float(candidate_v3["retrieval"]["supported_task_success_rate"])
        > float(baseline_v3["retrieval"]["supported_task_success_rate"])
        and float(candidate_v3["retrieval"]["no_support_rejection_rate"])
        >= float(baseline_v3["retrieval"]["no_support_rejection_rate"])
    )
    confirmation_passes = _does_not_regress(baseline_v2, candidate_v2)
    if not calibration_improves or not confirmation_passes:
        raise SystemExit("Candidate v2 failed development calibration or confirmation.")

    all_development = [*confirmation_v2, *development_v3]
    requirements, evidence = _training_text(base_pairs, all_development)
    final_gate = WordTfidfCosineScorer().fit(requirements, evidence)
    final_ranker = SemanticConceptBlendScorer(
        character_weight=CHARACTER_WEIGHT
    ).fit(requirements, evidence)
    combined_gate_scores = score_holdout_tasks(
        all_development, score=final_gate.score
    )
    final_threshold, _ = select_validation_threshold(
        all_development, combined_gate_scores
    )
    artifact = {
        "schema_version": 1,
        "model_type": "semantic_concept_blend_with_fixed_tfidf_gate",
        "ranker": final_ranker,
        "gate": final_gate,
        "gate_threshold": final_threshold,
        "ranking_floor": RANKING_FLOOR,
        "metadata": {
            "status": "development_tuned_candidate_for_fresh_reserve_v3",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "base_training_corpus_fingerprint": training_corpus_fingerprint(
                base_tasks, base_pairs
            ),
            "development_gold_sha256": {
                "v2": manifest_v2["gold_sha256"],
                "v3": manifest_v3["gold_sha256"],
            },
            "development_tasks": len(all_development),
            "character_weight": CHARACTER_WEIGHT,
            "ranking_floor": RANKING_FLOOR,
            "confirmation_policy": "v3_calibration_v2_non_regression",
            "feature_manifest": final_ranker.feature_manifest(),
            "fixed_gate_manifest": final_gate.feature_manifest(),
            "product_integration_allowed": False,
        },
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)
    report = {
        "schema_version": 1,
        "candidate": ARTIFACT_PATH.name,
        "candidate_sha256": hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest(),
        "character_weight": CHARACTER_WEIGHT,
        "ranking_floor": RANKING_FLOOR,
        "calibration_v3": {
            "baseline": baseline_v3["retrieval"],
            "candidate": candidate_v3["retrieval"],
            "improves": calibration_improves,
        },
        "confirmation_v2": {
            "source_overlap": 0,
            "baseline": baseline_v2["retrieval"],
            "candidate": candidate_v2["retrieval"],
            "does_not_regress": confirmation_passes,
        },
        "final_gate_threshold": final_threshold,
        "reserve_v2_read_or_used": False,
        "product_integration_allowed": False,
        "next_gate": "fresh_source_isolated_reserve_v3",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Candidate v2 SHA-256: {report['candidate_sha256']}")
    print(f"Development v3 improves: {calibration_improves}")
    print(f"Development v2 non-regression: {confirmation_passes}")
    print(f"Artifact: {ARTIFACT_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
