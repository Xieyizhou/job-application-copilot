"""Cross-fit semantic evidence ranking with a fixed lexical acceptance gate."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEVELOPMENT_DIR = (
    PROJECT_ROOT / "data" / "ml" / "annotations" / "real_development_v3"
)
TRAINING_DIR = (
    PROJECT_ROOT / "data" / "ml" / "processed" / "reviewed_evidence_training_v3"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "ml"
    / "generated"
    / "semantic_blend_development_v3_crossfit.json"
)
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "models"
    / "evidence_semantic_blend_candidate_v2.joblib"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.evidence_artifact import training_corpus_fingerprint  # noqa: E402
from ml.evidence_decision import paired_stratified_bootstrap_delta  # noqa: E402
from ml.evidence_models import WordTfidfCosineScorer  # noqa: E402
from ml.evidence_semantic_blend import (  # noqa: E402
    SEMANTIC_BLEND_CHARACTER_WEIGHTS,
    SemanticConceptBlendScorer,
    blend_semantic_scores,
    evaluate_ranker_with_fixed_gate,
)
from ml.evidence_two_stage import reviewed_multiclass_rows  # noqa: E402
from ml.real_holdout_evaluation import (  # noqa: E402
    assert_holdout_isolated,
    evaluate_scored_holdout,
    score_holdout_tasks,
    select_validation_threshold,
)


def _load_frozen_development() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold_path = DEVELOPMENT_DIR / "real_development_gold_v3.jsonl"
    manifest = json.loads(
        (DEVELOPMENT_DIR / "real_development_frozen_manifest_v3.json").read_text(
            encoding="utf-8"
        )
    )
    checksum = hashlib.sha256(gold_path.read_bytes()).hexdigest()
    if checksum != manifest["gold_sha256"]:
        raise SystemExit("Frozen development checksum mismatch; evaluation blocked.")
    return load_jsonl(gold_path), manifest


def _fold_assignments(tasks: list[dict[str, Any]]) -> dict[str, int]:
    ordered = sorted(tasks, key=lambda task: str(task["task_id"]))
    labels = [str(task["support_label"]) for task in ordered]
    assignments: dict[str, int] = {}
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    for fold, (_, test_indices) in enumerate(splitter.split(ordered, labels)):
        for index in test_indices:
            assignments[str(ordered[int(index)]["task_id"])] = fold
    return assignments


def _assert_fold_isolation(
    train_tasks: list[dict[str, Any]],
    test_tasks: list[dict[str, Any]],
) -> None:
    for field in ("source_job_hash", "source_resume_hash"):
        train_hashes = {str(task[field]) for task in train_tasks}
        test_hashes = {str(task[field]) for task in test_tasks}
        if train_hashes & test_hashes:
            raise SystemExit(f"Cross-fit {field} leakage detected.")


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


def _score_components(
    scorer: SemanticConceptBlendScorer,
    tasks: list[dict[str, Any]],
) -> dict[str, tuple[list[float], list[float]]]:
    output: dict[str, tuple[list[float], list[float]]] = {}
    for task in tasks:
        candidates = list(task["candidates"])
        requirements = [str(task["requirement"])] * len(candidates)
        evidence = [str(candidate["evidence"]) for candidate in candidates]
        character, concept = scorer.score_components(requirements, evidence)
        output[str(task["task_id"])] = (character.tolist(), concept.tolist())
    return output


def _blend_cached(
    components: dict[str, tuple[list[float], list[float]]],
    *,
    character_weight: float,
) -> dict[str, list[float]]:
    return {
        task_id: blend_semantic_scores(
            character,
            concept,
            character_weight=character_weight,
        ).tolist()
        for task_id, (character, concept) in components.items()
    }


def _select_parameters(
    tasks: list[dict[str, Any]],
    components: dict[str, tuple[list[float], list[float]]],
    gate_scores: dict[str, list[float]],
    gate_threshold: float,
    baseline: dict[str, Any],
) -> tuple[float, float, dict[str, Any]]:
    evaluated: list[tuple[float, float, dict[str, Any]]] = []
    for weight in SEMANTIC_BLEND_CHARACTER_WEIGHTS:
        ranking_scores = _blend_cached(components, character_weight=weight)
        top_scores = [
            max(ranking_scores[str(task["task_id"])]) for task in tasks
        ]
        floors = sorted(
            {
                0.0,
                *(
                    float(np.nextafter(score, float("inf")))
                    for score in top_scores
                ),
            }
        )
        evaluated.extend(
            (
                weight,
                floor,
                evaluate_ranker_with_fixed_gate(
                    tasks,
                    ranking_scores,
                    gate_scores,
                    gate_threshold=gate_threshold,
                    ranking_floor=floor,
                ),
            )
            for floor in floors
        )
    reference = baseline["retrieval"]

    def key(item: tuple[float, float, dict[str, Any]]) -> tuple[float, ...]:
        weight, floor, result = item
        retrieval = result["retrieval"]
        preserves = (
            float(retrieval["supported_task_success_rate"])
            >= float(reference["supported_task_success_rate"])
            and float(retrieval["recall_at_1"]) >= float(reference["recall_at_1"])
        )
        return (
            float(preserves),
            float(retrieval["supported_task_success_rate"]),
            float(retrieval["recall_at_1"]),
            float(retrieval["task_balanced_accuracy"]),
            -abs(weight - 0.25),
            -floor,
        )

    return max(evaluated, key=key)


def _aggregate(
    fold_results: Sequence[dict[str, Any]],
) -> dict[str, float | int]:
    support_tasks = sum(
        int(result["retrieval"]["support_tasks"]) for result in fold_results
    )
    no_support_tasks = sum(
        int(result["retrieval"]["no_support_tasks"]) for result in fold_results
    )

    def weighted(metric: str, denominator: str) -> float:
        total = sum(
            int(result["retrieval"][denominator]) for result in fold_results
        )
        return (
            sum(
                float(result["retrieval"][metric])
                * int(result["retrieval"][denominator])
                for result in fold_results
            )
            / total
        )

    supported = weighted("supported_task_success_rate", "support_tasks")
    rejected = weighted("no_support_rejection_rate", "no_support_tasks")
    total = support_tasks + no_support_tasks
    accuracy = sum(
        float(result["retrieval"]["task_decision_accuracy"])
        * (
            int(result["retrieval"]["support_tasks"])
            + int(result["retrieval"]["no_support_tasks"])
        )
        for result in fold_results
    ) / total
    return {
        "support_tasks": support_tasks,
        "no_support_tasks": no_support_tasks,
        "recall_at_1": weighted("recall_at_1", "support_tasks"),
        "recall_at_3": weighted("recall_at_3", "support_tasks"),
        "mean_reciprocal_rank": weighted(
            "mean_reciprocal_rank", "support_tasks"
        ),
        "supported_task_success_rate": supported,
        "no_support_rejection_rate": rejected,
        "task_decision_accuracy": accuracy,
        "task_balanced_accuracy": (supported + rejected) / 2,
    }


def _failure_ids(results: Sequence[dict[str, Any]]) -> set[str]:
    return {
        str(failure["task_id"])
        for result in results
        for failure in result["failures"]
    }


def main() -> None:
    tasks, manifest = _load_frozen_development()
    base_tasks = load_jsonl(TRAINING_DIR / "annotated_tasks.jsonl")
    base_pairs = load_jsonl(TRAINING_DIR / "training_pairs.jsonl")
    assert_holdout_isolated(tasks, base_tasks, base_pairs)
    assignments = _fold_assignments(tasks)
    baseline_results: list[dict[str, Any]] = []
    candidate_results: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    for fold in range(4):
        train_tasks = [
            task
            for task in tasks
            if assignments[str(task["task_id"])] != fold
        ]
        test_tasks = [
            task
            for task in tasks
            if assignments[str(task["task_id"])] == fold
        ]
        _assert_fold_isolation(train_tasks, test_tasks)
        requirements, evidence = _training_text(base_pairs, train_tasks)
        gate = WordTfidfCosineScorer().fit(requirements, evidence)
        semantic = SemanticConceptBlendScorer().fit(requirements, evidence)
        train_gate_scores = score_holdout_tasks(train_tasks, score=gate.score)
        gate_threshold, baseline_train = select_validation_threshold(
            train_tasks, train_gate_scores
        )
        train_components = _score_components(semantic, train_tasks)
        character_weight, ranking_floor, _ = _select_parameters(
            train_tasks,
            train_components,
            train_gate_scores,
            gate_threshold,
            baseline_train,
        )
        test_gate_scores = score_holdout_tasks(test_tasks, score=gate.score)
        baseline_test = evaluate_scored_holdout(
            test_tasks,
            test_gate_scores,
            threshold=gate_threshold,
        )
        candidate_test = evaluate_ranker_with_fixed_gate(
            test_tasks,
            _blend_cached(
                _score_components(semantic, test_tasks),
                character_weight=character_weight,
            ),
            test_gate_scores,
            gate_threshold=gate_threshold,
            ranking_floor=ranking_floor,
        )
        baseline_results.append(baseline_test)
        candidate_results.append(candidate_test)
        fold_reports.append(
            {
                "fold": fold,
                "train_tasks": len(train_tasks),
                "test_tasks": len(test_tasks),
                "gate_threshold": gate_threshold,
                "character_weight": character_weight,
                "ranking_floor": ranking_floor,
                "baseline": baseline_test,
                "semantic_blend": candidate_test,
            }
        )

    baseline = _aggregate(baseline_results)
    candidate = _aggregate(candidate_results)
    baseline_failures = _failure_ids(baseline_results)
    candidate_failures = _failure_ids(candidate_results)
    bootstrap = paired_stratified_bootstrap_delta(
        [str(task["support_label"]) != "No Support" for task in tasks],
        [str(task["task_id"]) not in baseline_failures for task in tasks],
        [str(task["task_id"]) not in candidate_failures for task in tasks],
    )
    point_estimate_wins = (
        float(candidate["task_balanced_accuracy"])
        > float(baseline["task_balanced_accuracy"])
        and float(candidate["task_decision_accuracy"])
        > float(baseline["task_decision_accuracy"])
        and float(candidate["recall_at_1"]) > float(baseline["recall_at_1"])
        and float(candidate["supported_task_success_rate"])
        >= float(baseline["supported_task_success_rate"])
        and float(candidate["no_support_rejection_rate"])
        >= float(baseline["no_support_rejection_rate"])
    )
    improved = point_estimate_wins and float(bootstrap["lower_90"]) > 0
    artifact_saved = False
    selected_weight: float | None = None
    selected_floor: float | None = None
    final_gate_threshold: float | None = None
    if improved:
        requirements, evidence = _training_text(base_pairs, tasks)
        gate = WordTfidfCosineScorer().fit(requirements, evidence)
        selection_semantic = SemanticConceptBlendScorer().fit(
            requirements, evidence
        )
        gate_scores = score_holdout_tasks(tasks, score=gate.score)
        final_gate_threshold, baseline_full = select_validation_threshold(
            tasks, gate_scores
        )
        selected_weight, selected_floor, _ = _select_parameters(
            tasks,
            _score_components(selection_semantic, tasks),
            gate_scores,
            final_gate_threshold,
            baseline_full,
        )
        ranker = SemanticConceptBlendScorer(
            character_weight=selected_weight
        ).fit(requirements, evidence)
        artifact = {
            "schema_version": 1,
            "model_type": "semantic_concept_blend_with_fixed_tfidf_gate",
            "ranker": ranker,
            "gate": gate,
            "gate_threshold": final_gate_threshold,
            "ranking_floor": selected_floor,
            "metadata": {
                "status": "candidate_for_fresh_source_isolated_reserve_v3",
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "base_training_corpus_fingerprint": training_corpus_fingerprint(
                    base_tasks, base_pairs
                ),
                "development_gold_sha256": manifest["gold_sha256"],
                "development_dataset": "real_development_v3",
                "development_tasks": len(tasks),
                "weight_selection": "nested_four_fold_development_only",
                "feature_manifest": ranker.feature_manifest(),
                "fixed_gate_manifest": gate.feature_manifest(),
                "product_integration_allowed": False,
            },
        }
        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, ARTIFACT_PATH)
        artifact_saved = True
    report = {
        "schema_version": 1,
        "dataset": "real_development_v3",
        "protocol": "four_fold_stratified_crossfit",
        "tasks": len(tasks),
        "base_training_tasks": len(base_tasks),
        "base_training_pairs": len(base_pairs),
        "fold_source_overlap": 0,
        "reserve_v2_read_or_used": False,
        "fixed_tfidf_baseline": baseline,
        "semantic_blend_fixed_gate": candidate,
        "folds": fold_reports,
        "selection": {
            "point_estimate_wins": point_estimate_wins,
            "paired_stratified_bootstrap": bootstrap,
            "status": (
                "candidate_for_new_source_isolated_reserve_v3"
                if improved
                else "blocked_crossfit_baseline_not_beaten"
            ),
            "artifact_saved": artifact_saved,
            "artifact_path": str(ARTIFACT_PATH) if artifact_saved else None,
            "selected_character_weight": selected_weight,
            "selected_ranking_floor": selected_floor,
            "gate_threshold": final_gate_threshold,
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, metrics in (("fixed_tfidf", baseline), ("semantic_blend", candidate)):
        print(
            f"{name}: balanced={metrics['task_balanced_accuracy']:.3f}, "
            f"accuracy={metrics['task_decision_accuracy']:.3f}, "
            f"Recall@1={metrics['recall_at_1']:.3f}, "
            f"supported={metrics['supported_task_success_rate']:.3f}, "
            f"No-support={metrics['no_support_rejection_rate']:.3f}"
        )
    print(f"Paired bootstrap lower 90%: {bootstrap['lower_90']:.4f}")
    print(
        "Selection: "
        + (
            "candidate_for_new_source_isolated_reserve_v3"
            if improved
            else "blocked_crossfit_baseline_not_beaten"
        )
    )
    print(f"Artifact saved: {artifact_saved}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
