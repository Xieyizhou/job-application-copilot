"""Evaluate a task-level rejector with resume-grouped cross-validation."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_BASE_DIR = (
    PROJECT_ROOT / "data" / "ml" / "processed"
    / "reviewed_evidence_training_v4_batch4"
)
DEFAULT_OPERATIONAL_DIR = (
    PROJECT_ROOT / "data" / "ml" / "processed"
    / "operational_development_v2"
)
DEFAULT_QUEUE = (
    PROJECT_ROOT / "data" / "ml" / "annotations"
    / "operational_development_v2" / "queue.jsonl"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "ml" / "generated"
    / "operational_task_rejector_v2.json"
)
DEFAULT_ARTIFACT = (
    PROJECT_ROOT / "data" / "ml" / "models"
    / "evidence_task_rejector_candidate_v2.joblib"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.evidence_artifact import training_corpus_fingerprint  # noqa: E402
from ml.evidence_decision import paired_stratified_bootstrap_delta  # noqa: E402
from ml.evidence_task_evaluation import (  # noqa: E402
    aggregate_fold_results,
    base_oof_candidate_probabilities,
    crossfit_rejector,
    fit_candidate,
    grouped_task_assignments,
    pair_rows,
    predict_tasks,
    summarize_failure_slices,
    task_feature_rows,
)
from ml.evidence_task_rejection import (  # noqa: E402
    evaluate_task_policy,
    select_task_rejection_threshold,
)
from ml.evidence_two_stage import reviewed_multiclass_rows  # noqa: E402
from ml.real_holdout_evaluation import assert_holdout_isolated  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument(
        "--operational-dir",
        type=Path,
        default=DEFAULT_OPERATIONAL_DIR,
    )
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--queue-path", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--random-state", type=int, default=20260730)
    return parser.parse_args()


def _inner_oof_candidate_probabilities(
    tasks: list[dict[str, Any]],
    base_rows: list[tuple[str, str, str]],
    *,
    random_state: int,
    n_splits: int = 3,
) -> dict[str, list[list[float]]]:
    """Build candidate features without exposing the outer test fold."""
    assignments = grouped_task_assignments(
        tasks,
        random_state=random_state,
        n_splits=n_splits,
    )
    probabilities: dict[str, list[list[float]]] = {}
    for fold in range(n_splits):
        inner_train = [
            task
            for task in tasks
            if assignments[str(task["task_id"])] != fold
        ]
        inner_test = [
            task
            for task in tasks
            if assignments[str(task["task_id"])] == fold
        ]
        model = fit_candidate(
            [*base_rows, *reviewed_multiclass_rows(inner_train)],
            random_state=random_state + fold,
        )
        probabilities.update(predict_tasks(model, inner_test))
    expected = {str(task["task_id"]) for task in tasks}
    if set(probabilities) != expected:
        raise ValueError("Inner candidate cross-fit did not cover every task.")
    return probabilities


def main() -> None:
    args = parse_args()
    if args.artifact_path.exists():
        raise SystemExit("Task-rejector artifact exists; refusing overwrite.")
    base_tasks = load_jsonl(args.base_dir / "annotated_tasks.jsonl")
    base_pairs = load_jsonl(args.base_dir / "training_pairs.jsonl")
    operational_tasks = load_jsonl(
        args.operational_dir / "annotated_tasks.jsonl"
    )
    operational_pairs = load_jsonl(
        args.operational_dir / "training_pairs.jsonl"
    )
    assert_holdout_isolated(operational_tasks, base_tasks, base_pairs)

    base_oof = base_oof_candidate_probabilities(
        base_tasks,
        base_pairs,
        random_state=args.random_state,
    )
    base_features, base_labels, base_groups = task_feature_rows(
        base_tasks,
        base_oof,
    )
    base_rejector, base_rejector_oof = crossfit_rejector(
        base_features,
        base_labels,
        base_groups,
        random_state=args.random_state,
    )
    base_threshold = select_task_rejection_threshold(
        base_labels,
        base_rejector_oof.tolist(),
    )

    assignments = grouped_task_assignments(
        operational_tasks,
        random_state=args.random_state,
    )
    base_rows = pair_rows(base_pairs)
    reference_results: list[dict[str, Any]] = []
    rejector_results: list[dict[str, Any]] = []
    operational_oof: dict[str, list[list[float]]] = {}
    fold_payloads: list[
        tuple[
            int,
            list[dict[str, Any]],
            list[dict[str, Any]],
            dict[str, list[list[float]]],
            dict[str, Any],
        ]
    ] = []
    fold_reports: list[dict[str, Any]] = []
    for fold in range(4):
        train_tasks = [
            task
            for task in operational_tasks
            if assignments[str(task["task_id"])] != fold
        ]
        test_tasks = [
            task
            for task in operational_tasks
            if assignments[str(task["task_id"])] == fold
        ]
        model = fit_candidate(
            [*base_rows, *reviewed_multiclass_rows(train_tasks)],
            random_state=args.random_state + fold,
        )
        test_probabilities = predict_tasks(model, test_tasks)
        operational_oof.update(test_probabilities)
        reference = evaluate_task_policy(
            test_tasks,
            test_probabilities,
        )
        fold_payloads.append(
            (fold, train_tasks, test_tasks, test_probabilities, reference)
        )
        reference_results.append(reference)

    for fold, train_tasks, test_tasks, test_probabilities, reference in (
        fold_payloads
    ):
        train_probabilities = _inner_oof_candidate_probabilities(
            train_tasks,
            base_rows,
            random_state=args.random_state + 10_000 + fold * 100,
        )
        train_features, train_labels, train_groups = task_feature_rows(
            train_tasks,
            train_probabilities,
        )
        fold_rejector, fold_rejector_oof = crossfit_rejector(
            [*base_features, *train_features],
            [*base_labels, *train_labels],
            [
                *[f"base:{group}" for group in base_groups],
                *[f"operational:{group}" for group in train_groups],
            ],
            random_state=args.random_state + 100 + fold,
        )
        fold_threshold = select_task_rejection_threshold(
            [*base_labels, *train_labels],
            fold_rejector_oof.tolist(),
        )
        test_features, _, _ = task_feature_rows(
            test_tasks,
            test_probabilities,
        )
        task_probabilities = fold_rejector.predict_proba(test_features)
        acceptance = {
            str(task["task_id"]): float(probability)
            for task, probability in zip(
                test_tasks,
                task_probabilities,
                strict=True,
            )
        }
        rejected = evaluate_task_policy(
            test_tasks,
            test_probabilities,
            acceptance_probabilities=acceptance,
            acceptance_threshold=fold_threshold,
        )
        rejector_results.append(rejected)
        fold_reports.append(
            {
                "fold": fold,
                "train_tasks": len(train_tasks),
                "test_tasks": len(test_tasks),
                "test_resume_groups": len(
                    {str(task["source_resume_hash"]) for task in test_tasks}
                ),
                "task_threshold": fold_threshold,
                "reference": reference["retrieval"],
                "task_rejector": rejected["retrieval"],
                "task_rejector_failure_counts": dict(
                    Counter(
                        failure["failure"]
                        for failure in rejected["failures"]
                    )
                ),
            }
        )

    reference_summary = aggregate_fold_results(reference_results)
    rejector_summary = aggregate_fold_results(rejector_results)
    reference_outcomes = {
        task_id: passed
        for result in reference_results
        for task_id, passed in result["outcomes"].items()
    }
    rejector_outcomes = {
        task_id: passed
        for result in rejector_results
        for task_id, passed in result["outcomes"].items()
    }
    task_ids = [str(task["task_id"]) for task in operational_tasks]
    bootstrap = paired_stratified_bootstrap_delta(
        [
            str(task["support_label"]) != "No Support"
            for task in operational_tasks
        ],
        [reference_outcomes[task_id] for task_id in task_ids],
        [rejector_outcomes[task_id] for task_id in task_ids],
        random_state=args.random_state,
    )
    point_estimate_wins = (
        float(rejector_summary["task_balanced_accuracy"])
        > float(reference_summary["task_balanced_accuracy"])
        and float(rejector_summary["recall_at_1"])
        >= float(reference_summary["recall_at_1"])
        and float(rejector_summary["supported_task_success_rate"])
        >= float(reference_summary["supported_task_success_rate"])
        and float(rejector_summary["no_support_rejection_rate"])
        >= float(reference_summary["no_support_rejection_rate"])
    )
    stable = point_estimate_wins and float(bootstrap["lower_90"]) > 0.0
    artifact_saved = False
    if stable:
        operational_features, operational_labels, operational_groups = (
            task_feature_rows(operational_tasks, operational_oof)
        )
        final_rejector, final_oof = crossfit_rejector(
            [*base_features, *operational_features],
            [*base_labels, *operational_labels],
            [
                *[f"base:{group}" for group in base_groups],
                *[
                    f"operational:{group}"
                    for group in operational_groups
                ],
            ],
            random_state=args.random_state,
        )
        final_threshold = select_task_rejection_threshold(
            [*base_labels, *operational_labels],
            final_oof.tolist(),
        )
        final_candidate = fit_candidate(
            [*base_rows, *pair_rows(operational_pairs)],
            random_state=args.random_state,
        )
        artifact = {
            "schema_version": 1,
            "model_type": "candidate_multiclass_plus_task_rejector",
            "candidate_model": final_candidate,
            "task_rejector": final_rejector,
            "task_threshold": final_threshold,
            "metadata": {
                "status": "candidate_for_new_source_isolated_reserve",
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "base_training_corpus_fingerprint": (
                    training_corpus_fingerprint(base_tasks, base_pairs)
                ),
                "operational_gold_sha256": json.loads(
                    (args.operational_dir / "manifest.json").read_text()
                )["gold_sha256"],
                "task_feature_manifest": (
                    final_rejector.feature_manifest()
                ),
                "product_integration_allowed": False,
            },
        }
        args.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, args.artifact_path)
        artifact_saved = True

    queue_metadata = {
        str(task["task_id"]): task for task in load_jsonl(args.queue_path)
    }
    report = {
        "schema_version": 1,
        "dataset": "operational_development_v2",
        "protocol": "nested_resume_grouped_candidate_and_task_crossfit",
        "inner_candidate_splits": 3,
        "outer_test_excluded_from_inner_candidate_fit": True,
        "base_task_rejector_training": "candidate_model_oof_features_only",
        "base_tasks": len(base_tasks),
        "base_pairs": len(base_pairs),
        "operational_tasks": len(operational_tasks),
        "operational_pairs": len(operational_pairs),
        "base_task_threshold": base_threshold,
        "base_task_feature_manifest": base_rejector.feature_manifest(),
        "candidate_acceptance_reference": reference_summary,
        "task_rejector": rejector_summary,
        "error_slices": {
            "candidate_acceptance_reference": summarize_failure_slices(
                operational_tasks,
                reference_results,
                queue_metadata,
            ),
            "task_rejector": summarize_failure_slices(
                operational_tasks,
                rejector_results,
                queue_metadata,
            ),
        },
        "folds": fold_reports,
        "selection": {
            "point_estimate_wins": point_estimate_wins,
            "paired_stratified_bootstrap": bootstrap,
            "stable_improvement": stable,
            "status": (
                "candidate_for_new_source_isolated_reserve"
                if stable
                else "blocked_operational_development_not_won"
            ),
            "artifact_saved": artifact_saved,
            "artifact_path": (
                str(args.artifact_path) if artifact_saved else None
            ),
        },
        "usage_boundary": (
            "Development-only task rejection. No product inference or final "
            "holdout is read."
        ),
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, result in (
        ("candidate_acceptance", reference_summary),
        ("task_rejector", rejector_summary),
    ):
        print(
            f"{name}: balanced={result['task_balanced_accuracy']:.3f}, "
            f"Recall@1={result['recall_at_1']:.3f}, "
            f"No-support={result['no_support_rejection_rate']:.3f}"
        )
    print(
        "Paired bootstrap 90% interval: "
        f"[{bootstrap['lower_90']:.3f}, {bootstrap['upper_90']:.3f}]"
    )
    selection = report["selection"]
    if not isinstance(selection, dict):
        raise TypeError("Selection report is invalid.")
    print(f"Selection: {selection['status']}")
    print(f"Report: {args.report_path}")


if __name__ == "__main__":
    main()
