"""Evaluate v3 successors with nested resume-grouped development only."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import joblib
from sklearn.metrics import confusion_matrix, f1_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_BASE_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "processed"
    / "reviewed_evidence_training_v4_batch4"
)
DEFAULT_V2_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "processed"
    / "operational_development_v2"
)
DEFAULT_V3_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "processed"
    / "operational_development_v3"
)
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "ml"
    / "generated"
    / "operational_development_v3_nested.json"
)
DEFAULT_ARTIFACT = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "models"
    / "operational_development_v3_successor.joblib"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.evidence_artifact import training_corpus_fingerprint  # noqa: E402
from ml.evidence_decision import paired_stratified_bootstrap_delta  # noqa: E402
from ml.evidence_models import LsaEmbeddingScorer  # noqa: E402
from ml.evidence_multiclass import SUPPORT_CLASSES  # noqa: E402
from ml.evidence_task_evaluation import (  # noqa: E402
    aggregate_fold_results,
    base_oof_candidate_probabilities,
    crossfit_rejector,
    fit_candidate,
    grouped_task_assignments,
    pair_rows,
    predict_tasks,
    task_feature_rows,
)
from ml.evidence_task_rejection import (  # noqa: E402
    evaluate_task_policy,
    select_task_rejection_threshold,
)
from ml.evidence_two_stage import (  # noqa: E402
    cache_two_stage_predictions,
    evaluate_two_stage,
    reviewed_multiclass_rows,
    select_two_stage_parameters,
)
from ml.real_holdout_evaluation import (  # noqa: E402
    evaluate_scored_holdout,
    score_holdout_tasks,
    select_validation_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--v2-dir", type=Path, default=DEFAULT_V2_DIR)
    parser.add_argument("--v3-dir", type=Path, default=DEFAULT_V3_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--random-state", type=int, default=20260731)
    return parser.parse_args()


def _assert_source_disjoint(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
) -> None:
    for field in ("source_resume_hash", "source_job_hash"):
        left = {str(task[field]) for task in first}
        right = {str(task[field]) for task in second}
        if left & right:
            raise ValueError(f"Operational datasets overlap on {field}.")


def _inner_oof_candidate_probabilities(
    tasks: list[dict[str, Any]],
    base_rows: list[tuple[str, str, str]],
    *,
    random_state: int,
    n_splits: int = 3,
) -> dict[str, list[list[float]]]:
    assignments = grouped_task_assignments(
        tasks,
        random_state=random_state,
        n_splits=n_splits,
    )
    probabilities: dict[str, list[list[float]]] = {}
    for fold in range(n_splits):
        train = [
            task
            for task in tasks
            if assignments[str(task["task_id"])] != fold
        ]
        test = [
            task
            for task in tasks
            if assignments[str(task["task_id"])] == fold
        ]
        model = fit_candidate(
            [*base_rows, *reviewed_multiclass_rows(train)],
            random_state=random_state + fold,
        )
        probabilities.update(predict_tasks(model, test))
    if set(probabilities) != {str(task["task_id"]) for task in tasks}:
        raise ValueError("Inner candidate cross-fit did not cover every task.")
    return probabilities


def _outcomes(
    result: dict[str, Any],
    tasks: list[dict[str, Any]],
) -> dict[str, bool]:
    if "outcomes" in result:
        return {
            str(task_id): bool(value)
            for task_id, value in result["outcomes"].items()
        }
    failures = {
        str(failure["task_id"]) for failure in result["failures"]
    }
    return {
        str(task["task_id"]): str(task["task_id"]) not in failures
        for task in tasks
    }


def _failure_types(result: dict[str, Any]) -> dict[str, str]:
    """Return task-id failure categories without retaining task text."""
    return {
        str(failure["task_id"]): str(failure["failure"])
        for failure in result.get("failures", [])
    }


def _slice_report(
    tasks: list[dict[str, Any]],
    outcomes: dict[str, bool],
    failures: dict[str, str],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "failure_counts": dict(Counter(failures.values()))
    }
    for field in ("role_family", "construction_stratum"):
        buckets: dict[str, dict[str, Any]] = {}
        for task in tasks:
            task_id = str(task["task_id"])
            value = str(task.get(field, "unknown"))
            bucket = buckets.setdefault(
                value,
                {"tasks": 0, "passed": 0, "failure_counts": Counter()},
            )
            bucket["tasks"] += 1
            if outcomes[task_id]:
                bucket["passed"] += 1
            else:
                bucket["failure_counts"][failures[task_id]] += 1
        report[f"by_{field}"] = {
            value: {
                "tasks": bucket["tasks"],
                "accuracy": bucket["passed"] / bucket["tasks"],
                "failure_counts": dict(bucket["failure_counts"]),
            }
            for value, bucket in sorted(buckets.items())
        }
    return report


def _candidate_metrics(
    labels: list[str],
    predictions: list[str],
) -> dict[str, Any]:
    return {
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=list(SUPPORT_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_recall": {
            label: float(value)
            for label, value in zip(
                SUPPORT_CLASSES,
                recall_score(
                    labels,
                    predictions,
                    labels=list(SUPPORT_CLASSES),
                    average=None,
                    zero_division=0,
                ),
                strict=True,
            )
        },
        "confusion_matrix": {
            "labels": list(SUPPORT_CLASSES),
            "matrix": confusion_matrix(
                labels,
                predictions,
                labels=list(SUPPORT_CLASSES),
            ).tolist(),
        },
    }


def _evaluate_dataset(
    name: str,
    tasks: list[dict[str, Any]],
    *,
    base_rows: list[tuple[str, str, str]],
    base_features: list[list[float]],
    base_labels: list[int],
    base_groups: list[str],
    random_state: int,
) -> dict[str, Any]:
    assignments = grouped_task_assignments(
        tasks,
        random_state=random_state,
        n_splits=4,
    )
    lsa_results: list[dict[str, Any]] = []
    two_stage_results: list[dict[str, Any]] = []
    reference_results: list[dict[str, Any]] = []
    rejector_results: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    candidate_labels: list[str] = []
    candidate_predictions: list[str] = []
    for fold in range(4):
        train = [
            task
            for task in tasks
            if assignments[str(task["task_id"])] != fold
        ]
        test = [
            task
            for task in tasks
            if assignments[str(task["task_id"])] == fold
        ]
        train_resumes = {str(task["source_resume_hash"]) for task in train}
        test_resumes = {str(task["source_resume_hash"]) for task in test}
        train_jobs = {str(task["source_job_hash"]) for task in train}
        test_jobs = {str(task["source_job_hash"]) for task in test}
        if train_resumes & test_resumes or train_jobs & test_jobs:
            raise ValueError("Outer fold source leakage detected.")
        train_rows = reviewed_multiclass_rows(train)
        all_rows = [*base_rows, *train_rows]
        requirements = [row[0] for row in all_rows]
        evidence = [row[1] for row in all_rows]

        lsa = LsaEmbeddingScorer(random_state=random_state + fold).fit(
            requirements,
            evidence,
        )
        lsa_train_scores = score_holdout_tasks(train, score=lsa.score)
        lsa_threshold, lsa_train_result = select_validation_threshold(
            train,
            lsa_train_scores,
        )
        lsa_test_result = evaluate_scored_holdout(
            test,
            score_holdout_tasks(test, score=lsa.score),
            threshold=lsa_threshold,
        )
        lsa_results.append(lsa_test_result)

        candidate = fit_candidate(
            all_rows,
            random_state=random_state + 100 + fold,
        )
        train_cached = cache_two_stage_predictions(candidate, train)
        parameters, two_stage_train_result = select_two_stage_parameters(
            train,
            train_cached,
        )
        test_cached = cache_two_stage_predictions(candidate, test)
        two_stage_test_result = evaluate_two_stage(
            test,
            test_cached,
            parameters,
        )
        two_stage_results.append(two_stage_test_result)

        reference_name = max(
            (
                ("lsa", lsa_train_result),
                ("two_stage", two_stage_train_result),
            ),
            key=lambda item: (
                float(item[1]["retrieval"]["task_balanced_accuracy"]),
                float(item[1]["retrieval"]["recall_at_1"]),
                float(item[1]["retrieval"]["no_support_rejection_rate"]),
            ),
        )[0]
        reference = (
            lsa_test_result
            if reference_name == "lsa"
            else two_stage_test_result
        )
        reference_results.append(reference)

        train_probabilities = _inner_oof_candidate_probabilities(
            train,
            base_rows,
            random_state=random_state + 10_000 + fold * 100,
        )
        train_features, train_labels, train_groups = task_feature_rows(
            train,
            train_probabilities,
        )
        rejector, rejector_oof = crossfit_rejector(
            [*base_features, *train_features],
            [*base_labels, *train_labels],
            [
                *[f"base:{group}" for group in base_groups],
                *[f"{name}:{group}" for group in train_groups],
            ],
            random_state=random_state + 1_000 + fold,
        )
        threshold = select_task_rejection_threshold(
            [*base_labels, *train_labels],
            rejector_oof.tolist(),
        )
        test_probabilities = predict_tasks(candidate, test)
        test_features, _, _ = task_feature_rows(test, test_probabilities)
        acceptance = {
            str(task["task_id"]): float(probability)
            for task, probability in zip(
                test,
                rejector.predict_proba(test_features),
                strict=True,
            )
        }
        rejected = evaluate_task_policy(
            test,
            test_probabilities,
            acceptance_probabilities=acceptance,
            acceptance_threshold=threshold,
        )
        rejector_results.append(rejected)

        for task in test:
            probabilities = test_probabilities[str(task["task_id"])]
            for candidate_row, probability in zip(
                task["candidates"],
                probabilities,
                strict=True,
            ):
                candidate_labels.append(
                    str(candidate_row["support_label"])
                )
                candidate_predictions.append(
                    SUPPORT_CLASSES[
                        max(
                            range(len(SUPPORT_CLASSES)),
                            key=lambda index: probability[index],
                        )
                    ]
                )
        fold_reports.append(
            {
                "fold": fold,
                "train_tasks": len(train),
                "test_tasks": len(test),
                "test_resume_groups": len(test_resumes),
                "reference_selected_on_outer_train": reference_name,
                "lsa_threshold": lsa_threshold,
                "task_rejector_threshold": threshold,
            }
        )

    summaries = {
        "lsa": aggregate_fold_results(lsa_results),
        "two_stage": aggregate_fold_results(two_stage_results),
        "predeclared_reference": aggregate_fold_results(reference_results),
        "task_rejector": aggregate_fold_results(rejector_results),
    }
    reference_outcomes = {
        task_id: outcome
        for fold, result in enumerate(reference_results)
        for task_id, outcome in _outcomes(
            result,
            [
                task
                for task in tasks
                if assignments[str(task["task_id"])] == fold
            ],
        ).items()
    }
    rejector_outcomes = {
        task_id: outcome
        for fold, result in enumerate(rejector_results)
        for task_id, outcome in _outcomes(
            result,
            [
                task
                for task in tasks
                if assignments[str(task["task_id"])] == fold
            ],
        ).items()
    }
    reference_failures = {
        task_id: failure
        for result in reference_results
        for task_id, failure in _failure_types(result).items()
    }
    rejector_failures = {
        task_id: failure
        for result in rejector_results
        for task_id, failure in _failure_types(result).items()
    }
    task_ids = [str(task["task_id"]) for task in tasks]
    bootstrap = paired_stratified_bootstrap_delta(
        [str(task["support_label"]) != "No Support" for task in tasks],
        [reference_outcomes[task_id] for task_id in task_ids],
        [rejector_outcomes[task_id] for task_id in task_ids],
        random_state=random_state,
    )
    reference_metrics = summaries["predeclared_reference"]
    rejector_metrics = summaries["task_rejector"]
    checks = {
        "positive_lower_90": float(bootstrap["lower_90"]) > 0.0,
        "supported_regression_within_5_points": (
            float(rejector_metrics["supported_task_success_rate"])
            >= float(reference_metrics["supported_task_success_rate"]) - 0.05
        ),
        "no_support_not_regressed": (
            float(rejector_metrics["no_support_rejection_rate"])
            >= float(reference_metrics["no_support_rejection_rate"])
        ),
        "recall_at_1_not_regressed": (
            float(rejector_metrics["recall_at_1"])
            >= float(reference_metrics["recall_at_1"])
        ),
    }
    return {
        "dataset": name,
        "protocol": "nested_four_fold_resume_grouped_crossfit",
        "tasks": len(tasks),
        "resume_groups": len(
            {str(task["source_resume_hash"]) for task in tasks}
        ),
        "summaries": summaries,
        "candidate_complete_metrics": _candidate_metrics(
            candidate_labels,
            candidate_predictions,
        ),
        "slices": {
            "predeclared_reference": _slice_report(
                tasks,
                reference_outcomes,
                reference_failures,
            ),
            "task_rejector": _slice_report(
                tasks,
                rejector_outcomes,
                rejector_failures,
            ),
        },
        "paired_stratified_bootstrap": bootstrap,
        "development_gate_checks": checks,
        "development_gate_passed": all(checks.values()),
        "folds": fold_reports,
    }


def main() -> None:
    args = parse_args()
    if args.artifact_path.exists():
        raise SystemExit("V3 successor artifact exists; refusing overwrite.")
    base_tasks = load_jsonl(args.base_dir / "annotated_tasks.jsonl")
    base_pairs = load_jsonl(args.base_dir / "training_pairs.jsonl")
    v2_tasks = load_jsonl(args.v2_dir / "annotated_tasks.jsonl")
    v2_pairs = load_jsonl(args.v2_dir / "training_pairs.jsonl")
    v3_tasks = load_jsonl(args.v3_dir / "annotated_tasks.jsonl")
    v3_pairs = load_jsonl(args.v3_dir / "training_pairs.jsonl")
    v3_manifest = json.loads(
        (args.v3_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if not v3_manifest.get("eligible_for_model_comparison"):
        raise SystemExit("V3 annotation or sufficiency gate blocks evaluation.")
    _assert_source_disjoint(v2_tasks, v3_tasks)
    base_oof = base_oof_candidate_probabilities(
        base_tasks,
        base_pairs,
        random_state=args.random_state,
    )
    base_features, base_labels, base_groups = task_feature_rows(
        base_tasks,
        base_oof,
    )
    base_rows = pair_rows(base_pairs)
    datasets = {
        "operational_v2": v2_tasks,
        "operational_v3": v3_tasks,
        "combined_v2_v3": [*v2_tasks, *v3_tasks],
    }
    results = {
        name: _evaluate_dataset(
            name,
            tasks,
            base_rows=base_rows,
            base_features=base_features,
            base_labels=base_labels,
            base_groups=base_groups,
            random_state=args.random_state + index * 1_000,
        )
        for index, (name, tasks) in enumerate(datasets.items())
    }
    v2_reference = results["operational_v2"]["summaries"][
        "predeclared_reference"
    ]
    v2_successor = results["operational_v2"]["summaries"]["task_rejector"]
    v2_nonregression = (
        float(v2_successor["supported_task_success_rate"])
        >= float(v2_reference["supported_task_success_rate"]) - 0.05
        and float(v2_successor["no_support_rejection_rate"])
        >= float(v2_reference["no_support_rejection_rate"])
        and float(v2_successor["recall_at_1"])
        >= float(v2_reference["recall_at_1"])
    )
    promotion_checks = {
        "v3_development_gate": bool(
            results["operational_v3"]["development_gate_passed"]
        ),
        "combined_development_gate": bool(
            results["combined_v2_v3"]["development_gate_passed"]
        ),
        "v2_nonregression": v2_nonregression,
        "annotation_agreement_gate": bool(
            v3_manifest["agreement_gate_passed"]
        ),
        "dataset_sufficiency_gate": bool(
            v3_manifest["dataset_sufficiency_gate_passed"]
        ),
    }
    promoted = all(promotion_checks.values())
    artifact_saved = False
    if promoted:
        all_tasks = [*v2_tasks, *v3_tasks]
        all_pairs = [*v2_pairs, *v3_pairs]
        candidate = fit_candidate(
            [*base_rows, *pair_rows(all_pairs)],
            random_state=args.random_state,
        )
        all_oof = _inner_oof_candidate_probabilities(
            all_tasks,
            base_rows,
            random_state=args.random_state + 90_000,
            n_splits=4,
        )
        features, labels, groups = task_feature_rows(all_tasks, all_oof)
        rejector, rejector_oof = crossfit_rejector(
            [*base_features, *features],
            [*base_labels, *labels],
            [
                *[f"base:{group}" for group in base_groups],
                *[f"operational:{group}" for group in groups],
            ],
            random_state=args.random_state,
        )
        threshold = select_task_rejection_threshold(
            [*base_labels, *labels],
            rejector_oof.tolist(),
        )
        artifact = {
            "schema_version": 1,
            "model_type": "candidate_multiclass_plus_task_rejector",
            "candidate_model": candidate,
            "task_rejector": rejector,
            "task_threshold": threshold,
            "metadata": {
                "status": "candidate_for_new_source_isolated_holdout",
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "base_training_corpus_fingerprint": (
                    training_corpus_fingerprint(base_tasks, base_pairs)
                ),
                "operational_v3_gold_sha256": v3_manifest["gold_sha256"],
                "task_feature_manifest": rejector.feature_manifest(),
                "product_integration_allowed": False,
                "demo_integration_allowed": False,
                "application_loadable": False,
            },
        }
        args.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, args.artifact_path)
        artifact_saved = True
    report = {
        "schema_version": 1,
        "dataset": "operational_development_v3",
        "protocol": "nested_resume_grouped_candidate_and_task_crossfit",
        "base_tasks": len(base_tasks),
        "base_pairs": len(base_pairs),
        "results": results,
        "promotion_checks": promotion_checks,
        "promotion_gate_passed": promoted,
        "selection_status": (
            "candidate_for_new_source_isolated_holdout"
            if promoted
            else "blocked_operational_development_not_won"
        ),
        "artifact_saved": artifact_saved,
        "artifact_path": str(args.artifact_path) if artifact_saved else None,
        "product_integration_allowed": False,
        "demo_integration_allowed": False,
        "frozen_reserve_labels_read": False,
        "shadow_diagnostics_used_for_selection": False,
        "usage_boundary": (
            "Offline ML development only. The application and Demo do not "
            "load this report or artifact."
        ),
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, result in results.items():
        reference = result["summaries"]["predeclared_reference"]
        successor = result["summaries"]["task_rejector"]
        print(
            f"{name}: reference={reference['task_balanced_accuracy']:.3f}, "
            f"successor={successor['task_balanced_accuracy']:.3f}, "
            f"gate={result['development_gate_passed']}"
        )
    print(f"Selection: {report['selection_status']}")
    print(f"Report: {args.report_path}")


if __name__ == "__main__":
    main()
