"""Cross-fit the local LSA shortlist and three-class evidence reranker."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
from sklearn.model_selection import StratifiedGroupKFold


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
    / "evidence_rank_preserving_candidate_v1.joblib"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.evidence_models import WordTfidfCosineScorer  # noqa: E402
from ml.evidence_artifact import training_corpus_fingerprint  # noqa: E402
from ml.evidence_multiclass import MulticlassEvidenceReranker  # noqa: E402
from ml.evidence_two_stage import (  # noqa: E402
    RankPreservingGateParameters,
    TwoStageParameters,
    cache_two_stage_predictions,
    evaluate_rank_preserving_gate,
    evaluate_two_stage,
    reviewed_multiclass_rows,
    select_rank_preserving_gate,
    select_two_stage_parameters,
)
from ml.evidence_decision import paired_stratified_bootstrap_delta  # noqa: E402
from ml.real_holdout_evaluation import (  # noqa: E402
    assert_holdout_isolated,
    evaluate_scored_holdout,
    score_holdout_tasks,
    select_validation_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version", type=int, choices=(2, 3), default=2)
    parser.add_argument("--annotation-dir", type=Path)
    parser.add_argument("--training-dir", type=Path, default=TRAINING_DIR)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--artifact-path", type=Path, default=ARTIFACT_PATH)
    parser.add_argument(
        "--gold-filename",
        default=None,
    )
    parser.add_argument(
        "--manifest-filename",
        default=None,
    )
    parser.add_argument("--dataset-name", default=None)
    return parser.parse_args()


def _load_frozen_development(
    base: Path,
    version: str,
    *,
    gold_filename: str | None = None,
    manifest_filename: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold_path = base / (
        gold_filename or f"real_development_gold_{version}.jsonl"
    )
    manifest = json.loads(
        (
            base
            / (
                manifest_filename
                or f"real_development_frozen_manifest_{version}.json"
            )
        ).read_text(
            encoding="utf-8"
        )
    )
    checksum = hashlib.sha256(gold_path.read_bytes()).hexdigest()
    if checksum != manifest["gold_sha256"]:
        raise SystemExit("Frozen development checksum mismatch; evaluation blocked.")
    return load_jsonl(gold_path), manifest


def _fold_assignments(
    tasks: list[dict[str, Any]],
    *,
    folds: int = 4,
) -> dict[str, int]:
    ordered = sorted(tasks, key=lambda task: str(task["task_id"]))
    labels = [
        (
            "no_support"
            if str(task["support_label"]) == "No Support"
            else "supported"
        )
        for task in ordered
    ]
    groups = [str(task["source_resume_hash"]) for task in ordered]
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42,
    )
    assignments: dict[str, int] = {}
    for fold, (_, test_indices) in enumerate(
        splitter.split(ordered, labels, groups)
    ):
        for index in test_indices:
            assignments[str(ordered[int(index)]["task_id"])] = fold
    return assignments


def _aggregate_retrieval(
    fold_results: list[dict[str, Any]],
) -> dict[str, float | int]:
    support_tasks = sum(
        int(result["retrieval"]["support_tasks"]) for result in fold_results
    )
    no_support_tasks = sum(
        int(result["retrieval"]["no_support_tasks"]) for result in fold_results
    )
    total = support_tasks + no_support_tasks

    def weighted(metric: str, denominator: str) -> float:
        count = sum(int(result["retrieval"][denominator]) for result in fold_results)
        if not count:
            return 0.0
        return sum(
            float(result["retrieval"][metric])
            * int(result["retrieval"][denominator])
            for result in fold_results
        ) / count

    supported_rate = weighted("supported_task_success_rate", "support_tasks")
    rejection_rate = weighted("no_support_rejection_rate", "no_support_tasks")
    task_accuracy = (
        sum(
            float(result["retrieval"]["task_decision_accuracy"])
            * (
                int(result["retrieval"]["support_tasks"])
                + int(result["retrieval"]["no_support_tasks"])
            )
            for result in fold_results
        )
        / total
    )
    return {
        "support_tasks": support_tasks,
        "no_support_tasks": no_support_tasks,
        "recall_at_1": weighted("recall_at_1", "support_tasks"),
        "recall_at_3": weighted("recall_at_3", "support_tasks"),
        "mean_reciprocal_rank": weighted(
            "mean_reciprocal_rank",
            "support_tasks",
        ),
        "supported_task_success_rate": supported_rate,
        "no_support_rejection_rate": rejection_rate,
        "task_decision_accuracy": task_accuracy,
        "task_balanced_accuracy": (supported_rate + rejection_rate) / 2,
    }


def _assert_fold_isolation(
    train_tasks: list[dict[str, Any]],
    test_tasks: list[dict[str, Any]],
) -> None:
    for field in ("source_job_hash", "source_resume_hash"):
        train_hashes = {str(task[field]) for task in train_tasks}
        test_hashes = {str(task[field]) for task in test_tasks}
        if train_hashes & test_hashes:
            raise SystemExit(f"Cross-fit {field} leakage detected.")


def main() -> None:
    args = parse_args()
    version = f"v{args.dataset_version}"
    base = args.annotation_dir or (
        PROJECT_ROOT
        / "data"
        / "ml"
        / "annotations"
        / f"real_development_{version}"
    )
    report_path = args.report_path or (
        PROJECT_ROOT
        / "reports"
        / "ml"
        / "generated"
        / f"two_stage_development_{version}_crossfit.json"
    )
    tasks, manifest = _load_frozen_development(
        base,
        version,
        gold_filename=args.gold_filename,
        manifest_filename=args.manifest_filename,
    )
    base_tasks = load_jsonl(args.training_dir / "annotated_tasks.jsonl")
    base_pairs = load_jsonl(args.training_dir / "training_pairs.jsonl")
    assert_holdout_isolated(tasks, base_tasks, base_pairs)
    assignments = _fold_assignments(tasks)

    fold_reports: list[dict[str, Any]] = []
    baseline_results: list[dict[str, Any]] = []
    two_stage_results: list[dict[str, Any]] = []
    rank_preserving_results: list[dict[str, Any]] = []
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
        real_pairs = reviewed_multiclass_rows(train_tasks)
        requirements = [
            *[str(pair["requirement"]) for pair in base_pairs],
            *[row[0] for row in real_pairs],
        ]
        evidence = [
            *[str(pair["evidence"]) for pair in base_pairs],
            *[row[1] for row in real_pairs],
        ]
        labels = [
            *[str(pair["support_label"]) for pair in base_pairs],
            *[row[2] for row in real_pairs],
        ]

        baseline = WordTfidfCosineScorer().fit(requirements, evidence)
        baseline_train_scores = score_holdout_tasks(
            train_tasks,
            score=baseline.score,
        )
        baseline_threshold, baseline_train_result = select_validation_threshold(
            train_tasks,
            baseline_train_scores,
        )
        baseline_test_scores = score_holdout_tasks(
            test_tasks,
            score=baseline.score,
        )
        baseline_result = evaluate_scored_holdout(
            test_tasks,
            baseline_test_scores,
            threshold=baseline_threshold,
        )
        baseline_results.append(baseline_result)

        model = MulticlassEvidenceReranker(random_state=42 + fold).fit(
            requirements,
            evidence,
            labels,
        )
        train_cached = cache_two_stage_predictions(model, train_tasks)
        parameters, _ = select_two_stage_parameters(
            train_tasks,
            train_cached,
        )
        test_cached = cache_two_stage_predictions(model, test_tasks)
        two_stage_result = evaluate_two_stage(
            test_tasks,
            test_cached,
            parameters,
        )
        two_stage_results.append(two_stage_result)

        rank_preserving_train = cache_two_stage_predictions(
            model,
            train_tasks,
            retrieval_score=baseline.score,
        )
        baseline_train_metrics = baseline_train_result["retrieval"]
        rank_preserving_parameters, _ = select_rank_preserving_gate(
            train_tasks,
            rank_preserving_train,
            retrieval_threshold=baseline_threshold,
            reference_supported_success=float(
                baseline_train_metrics["supported_task_success_rate"]
            ),
            reference_rejection_rate=float(
                baseline_train_metrics["no_support_rejection_rate"]
            ),
        )
        rank_preserving_test = cache_two_stage_predictions(
            model,
            test_tasks,
            retrieval_score=baseline.score,
        )
        rank_preserving_result = evaluate_rank_preserving_gate(
            test_tasks,
            rank_preserving_test,
            rank_preserving_parameters,
        )
        rank_preserving_results.append(rank_preserving_result)
        fold_reports.append(
            {
                "fold": fold,
                "train_tasks": len(train_tasks),
                "test_tasks": len(test_tasks),
                "real_training_pairs": len(real_pairs),
                "baseline_threshold": baseline_threshold,
                "two_stage_parameters": two_stage_result["parameters"],
                "rank_preserving_parameters": (
                    rank_preserving_result["parameters"]
                ),
                "baseline": baseline_result,
                "two_stage": two_stage_result,
                "rank_preserving": rank_preserving_result,
            }
        )

    baseline_summary = _aggregate_retrieval(baseline_results)
    two_stage_summary = _aggregate_retrieval(two_stage_results)
    rank_preserving_summary = _aggregate_retrieval(rank_preserving_results)

    def beats_baseline(candidate: dict[str, float | int]) -> bool:
        return (
            float(candidate["task_balanced_accuracy"])
            > float(baseline_summary["task_balanced_accuracy"])
            and float(candidate["recall_at_1"])
            >= float(baseline_summary["recall_at_1"])
            and float(candidate["supported_task_success_rate"])
            >= float(baseline_summary["supported_task_success_rate"])
            and float(candidate["no_support_rejection_rate"])
            >= float(baseline_summary["no_support_rejection_rate"])
        )

    baseline_failures = {
        str(failure["task_id"])
        for result in baseline_results
        for failure in result["failures"]
    }
    support_mask = [
        str(task["support_label"]) != "No Support" for task in tasks
    ]
    baseline_outcomes = [
        str(task["task_id"]) not in baseline_failures for task in tasks
    ]
    candidate_results = {
        "two_stage": (two_stage_summary, two_stage_results),
        "rank_preserving_acceptance": (
            rank_preserving_summary,
            rank_preserving_results,
        ),
    }
    candidate_selection: dict[str, dict[str, Any]] = {}
    for name, (summary, fold_results) in candidate_results.items():
        failures = {
            str(failure["task_id"])
            for result in fold_results
            for failure in result["failures"]
        }
        bootstrap = paired_stratified_bootstrap_delta(
            support_mask,
            baseline_outcomes,
            [
                str(task["task_id"]) not in failures
                for task in tasks
            ],
        )
        point_estimate_wins = beats_baseline(summary)
        candidate_selection[name] = {
            "point_estimate_wins": point_estimate_wins,
            "paired_stratified_bootstrap": bootstrap,
            "stable_improvement": (
                point_estimate_wins
                and float(bootstrap["lower_90"]) > 0.0
            ),
        }
    eligible = [
        name
        for name, result in candidate_selection.items()
        if result["stable_improvement"]
    ]
    selected_candidate = (
        max(
            eligible,
            key=lambda name: (
                float(candidate_results[name][0]["task_balanced_accuracy"]),
                float(candidate_results[name][0]["recall_at_1"]),
                float(
                    candidate_results[name][0][
                        "no_support_rejection_rate"
                    ]
                ),
            ),
        )
        if eligible
        else None
    )
    improved = selected_candidate is not None
    selection_status = (
        "candidate_for_new_source_isolated_reserve"
        if improved
        else "blocked_crossfit_baseline_not_beaten"
    )
    artifact_saved = False
    if improved:
        real_pairs = reviewed_multiclass_rows(tasks)
        requirements = [
            *[str(pair["requirement"]) for pair in base_pairs],
            *[row[0] for row in real_pairs],
        ]
        evidence = [
            *[str(pair["evidence"]) for pair in base_pairs],
            *[row[1] for row in real_pairs],
        ]
        labels = [
            *[str(pair["support_label"]) for pair in base_pairs],
            *[row[2] for row in real_pairs],
        ]
        final_ranker = WordTfidfCosineScorer().fit(requirements, evidence)
        final_model = MulticlassEvidenceReranker(random_state=42).fit(
            requirements,
            evidence,
            labels,
        )
        final_parameters: (
            TwoStageParameters | RankPreservingGateParameters
        )
        if selected_candidate == "two_stage":
            final_cached = cache_two_stage_predictions(final_model, tasks)
            final_parameters, _ = select_two_stage_parameters(
                tasks,
                final_cached,
            )
            model_type = "lsa_multiclass_two_stage"
        else:
            baseline_scores = score_holdout_tasks(
                tasks,
                score=final_ranker.score,
            )
            baseline_threshold, baseline_result = (
                select_validation_threshold(tasks, baseline_scores)
            )
            final_cached = cache_two_stage_predictions(
                final_model,
                tasks,
                retrieval_score=final_ranker.score,
            )
            baseline_metrics = baseline_result["retrieval"]
            final_parameters, _ = select_rank_preserving_gate(
                tasks,
                final_cached,
                retrieval_threshold=baseline_threshold,
                reference_supported_success=float(
                    baseline_metrics["supported_task_success_rate"]
                ),
                reference_rejection_rate=float(
                    baseline_metrics["no_support_rejection_rate"]
                ),
            )
            model_type = "tfidf_rank_preserving_multiclass_gate"
        artifact = {
            "schema_version": 1,
            "model_type": model_type,
            "ranker": final_ranker,
            "support_model": final_model,
            "parameters": final_parameters,
            "metadata": {
                "status": "candidate_for_fresh_source_isolated_reserve",
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "base_training_corpus_fingerprint": training_corpus_fingerprint(
                    base_tasks,
                    base_pairs,
                ),
                "development_gold_sha256": manifest["gold_sha256"],
                "development_dataset": (
                    args.dataset_name or f"real_development_{version}"
                ),
                "development_tasks": len(tasks),
                "selected_candidate": selected_candidate,
                "feature_manifest": final_model.feature_manifest(),
                "product_integration_allowed": False,
            },
        }
        args.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, args.artifact_path)
        artifact_saved = True
    report = {
        "schema_version": 1,
        "dataset": args.dataset_name or f"real_development_{version}",
        "protocol": "four_fold_stratified_resume_group_crossfit",
        "tasks": len(tasks),
        "base_training_tasks": len(base_tasks),
        "base_training_pairs": len(base_pairs),
        "fold_source_overlap": 0,
        "unselected_supported_candidates_used_as_negatives": False,
        "baseline": baseline_summary,
        "two_stage": two_stage_summary,
        "rank_preserving_acceptance": rank_preserving_summary,
        "folds": fold_reports,
        "selection": {
            "selected_candidate": selected_candidate,
            "beats_crossfit_tfidf": improved,
            "candidates": candidate_selection,
            "status": selection_status,
            "artifact_saved": artifact_saved,
            "artifact_path": str(args.artifact_path) if artifact_saved else None,
        },
        "usage_boundary": (
            "Development cross-fit only. No final holdout or consumed reserve "
            "was read. A fresh reserve is required before product integration."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, result in (
        ("crossfit_tfidf", baseline_summary),
        ("crossfit_lsa_multiclass", two_stage_summary),
        ("crossfit_tfidf_rank_preserving_gate", rank_preserving_summary),
    ):
        print(
            f"{name}: balanced={result['task_balanced_accuracy']:.3f}, "
            f"task accuracy={result['task_decision_accuracy']:.3f}, "
            f"Recall@1={result['recall_at_1']:.3f}, "
            f"Recall@3={result['recall_at_3']:.3f}, "
            f"No-support={result['no_support_rejection_rate']:.3f}"
        )
    print(f"Selection: {selection_status}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
