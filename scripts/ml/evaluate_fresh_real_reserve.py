"""Evaluate the precommitted rank-preserving candidate on fresh reserve v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
BASE = PROJECT_ROOT / "data" / "ml" / "annotations" / "real_reserve_v2"
TRAINING_DIR = (
    PROJECT_ROOT / "data" / "ml" / "processed" / "reviewed_evidence_training_v3"
)
MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "models"
    / "evidence_rank_preserving_candidate_v1.joblib"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "ml"
    / "generated"
    / "fresh_real_reserve_v2_metrics.json"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.evidence_artifact import training_corpus_fingerprint  # noqa: E402
from ml.evidence_decision import paired_stratified_bootstrap_delta  # noqa: E402
from ml.evidence_two_stage import (  # noqa: E402
    RankPreservingGateParameters,
    cache_two_stage_predictions,
    evaluate_rank_preserving_gate,
)
from ml.real_holdout_evaluation import (  # noqa: E402
    assert_holdout_isolated,
    evaluate_scored_holdout,
    score_holdout_tasks,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pair_rows(tasks: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "requirement": str(task["requirement"]),
            "evidence": str(candidate["evidence"]),
        }
        for task in tasks
        for candidate in task["candidates"]
    ]


def _prior_gold_tasks() -> list[dict[str, Any]]:
    paths = [
        PROJECT_ROOT
        / "data"
        / "ml"
        / "annotations"
        / "real_holdout_v1"
        / name
        for name in (
            "real_holdout_gold_v1.jsonl",
            "real_validation_gold_v1.jsonl",
            "real_reserve_gold_v1.jsonl",
        )
    ]
    paths.extend(
        [
            PROJECT_ROOT
            / "data"
            / "ml"
            / "annotations"
            / f"real_development_{version}"
            / f"real_development_gold_{version}.jsonl"
            for version in ("v2", "v3")
        ]
    )
    return [task for path in paths for task in load_jsonl(path)]


def _validate_artifact(
    artifact: Any,
    *,
    base_tasks: list[dict[str, Any]],
    base_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise SystemExit("Candidate artifact must be an object.")
    if (
        int(artifact.get("schema_version", 0)) != 1
        or artifact.get("model_type")
        != "tfidf_rank_preserving_multiclass_gate"
    ):
        raise SystemExit("Unexpected fresh-reserve candidate artifact.")
    if not callable(getattr(artifact.get("ranker"), "score", None)):
        raise SystemExit("Candidate ranker is missing its score method.")
    if not callable(
        getattr(artifact.get("support_model"), "predict_class_proba", None)
    ):
        raise SystemExit("Candidate support model is invalid.")
    if not isinstance(artifact.get("parameters"), RankPreservingGateParameters):
        raise SystemExit("Candidate gate parameters are invalid.")
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        raise SystemExit("Candidate metadata is missing.")
    expected_fingerprint = training_corpus_fingerprint(base_tasks, base_pairs)
    if metadata.get("base_training_corpus_fingerprint") != expected_fingerprint:
        raise SystemExit("Candidate base-training commitment mismatch.")
    development_manifest = json.loads(
        (
            PROJECT_ROOT
            / "data"
            / "ml"
            / "annotations"
            / "real_development_v3"
            / "real_development_frozen_manifest_v3.json"
        ).read_text(encoding="utf-8")
    )
    if metadata.get("development_gold_sha256") != development_manifest["gold_sha256"]:
        raise SystemExit("Candidate development commitment mismatch.")
    return artifact


def main() -> None:
    gold_path = BASE / "real_reserve_gold_v2.jsonl"
    manifest_path = BASE / "real_reserve_frozen_manifest_v2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _sha256(gold_path) != manifest["gold_sha256"]:
        raise SystemExit("Frozen reserve checksum mismatch; evaluation blocked.")
    if _sha256(MODEL_PATH) != manifest["precommitted_candidate_sha256"]:
        raise SystemExit("Precommitted candidate checksum mismatch; evaluation blocked.")

    tasks = load_jsonl(gold_path)
    base_tasks = load_jsonl(TRAINING_DIR / "annotated_tasks.jsonl")
    base_pairs = load_jsonl(TRAINING_DIR / "training_pairs.jsonl")
    prior_tasks = _prior_gold_tasks()
    assert_holdout_isolated(
        tasks,
        [*base_tasks, *prior_tasks],
        [*base_pairs, *_pair_rows(prior_tasks)],
    )
    artifact = _validate_artifact(
        joblib.load(MODEL_PATH),
        base_tasks=base_tasks,
        base_pairs=base_pairs,
    )
    parameters = artifact["parameters"]
    ranker = artifact["ranker"]
    support_model = artifact["support_model"]

    baseline_scores = score_holdout_tasks(tasks, score=ranker.score)
    baseline = evaluate_scored_holdout(
        tasks,
        baseline_scores,
        threshold=parameters.retrieval_threshold,
    )
    cached = cache_two_stage_predictions(
        support_model,
        tasks,
        retrieval_score=ranker.score,
    )
    candidate = evaluate_rank_preserving_gate(tasks, cached, parameters)
    baseline_failures = {
        str(failure["task_id"]) for failure in baseline["failures"]
    }
    candidate_failures = {
        str(failure["task_id"]) for failure in candidate["failures"]
    }
    bootstrap = paired_stratified_bootstrap_delta(
        [str(task["support_label"]) != "No Support" for task in tasks],
        [str(task["task_id"]) not in baseline_failures for task in tasks],
        [str(task["task_id"]) not in candidate_failures for task in tasks],
    )
    baseline_retrieval = baseline["retrieval"]
    candidate_retrieval = candidate["retrieval"]
    point_estimate_wins = (
        candidate_retrieval["task_balanced_accuracy"]
        > baseline_retrieval["task_balanced_accuracy"]
        and candidate_retrieval["task_decision_accuracy"]
        > baseline_retrieval["task_decision_accuracy"]
        and candidate_retrieval["recall_at_1"]
        >= baseline_retrieval["recall_at_1"]
        and candidate_retrieval["supported_task_success_rate"]
        >= baseline_retrieval["supported_task_success_rate"]
        and candidate_retrieval["no_support_rejection_rate"]
        >= baseline_retrieval["no_support_rejection_rate"]
    )
    passed = point_estimate_wins and float(bootstrap["lower_90"]) > 0
    report = {
        "schema_version": 1,
        "dataset": "fresh_real_reserve_v2",
        "tasks": len(tasks),
        "gold_sha256": manifest["gold_sha256"],
        "candidate_sha256": manifest["precommitted_candidate_sha256"],
        "review_policy": manifest.get("review_policy"),
        "reviewer_exact_agreement_rate": manifest.get(
            "reviewer_exact_agreement_rate"
        ),
        "source_and_content_isolation_verified": True,
        "fixed_tfidf_baseline": baseline,
        "precommitted_rank_preserving_candidate": candidate,
        "selection_gate": {
            "point_estimate_wins": point_estimate_wins,
            "paired_stratified_bootstrap": bootstrap,
            "passed": passed,
            "status": (
                "eligible_for_failure_review_and_shadow_mode"
                if passed
                else "candidate_rejected_on_fresh_reserve"
            ),
        },
        "product_integration_allowed": False,
        "limitation": (
            "The reserve is a one-time direction test. Results must not be used "
            "to retune this candidate; product use requires failure review and a "
            "separate shadow-mode decision."
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, result in (("fixed_tfidf", baseline), ("candidate", candidate)):
        metrics = result["retrieval"]
        print(
            f"{name}: balanced={metrics['task_balanced_accuracy']:.3f}, "
            f"accuracy={metrics['task_decision_accuracy']:.3f}, "
            f"Recall@1={metrics['recall_at_1']:.3f}, "
            f"supported={metrics['supported_task_success_rate']:.3f}, "
            f"No-support={metrics['no_support_rejection_rate']:.3f}"
        )
    print(f"Paired bootstrap lower 90%: {bootstrap['lower_90']:.4f}")
    print(f"Selection gate: {report['selection_gate']['status']}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
