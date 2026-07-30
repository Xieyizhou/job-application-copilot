"""Run the one-time small-reserve check for the precommitted candidate."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
BASE = PROJECT_ROOT / "data" / "ml" / "annotations" / "real_holdout_v1"
DATASET_DIR = (
    PROJECT_ROOT / "data" / "ml" / "processed" / "reviewed_evidence_training_v3"
)
MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "models"
    / "evidence_validation_candidate_v1.joblib"
)
REPORT_PATH = (
    PROJECT_ROOT / "reports" / "ml" / "generated" / "real_reserve_v1_metrics.json"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.annotation_experiment import NO_PORTABLE_MODEL  # noqa: E402
from ml.evidence import MIN_ACCEPTED_SIMILARITY, score_evidence_pair  # noqa: E402
from ml.evidence_artifact import validate_evidence_artifact  # noqa: E402
from ml.real_holdout_evaluation import (  # noqa: E402
    assert_holdout_isolated,
    evaluate_holdout_method,
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


def main() -> None:
    gold_path = BASE / "real_reserve_gold_v1.jsonl"
    manifest_path = BASE / "real_reserve_frozen_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _sha256(gold_path) != manifest["gold_sha256"]:
        raise SystemExit("Frozen reserve checksum mismatch; evaluation blocked.")
    if _sha256(MODEL_PATH) != manifest["precommitted_candidate_sha256"]:
        raise SystemExit("Precommitted candidate checksum mismatch; evaluation blocked.")

    tasks = load_jsonl(gold_path)
    training_tasks = load_jsonl(DATASET_DIR / "annotated_tasks.jsonl")
    training_pairs = load_jsonl(DATASET_DIR / "training_pairs.jsonl")
    validation_tasks = load_jsonl(BASE / "real_validation_gold_v1.jsonl")
    prior_holdout_tasks = load_jsonl(BASE / "real_holdout_gold_v1.jsonl")
    prior_tasks = [*training_tasks, *validation_tasks, *prior_holdout_tasks]
    prior_pairs = [
        *training_pairs,
        *_pair_rows(validation_tasks),
        *_pair_rows(prior_holdout_tasks),
    ]
    assert_holdout_isolated(tasks, prior_tasks, prior_pairs)

    artifact = validate_evidence_artifact(
        joblib.load(MODEL_PATH),
        tasks=training_tasks,
        pairs=training_pairs,
    )
    validation_manifest = json.loads(
        (BASE / "real_validation_frozen_manifest_v1.json").read_text(encoding="utf-8")
    )
    if (
        artifact["metadata"].get("validation_gold_sha256")
        != validation_manifest["gold_sha256"]
    ):
        raise SystemExit("Candidate validation commitment mismatch.")

    def concept_score(
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> list[float]:
        return [
            float(
                score_evidence_pair(
                    requirement,
                    candidate,
                    model_path=NO_PORTABLE_MODEL,
                )["similarity"]
            )
            for requirement, candidate in zip(requirements, evidence, strict=True)
        ]

    baseline = evaluate_holdout_method(
        tasks,
        score=concept_score,
        threshold=MIN_ACCEPTED_SIMILARITY,
    )
    candidate = evaluate_holdout_method(
        tasks,
        score=artifact["model"].predict_proba,
        threshold=float(artifact["threshold"]),
    )
    base_retrieval = baseline["retrieval"]
    candidate_retrieval = candidate["retrieval"]
    direction_confirmed = (
        candidate_retrieval["task_balanced_accuracy"]
        > base_retrieval["task_balanced_accuracy"]
        and candidate_retrieval["task_decision_accuracy"]
        > base_retrieval["task_decision_accuracy"]
        and candidate_retrieval["recall_at_1"] >= base_retrieval["recall_at_1"]
        and candidate_retrieval["no_support_rejection_rate"]
        >= base_retrieval["no_support_rejection_rate"]
    )
    report = {
        "schema_version": 1,
        "dataset": "real_reserve_v1",
        "tasks": len(tasks),
        "gold_sha256": manifest["gold_sha256"],
        "candidate_sha256": manifest["precommitted_candidate_sha256"],
        "candidate_model_type": artifact["model_type"],
        "candidate_threshold": artifact["threshold"],
        "exact_prior_content_overlap": 0,
        "source_group_overlap": 0,
        "fixed_baseline": baseline,
        "precommitted_candidate": candidate,
        "direction_gate": {
            "passed": direction_confirmed,
            "status": (
                "direction_confirmed_expand_reserve"
                if direction_confirmed
                else "direction_not_confirmed"
            ),
        },
        "promotion_eligible": False,
        "limitation": (
            "Only 16 reserve tasks. This one-time result may guide whether to build a "
            "new 40+ task final reserve, but cannot promote product inference."
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, result in (
        ("concept_lexical_rule", baseline),
        (str(artifact["model_type"]), candidate),
    ):
        retrieval = result["retrieval"]
        pair = result["pair_classification"]
        print(
            f"{name}: F1={pair['f1']:.3f}, "
            f"balanced task={retrieval['task_balanced_accuracy']:.3f}, "
            f"task accuracy={retrieval['task_decision_accuracy']:.3f}, "
            f"Recall@1={retrieval['recall_at_1']:.3f}, "
            f"No-support={retrieval['no_support_rejection_rate']:.3f}"
        )
    print(f"Direction gate: {report['direction_gate']['status']}")
    print("Promotion eligible: no (16-task reserve)")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
