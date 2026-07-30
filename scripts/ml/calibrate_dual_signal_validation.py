"""Calibrate independent TF-IDF ranking and transparent support acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DATASET_DIR = (
    PROJECT_ROOT / "data" / "ml" / "processed" / "reviewed_evidence_training_v3"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.evidence import (  # noqa: E402
    MIN_ACCEPTED_SIMILARITY,
    score_transparent_evidence_pair,
)
from ml.evidence_decision import (  # noqa: E402
    SupportSignal,
    evaluate_ranked_support_gate,
    select_support_gate_threshold,
    support_signal_from_probability,
    support_signal_from_score,
)
from ml.evidence_models import WordTfidfCosineScorer  # noqa: E402
from ml.evidence_support import TransparentSupportClassifier  # noqa: E402
from ml.real_holdout_evaluation import (  # noqa: E402
    assert_holdout_isolated,
    score_holdout_tasks,
    select_validation_threshold,
)


DATASETS = {
    "validation": {
        "dataset_id": "real_validation_v1",
        "base": PROJECT_ROOT / "data" / "ml" / "annotations" / "real_holdout_v1",
        "gold": "real_validation_gold_v1.jsonl",
        "manifest": "real_validation_frozen_manifest_v1.json",
        "report": "dual_signal_validation_v1_calibration.json",
    },
    "development": {
        "dataset_id": "real_development_v2",
        "base": (
            PROJECT_ROOT
            / "data"
            / "ml"
            / "annotations"
            / "real_development_v2"
        ),
        "gold": "real_development_gold_v2.jsonl",
        "manifest": "real_development_frozen_manifest_v2.json",
        "report": "dual_signal_development_v2_calibration.json",
    },
}


def _load_dataset(
    dataset: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    config = DATASETS[dataset]
    base_value = config["base"]
    if not isinstance(base_value, Path):
        raise TypeError(f"Invalid base path for {dataset}.")
    base = base_value
    gold_path = base / str(config["gold"])
    manifest_path = base / str(config["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if hashlib.sha256(gold_path.read_bytes()).hexdigest() != manifest["gold_sha256"]:
        raise SystemExit(f"Frozen {dataset} checksum mismatch; calibration blocked.")
    return load_jsonl(gold_path), manifest, config


def _transparent_signals(
    tasks: list[dict[str, Any]],
) -> dict[str, list[SupportSignal]]:
    signals: dict[str, list[SupportSignal]] = {}
    for task in tasks:
        requirement = str(task["requirement"])
        signals[str(task["task_id"])] = [
            support_signal_from_score(
                score_transparent_evidence_pair(
                    requirement, str(candidate["evidence"])
                )
            )
            for candidate in task["candidates"]
        ]
    return signals


def main(dataset: str = "validation") -> None:
    tasks, manifest, config = _load_dataset(dataset)
    training_tasks = load_jsonl(DATASET_DIR / "annotated_tasks.jsonl")
    pairs = load_jsonl(DATASET_DIR / "training_pairs.jsonl")
    assert_holdout_isolated(tasks, training_tasks, pairs)

    requirements = [str(pair["requirement"]) for pair in pairs]
    evidence = [str(pair["evidence"]) for pair in pairs]
    labels = [int(pair["binary_label"]) for pair in pairs]
    ranker = WordTfidfCosineScorer().fit(requirements, evidence)
    support_classifier = TransparentSupportClassifier().fit(
        requirements,
        evidence,
        labels,
    )
    ranking_scores = score_holdout_tasks(tasks, score=ranker.score)
    support_signals = _transparent_signals(tasks)
    learned_support_scores = score_holdout_tasks(
        tasks,
        score=support_classifier.predict_proba,
    )
    learned_support_signals = {
        task_id: [
            support_signal_from_probability(value) for value in values
        ]
        for task_id, values in learned_support_scores.items()
    }

    rank_threshold, rank_only = select_validation_threshold(
        tasks,
        ranking_scores,
    )
    fixed_gate = evaluate_ranked_support_gate(
        tasks,
        ranking_scores,
        support_signals,
        support_threshold=MIN_ACCEPTED_SIMILARITY,
    )
    gate_threshold, calibrated_gate = select_support_gate_threshold(
        tasks,
        ranking_scores,
        support_signals,
    )
    learned_threshold, learned_gate = select_support_gate_threshold(
        tasks,
        ranking_scores,
        learned_support_signals,
    )

    rank_retrieval = rank_only["retrieval"]
    gate_retrieval = learned_gate["retrieval"]
    improves_rank_only = (
        gate_retrieval["task_balanced_accuracy"]
        > rank_retrieval["task_balanced_accuracy"]
        and gate_retrieval["recall_at_1"] == rank_retrieval["recall_at_1"]
    )
    report = {
        "schema_version": 1,
        "dataset": config["dataset_id"],
        "dataset_role": dataset,
        "gold_sha256": manifest["gold_sha256"],
        "tasks": len(tasks),
        "training_tasks": len(training_tasks),
        "training_pairs": len(pairs),
        "exact_training_overlap": 0,
        "source_group_overlap": 0,
        "architecture": {
            "ranking": "word_tfidf_cosine",
            "acceptance": "explicit_feature_logistic_support_gate",
            "ranking_affects_acceptance": False,
            "support_feature_groups": [
                "lexical_coverage",
                "concept_overlap",
                "numeric_constraint_signal",
                "compound_requirement_signal",
            ],
        },
        "rank_only_reference": {
            **rank_only,
            "threshold": rank_threshold,
        },
        "fixed_support_gate": fixed_gate,
        "calibrated_support_gate": calibrated_gate,
        "learned_feature_support_gate": {
            **learned_gate,
            "feature_manifest": support_classifier.feature_manifest(),
        },
        "selection": {
            "selected_method": "dual_signal_tfidf_feature_gate",
            "support_threshold": learned_threshold,
            "improves_validation_rank_only": improves_rank_only,
            "status": (
                "candidate_for_new_source_isolated_reserve"
                if improves_rank_only
                else "blocked_validation_not_improved"
            ),
        },
        "usage_boundary": (
            "Threshold and architecture development only. Frozen final holdout "
            "and prior reserve results were not read or used for selection."
        ),
    }
    report_path = (
        PROJECT_ROOT / "reports" / "ml" / "generated" / str(config["report"])
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for label, result in (
        ("rank_only", rank_only),
        ("fixed_dual_gate", fixed_gate),
        ("calibrated_dual_gate", calibrated_gate),
        ("learned_feature_gate", learned_gate),
    ):
        retrieval = result["retrieval"]
        print(
            f"{label}: balanced={retrieval['task_balanced_accuracy']:.3f}, "
            f"task accuracy={retrieval['task_decision_accuracy']:.3f}, "
            f"Recall@1={retrieval['recall_at_1']:.3f}, "
            f"No-support={retrieval['no_support_rejection_rate']:.3f}"
        )
    print(f"Support threshold: {learned_threshold:.4f}")
    print(f"Selection: {report['selection']['status']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate ranking and support acceptance on frozen development data."
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASETS),
        default="validation",
        help="Frozen development dataset to evaluate (default: validation).",
    )
    main(parser.parse_args().dataset)
