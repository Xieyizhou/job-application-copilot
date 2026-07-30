"""Compare locally fitted evidence scorers on adjudicated real validation gold."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
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
from ml.annotation_experiment import NO_PORTABLE_MODEL  # noqa: E402
from ml.evidence import MIN_ACCEPTED_SIMILARITY, score_evidence_pair  # noqa: E402
from ml.evidence_models import (  # noqa: E402
    HybridEvidenceReranker,
    LexicalGuardedReranker,
    LsaEmbeddingScorer,
    PairwiseHybridReranker,
    WordTfidfCosineScorer,
)
from ml.real_holdout_evaluation import (  # noqa: E402
    assert_holdout_isolated,
    evaluate_holdout_method,
    score_holdout_tasks,
    select_validation_threshold,
)


DATASETS = {
    "validation": {
        "dataset_id": "real_validation_v1",
        "base": PROJECT_ROOT / "data" / "ml" / "annotations" / "real_holdout_v1",
        "gold": "real_validation_gold_v1.jsonl",
        "manifest": "real_validation_frozen_manifest_v1.json",
        "report": "real_validation_v1_calibration.json",
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
        "report": "real_development_v2_calibration.json",
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
    checksum = hashlib.sha256(gold_path.read_bytes()).hexdigest()
    if checksum != manifest["gold_sha256"]:
        raise SystemExit(f"Frozen {dataset} checksum mismatch; calibration blocked.")
    return load_jsonl(gold_path), manifest, config


def main(dataset: str = "validation") -> None:
    tasks, manifest, config = _load_dataset(dataset)
    training_tasks = load_jsonl(DATASET_DIR / "annotated_tasks.jsonl")
    pairs = load_jsonl(DATASET_DIR / "training_pairs.jsonl")
    assert_holdout_isolated(tasks, training_tasks, pairs)
    requirements = [str(pair["requirement"]) for pair in pairs]
    evidence = [str(pair["evidence"]) for pair in pairs]
    labels = [int(pair["binary_label"]) for pair in pairs]

    word = WordTfidfCosineScorer().fit(requirements, evidence)
    lsa = LsaEmbeddingScorer(random_state=42).fit(requirements, evidence)
    hybrid = HybridEvidenceReranker(random_state=42).fit(
        requirements,
        evidence,
        labels,
    )
    guarded = LexicalGuardedReranker(random_state=42).fit(
        requirements,
        evidence,
        labels,
    )
    pairwise = PairwiseHybridReranker(random_state=42).fit(
        requirements,
        evidence,
        labels,
        training_tasks,
    )
    scorers = {
        "tfidf_cosine": word.score,
        "lsa_embedding": lsa.score,
        "hybrid_lsa_reranker": hybrid.predict_proba,
        "lexical_guarded_reranker": guarded.predict_proba,
        "pairwise_hybrid_reranker": pairwise.predict_proba,
    }

    def concept_score(
        aligned_requirements: Sequence[str],
        aligned_evidence: Sequence[str],
    ) -> list[float]:
        return [
            float(
                score_evidence_pair(
                    requirement,
                    candidate,
                    model_path=NO_PORTABLE_MODEL,
                )["similarity"]
            )
            for requirement, candidate in zip(
                aligned_requirements,
                aligned_evidence,
                strict=True,
            )
        ]

    fixed_baseline = evaluate_holdout_method(
        tasks,
        score=concept_score,
        threshold=MIN_ACCEPTED_SIMILARITY,
    )
    calibrated: dict[str, dict[str, Any]] = {}
    for name, scorer in scorers.items():
        scored = score_holdout_tasks(tasks, score=scorer)
        threshold, result = select_validation_threshold(tasks, scored)
        calibrated[name] = {
            **result,
            "threshold": threshold,
            "threshold_source": config["dataset_id"],
        }

    def selection_key(name: str) -> tuple[float, ...]:
        retrieval = calibrated[name]["retrieval"]
        return (
            float(retrieval["task_balanced_accuracy"]),
            float(retrieval["task_decision_accuracy"]),
            float(retrieval["recall_at_1"]),
            float(retrieval["no_support_rejection_rate"]),
        )

    selected = max(calibrated, key=selection_key)
    baseline_retrieval = fixed_baseline["retrieval"]
    selected_retrieval = calibrated[selected]["retrieval"]
    beats_baseline = (
        selected_retrieval["task_balanced_accuracy"]
        > baseline_retrieval["task_balanced_accuracy"]
        and selected_retrieval["recall_at_1"] >= baseline_retrieval["recall_at_1"]
        and selected_retrieval["no_support_rejection_rate"]
        >= baseline_retrieval["no_support_rejection_rate"]
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
        "fixed_baseline": fixed_baseline,
        "calibrated_methods": calibrated,
        "selection": {
            "selected_method": selected,
            "selected_threshold": calibrated[selected]["threshold"],
            "beats_fixed_baseline": beats_baseline,
            "status": (
                "candidate_for_fresh_reserve_test"
                if beats_baseline
                else "blocked_baseline_not_beaten"
            ),
        },
        "usage_boundary": (
            "Development calibration only. Do not report these values as final "
            "generalization metrics."
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
    base = fixed_baseline["retrieval"]
    print(
        "concept_lexical_rule: "
        f"balanced={base['task_balanced_accuracy']:.3f}, "
        f"task accuracy={base['task_decision_accuracy']:.3f}, "
        f"Recall@1={base['recall_at_1']:.3f}, "
        f"No-support={base['no_support_rejection_rate']:.3f}"
    )
    for name, result in calibrated.items():
        retrieval = result["retrieval"]
        print(
            f"{name}: threshold={result['threshold']:.4f}, "
            f"balanced={retrieval['task_balanced_accuracy']:.3f}, "
            f"task accuracy={retrieval['task_decision_accuracy']:.3f}, "
            f"Recall@1={retrieval['recall_at_1']:.3f}, "
            f"No-support={retrieval['no_support_rejection_rate']:.3f}"
        )
    print(f"Selection: {selected} ({report['selection']['status']})")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare local evidence scorers on frozen development data."
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASETS),
        default="validation",
        help="Frozen development dataset to evaluate (default: validation).",
    )
    main(parser.parse_args().dataset)
