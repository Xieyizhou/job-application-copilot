"""Evaluate fixed evidence candidates on a source-isolated development set."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_TRAINING_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "processed"
    / "reviewed_evidence_training_v4_batch4"
)
DEFAULT_DEVELOPMENT_DIR = (
    PROJECT_ROOT / "data" / "ml" / "annotations" / "real_development_v3"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "ml"
    / "generated"
    / "sentence_embedding_real_development_v3.json"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.evidence_decision import paired_stratified_bootstrap_delta  # noqa: E402
from ml.evidence_grouped_evaluation import (  # noqa: E402
    task_retrieval_evaluation,
)
from ml.evidence_multiclass import (  # noqa: E402
    SUPPORT_CLASSES,
    MulticlassEvidenceReranker,
)
from ml.evidence_sentence_embedding import (  # noqa: E402
    DEFAULT_SENTENCE_MODEL,
    DEFAULT_SENTENCE_MODEL_REVISION,
    CachedSentenceEncoder,
    FrozenSentenceEmbeddingClassifier,
    load_sentence_encoder,
)
from ml.real_holdout_evaluation import assert_holdout_isolated  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument(
        "--development-dir",
        type=Path,
        default=DEFAULT_DEVELOPMENT_DIR,
    )
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--gold-filename",
        default="real_development_gold_v3.jsonl",
    )
    parser.add_argument(
        "--manifest-filename",
        default="real_development_frozen_manifest_v3.json",
    )
    parser.add_argument("--dataset-name", default="real_development_v3")
    parser.add_argument(
        "--dataset-role",
        default="source_isolated_development",
    )
    parser.add_argument("--sentence-model", default=DEFAULT_SENTENCE_MODEL)
    parser.add_argument(
        "--sentence-model-revision",
        default=DEFAULT_SENTENCE_MODEL_REVISION,
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--random-state", type=int, default=20260729)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flatten_tasks(
    tasks: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "pair_id": f"{task['task_id']}:{candidate['candidate_id']}",
            "task_id": str(task["task_id"]),
            "requirement": str(task["requirement"]),
            "evidence": str(candidate["evidence"]),
        }
        for task in tasks
        for candidate in task["candidates"]
    ]


def _evaluate_model(
    model: Any,
    tasks: list[dict[str, Any]],
    pairs: list[dict[str, str]],
) -> tuple[dict[str, Any], list[str], list[float], dict[str, bool]]:
    requirements = [pair["requirement"] for pair in pairs]
    evidence = [pair["evidence"] for pair in pairs]
    probabilities = np.asarray(
        model.predict_class_proba(requirements, evidence),
        dtype=np.float64,
    )
    if probabilities.shape != (len(pairs), len(SUPPORT_CLASSES)):
        raise SystemExit("Candidate model returned invalid class probabilities.")
    predictions = np.asarray(SUPPORT_CLASSES)[
        np.argmax(probabilities, axis=1)
    ].tolist()
    support_scores = (
        probabilities[:, 0] + 0.5 * probabilities[:, 1]
    ).astype(float).tolist()
    metrics, outcomes = task_retrieval_evaluation(
        tasks,
        pairs,
        predictions,
        support_scores,
    )
    return (
        metrics,
        [str(value) for value in predictions],
        support_scores,
        outcomes,
    )


def main() -> None:
    args = parse_args()
    gold_path = args.development_dir / args.gold_filename
    manifest_path = args.development_dir / args.manifest_filename
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _sha256(gold_path) != manifest["gold_sha256"]:
        raise SystemExit("Real development checksum mismatch; evaluation blocked.")

    training_tasks = load_jsonl(args.training_dir / "annotated_tasks.jsonl")
    training_pairs = load_jsonl(args.training_dir / "training_pairs.jsonl")
    development_tasks = load_jsonl(gold_path)
    assert_holdout_isolated(
        development_tasks,
        training_tasks,
        training_pairs,
    )
    requirements = [str(pair["requirement"]) for pair in training_pairs]
    evidence = [str(pair["evidence"]) for pair in training_pairs]
    labels = [str(pair["support_label"]) for pair in training_pairs]
    development_pairs = _flatten_tasks(development_tasks)

    lsa = MulticlassEvidenceReranker(
        random_state=args.random_state
    ).fit(requirements, evidence, labels)
    lsa_metrics, _, _, lsa_outcomes = _evaluate_model(
        lsa,
        development_tasks,
        development_pairs,
    )

    encoder = CachedSentenceEncoder(
        load_sentence_encoder(
            model_name=args.sentence_model,
            revision=args.sentence_model_revision,
            local_files_only=args.local_files_only,
        )
    )
    pure = FrozenSentenceEmbeddingClassifier(
        encoder,
        random_state=args.random_state,
    ).fit(requirements, evidence, labels)
    hybrid = FrozenSentenceEmbeddingClassifier(
        encoder,
        include_transparent_features=True,
        random_state=args.random_state,
    ).fit(requirements, evidence, labels)
    pure_metrics, pure_predictions, _, _ = _evaluate_model(
        pure,
        development_tasks,
        development_pairs,
    )
    hybrid_metrics, _, hybrid_scores, _ = _evaluate_model(
        hybrid,
        development_tasks,
        development_pairs,
    )
    two_stage_metrics, two_stage_outcomes = task_retrieval_evaluation(
        development_tasks,
        development_pairs,
        pure_predictions,
        hybrid_scores,
    )
    task_ids = [str(task["task_id"]) for task in development_tasks]
    bootstrap = paired_stratified_bootstrap_delta(
        [
            str(task["support_label"]) != "No Support"
            for task in development_tasks
        ],
        [lsa_outcomes[task_id] for task_id in task_ids],
        [two_stage_outcomes[task_id] for task_id in task_ids],
        random_state=args.random_state,
    )
    point_estimate_wins = (
        two_stage_metrics["task_balanced_accuracy"]
        > lsa_metrics["task_balanced_accuracy"]
        and two_stage_metrics["recall_at_1"] >= lsa_metrics["recall_at_1"]
        and two_stage_metrics["no_support_rejection_rate"]
        >= lsa_metrics["no_support_rejection_rate"]
    )
    stable_improvement = (
        point_estimate_wins and float(bootstrap["lower_90"]) > 0.0
    )

    methods: dict[str, dict[str, Any]] = {
        "lsa_transparent_multiclass": lsa_metrics,
        "frozen_sentence_embedding": pure_metrics,
        "frozen_sentence_embedding_hybrid": hybrid_metrics,
        "frozen_embedding_two_stage": two_stage_metrics,
    }
    development_gate: dict[str, Any] = {
        "reference": "lsa_transparent_multiclass",
        "point_estimate_wins": point_estimate_wins,
        "paired_stratified_bootstrap": bootstrap,
        "stable_improvement": stable_improvement,
        "status": (
            "eligible_to_construct_new_source_isolated_reserve"
            if stable_improvement
            else "continue_development_without_opening_new_reserve"
        ),
    }
    report = {
        "schema_version": 1,
        "dataset": args.dataset_name,
        "dataset_role": args.dataset_role,
        "tasks": len(development_tasks),
        "pairs": len(development_pairs),
        "label_counts": dict(
            Counter(str(task["support_label"]) for task in development_tasks)
        ),
        "training_tasks": len(training_tasks),
        "training_pairs": len(training_pairs),
        "input_sha256": {
            "development_gold": manifest["gold_sha256"],
            "training_tasks": _sha256(
                args.training_dir / "annotated_tasks.jsonl"
            ),
            "training_pairs": _sha256(
                args.training_dir / "training_pairs.jsonl"
            ),
        },
        "source_and_content_isolation_verified": True,
        "sentence_encoder": {
            "model": args.sentence_model,
            "revision": args.sentence_model_revision,
            "weights_frozen": True,
        },
        "methods": methods,
        "locked_candidate": "frozen_embedding_two_stage",
        "threshold_tuning": False,
        "development_gate": development_gate,
        "promotion_eligible": False,
        "contains_real_candidate_profiles": bool(
            manifest.get("contains_real_candidate_profiles", True)
        ),
        "limitation": (
            "This split permits architecture comparison and calibration. It is "
            "not an untouched final holdout and cannot authorize product use."
        ),
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, metrics in methods.items():
        print(
            f"{name}: balanced={metrics['task_balanced_accuracy']:.3f}, "
            f"Recall@1={metrics['recall_at_1']:.3f}, "
            f"Recall@3={metrics['recall_at_3']:.3f}, "
            f"No-support={metrics['no_support_rejection_rate']:.3f}"
        )
    print(
        f"Paired bootstrap 90% interval: "
        f"[{bootstrap['lower_90']:.3f}, {bootstrap['upper_90']:.3f}]"
    )
    print(f"Development gate: {development_gate['status']}")
    print("Promotion eligible: no")
    print(f"Report: {args.report_path}")


if __name__ == "__main__":
    main()
