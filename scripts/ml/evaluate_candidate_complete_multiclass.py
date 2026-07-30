"""Compare three-class pair baselines with task-grouped cross-validation."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "processed"
    / "reviewed_evidence_training_v4_batch1"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "ml"
    / "generated"
    / "candidate_complete_multiclass_v4_batch1.json"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.evidence_multiclass import (  # noqa: E402
    SUPPORT_CLASSES,
    MulticlassEvidenceReranker,
    PairTextTfidfClassifier,
)
from ml.evidence_grouped_evaluation import (  # noqa: E402
    classification_metrics as _metrics,
    task_retrieval_metrics as _task_retrieval_metrics,
)
from ml.evidence_sentence_embedding import (  # noqa: E402
    DEFAULT_SENTENCE_MODEL,
    DEFAULT_SENTENCE_MODEL_REVISION,
    CachedSentenceEncoder,
    FrozenSentenceEmbeddingClassifier,
    load_sentence_encoder,
)


ModelFactory = Callable[[int], Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=20260729)
    parser.add_argument("--sentence-model", default=DEFAULT_SENTENCE_MODEL)
    parser.add_argument(
        "--sentence-model-revision",
        default=DEFAULT_SENTENCE_MODEL_REVISION,
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-sentence-embedding", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cross_validated_predictions(
    pairs: list[dict[str, Any]],
    *,
    splitter: StratifiedGroupKFold,
    factory: ModelFactory,
    random_state: int,
) -> tuple[list[str], list[float], list[dict[str, Any]]]:
    requirements = np.asarray([str(pair["requirement"]) for pair in pairs])
    evidence = np.asarray([str(pair["evidence"]) for pair in pairs])
    labels = np.asarray([str(pair["support_label"]) for pair in pairs])
    groups = np.asarray([str(pair["evaluation_group"]) for pair in pairs])
    predictions = np.empty(len(pairs), dtype=object)
    support_scores = np.empty(len(pairs), dtype=np.float64)
    folds: list[dict[str, Any]] = []
    for fold, (train, test) in enumerate(
        splitter.split(requirements, labels, groups),
        start=1,
    ):
        model = factory(random_state + fold)
        model.fit(
            requirements[train].tolist(),
            evidence[train].tolist(),
            labels[train].tolist(),
        )
        probabilities = model.predict_class_proba(
            requirements[test].tolist(),
            evidence[test].tolist(),
        )
        predictions[test] = np.asarray(SUPPORT_CLASSES)[
            np.argmax(probabilities, axis=1)
        ]
        support_scores[test] = probabilities[:, 0] + 0.5 * probabilities[:, 1]
        folds.append(
            {
                "fold": fold,
                "train_pairs": int(len(train)),
                "test_pairs": int(len(test)),
                "test_groups": int(len(set(groups[test]))),
                "test_label_counts": dict(Counter(labels[test].tolist())),
            }
        )
    return (
        [str(value) for value in predictions],
        support_scores.astype(float).tolist(),
        folds,
    )


def main() -> None:
    args = parse_args()
    if args.folds < 3:
        raise SystemExit("Use at least three task-grouped folds.")
    pair_path = args.dataset_dir / "training_pairs.jsonl"
    task_path = args.dataset_dir / "annotated_tasks.jsonl"
    pairs = load_jsonl(pair_path)
    tasks = load_jsonl(task_path)
    if not pairs or not tasks:
        raise SystemExit("Candidate-complete training corpus is missing.")
    labels = [str(pair.get("support_label", "")) for pair in pairs]
    if set(labels) != set(SUPPORT_CLASSES):
        raise SystemExit("Training pairs must contain every support class.")
    if any(not str(pair.get("evaluation_group", "")) for pair in pairs):
        raise SystemExit("Every pair needs an evaluation group.")

    splitter = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.random_state,
    )
    majority_predictions: list[str] = ["No Support"] * len(labels)
    methods: dict[str, Any] = {
        "majority_no_support": {
            "metrics": _metrics(labels, majority_predictions),
            "retrieval": _task_retrieval_metrics(
                tasks,
                pairs,
                majority_predictions,
                [0.0] * len(pairs),
            ),
            "folds": [],
        }
    }
    factories: dict[str, ModelFactory] = {
        "pair_text_tfidf": lambda seed: PairTextTfidfClassifier(
            random_state=seed
        ),
        "lsa_transparent_multiclass": lambda seed: MulticlassEvidenceReranker(
            random_state=seed
        ),
    }
    sentence_encoder_manifest: dict[str, Any] | None = None
    if not args.skip_sentence_embedding:
        sentence_encoder = CachedSentenceEncoder(
            load_sentence_encoder(
                model_name=args.sentence_model,
                revision=args.sentence_model_revision,
                local_files_only=args.local_files_only,
            )
        )
        factories.update(
            {
                "frozen_sentence_embedding": (
                    lambda seed: FrozenSentenceEmbeddingClassifier(
                        sentence_encoder,
                        random_state=seed,
                    )
                ),
                "frozen_sentence_embedding_hybrid": (
                    lambda seed: FrozenSentenceEmbeddingClassifier(
                        sentence_encoder,
                        include_transparent_features=True,
                        random_state=seed,
                    )
                ),
            }
        )
        sentence_encoder_manifest = {
            "model": args.sentence_model,
            "revision": args.sentence_model_revision,
            "weights_frozen": True,
            "local_files_only": bool(args.local_files_only),
        }
    method_outputs: dict[str, tuple[list[str], list[float]]] = {}
    for name, factory in factories.items():
        predictions, support_scores, folds = _cross_validated_predictions(
            pairs,
            splitter=splitter,
            factory=factory,
            random_state=args.random_state,
        )
        methods[name] = {
            "metrics": _metrics(labels, predictions),
            "retrieval": _task_retrieval_metrics(
                tasks,
                pairs,
                predictions,
                support_scores,
            ),
            "folds": folds,
        }
        method_outputs[name] = (predictions, support_scores)
    pure_name = "frozen_sentence_embedding"
    hybrid_name = "frozen_sentence_embedding_hybrid"
    if pure_name in method_outputs and hybrid_name in method_outputs:
        pure_predictions, _ = method_outputs[pure_name]
        _, hybrid_scores = method_outputs[hybrid_name]
        methods["frozen_embedding_two_stage"] = {
            "metrics": _metrics(labels, pure_predictions),
            "retrieval": _task_retrieval_metrics(
                tasks,
                pairs,
                pure_predictions,
                hybrid_scores,
            ),
            "folds": methods[pure_name]["folds"],
            "components": {
                "support_classification": pure_name,
                "candidate_ranking": hybrid_name,
                "threshold_tuning": False,
            },
        }
    selected = max(
        factories,
        key=lambda name: (
            float(methods[name]["metrics"]["macro_f1"]),
            float(methods[name]["metrics"]["balanced_accuracy"]),
        ),
    )
    report = {
        "schema_version": 1,
        "experiment": "candidate_complete_three_class_grouped_comparison",
        "dataset": args.dataset_dir.name,
        "tasks": len(tasks),
        "pairs": len(pairs),
        "evaluation_groups": len(
            {str(pair["evaluation_group"]) for pair in pairs}
        ),
        "label_counts": dict(Counter(labels)),
        "candidate_label_coverage_counts": dict(
            Counter(
                str(task.get("candidate_label_coverage", "legacy"))
                for task in tasks
            )
        ),
        "input_sha256": {
            "tasks": _sha256(task_path),
            "pairs": _sha256(pair_path),
        },
        "protocol": (
            f"{args.folds}-fold stratified task-group cross-validation. No holdout, "
            "reserve, threshold tuning, or product inference is used."
        ),
        "sentence_encoder": sentence_encoder_manifest,
        "methods": methods,
        "selected_development_method": selected,
        "selected_operational_candidate": (
            "frozen_embedding_two_stage"
            if "frozen_embedding_two_stage" in methods
            else selected
        ),
        "promotion_eligible": False,
        "next_gate": (
            "Complete additional candidate-label batches, then compare a frozen "
            "sentence-embedding baseline on the same grouped protocol."
        ),
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, result in methods.items():
        metrics = result["metrics"]
        print(
            f"{name}: macro F1={metrics['macro_f1']:.3f}, "
            f"balanced accuracy={metrics['balanced_accuracy']:.3f}, "
            f"Recall@1={result['retrieval']['recall_at_1']:.3f}"
        )
    print(f"Development winner: {selected}")
    print("Promotion eligible: no")
    print(f"Report: {args.report_path}")


if __name__ == "__main__":
    main()
