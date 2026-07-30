"""Run the one-time reserve v4 test for the precommitted candidate."""

from __future__ import annotations

from collections import Counter
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
BASE = PROJECT_ROOT / "data" / "ml" / "annotations" / "real_reserve_v4"
TRAINING_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "processed"
    / "reviewed_evidence_training_v4_batch4"
)
CANDIDATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "models"
    / "evidence_sentence_embedding_two_stage_v1.json"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "ml"
    / "generated"
    / "sentence_embedding_real_reserve_v4.json"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.evidence_decision import paired_stratified_bootstrap_delta  # noqa: E402
from ml.evidence_grouped_evaluation import task_retrieval_evaluation  # noqa: E402
from ml.evidence_multiclass import SUPPORT_CLASSES, MulticlassEvidenceReranker  # noqa: E402
from ml.evidence_sentence_embedding import (  # noqa: E402
    CachedSentenceEncoder,
    FrozenSentenceEmbeddingClassifier,
    load_sentence_encoder,
)
from ml.real_holdout_evaluation import assert_holdout_isolated  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flatten_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, str]]:
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


def _evaluate(
    model: Any,
    tasks: list[dict[str, Any]],
    pairs: list[dict[str, str]],
) -> tuple[dict[str, Any], list[str], list[float], dict[str, bool]]:
    probabilities = np.asarray(
        model.predict_class_proba(
            [pair["requirement"] for pair in pairs],
            [pair["evidence"] for pair in pairs],
        ),
        dtype=np.float64,
    )
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
    return metrics, [str(value) for value in predictions], support_scores, outcomes


def _selection_gate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    point_estimate_wins = (
        candidate["task_balanced_accuracy"]
        > baseline["task_balanced_accuracy"]
        and candidate["recall_at_1"] >= baseline["recall_at_1"]
        and candidate["no_support_rejection_rate"]
        >= baseline["no_support_rejection_rate"]
    )
    passed = point_estimate_wins and float(bootstrap["lower_90"]) > 0.0
    return {
        "reference": "lsa_transparent_multiclass",
        "point_estimate_wins": point_estimate_wins,
        "paired_stratified_bootstrap": bootstrap,
        "passed": passed,
        "status": (
            "eligible_for_failure_review_and_shadow_mode"
            if passed
            else "candidate_rejected_on_fresh_reserve"
        ),
    }


def _failure_review(
    tasks: list[dict[str, Any]],
    pairs: list[dict[str, str]],
    predictions: list[str],
    support_scores: list[float],
    baseline_outcomes: dict[str, bool],
    candidate_outcomes: dict[str, bool],
) -> dict[str, Any]:
    pair_indices: defaultdict[str, list[int]] = defaultdict(list)
    for index, pair in enumerate(pairs):
        pair_indices[str(pair["task_id"])].append(index)
    failures: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        if candidate_outcomes[task_id]:
            continue
        indices = pair_indices[task_id]
        ordered = sorted(
            indices,
            key=lambda index: (
                -support_scores[index],
                str(pairs[index]["pair_id"]),
            ),
        )
        accepted = any(predictions[index] != "No Support" for index in indices)
        gold_label = str(task["support_label"])
        if gold_label == "No Support":
            failure_type = "false_accept"
            gold_rank = None
        else:
            selected_id = str(task["selected_candidate_id"])
            selected_evidence = next(
                str(candidate["evidence"])
                for candidate in task["candidates"]
                if str(candidate["candidate_id"]) == selected_id
            )
            gold_index = next(
                index
                for index in indices
                if pairs[index]["evidence"] == selected_evidence
            )
            gold_rank = ordered.index(gold_index) + 1
            failure_type = "false_reject" if not accepted else "wrong_rank"
        failures.append(
            {
                "task_id": task_id,
                "role_family": str(task["role_family"]),
                "gold_label": gold_label,
                "failure_type": failure_type,
                "gold_rank": gold_rank,
                "top_candidate_id": str(
                    pairs[ordered[0]]["pair_id"]
                ).rsplit(":", maxsplit=1)[-1],
            }
        )
    paired_counts = Counter(
        (
            "both_pass"
            if baseline_outcomes[task_id] and candidate_outcomes[task_id]
            else (
                "candidate_improved"
                if not baseline_outcomes[task_id]
                and candidate_outcomes[task_id]
                else (
                    "candidate_regressed"
                    if baseline_outcomes[task_id]
                    and not candidate_outcomes[task_id]
                    else "both_fail"
                )
            )
        )
        for task_id in baseline_outcomes
    )
    return {
        "candidate_failures": len(failures),
        "failure_type_counts": dict(
            Counter(str(item["failure_type"]) for item in failures)
        ),
        "failure_role_counts": dict(
            Counter(str(item["role_family"]) for item in failures)
        ),
        "paired_outcome_counts": dict(paired_counts),
        "tasks": failures,
        "review_policy": (
            "Diagnostic review only. Do not retune this candidate from these "
            "reserve failures."
        ),
    }


def _validate_commitments(
    manifest: dict[str, Any],
    specification: dict[str, Any],
) -> None:
    if _sha256(CANDIDATE_PATH) != manifest["precommitted_candidate_sha256"]:
        raise SystemExit("Precommitted candidate checksum mismatch.")
    construction = json.loads(
        (BASE / "real_reserve_manifest_v4.json").read_text(encoding="utf-8")
    )
    if (
        construction["precommitted_candidate_sha256"]
        != manifest["precommitted_candidate_sha256"]
    ):
        raise SystemExit("Construction and frozen candidate commitments differ.")
    training = specification["training"]
    if (
        _sha256(TRAINING_DIR / "annotated_tasks.jsonl")
        != training["annotated_tasks_sha256"]
        or _sha256(TRAINING_DIR / "training_pairs.jsonl")
        != training["training_pairs_sha256"]
    ):
        raise SystemExit("Frozen training-corpus commitment mismatch.")
    for filename, expected in specification["source_code_sha256"].items():
        path = SRC_DIR / "ml" / filename
        if _sha256(path) != expected:
            raise SystemExit(f"Frozen candidate code changed: {filename}.")
    development = specification["development_evidence"]
    development_path = (
        PROJECT_ROOT / "reports" / "ml" / "generated" / development["report_file"]
    )
    if _sha256(development_path) != development["report_sha256"]:
        raise SystemExit("Development-report commitment mismatch.")


def main() -> None:
    gold_path = BASE / "real_reserve_gold_v4.jsonl"
    manifest_path = BASE / "real_reserve_frozen_manifest_v4.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _sha256(gold_path) != manifest["gold_sha256"]:
        raise SystemExit("Frozen reserve checksum mismatch.")
    specification = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
    _validate_commitments(manifest, specification)

    tasks = load_jsonl(gold_path)
    training_tasks = load_jsonl(TRAINING_DIR / "annotated_tasks.jsonl")
    training_pairs = load_jsonl(TRAINING_DIR / "training_pairs.jsonl")
    assert_holdout_isolated(tasks, training_tasks, training_pairs)
    requirements = [str(pair["requirement"]) for pair in training_pairs]
    evidence = [str(pair["evidence"]) for pair in training_pairs]
    labels = [str(pair["support_label"]) for pair in training_pairs]
    pairs = _flatten_tasks(tasks)

    baseline_model = MulticlassEvidenceReranker(
        random_state=20260729
    ).fit(requirements, evidence, labels)
    baseline, _, _, baseline_outcomes = _evaluate(
        baseline_model,
        tasks,
        pairs,
    )
    encoder_spec = specification["sentence_encoder"]
    encoder = CachedSentenceEncoder(
        load_sentence_encoder(
            model_name=encoder_spec["model"],
            revision=encoder_spec["revision"],
            local_files_only=True,
        )
    )
    pure_model = FrozenSentenceEmbeddingClassifier(
        encoder,
        random_state=20260729,
    ).fit(requirements, evidence, labels)
    hybrid_model = FrozenSentenceEmbeddingClassifier(
        encoder,
        include_transparent_features=True,
        random_state=20260729,
    ).fit(requirements, evidence, labels)
    _, pure_predictions, _, _ = _evaluate(pure_model, tasks, pairs)
    _, _, hybrid_scores, _ = _evaluate(hybrid_model, tasks, pairs)
    candidate, candidate_outcomes = task_retrieval_evaluation(
        tasks,
        pairs,
        pure_predictions,
        hybrid_scores,
    )
    task_ids = [str(task["task_id"]) for task in tasks]
    bootstrap = paired_stratified_bootstrap_delta(
        [str(task["support_label"]) != "No Support" for task in tasks],
        [baseline_outcomes[task_id] for task_id in task_ids],
        [candidate_outcomes[task_id] for task_id in task_ids],
        random_state=20260729,
    )
    gate = _selection_gate(baseline, candidate, bootstrap)
    failure_review = _failure_review(
        tasks,
        pairs,
        pure_predictions,
        hybrid_scores,
        baseline_outcomes,
        candidate_outcomes,
    )
    report = {
        "schema_version": 1,
        "dataset": "real_reserve_v4",
        "dataset_role": "one_time_source_isolated_reserve",
        "tasks": len(tasks),
        "pairs": len(pairs),
        "label_counts": dict(
            Counter(str(task["support_label"]) for task in tasks)
        ),
        "gold_sha256": manifest["gold_sha256"],
        "candidate_sha256": manifest["precommitted_candidate_sha256"],
        "source_and_content_isolation_verified": True,
        "reviewer_exact_agreement_rate": manifest.get(
            "reviewer_exact_agreement_rate"
        ),
        "human_adjudications": manifest.get("human_adjudications"),
        "human_agreement_audits": manifest.get("human_agreement_audits"),
        "fixed_lsa_baseline": baseline,
        "precommitted_frozen_embedding_two_stage": candidate,
        "selection_gate": gate,
        "failure_review": failure_review,
        "threshold_tuning": False,
        "product_integration_allowed": False,
        "limitation": (
            "This is a one-time reserve evaluation. Its labels and failures "
            "cannot be used to retune this candidate. A passing result permits "
            "failure review and shadow-mode planning, not automatic product use."
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, metrics in (("fixed_lsa", baseline), ("candidate", candidate)):
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
    print(f"Selection gate: {gate['status']}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
