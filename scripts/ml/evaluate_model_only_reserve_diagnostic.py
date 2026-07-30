"""Compare two model-only reviews and run a non-promotional reserve diagnostic."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
from sklearn.metrics import cohen_kappa_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
BASE = PROJECT_ROOT / "data" / "ml" / "annotations" / "real_reserve_v3"
MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "models"
    / "evidence_semantic_blend_candidate_v2.joblib"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "ml"
    / "generated"
    / "model_only_reserve_v3_diagnostic.json"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import latest_task_states, load_jsonl, validate_queue  # noqa: E402
from ml.evidence_semantic_blend import (  # noqa: E402
    SemanticConceptBlendScorer,
    evaluate_ranker_with_fixed_gate,
)
from ml.real_holdout_evaluation import (  # noqa: E402
    evaluate_scored_holdout,
    score_holdout_tasks,
)
from ml.real_review import reviewer_packet_manifest  # noqa: E402


FINAL_LABELS = {"Direct", "Partial", "No Support"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_states(
    tasks: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    reviewer: str,
) -> dict[str, dict[str, Any]]:
    states = latest_task_states(events)
    task_ids = {str(task["task_id"]) for task in tasks}
    if set(states) != task_ids:
        raise SystemExit(f"{reviewer} does not exactly cover all reserve tasks.")
    by_id = {str(task["task_id"]): task for task in tasks}
    for task_id, state in states.items():
        label = str(state.get("support_label", ""))
        selected = state.get("selected_candidate_id")
        candidates = {
            str(candidate["candidate_id"])
            for candidate in by_id[task_id]["candidates"]
        }
        if label not in FINAL_LABELS:
            raise SystemExit(f"{reviewer} has a non-final label for {task_id}.")
        if label == "No Support" and selected is not None:
            raise SystemExit(f"{reviewer} selected evidence for No Support.")
        if label != "No Support" and str(selected) not in candidates:
            raise SystemExit(f"{reviewer} selected invalid evidence for {task_id}.")
    return states


def _decision(state: dict[str, Any]) -> tuple[str, str | None]:
    selected = state.get("selected_candidate_id")
    return (
        str(state["support_label"]),
        str(selected) if selected is not None else None,
    )


def _consensus_tasks(
    tasks: list[dict[str, Any]],
    a_states: dict[str, dict[str, Any]],
    b_states: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    consensus: list[dict[str, Any]] = []
    disagreements: list[str] = []
    for task in tasks:
        task_id = str(task["task_id"])
        if _decision(a_states[task_id]) != _decision(b_states[task_id]):
            disagreements.append(task_id)
            continue
        label, selected = _decision(a_states[task_id])
        consensus.append(
            {
                **task,
                "support_label": label,
                "selected_candidate_id": selected,
            }
        )
    return consensus, disagreements


def _validate_artifact(artifact: Any) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise SystemExit("Candidate artifact must be an object.")
    if artifact.get("model_type") != (
        "semantic_concept_blend_with_fixed_tfidf_gate"
    ):
        raise SystemExit("Unexpected diagnostic candidate type.")
    if not isinstance(artifact.get("ranker"), SemanticConceptBlendScorer):
        raise SystemExit("Candidate semantic ranker is invalid.")
    if not callable(getattr(artifact.get("gate"), "score", None)):
        raise SystemExit("Candidate lexical gate is invalid.")
    return artifact


def main() -> None:
    source = validate_queue(load_jsonl(BASE / "real_reserve_queue_v3.jsonl"))
    reviewer_a_queue = validate_queue(
        load_jsonl(BASE / "real_reserve_reviewer_a_queue_v3.jsonl")
    )
    reviewer_b_queue = validate_queue(
        load_jsonl(BASE / "real_reserve_reviewer_b_queue_v3.jsonl")
    )
    reviewer_packet_manifest(
        source,
        {"reviewer_a": reviewer_a_queue, "reviewer_b": reviewer_b_queue},
    )
    a_states = _validated_states(
        source,
        load_jsonl(BASE / "real_reserve_model_reviewer_a_decisions_v3.jsonl"),
        reviewer="Model Reviewer A",
    )
    b_states = _validated_states(
        source,
        load_jsonl(BASE / "real_reserve_model_reviewer_b_decisions_v3.jsonl"),
        reviewer="Model Reviewer B",
    )
    task_ids = [str(task["task_id"]) for task in source]
    labels_a = [str(a_states[task_id]["support_label"]) for task_id in task_ids]
    labels_b = [str(b_states[task_id]["support_label"]) for task_id in task_ids]
    label_agreements = sum(
        label_a == label_b
        for label_a, label_b in zip(labels_a, labels_b, strict=True)
    )
    consensus, disagreements = _consensus_tasks(source, a_states, b_states)
    if not consensus:
        raise SystemExit("No exact model-review consensus tasks are available.")

    manifest = json.loads(
        (BASE / "real_reserve_manifest_v3.json").read_text(encoding="utf-8")
    )
    if _sha256(MODEL_PATH) != manifest["precommitted_candidate_sha256"]:
        raise SystemExit("Precommitted candidate checksum mismatch.")
    artifact = _validate_artifact(joblib.load(MODEL_PATH))
    gate = artifact["gate"]
    ranker = artifact["ranker"]
    gate_scores = score_holdout_tasks(consensus, score=gate.score)
    baseline = evaluate_scored_holdout(
        consensus,
        gate_scores,
        threshold=float(artifact["gate_threshold"]),
    )
    candidate = evaluate_ranker_with_fixed_gate(
        consensus,
        score_holdout_tasks(consensus, score=ranker.score),
        gate_scores,
        gate_threshold=float(artifact["gate_threshold"]),
        ranking_floor=float(artifact["ranking_floor"]),
    )
    baseline_metrics = baseline["retrieval"]
    candidate_metrics = candidate["retrieval"]
    report = {
        "schema_version": 1,
        "dataset": "real_reserve_v3_model_only_diagnostic",
        "tasks": len(source),
        "human_labels": 0,
        "reviewer_a_label_counts": dict(Counter(labels_a)),
        "reviewer_b_label_counts": dict(Counter(labels_b)),
        "label_agreements": label_agreements,
        "label_agreement_rate": label_agreements / len(source),
        "label_cohen_kappa": float(cohen_kappa_score(labels_a, labels_b)),
        "exact_consensus_tasks": len(consensus),
        "exact_consensus_rate": len(consensus) / len(source),
        "disagreement_tasks": disagreements,
        "consensus_label_counts": dict(
            Counter(str(task["support_label"]) for task in consensus)
        ),
        "precommitted_candidate_sha256": manifest[
            "precommitted_candidate_sha256"
        ],
        "fixed_tfidf_baseline": baseline,
        "semantic_blend_candidate": candidate,
        "diagnostic_direction": {
            "balanced_accuracy_delta": (
                float(candidate_metrics["task_balanced_accuracy"])
                - float(baseline_metrics["task_balanced_accuracy"])
            ),
            "recall_at_1_delta": (
                float(candidate_metrics["recall_at_1"])
                - float(baseline_metrics["recall_at_1"])
            ),
            "recall_at_3_delta": (
                float(candidate_metrics["recall_at_3"])
                - float(baseline_metrics["recall_at_3"])
            ),
            "mean_reciprocal_rank_delta": (
                float(candidate_metrics["mean_reciprocal_rank"])
                - float(baseline_metrics["mean_reciprocal_rank"])
            ),
            "supported_success_delta": (
                float(candidate_metrics["supported_task_success_rate"])
                - float(baseline_metrics["supported_task_success_rate"])
            ),
            "no_support_rejection_delta": (
                float(candidate_metrics["no_support_rejection_rate"])
                - float(baseline_metrics["no_support_rejection_rate"])
            ),
        },
        "diagnostic_flags": (
            ["recall_at_3_regressed"]
            if float(candidate_metrics["recall_at_3"])
            < float(baseline_metrics["recall_at_3"])
            else []
        ),
        "promotion_eligible": False,
        "training_use": "prohibited",
        "limitation": (
            "Both labels were produced by model reviewers. Exact-consensus filtering "
            "favours easier cases and cannot replace independent human gold."
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Label agreement: {label_agreements}/{len(source)}")
    print(f"Exact consensus: {len(consensus)}/{len(source)}")
    print(f"Cohen kappa: {report['label_cohen_kappa']:.3f}")
    print(
        "Balanced accuracy: "
        f"{baseline_metrics['task_balanced_accuracy']:.3f} -> "
        f"{candidate_metrics['task_balanced_accuracy']:.3f}"
    )
    print(
        f"Recall@1: {baseline_metrics['recall_at_1']:.3f} -> "
        f"{candidate_metrics['recall_at_1']:.3f}"
    )
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
