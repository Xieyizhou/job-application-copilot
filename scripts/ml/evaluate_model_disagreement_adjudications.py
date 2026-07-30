"""Evaluate a fixed candidate on human labels from the model-disagreement slice.

The slice was selected because two diagnostic model reviewers disagreed.  It is
therefore useful for error analysis only, not model selection, promotion, or
training.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
BASE = PROJECT_ROOT / "data" / "ml" / "annotations" / "real_reserve_v3"
MODEL_PATH = (
    PROJECT_ROOT / "data" / "ml" / "models" / "evidence_semantic_blend_candidate_v2.joblib"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "ml"
    / "generated"
    / "model_disagreement_adjudication_v3_diagnostic.json"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import latest_task_states, load_jsonl, validate_queue  # noqa: E402
from ml.evidence_semantic_blend import evaluate_ranker_with_fixed_gate  # noqa: E402
from ml.real_holdout_evaluation import (  # noqa: E402
    evaluate_scored_holdout,
    score_holdout_tasks,
)


FINAL_LABELS = {"Direct", "Partial", "No Support"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _human_gold_tasks() -> list[dict[str, Any]]:
    queue = validate_queue(
        load_jsonl(BASE / "real_reserve_model_diagnostic_adjudication_queue_v3.jsonl")
    )
    states = latest_task_states(
        load_jsonl(
            BASE / "real_reserve_model_diagnostic_adjudication_decisions_v3.jsonl"
        )
    )
    task_ids = {str(task["task_id"]) for task in queue}
    if set(states) != task_ids:
        raise SystemExit("Human adjudication is incomplete; diagnostic evaluation blocked.")
    gold: list[dict[str, Any]] = []
    for task in queue:
        state = states[str(task["task_id"])]
        label = str(state.get("support_label", ""))
        selected = state.get("selected_candidate_id")
        candidate_ids = {str(candidate["candidate_id"]) for candidate in task["candidates"]}
        if label not in FINAL_LABELS:
            raise SystemExit("Human adjudication has a non-final support label.")
        if label == "No Support" and selected is not None:
            raise SystemExit("No Support cannot select resume evidence.")
        if label != "No Support" and str(selected) not in candidate_ids:
            raise SystemExit("Human adjudication selected invalid resume evidence.")
        gold.append({**task, "support_label": label, "selected_candidate_id": selected})
    return gold


def _artifact() -> dict[str, Any]:
    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict):
        raise SystemExit("Candidate artifact must be an object.")
    if artifact.get("model_type") != "semantic_concept_blend_with_fixed_tfidf_gate":
        raise SystemExit("Unexpected candidate artifact.")
    if not callable(getattr(artifact.get("gate"), "score", None)):
        raise SystemExit("Candidate lexical gate is invalid.")
    if not callable(getattr(artifact.get("ranker"), "score", None)):
        raise SystemExit("Candidate semantic ranker is invalid.")
    return artifact


def main() -> None:
    tasks = _human_gold_tasks()
    artifact = _artifact()
    gate_scores = score_holdout_tasks(tasks, score=artifact["gate"].score)
    baseline = evaluate_scored_holdout(
        tasks,
        gate_scores,
        threshold=float(artifact["gate_threshold"]),
    )
    candidate = evaluate_ranker_with_fixed_gate(
        tasks,
        score_holdout_tasks(tasks, score=artifact["ranker"].score),
        gate_scores,
        gate_threshold=float(artifact["gate_threshold"]),
        ranking_floor=float(artifact["ranking_floor"]),
    )
    report = {
        "schema_version": 1,
        "dataset": "real_reserve_v3_human_adjudicated_model_disagreement_slice",
        "tasks": len(tasks),
        "label_counts": dict(Counter(str(task["support_label"]) for task in tasks)),
        "candidate_sha256": _sha256(MODEL_PATH),
        "fixed_tfidf_baseline": baseline,
        "semantic_blend_candidate": candidate,
        "diagnostic_direction": {
            "balanced_accuracy_delta": (
                float(candidate["retrieval"]["task_balanced_accuracy"])
                - float(baseline["retrieval"]["task_balanced_accuracy"])
            ),
            "recall_at_1_delta": (
                float(candidate["retrieval"]["recall_at_1"])
                - float(baseline["retrieval"]["recall_at_1"])
            ),
        },
        "promotion_eligible": False,
        "training_use": "prohibited",
        "limitation": (
            "This four-task slice was selected after model disagreement. It is an "
            "error-analysis slice, not an independent or representative evaluation set."
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Human-adjudicated diagnostic tasks: {len(tasks)}")
    print(
        "Balanced accuracy: "
        f"{baseline['retrieval']['task_balanced_accuracy']:.3f} -> "
        f"{candidate['retrieval']['task_balanced_accuracy']:.3f}"
    )
    print(
        "Recall@1: "
        f"{baseline['retrieval']['recall_at_1']:.3f} -> "
        f"{candidate['retrieval']['recall_at_1']:.3f}"
    )
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
