"""Freeze the precommitted sentence-embedding candidate specification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "ml"
    / "generated"
    / "sentence_embedding_real_development_v3.json"
)
DEFAULT_TRAINING_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "processed"
    / "reviewed_evidence_training_v4_batch4"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "models"
    / "evidence_sentence_embedding_two_stage_v1.json"
)
MODEL_MODULE = PROJECT_ROOT / "src" / "ml" / "evidence_sentence_embedding.py"
METRICS_MODULE = PROJECT_ROOT / "src" / "ml" / "evidence_grouped_evaluation.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_specification(
    report: dict[str, object],
    *,
    report_path: Path,
    training_dir: Path,
) -> dict[str, object]:
    gate = report.get("development_gate")
    if not isinstance(gate, dict) or not gate.get("stable_improvement"):
        raise SystemExit("Development gate is not stable; candidate freeze blocked.")
    if report.get("locked_candidate") != "frozen_embedding_two_stage":
        raise SystemExit("Development report does not lock the expected candidate.")
    encoder = report.get("sentence_encoder")
    if not isinstance(encoder, dict):
        raise SystemExit("Development report lacks sentence encoder metadata.")
    return {
        "schema_version": 1,
        "candidate_id": "frozen_embedding_two_stage_v1",
        "candidate_status": "precommitted_for_one_time_reserve_evaluation",
        "training": {
            "directory": training_dir.name,
            "annotated_tasks_sha256": _sha256(
                training_dir / "annotated_tasks.jsonl"
            ),
            "training_pairs_sha256": _sha256(
                training_dir / "training_pairs.jsonl"
            ),
        },
        "sentence_encoder": encoder,
        "classification": {
            "head": "balanced_logistic_regression",
            "random_state": 20260729,
            "encoder_weights_frozen": True,
            "acceptance": (
                "accept iff any candidate pure-model argmax is not No Support"
            ),
            "ranking": (
                "descending hybrid P(Direct) + 0.5 * P(Partial)"
            ),
        },
        "source_code_sha256": {
            MODEL_MODULE.name: _sha256(MODEL_MODULE),
            METRICS_MODULE.name: _sha256(METRICS_MODULE),
        },
        "development_evidence": {
            "report_file": report_path.name,
            "report_sha256": _sha256(report_path),
            "dataset": report.get("dataset"),
            "methods": report.get("methods"),
            "paired_stratified_bootstrap": gate.get(
                "paired_stratified_bootstrap"
            ),
        },
        "reserve_success_criteria": {
            "candidate_task_balanced_accuracy_gt_lsa": True,
            "candidate_recall_at_1_gte_lsa": True,
            "candidate_no_support_rejection_gte_lsa": True,
            "paired_bootstrap_lower_90_gt_zero": True,
        },
        "threshold_tuning_after_freeze": False,
        "product_integration_allowed": False,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit("Candidate specification exists; refusing overwrite.")
    report = json.loads(args.report_path.read_text(encoding="utf-8"))
    specification = build_specification(
        report,
        report_path=args.report_path,
        training_dir=args.training_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(specification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Candidate: {specification['candidate_id']}")
    print(f"Specification SHA-256: {_sha256(args.output)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
