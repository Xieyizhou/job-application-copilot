"""Freeze the validation-selected evidence candidate before reserve review."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DATASET_DIR = (
    PROJECT_ROOT / "data" / "ml" / "processed" / "reviewed_evidence_training_v3"
)
CALIBRATION_REPORT = (
    PROJECT_ROOT / "reports" / "ml" / "generated" / "real_validation_v1_calibration.json"
)
MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "models"
    / "evidence_validation_candidate_v1.joblib"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.evidence_artifact import (  # noqa: E402
    EVIDENCE_ARTIFACT_SCHEMA_VERSION,
    SUPPORTED_ARTIFACT_MODEL_TYPES,
    training_corpus_fingerprint,
)
from ml.evidence_models import fit_selected_reranker  # noqa: E402


def main() -> None:
    if MODEL_PATH.exists():
        raise SystemExit("Validation candidate artifact exists; refusing overwrite.")
    report = json.loads(CALIBRATION_REPORT.read_text(encoding="utf-8"))
    selection = dict(report["selection"])
    if selection.get("status") != "candidate_for_fresh_reserve_test":
        raise SystemExit("Validation did not approve a candidate for reserve testing.")
    method = str(selection["selected_method"])
    if method not in SUPPORTED_ARTIFACT_MODEL_TYPES:
        raise SystemExit(f"Selected method is not artifact capable: {method}")
    tasks = load_jsonl(DATASET_DIR / "annotated_tasks.jsonl")
    pairs = load_jsonl(DATASET_DIR / "training_pairs.jsonl")
    requirements = [str(pair["requirement"]) for pair in pairs]
    evidence = [str(pair["evidence"]) for pair in pairs]
    labels = [int(pair["binary_label"]) for pair in pairs]
    model = fit_selected_reranker(
        method,
        requirements,
        evidence,
        labels,
        tasks,
        random_state=42,
    )
    fingerprint = training_corpus_fingerprint(tasks, pairs)
    threshold = float(selection["selected_threshold"])
    artifact = {
        "schema_version": EVIDENCE_ARTIFACT_SCHEMA_VERSION,
        "model_type": method,
        "model": model,
        "threshold": threshold,
        "metadata": {
            "model_version": f"real-validation-v1-{method}",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_source": "reviewed_evidence_training_v3",
            "training_task_count": len(tasks),
            "training_pair_count": len(pairs),
            "training_corpus_fingerprint": fingerprint,
            "validation_dataset": report["dataset"],
            "validation_gold_sha256": report["gold_sha256"],
            "validation_report": str(CALIBRATION_REPORT),
            "status": "frozen_candidate_pending_fresh_reserve",
            "threshold": threshold,
            "selection": selection,
            "feature_manifest": model.feature_manifest(),
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"Frozen candidate: {method}")
    print(f"Threshold: {threshold:.12f}")
    print(f"Training fingerprint: {fingerprint}")
    print(f"Validation checksum: {report['gold_sha256']}")
    print(f"Artifact: {MODEL_PATH}")


if __name__ == "__main__":
    main()
