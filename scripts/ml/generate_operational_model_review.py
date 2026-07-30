"""Generate a blind model-assisted Reviewer B for operational v2."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
BASE = (
    PROJECT_ROOT / "data" / "ml" / "annotations"
    / "operational_development_v2"
)
QUEUE_PATH = BASE / "reviewer_b_queue.jsonl"
OUTPUT_PATH = BASE / "reviewer_b_decisions.jsonl"
ARTIFACT_PATH = (
    PROJECT_ROOT / "data" / "ml" / "models"
    / "evidence_sentence_embedding_shadow_v1.joblib"
)
SPEC_PATH = (
    PROJECT_ROOT / "data" / "ml" / "models"
    / "evidence_sentence_embedding_two_stage_v1.json"
)
TRAINING_DIR = (
    PROJECT_ROOT / "data" / "ml" / "processed"
    / "reviewed_evidence_training_v4_batch4"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_queue  # noqa: E402
from ml.evidence_sentence_artifact import (  # noqa: E402
    file_sha256,
    predict_two_stage,
    validate_sentence_artifact,
)
from ml.evidence_sentence_embedding import (  # noqa: E402
    CachedSentenceEncoder,
    load_sentence_encoder,
)
from ml.operational_review import model_candidate_decision  # noqa: E402


def main() -> None:
    if OUTPUT_PATH.exists():
        raise SystemExit("Operational Reviewer B decisions exist; refusing overwrite.")
    tasks = load_queue(QUEUE_PATH)
    artifact = validate_sentence_artifact(
        joblib.load(ARTIFACT_PATH),
        candidate_spec_path=SPEC_PATH,
        training_dir=TRAINING_DIR,
    )
    encoder_spec = artifact["metadata"]["sentence_encoder"]
    encoder = CachedSentenceEncoder(
        load_sentence_encoder(
            model_name=encoder_spec["model"],
            revision=encoder_spec["revision"],
            local_files_only=True,
        )
    )
    pairs = [
        (
            str(task["task_id"]),
            str(task["requirement"]),
            str(candidate["evidence"]),
        )
        for task in tasks
        for candidate in task["candidates"]
    ]
    predictions = predict_two_stage(
        artifact,
        encoder,
        [requirement for _, requirement, _ in pairs],
        [evidence for _, _, evidence in pairs],
    )
    events: list[dict[str, object]] = []
    offset = 0
    annotated_at = datetime.now(timezone.utc).isoformat()
    artifact_sha256 = file_sha256(ARTIFACT_PATH)
    for task in tasks:
        count = len(task["candidates"])
        decision = model_candidate_decision(
            task,
            predictions["predictions"][offset : offset + count],
            predictions["hybrid_support_scores"][offset : offset + count],
        )
        task_id = str(task["task_id"])
        events.append(
            {
                "schema_version": 1,
                "event_id": hashlib.sha256(
                    f"{task_id}:{artifact_sha256}".encode()
                ).hexdigest(),
                "task_id": task_id,
                "action": "label",
                **decision,
                "note": (
                    "Independent frozen-model diagnostic; not human gold "
                    "and not eligible for product decisions."
                ),
                "annotated_at": annotated_at,
            }
        )
        offset += count
    OUTPUT_PATH.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )
    print(f"Reviewer B tasks: {len(events)}")
    print(f"Artifact SHA-256: {artifact_sha256}")
    print("Reviewer A decisions loaded: no")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
