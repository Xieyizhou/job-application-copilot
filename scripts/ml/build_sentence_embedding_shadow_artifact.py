"""Package the reserve-qualified two-stage model for local shadow use."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
TRAINING_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "processed"
    / "reviewed_evidence_training_v4_batch4"
)
SPEC_PATH = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "models"
    / "evidence_sentence_embedding_two_stage_v1.json"
)
RESERVE_REPORT = (
    PROJECT_ROOT
    / "reports"
    / "ml"
    / "generated"
    / "sentence_embedding_real_reserve_v4.json"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "models"
    / "evidence_sentence_embedding_shadow_v1.joblib"
)
MANIFEST_PATH = OUTPUT_PATH.with_suffix(".manifest.json")
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.evidence_sentence_artifact import (  # noqa: E402
    ARTIFACT_MODEL_TYPE,
    ARTIFACT_SCHEMA_VERSION,
    file_sha256,
    predict_two_stage,
    validate_sentence_artifact,
)
from ml.evidence_grouped_evaluation import task_retrieval_metrics  # noqa: E402
from ml.evidence_sentence_embedding import (  # noqa: E402
    CachedSentenceEncoder,
    FrozenSentenceEmbeddingClassifier,
    load_sentence_encoder,
)


def main() -> None:
    if OUTPUT_PATH.exists() or MANIFEST_PATH.exists():
        raise SystemExit("Shadow artifact exists; refusing overwrite.")
    specification = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    reserve_report = json.loads(RESERVE_REPORT.read_text(encoding="utf-8"))
    gate = reserve_report["selection_gate"]
    if not gate["passed"]:
        raise SystemExit("Reserve gate did not approve shadow packaging.")
    if (
        reserve_report["candidate_sha256"] != file_sha256(SPEC_PATH)
        or reserve_report["product_integration_allowed"]
    ):
        raise SystemExit("Reserve candidate commitment or boundary is invalid.")

    pairs = load_jsonl(TRAINING_DIR / "training_pairs.jsonl")
    requirements = [str(pair["requirement"]) for pair in pairs]
    evidence = [str(pair["evidence"]) for pair in pairs]
    labels = [str(pair["support_label"]) for pair in pairs]
    encoder_spec = specification["sentence_encoder"]
    encoder = CachedSentenceEncoder(
        load_sentence_encoder(
            model_name=encoder_spec["model"],
            revision=encoder_spec["revision"],
            local_files_only=True,
        )
    )
    pure = FrozenSentenceEmbeddingClassifier(
        encoder,
        random_state=20260729,
    ).fit(requirements, evidence, labels)
    hybrid = FrozenSentenceEmbeddingClassifier(
        encoder,
        include_transparent_features=True,
        random_state=20260729,
    ).fit(requirements, evidence, labels)
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_type": ARTIFACT_MODEL_TYPE,
        "pure_classifier": pure.classifier,
        "hybrid_classifier": hybrid.classifier,
        "hybrid_word_scorer": hybrid.word,
        "metadata": {
            "artifact_version": "shadow-v1",
            "packaged_at": datetime.now(timezone.utc).isoformat(),
            "candidate_spec_sha256": file_sha256(SPEC_PATH),
            "reserve_report_sha256": file_sha256(RESERVE_REPORT),
            "reserve_gold_sha256": reserve_report["gold_sha256"],
            "reserve_gate": gate,
            "training": {
                "annotated_tasks_sha256": file_sha256(
                    TRAINING_DIR / "annotated_tasks.jsonl"
                ),
                "training_pairs_sha256": file_sha256(
                    TRAINING_DIR / "training_pairs.jsonl"
                ),
            },
            "training_tasks": reserve_report.get("training_tasks", 207),
            "training_pairs": len(pairs),
            "random_state": 20260729,
            "sentence_encoder": encoder_spec,
            "acceptance": (
                "accept iff any pure-model candidate is not No Support"
            ),
            "ranking": "hybrid P(Direct) + 0.5 * P(Partial)",
            "mode": "local_shadow_only",
            "product_decisions_allowed": False,
        },
    }
    validate_sentence_artifact(
        artifact,
        candidate_spec_path=SPEC_PATH,
        training_dir=TRAINING_DIR,
    )
    reserve_tasks = load_jsonl(
        PROJECT_ROOT
        / "data"
        / "ml"
        / "annotations"
        / "real_reserve_v4"
        / "real_reserve_gold_v4.jsonl"
    )
    reserve_pairs = [
        {
            "pair_id": f"{task['task_id']}:{candidate['candidate_id']}",
            "task_id": str(task["task_id"]),
            "requirement": str(task["requirement"]),
            "evidence": str(candidate["evidence"]),
        }
        for task in reserve_tasks
        for candidate in task["candidates"]
    ]
    predictions = predict_two_stage(
        artifact,
        encoder,
        [pair["requirement"] for pair in reserve_pairs],
        [pair["evidence"] for pair in reserve_pairs],
    )
    reproduced = task_retrieval_metrics(
        reserve_tasks,
        reserve_pairs,
        predictions["predictions"],
        predictions["hybrid_support_scores"],
    )
    expected = reserve_report["precommitted_frozen_embedding_two_stage"]
    if any(abs(float(reproduced[key]) - float(expected[key])) > 1e-12 for key in expected):
        raise SystemExit("Packaged heads do not reproduce the reserve metrics.")
    artifact["metadata"]["reserve_reproduction"] = {
        "verified": True,
        "metrics": reproduced,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, OUTPUT_PATH)
    reloaded = validate_sentence_artifact(
        joblib.load(OUTPUT_PATH),
        candidate_spec_path=SPEC_PATH,
        training_dir=TRAINING_DIR,
    )
    manifest = {
        "schema_version": 1,
        "artifact_file": OUTPUT_PATH.name,
        "artifact_sha256": file_sha256(OUTPUT_PATH),
        "model_type": reloaded["model_type"],
        "metadata": reloaded["metadata"],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Artifact SHA-256: {manifest['artifact_sha256']}")
    print(f"Artifact: {OUTPUT_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
