"""Freeze human-authority operational v2 development gold."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
BASE = (
    PROJECT_ROOT / "data" / "ml" / "annotations"
    / "operational_development_v2"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "ml" / "processed"
    / "operational_development_v2"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import latest_task_states, load_jsonl, load_queue  # noqa: E402
from ml.annotation_dataset import (  # noqa: E402
    build_annotated_tasks,
    build_training_pairs,
)
from ml.operational_review import operational_review_agreement  # noqa: E402
from ml.real_holdout import jsonl_bytes, sha256_bytes  # noqa: E402
from ml.real_review import reviewer_packet_manifest  # noqa: E402


def main() -> None:
    outputs = {
        "gold": OUTPUT_DIR / "annotated_tasks.jsonl",
        "pairs": OUTPUT_DIR / "training_pairs.jsonl",
        "manifest": OUTPUT_DIR / "manifest.json",
    }
    if any(path.exists() for path in outputs.values()):
        raise SystemExit("Operational v2 gold exists; refusing overwrite.")
    source = load_queue(BASE / "queue.jsonl")
    reviewer_packet_manifest(
        source,
        {
            "reviewer_a": load_queue(BASE / "reviewer_a_queue.jsonl"),
            "reviewer_b": load_queue(BASE / "reviewer_b_queue.jsonl"),
        },
    )
    reviewer_a = latest_task_states(
        load_jsonl(BASE / "reviewer_a_decisions.jsonl")
    )
    reviewer_b = latest_task_states(
        load_jsonl(BASE / "reviewer_b_decisions.jsonl")
    )
    agreement = operational_review_agreement(
        source,
        reviewer_a,
        reviewer_b,
    )
    gold = build_annotated_tasks(
        source,
        reviewer_a,
        random_state=20260730,
        require_complete=True,
        require_candidate_labels=True,
    )
    pairs = build_training_pairs(gold)
    gold_payload = jsonl_bytes(gold)
    pair_payload = jsonl_bytes(pairs)
    manifest = {
        "schema_version": 1,
        "dataset_id": "operational_development_v2",
        "dataset_role": "successor_training_and_development_only",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        **agreement,
        "gold_tasks": len(gold),
        "training_pairs": len(pairs),
        "positive_pairs": sum(int(pair["binary_label"]) for pair in pairs),
        "negative_pairs": sum(
            int(pair["binary_label"]) == 0 for pair in pairs
        ),
        "gold_sha256": sha256_bytes(gold_payload),
        "training_pairs_sha256": sha256_bytes(pair_payload),
        "reviewer_b_role": "independent_rejected_shadow_v1_diagnostic",
        "reviewer_b_can_define_gold": False,
        "disagreement_policy": "reviewer_a_human_decision_is_final",
        "contains_real_candidate_profiles": False,
        "eligible_for_final_evaluation": False,
        "product_integration_allowed": False,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs["gold"].write_bytes(gold_payload)
    outputs["pairs"].write_bytes(pair_payload)
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Gold tasks: {manifest['gold_tasks']}")
    print(f"Training pairs: {manifest['training_pairs']}")
    print(
        "Candidate agreement: "
        f"{manifest['candidate_label_agreements']}/"
        f"{manifest['candidate_judgments']}"
    )
    print(f"Disagreement tasks: {manifest['disagreement_tasks']}")
    print(f"Gold SHA-256: {manifest['gold_sha256']}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
