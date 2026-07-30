"""Freeze reviewed acceptance tasks into a local training supplement."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
BASE = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "annotations"
    / "acceptance_supplement_v1"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.real_holdout import finalize_real_reviews, jsonl_bytes, sha256_bytes  # noqa: E402


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    paths = {
        "reviewer_a_queue": (
            BASE / "acceptance_supplement_reviewer_a_queue_v1.jsonl"
        ),
        "reviewer_b_queue": (
            BASE / "acceptance_supplement_reviewer_b_queue_v1.jsonl"
        ),
        "reviewer_a_events": (
            BASE / "acceptance_supplement_reviewer_a_decisions_v1.jsonl"
        ),
        "reviewer_b_events": (
            BASE / "acceptance_supplement_reviewer_b_decisions_v1.jsonl"
        ),
        "adjudication_queue": (
            BASE / "acceptance_supplement_adjudication_queue_v1.jsonl"
        ),
        "adjudication_events": (
            BASE / "acceptance_supplement_adjudication_decisions_v1.jsonl"
        ),
    }
    output = BASE / "acceptance_supplement_gold_v1.jsonl"
    manifest_path = BASE / "acceptance_supplement_frozen_manifest_v1.json"
    if output.exists() or manifest_path.exists():
        raise SystemExit("Frozen acceptance supplement exists; refusing overwrite.")
    gold, report = finalize_real_reviews(
        load_jsonl(paths["reviewer_a_queue"]),
        load_jsonl(paths["reviewer_b_queue"]),
        load_jsonl(paths["reviewer_a_events"]),
        load_jsonl(paths["reviewer_b_events"]),
        load_jsonl(paths["adjudication_queue"]),
        load_jsonl(paths["adjudication_events"]),
        dataset_role="training_supplement",
    )
    payload = jsonl_bytes(gold)
    output.write_bytes(payload)
    manifest = {
        **report,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "gold_file": output.name,
        "gold_sha256": sha256_bytes(payload),
        "input_sha256": {
            name: _file_sha256(path) for name, path in sorted(paths.items())
        },
        "selection_policy": (
            "Active-learning training supplement. Prohibited for threshold "
            "selection, model comparison, reserve, or final evaluation."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Frozen training-supplement tasks: {len(gold)}")
    print(f"Human adjudications: {report['human_adjudications']}")
    print(f"Gold SHA-256: {manifest['gold_sha256']}")
    print(f"Gold: {output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
