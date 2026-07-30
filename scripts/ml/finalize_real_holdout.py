"""Freeze completed real-text reviews and human adjudications into local gold."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_DIR = PROJECT_ROOT / "data" / "ml" / "annotations" / "real_holdout_v1"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.real_holdout import finalize_real_holdout, jsonl_bytes, sha256_bytes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-dir", type=Path, default=DEFAULT_DIR)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    base = parse_args().holdout_dir
    paths = {
        "reviewer_a_queue": base / "real_holdout_reviewer_a_queue_v1.jsonl",
        "reviewer_b_queue": base / "real_holdout_reviewer_b_queue_v1.jsonl",
        "reviewer_a_events": base / "reviewer_a_decisions_v1.jsonl",
        "reviewer_b_events": base / "reviewer_b_decisions_v1.jsonl",
        "adjudication_queue": base / "real_holdout_adjudication_queue_v1.jsonl",
        "adjudication_events": base / "real_holdout_adjudication_decisions_v1.jsonl",
    }
    output = base / "real_holdout_gold_v1.jsonl"
    manifest_path = base / "real_holdout_frozen_manifest_v1.json"
    if output.exists() or manifest_path.exists():
        raise SystemExit("Frozen holdout already exists; refusing to overwrite it.")
    gold, report = finalize_real_holdout(
        load_jsonl(paths["reviewer_a_queue"]),
        load_jsonl(paths["reviewer_b_queue"]),
        load_jsonl(paths["reviewer_a_events"]),
        load_jsonl(paths["reviewer_b_events"]),
        load_jsonl(paths["adjudication_queue"]),
        load_jsonl(paths["adjudication_events"]),
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
        "evaluation_policy": (
            "Never use this holdout for fitting, threshold selection, model selection, "
            "prompt refinement, or training-data filtering."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Frozen real holdout tasks: {len(gold)}")
    print(f"Human adjudications: {report['human_adjudications']}")
    print(f"Gold SHA-256: {manifest['gold_sha256']}")
    print(f"Gold: {output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
