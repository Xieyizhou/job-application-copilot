"""Freeze completed development-extension reviews into local development gold."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.real_holdout import finalize_real_reviews, jsonl_bytes, sha256_bytes  # noqa: E402


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version", type=int, choices=(2, 3), default=2)
    parser.add_argument("--annotation-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version = f"v{args.dataset_version}"
    base = args.annotation_dir or (
        PROJECT_ROOT
        / "data"
        / "ml"
        / "annotations"
        / f"real_development_{version}"
    )
    paths = {
        "reviewer_a_queue": base / f"real_development_reviewer_a_queue_{version}.jsonl",
        "reviewer_b_queue": base / f"real_development_reviewer_b_queue_{version}.jsonl",
        "reviewer_a_events": base / f"real_development_reviewer_a_decisions_{version}.jsonl",
        "reviewer_b_events": base / f"real_development_reviewer_b_decisions_{version}.jsonl",
        "adjudication_queue": base / f"real_development_adjudication_queue_{version}.jsonl",
        "adjudication_events": (
            base / f"real_development_adjudication_decisions_{version}.jsonl"
        ),
    }
    output = base / f"real_development_gold_{version}.jsonl"
    manifest_path = base / f"real_development_frozen_manifest_{version}.json"
    if output.exists() or manifest_path.exists():
        raise SystemExit("Frozen development gold exists; refusing overwrite.")
    gold, report = finalize_real_reviews(
        load_jsonl(paths["reviewer_a_queue"]),
        load_jsonl(paths["reviewer_b_queue"]),
        load_jsonl(paths["reviewer_a_events"]),
        load_jsonl(paths["reviewer_b_events"]),
        load_jsonl(paths["adjudication_queue"]),
        load_jsonl(paths["adjudication_events"]),
        dataset_role="development",
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
        "development_policy": (
            "May compare architectures and calibrate thresholds. Must not be "
            "reported as an untouched final generalization set."
        ),
        "dataset_version": args.dataset_version,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Frozen development tasks: {len(gold)}")
    print(f"Human adjudications: {report['human_adjudications']}")
    print(f"Gold SHA-256: {manifest['gold_sha256']}")
    print(f"Gold: {output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
