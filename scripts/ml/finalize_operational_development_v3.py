"""Freeze adjudicated operational v3 gold and enforce development gates."""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_ANNOTATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "annotations"
    / "operational_development_v3"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "processed"
    / "operational_development_v3"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import latest_task_states, load_jsonl, load_queue  # noqa: E402
from ml.annotation_dataset import (  # noqa: E402
    build_annotated_tasks,
    build_training_pairs,
)
from ml.operational_development_v3 import (  # noqa: E402
    resolved_single_reviewer_states,
    sufficiency_report,
)
from ml.real_holdout import jsonl_bytes, sha256_bytes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=20260731)
    return parser.parse_args()


def _enrich_gold(
    gold: list[dict[str, Any]],
    source: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_by_id = {str(task["task_id"]): task for task in source}
    enriched: list[dict[str, Any]] = []
    for task in gold:
        source_task = source_by_id[str(task["task_id"])]
        row = dict(task)
        row.update(
            {
                "operational_resume_group": str(
                    source_task["operational_resume_group"]
                ),
                "construction_stratum": str(
                    source_task["construction_stratum"]
                ),
                "taxonomy_reference": list(
                    source_task["taxonomy_reference"]
                ),
                "source_dataset_revision": str(
                    source_task["source_dataset_revision"]
                ),
                "profile_role_family": str(
                    source_task["profile_role_family"]
                ),
                "pairing_type": str(source_task["pairing_type"]),
                "presentation_id": str(source_task["presentation_id"]),
                "hidden_repeat_of": None,
                "evaluation_group": (
                    "resume:" + str(source_task["source_resume_hash"])
                ),
            }
        )
        enriched.append(row)
    return sorted(enriched, key=lambda row: str(row["task_id"]))


def main() -> None:
    args = parse_args()
    outputs = {
        "gold": args.output_dir / "annotated_tasks.jsonl",
        "pairs": args.output_dir / "training_pairs.jsonl",
        "manifest": args.output_dir / "manifest.json",
    }
    if any(path.exists() for path in outputs.values()):
        raise SystemExit("Operational v3 gold exists; refusing overwrite.")
    source = load_queue(args.annotation_dir / "queue.jsonl")
    source_manifest = json.loads(
        (args.annotation_dir / "manifest.json").read_text(encoding="utf-8")
    )
    source_checksum = sha256_bytes(jsonl_bytes(source))
    if source_checksum != source_manifest["task_sha256"]:
        raise SystemExit("Operational v3 source queue checksum mismatch.")
    agreement_path = args.annotation_dir / "repeat_agreement.json"
    if not agreement_path.is_file():
        raise SystemExit("Freeze repeat agreement before finalizing v3.")
    repeat_report = json.loads(agreement_path.read_text(encoding="utf-8"))
    reviewer_states = latest_task_states(
        load_jsonl(args.annotation_dir / "reviewer_decisions.jsonl")
    )
    adjudication_states = latest_task_states(
        load_jsonl(args.annotation_dir / "self_adjudication_decisions.jsonl")
    )
    expected_adjudications = {
        str(task_id)
        for task_id in repeat_report.get("disagreement_task_ids", [])
    }
    if set(adjudication_states) - expected_adjudications:
        raise SystemExit("Adjudication events contain unexpected task ids.")
    resolved = resolved_single_reviewer_states(
        source,
        reviewer_states,
        adjudication_states,
        repeat_report,
    )
    gold = build_annotated_tasks(
        source,
        resolved,
        random_state=args.random_state,
        require_complete=True,
        require_candidate_labels=True,
    )
    gold = _enrich_gold(gold, source)
    pairs = build_training_pairs(gold)
    for pair in pairs:
        task = next(
            row for row in gold if str(row["task_id"]) == str(pair["task_id"])
        )
        pair["evaluation_group"] = str(task["evaluation_group"])
        pair["construction_stratum"] = str(task["construction_stratum"])
    if len(gold) != 120 or len(pairs) != 480:
        raise SystemExit("Operational v3 must freeze 120 tasks and 480 pairs.")
    sufficiency = sufficiency_report(gold)
    agreement_passed = bool(repeat_report["agreement_gate_passed"])
    eligible = (
        agreement_passed
        and bool(sufficiency["dataset_sufficiency_gate_passed"])
    )
    gold_payload = jsonl_bytes(gold)
    pair_payload = jsonl_bytes(pairs)
    manifest = {
        "schema_version": 1,
        "dataset_id": "operational_development_v3",
        "dataset_role": "successor_training_and_nested_development_only",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": hashlib.sha256(
            (args.annotation_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "repeat_agreement_sha256": hashlib.sha256(
            agreement_path.read_bytes()
        ).hexdigest(),
        "gold_tasks": len(gold),
        "training_pairs": len(pairs),
        "candidate_label_coverage": "complete",
        "task_label_counts": dict(
            Counter(str(task["support_label"]) for task in gold)
        ),
        "candidate_label_counts": dict(
            Counter(str(pair["support_label"]) for pair in pairs)
        ),
        "gold_sha256": sha256_bytes(gold_payload),
        "training_pairs_sha256": sha256_bytes(pair_payload),
        "repeat_agreement": repeat_report,
        "dataset_sufficiency": sufficiency,
        "agreement_gate_passed": agreement_passed,
        "dataset_sufficiency_gate_passed": bool(
            sufficiency["dataset_sufficiency_gate_passed"]
        ),
        "eligible_for_model_comparison": eligible,
        "eligible_for_final_evaluation": False,
        "product_integration_allowed": False,
        "post_label_selection_performed": False,
        "status": (
            "ready_for_nested_development_comparison"
            if eligible
            else "annotation_pilot_blocked_by_quality_gate"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs["gold"].write_bytes(gold_payload)
    outputs["pairs"].write_bytes(pair_payload)
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Gold tasks: {manifest['gold_tasks']}")
    print(f"Training pairs: {manifest['training_pairs']}")
    print(f"Task labels: {manifest['task_label_counts']}")
    print(f"Agreement gate: {manifest['agreement_gate_passed']}")
    print(
        "Dataset sufficiency gate: "
        f"{manifest['dataset_sufficiency_gate_passed']}"
    )
    print(f"Status: {manifest['status']}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
