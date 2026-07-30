"""Freeze completed reserve reviews for one-time candidate evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_V1_BASE = PROJECT_ROOT / "data" / "ml" / "annotations" / "real_holdout_v1"
DEFAULT_V1_MODEL = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "models"
    / "evidence_validation_candidate_v1.joblib"
)
DEFAULT_V2_MODEL = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "models"
    / "evidence_rank_preserving_candidate_v1.joblib"
)
DEFAULT_V4_MODEL = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "models"
    / "evidence_sentence_embedding_two_stage_v1.json"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.real_holdout import finalize_real_reviews, jsonl_bytes, sha256_bytes  # noqa: E402


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-version",
        type=int,
        choices=(1, 2, 4),
        default=1,
    )
    parser.add_argument("--annotation-dir", type=Path)
    parser.add_argument("--model-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    version = f"v{args.dataset_version}"
    default_bases = {
        1: DEFAULT_V1_BASE,
        2: PROJECT_ROOT / "data" / "ml" / "annotations" / "real_reserve_v2",
        4: PROJECT_ROOT / "data" / "ml" / "annotations" / "real_reserve_v4",
    }
    default_models = {
        1: DEFAULT_V1_MODEL,
        2: DEFAULT_V2_MODEL,
        4: DEFAULT_V4_MODEL,
    }
    base = args.annotation_dir or default_bases[args.dataset_version]
    model_path = args.model_path or default_models[args.dataset_version]
    paths = {
        "reviewer_a_queue": base / f"real_reserve_reviewer_a_queue_{version}.jsonl",
        "reviewer_b_queue": base / f"real_reserve_reviewer_b_queue_{version}.jsonl",
        "reviewer_a_events": base / f"real_reserve_reviewer_a_decisions_{version}.jsonl",
        "reviewer_b_events": base / f"real_reserve_reviewer_b_decisions_{version}.jsonl",
        "adjudication_queue": base / f"real_reserve_adjudication_queue_{version}.jsonl",
        "adjudication_events": (
            base / f"real_reserve_adjudication_decisions_{version}.jsonl"
        ),
    }
    output = base / f"real_reserve_gold_{version}.jsonl"
    manifest_path = base / f"real_reserve_frozen_manifest_{version}.json"
    if output.exists() or manifest_path.exists():
        raise SystemExit("Frozen reserve gold exists; refusing overwrite.")
    if not model_path.is_file():
        raise SystemExit("Precommitted candidate artifact is missing.")
    construction_manifest_path = (
        base / f"real_reserve_manifest_{version}.json"
    )
    if construction_manifest_path.is_file():
        construction_manifest = json.loads(
            construction_manifest_path.read_text(encoding="utf-8")
        )
        committed_sha256 = construction_manifest.get(
            "precommitted_candidate_sha256"
        )
        if committed_sha256 and committed_sha256 != _file_sha256(model_path):
            raise SystemExit(
                "Candidate differs from the reserve construction commitment."
            )
    agreement_path = base / f"real_reserve_agreement_report_{version}.json"
    agreement = (
        json.loads(agreement_path.read_text(encoding="utf-8"))
        if agreement_path.is_file()
        else {}
    )
    gold, report = finalize_real_reviews(
        load_jsonl(paths["reviewer_a_queue"]),
        load_jsonl(paths["reviewer_b_queue"]),
        load_jsonl(paths["reviewer_a_events"]),
        load_jsonl(paths["reviewer_b_events"]),
        load_jsonl(paths["adjudication_queue"]),
        load_jsonl(paths["adjudication_events"]),
        dataset_role="reserve",
        agreement_audit_task_ids=agreement.get(
            "agreement_audit_task_ids",
            [],
        ),
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
        "precommitted_candidate_file": model_path.name,
        "precommitted_candidate_sha256": _file_sha256(model_path),
        "evaluation_policy": (
            (
                "One-time direction check only; 16 tasks are insufficient for "
                "product promotion and may not be used for threshold or model "
                "selection."
            )
            if args.dataset_version == 1
            else (
                "One-time evaluation on 48 fresh source-isolated tasks. These labels "
                "may not be used to change the precommitted model, thresholds, or "
                "selection criteria."
                if args.dataset_version == 2
                else (
                    "One-time evaluation on 96 fresh source-isolated tasks against "
                    "the precommitted candidate and success criteria. Labels may "
                    "not be used for tuning or model selection."
                )
            )
        ),
    }
    if agreement_path.is_file():
        manifest["review_policy"] = agreement.get("adjudication_policy")
        manifest["reviewer_exact_agreement_rate"] = agreement.get(
            "exact_agreement_rate"
        )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Frozen reserve tasks: {len(gold)}")
    print(f"Human adjudications: {report['human_adjudications']}")
    print(f"Human agreement audits: {report['human_agreement_audits']}")
    print(f"Gold SHA-256: {manifest['gold_sha256']}")
    print(f"Candidate SHA-256: {manifest['precommitted_candidate_sha256']}")
    print(f"Gold: {output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
