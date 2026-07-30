"""Compare local evidence methods without changing application decisions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
SHADOW_ROOT = PROJECT_ROOT / "data" / "ml" / "shadow"
DEFAULT_ARTIFACT = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "models"
    / "evidence_sentence_embedding_shadow_v1.joblib"
)
SPEC_PATH = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "models"
    / "evidence_sentence_embedding_two_stage_v1.json"
)
TRAINING_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "processed"
    / "reviewed_evidence_training_v4_batch4"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.evidence_sentence_artifact import (  # noqa: E402
    file_sha256,
    validate_sentence_artifact,
)
from ml.evidence_sentence_embedding import (  # noqa: E402
    CachedSentenceEncoder,
    load_sentence_encoder,
)
from ml.evidence_shadow import build_shadow_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-text", type=Path, required=True)
    parser.add_argument("--resume-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--max-requirements", type=int, default=8)
    return parser.parse_args()


def _assert_private_output(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(SHADOW_ROOT.resolve())
    except ValueError as error:
        raise SystemExit(
            f"Shadow output must stay under ignored directory {SHADOW_ROOT}."
        ) from error
    if resolved.exists():
        raise SystemExit("Shadow output exists; refusing overwrite.")
    return resolved


def main() -> None:
    args = parse_args()
    output = _assert_private_output(args.output)
    artifact = validate_sentence_artifact(
        joblib.load(args.artifact),
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
    report = build_shadow_report(
        args.job_text.read_text(encoding="utf-8"),
        args.resume_text.read_text(encoding="utf-8"),
        artifact=artifact,
        encoder=encoder,
        max_requirements=args.max_requirements,
    )
    report.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "job_text_sha256": file_sha256(args.job_text),
            "resume_text_sha256": file_sha256(args.resume_text),
            "artifact_sha256": file_sha256(args.artifact),
            "input_paths_recorded": False,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Requirements: {report['requirement_count']}")
    print(f"Evidence records: {report['evidence_count']}")
    print(f"Comparisons: {report['comparison_counts']}")
    print("Product state modified: no")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
