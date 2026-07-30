"""Run a balanced, source-isolated local evidence shadow batch."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
SHADOW_ROOT = PROJECT_ROOT / "data" / "ml" / "shadow"
ANNOTATION_DIR = PROJECT_ROOT / "data" / "ml" / "annotations"
TRAINING_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "processed"
    / "reviewed_evidence_training_v4_batch4"
)
DJINNI_PATH = (
    PROJECT_ROOT / "data" / "ml" / "external" / "djinni" / "raw"
    / "djinni_jobs.parquet"
)
ARTIFACT_PATH = (
    PROJECT_ROOT / "data" / "ml" / "models"
    / "evidence_sentence_embedding_shadow_v1.joblib"
)
SPEC_PATH = (
    PROJECT_ROOT / "data" / "ml" / "models"
    / "evidence_sentence_embedding_two_stage_v1.json"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.evidence import extract_requirement_records  # noqa: E402
from ml.evidence_sentence_artifact import (  # noqa: E402
    file_sha256,
    validate_sentence_artifact,
)
from ml.evidence_sentence_embedding import (  # noqa: E402
    CachedSentenceEncoder,
    load_sentence_encoder,
)
from ml.evidence_shadow import build_shadow_report  # noqa: E402
from ml.real_development import (  # noqa: E402
    ROLE_FAMILIES,
    infer_role_family,
    source_hash,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-text", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs-per-family", type=int, default=10)
    parser.add_argument("--max-requirements", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=20260729)
    return parser.parse_args()


def _private_new_directory(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(SHADOW_ROOT.resolve())
    except ValueError as error:
        raise SystemExit(
            f"Shadow output must stay under ignored directory {SHADOW_ROOT}."
        ) from error
    if resolved.exists():
        raise SystemExit("Shadow batch output exists; refusing overwrite.")
    return resolved


def _excluded_source_hashes() -> set[str]:
    excluded: set[str] = set()
    for path in ANNOTATION_DIR.rglob("*.jsonl"):
        excluded.update(
            str(row["source_job_hash"])
            for row in load_jsonl(path)
            if row.get("source_job_hash")
        )
    excluded.update(
        str(row["source_job_hash"])
        for row in load_jsonl(TRAINING_DIR / "annotated_tasks.jsonl")
        if row.get("source_job_hash")
    )
    return excluded


def _selection_rank(raw_id: str, random_state: int) -> str:
    return hashlib.sha256(f"{random_state}:{raw_id}".encode()).hexdigest()


def _select_jobs(
    *,
    jobs_per_family: int,
    random_state: int,
    excluded: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    frame = pd.read_parquet(
        DJINNI_PATH,
        columns=["id", "Position", "Primary Keyword", "Long Description"],
    )
    rows = frame.to_dict("records")
    rows.sort(
        key=lambda row: _selection_rank(str(row.get("id", "")), random_state)
    )
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_descriptions: set[str] = set()
    excluded_source_rows = 0
    for row in rows:
        raw_id = str(row.get("id", ""))
        job_hash = source_hash("djinni_job", raw_id)
        if not raw_id or job_hash in excluded:
            excluded_source_rows += 1
            continue
        family = infer_role_family(
            str(row.get("Position", "")),
            str(row.get("Primary Keyword", "")),
        )
        if family not in ROLE_FAMILIES or counts[family] >= jobs_per_family:
            continue
        description = str(row.get("Long Description", "")).strip()
        description_hash = hashlib.sha256(description.encode()).hexdigest()
        if (
            len(description.split()) < 120
            or description_hash in seen_descriptions
            or len(extract_requirement_records(description)) < 2
        ):
            continue
        selected.append(
            {
                "source_job_hash": job_hash,
                "role_family": family,
                "job_text": description,
            }
        )
        counts[family] += 1
        seen_descriptions.add(description_hash)
        if all(counts[family] >= jobs_per_family for family in ROLE_FAMILIES):
            break
    if any(counts[family] < jobs_per_family for family in ROLE_FAMILIES):
        raise SystemExit(
            f"Could not build balanced shadow batch: {dict(counts)}."
        )
    return selected, {
        "source_rows_excluded": excluded_source_rows,
        "eligible_selected": len(selected),
    }


def main() -> None:
    args = parse_args()
    output_dir = _private_new_directory(args.output_dir)
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
    selected, construction = _select_jobs(
        jobs_per_family=args.jobs_per_family,
        random_state=args.random_state,
        excluded=_excluded_source_hashes(),
    )
    resume_text = args.resume_text.read_text(encoding="utf-8")
    output_dir.mkdir(parents=True)
    comparison_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    review_jobs: list[dict[str, Any]] = []
    for index, job in enumerate(selected, start=1):
        report = build_shadow_report(
            str(job["job_text"]),
            resume_text,
            artifact=artifact,
            encoder=encoder,
            max_requirements=args.max_requirements,
        )
        filename = f"{index:02d}_{str(job['source_job_hash'])[:16]}.json"
        report.update(
            {
                "source_dataset": "djinni_public",
                "source_job_hash": job["source_job_hash"],
                "role_family": job["role_family"],
                "artifact_sha256": file_sha256(ARTIFACT_PATH),
                "resume_text_sha256": file_sha256(args.resume_text),
                "input_paths_recorded": False,
            }
        )
        (output_dir / filename).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        comparison_counts.update(report["comparison_counts"])
        role_counts[str(job["role_family"])] += 1
        disagreements = sum(
            count
            for status, count in report["comparison_counts"].items()
            if status not in {"both_accept_same_evidence", "both_reject"}
        )
        if disagreements:
            review_jobs.append(
                {
                    "report_file": filename,
                    "source_job_hash": job["source_job_hash"],
                    "role_family": job["role_family"],
                    "disagreements": disagreements,
                    "comparison_counts": report["comparison_counts"],
                }
            )
    summary = {
        "schema_version": 1,
        "mode": "local_shadow_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "jobs": len(selected),
        "jobs_per_family": args.jobs_per_family,
        "role_counts": dict(role_counts),
        "comparison_counts": dict(comparison_counts),
        "jobs_requiring_review": len(review_jobs),
        "review_queue": review_jobs,
        "construction": {
            **construction,
            "source_dataset": "djinni_public",
            "source_hash_overlap": 0,
            "selection_uses_model_predictions": False,
            "minimum_description_words": 120,
            "minimum_extracted_requirements": 2,
            "random_state": args.random_state,
        },
        "artifact_sha256": file_sha256(ARTIFACT_PATH),
        "resume_text_sha256": file_sha256(args.resume_text),
        "input_paths_recorded": False,
        "product_state_modified": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Jobs: {summary['jobs']}")
    print(f"Roles: {summary['role_counts']}")
    print(f"Comparisons: {summary['comparison_counts']}")
    print(f"Jobs requiring review: {summary['jobs_requiring_review']}")
    print("Product state modified: no")
    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
