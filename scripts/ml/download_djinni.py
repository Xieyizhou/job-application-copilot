"""Download the public Djinni job and candidate datasets locally."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_DIR = PROJECT_ROOT / "data" / "ml" / "external" / "djinni"
CACHE_DIR = BASE_DIR / "cache"
RAW_DIR = BASE_DIR / "raw"

JOB_DATASET = (
    "lang-uk/"
    "recruitment-dataset-job-descriptions-english"
)

PROFILE_DATASET = (
    "lang-uk/"
    "recruitment-dataset-candidate-profiles-english"
)


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading job descriptions...")

    jobs = load_dataset(
        JOB_DATASET,
        split="train",
        cache_dir=str(CACHE_DIR),
    )

    jobs_path = RAW_DIR / "djinni_jobs.parquet"
    jobs.to_parquet(str(jobs_path))

    print(f"Job rows: {len(jobs):,}")
    print(f"Job columns: {jobs.column_names}")
    print(f"Saved: {jobs_path}")
    print()

    print("Downloading candidate profiles...")

    profiles = load_dataset(
        PROFILE_DATASET,
        split="train",
        cache_dir=str(CACHE_DIR),
    )

    profiles_path = RAW_DIR / "djinni_profiles.parquet"
    profiles.to_parquet(str(profiles_path))

    print(f"Profile rows: {len(profiles):,}")
    print(f"Profile columns: {profiles.column_names}")
    print(f"Saved: {profiles_path}")

    manifest = {
        "job_dataset": JOB_DATASET,
        "job_rows": len(jobs),
        "job_columns": jobs.column_names,
        "profile_dataset": PROFILE_DATASET,
        "profile_rows": len(profiles),
        "profile_columns": profiles.column_names,
        "license": "MIT",
    }

    manifest_path = BASE_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
