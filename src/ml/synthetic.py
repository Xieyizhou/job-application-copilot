"""Load and convert the local synthetic candidate-matching dataset."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SyntheticPair:
    """One generated binary relevance example."""

    resume_id: str
    job_id: str
    resume_text: str
    job_text: str
    label: int
    subset: str


def _list_text(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return ". ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def render_resume(row: dict[str, Any]) -> str:
    """Render structured synthetic resume fields as model input text."""
    return "\n".join(
        part
        for part in [
            f"Target role: {row.get('role', '')}",
            f"Seniority: {row.get('seniority', '')}",
            f"Industry: {row.get('industry', '')}",
            f"Years of experience: {row.get('years_experience', '')}",
            f"Education: {row.get('education', '')}",
            f"Skills: {_list_text(row.get('skills'))}",
            _list_text(row.get("summary")),
            _list_text(row.get("experience_bullets")),
        ]
        if part.strip(" :")
    )


def render_job(row: dict[str, Any]) -> str:
    """Render structured synthetic job fields as model input text."""
    return "\n".join(
        part
        for part in [
            f"Role: {row.get('job_title', '')}",
            f"Seniority: {row.get('seniority', '')}",
            f"Industry: {row.get('industry', '')}",
            f"Must have skills: {_list_text(row.get('must_have_skills'))}",
            f"Nice to have skills: {_list_text(row.get('nice_to_have_skills'))}",
            _list_text(row.get("description")),
            _list_text(row.get("responsibilities")),
            _list_text(row.get("requirements")),
        ]
        if part.strip(" :")
    )


def stable_job_subset(job_id: str) -> str:
    """Assign a job to train/validation/test without cross-job leakage."""
    bucket = int(hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def load_synthetic_pairs(
    dataset_dir: Path,
    *,
    negatives_per_positive: float = 1.0,
    random_state: int = 42,
    max_jobs: int | None = None,
) -> list[SyntheticPair]:
    """Create balanced examples with job-grouped, deterministic splits."""
    if negatives_per_positive <= 0:
        raise ValueError("negatives_per_positive must be positive")
    resumes = pd.read_parquet(dataset_dir / "resumes-00000-of-00001.parquet")
    jobs = pd.read_parquet(dataset_dir / "jobs-00000-of-00001.parquet")
    matches = pd.read_parquet(dataset_dir / "matches-00000-of-00001.parquet")
    if max_jobs is not None:
        jobs = jobs.sort_values("job_id").head(max_jobs)
    resume_rows = {str(row["resume_id"]): row for row in resumes.to_dict("records")}
    job_rows = {str(row["job_id"]): row for row in jobs.to_dict("records")}
    match_rows = {str(row["job_id"]): row for row in matches.to_dict("records")}
    all_resume_ids = sorted(resume_rows)
    resume_texts = {key: render_resume(value) for key, value in resume_rows.items()}
    job_texts = {key: render_job(value) for key, value in job_rows.items()}
    pairs: list[SyntheticPair] = []

    for job_id in sorted(job_rows):
        relevant_value = match_rows[job_id]["relevant_resume_ids"]
        if hasattr(relevant_value, "tolist"):
            relevant_value = relevant_value.tolist()
        positive_ids = sorted(str(value) for value in relevant_value)
        positive_set = set(positive_ids)
        negative_count = max(1, round(len(positive_ids) * negatives_per_positive))
        candidates = [resume_id for resume_id in all_resume_ids if resume_id not in positive_set]
        job_seed = random_state ^ int(hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:8], 16)
        negative_ids = random.Random(job_seed).sample(candidates, min(negative_count, len(candidates)))
        subset = stable_job_subset(job_id)
        for resume_id, label in [*((value, 1) for value in positive_ids), *((value, 0) for value in negative_ids)]:
            pairs.append(
                SyntheticPair(
                    resume_id=resume_id,
                    job_id=job_id,
                    resume_text=resume_texts[resume_id],
                    job_text=job_texts[job_id],
                    label=label,
                    subset=subset,
                )
            )
    return pairs
