"""Prepare a blind, role-balanced review of shadow disagreements."""

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_BATCH = PROJECT_ROOT / "data" / "ml" / "shadow" / "real_batch_v2_candidate4"
DEFAULT_RESUME = (
    PROJECT_ROOT / "data" / "local_workspace" / "candidate"
    / "candidate_source.md"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import validate_queue, write_queue  # noqa: E402
from ml.evidence import extract_resume_evidence_records  # noqa: E402
from ml.real_development import ROLE_FAMILIES  # noqa: E402


DISAGREEMENT_STATUSES = {
    "shadow_only_accept",
    "baseline_only_accept",
    "both_accept_different_evidence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH)
    parser.add_argument("--resume-text", type=Path, default=DEFAULT_RESUME)
    parser.add_argument("--tasks-per-family", type=int, default=6)
    parser.add_argument("--random-state", type=int, default=20260729)
    return parser.parse_args()


def _stable_rank(value: str, random_state: int) -> str:
    return hashlib.sha256(f"{random_state}:{value}".encode()).hexdigest()


def _load_disagreements(batch_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(batch_dir.glob("[0-9][0-9]_*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        for comparison in report["comparisons"]:
            if comparison["comparison"] in DISAGREEMENT_STATUSES:
                rows.append(
                    {
                        **comparison,
                        "source_job_hash": report["source_job_hash"],
                        "role_family": report["role_family"],
                        "source_report": path.name,
                    }
                )
    return rows


def _balanced_sample(
    disagreements: list[dict[str, Any]],
    *,
    tasks_per_family: int,
    random_state: int,
) -> list[dict[str, Any]]:
    by_family: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in disagreements:
        by_family[str(row["role_family"])].append(row)
    selected: list[dict[str, Any]] = []
    for family in ROLE_FAMILIES:
        ranked = sorted(
            by_family[family],
            key=lambda row: (
                row["comparison"] == "shadow_only_accept",
                _stable_rank(
                    f"{row['source_job_hash']}:{row['requirement']}",
                    random_state,
                ),
            ),
        )
        if len(ranked) < tasks_per_family:
            raise SystemExit(
                f"Only {len(ranked)} {family} disagreements are available."
            )
        selected.extend(ranked[:tasks_per_family])
    random.Random(random_state).shuffle(selected)
    return selected


def _candidate_rows(
    row: dict[str, Any],
    resume_evidence: list[str],
    *,
    random_state: int,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    sources: list[tuple[str, str]] = []
    shadow_evidence = str(row["shadow"].get("evidence", "")).strip()
    baseline_evidence = str(row["baseline"].get("evidence", "")).strip()
    if shadow_evidence:
        sources.append(("shadow", shadow_evidence))
    if baseline_evidence and baseline_evidence != shadow_evidence:
        sources.append(("baseline", baseline_evidence))
    fillers = sorted(
        (
            evidence
            for evidence in resume_evidence
            if evidence not in {item[1] for item in sources}
        ),
        key=lambda evidence: _stable_rank(
            f"{row['requirement']}:{evidence}",
            random_state,
        ),
    )
    while len(sources) < 2 and fillers:
        sources.append(("filler", fillers.pop(0)))
    if len(sources) < 2:
        raise SystemExit("Resume needs at least two distinct evidence records.")
    candidates: list[dict[str, str]] = []
    origins: dict[str, str] = {}
    for origin, evidence in sources[:2]:
        candidate_id = hashlib.sha256(
            f"{row['source_job_hash']}:{row['requirement']}:{evidence}".encode()
        ).hexdigest()[:16]
        candidates.append(
            {"candidate_id": candidate_id, "evidence": evidence}
        )
        origins[candidate_id] = origin
    random.Random(
        f"{random_state}:{row['source_job_hash']}:{row['requirement']}"
    ).shuffle(candidates)
    return candidates, origins


def main() -> None:
    args = parse_args()
    review_dir = args.batch_dir / "human_review"
    queue_path = review_dir / "queue.jsonl"
    construction_path = review_dir / "construction.json"
    manifest_path = review_dir / "manifest.json"
    if any(path.exists() for path in (queue_path, construction_path, manifest_path)):
        raise SystemExit("Shadow review packet exists; refusing overwrite.")
    resume_evidence = [
        str(record["text"])
        for record in extract_resume_evidence_records(
            args.resume_text.read_text(encoding="utf-8")
        )
    ]
    sampled = _balanced_sample(
        _load_disagreements(args.batch_dir),
        tasks_per_family=args.tasks_per_family,
        random_state=args.random_state,
    )
    tasks: list[dict[str, Any]] = []
    construction: list[dict[str, Any]] = []
    resume_hash = hashlib.sha256(args.resume_text.read_bytes()).hexdigest()
    for row in sampled:
        candidates, origins = _candidate_rows(
            row,
            resume_evidence,
            random_state=args.random_state,
        )
        task_id = hashlib.sha256(
            f"shadow-review:{row['source_job_hash']}:{row['requirement']}".encode()
        ).hexdigest()[:24]
        tasks.append(
            {
                "schema_version": 1,
                "task_id": task_id,
                "requirement": row["requirement"],
                "role_family": row["role_family"],
                "source_dataset": "local_shadow_review_v1",
                "source_job_hash": row["source_job_hash"],
                "source_resume_hash": resume_hash,
                "candidates": candidates,
            }
        )
        construction.append(
            {
                "task_id": task_id,
                "comparison": row["comparison"],
                "source_report": row["source_report"],
                "candidate_origins": origins,
            }
        )
    checked = validate_queue(tasks)
    review_dir.mkdir(parents=True, exist_ok=True)
    write_queue(checked, queue_path)
    construction_path.write_text(
        json.dumps(construction, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "tasks": len(checked),
        "candidates": sum(len(task["candidates"]) for task in checked),
        "role_counts": {
            family: sum(
                str(task["role_family"]) == family for task in checked
            )
            for family in ROLE_FAMILIES
        },
        "reviewer_visibility": "method origins hidden",
        "selection_uses_model_probabilities": False,
        "construction_file_visible_to_reviewer": False,
        "human_decisions_are_shadow_diagnostics_only": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Tasks: {manifest['tasks']}")
    print(f"Candidates: {manifest['candidates']}")
    print(f"Roles: {manifest['role_counts']}")
    print(f"Queue: {queue_path}")


if __name__ == "__main__":
    main()
