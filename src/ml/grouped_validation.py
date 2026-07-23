"""Leakage-resistant grouped splits for anonymous real requirement/evidence tasks."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from typing import Any, Iterable


SPLIT_NAMES = ("train", "validation", "holdout")


class GroupedValidationError(ValueError):
    """Raised when real-data grouping metadata is missing or inconsistent."""


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _task_id(row: dict[str, Any]) -> str:
    identifier = str(row.get("task_id") or row.get("gold_id") or "").strip()
    if not identifier:
        raise GroupedValidationError("Every real-data task requires task_id or gold_id.")
    return identifier


def _group_nodes(row: dict[str, Any]) -> list[str]:
    resume_hash = str(row.get("source_resume_hash", "")).strip()
    job_hash = str(row.get("source_job_hash", "")).strip()
    if not resume_hash or not job_hash:
        raise GroupedValidationError(
            f"Task {_task_id(row)} needs anonymous resume and job hashes."
        )
    nodes = [f"resume:{resume_hash}", f"job:{job_hash}"]
    semantic_group = str(
        row.get("near_duplicate_cluster")
        or row.get("semantic_case_group_id")
        or ""
    ).strip()
    if semantic_group:
        nodes.append(f"semantic:{semantic_group}")
    return nodes


def grouped_split(
    rows: Iterable[dict[str, Any]],
    *,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    random_state: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Split connected resume/job/semantic components without cross-split leakage."""
    records = list(rows)
    if len(fractions) != 3 or any(value <= 0 for value in fractions):
        raise GroupedValidationError("Provide three positive split fractions.")
    fraction_total = sum(fractions)
    normalized_fractions = tuple(value / fraction_total for value in fractions)
    identifiers = [_task_id(row) for row in records]
    if len(identifiers) != len(set(identifiers)):
        raise GroupedValidationError("Real-data task identifiers must be unique.")

    groups = _DisjointSet()
    nodes_by_task: dict[str, list[str]] = {}
    for row in records:
        task_id = _task_id(row)
        nodes = _group_nodes(row)
        nodes_by_task[task_id] = nodes
        for node in nodes[1:]:
            groups.union(nodes[0], node)

    components: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        task_id = _task_id(row)
        components[groups.find(nodes_by_task[task_id][0])].append(row)
    ordered_components = sorted(
        components.values(),
        key=lambda component: (
            -len(component),
            hashlib.sha256(
                (
                    str(random_state)
                    + ":"
                    + ",".join(sorted(_task_id(row) for row in component))
                ).encode("utf-8")
            ).hexdigest(),
        ),
    )

    target_counts = {
        split: len(records) * fraction
        for split, fraction in zip(SPLIT_NAMES, normalized_fractions, strict=True)
    }
    split_rows: dict[str, list[dict[str, Any]]] = {
        split: []
        for split in SPLIT_NAMES
    }
    for component in ordered_components:
        split = min(
            SPLIT_NAMES,
            key=lambda name: (
                (len(split_rows[name]) + len(component)) / target_counts[name],
                len(split_rows[name]),
                name,
            ),
        )
        split_rows[split].extend(component)
    for split in SPLIT_NAMES:
        split_rows[split] = sorted(split_rows[split], key=_task_id)
    return split_rows


def grouped_split_report(
    split_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Return overlap checks and a checksum for the frozen holdout identities."""
    missing_splits = set(SPLIT_NAMES) - set(split_rows)
    if missing_splits:
        raise GroupedValidationError(f"Missing splits: {sorted(missing_splits)}")
    resume_sets: dict[str, set[str]] = {}
    job_sets: dict[str, set[str]] = {}
    semantic_sets: dict[str, set[str]] = {}
    for split in SPLIT_NAMES:
        rows = split_rows[split]
        resume_sets[split] = {str(row["source_resume_hash"]) for row in rows}
        job_sets[split] = {str(row["source_job_hash"]) for row in rows}
        semantic_sets[split] = {
            str(
                row.get("near_duplicate_cluster")
                or row.get("semantic_case_group_id")
            )
            for row in rows
            if row.get("near_duplicate_cluster") or row.get("semantic_case_group_id")
        }

    def overlaps(sets: dict[str, set[str]]) -> dict[str, int]:
        return {
            f"{left}_{right}": len(sets[left] & sets[right])
            for index, left in enumerate(SPLIT_NAMES)
            for right in SPLIT_NAMES[index + 1 :]
        }

    holdout_ids = sorted(_task_id(row) for row in split_rows["holdout"])
    holdout_checksum = hashlib.sha256("\n".join(holdout_ids).encode("utf-8")).hexdigest()
    report: dict[str, Any] = {
        "schema_version": 1,
        "split_counts": {
            split: len(split_rows[split])
            for split in SPLIT_NAMES
        },
        "split_label_counts": {
            split: dict(
                Counter(str(row.get("support_label", "unknown")) for row in split_rows[split])
            )
            for split in SPLIT_NAMES
        },
        "resume_overlap": overlaps(resume_sets),
        "job_overlap": overlaps(job_sets),
        "semantic_overlap": overlaps(semantic_sets),
        "holdout_checksum_sha256": holdout_checksum,
        "holdout_is_fixed": True,
        "warnings": [],
    }
    if len(split_rows["holdout"]) < 40:
        report["warnings"].append(
            "The fixed real-data holdout contains fewer than 40 unique tasks."
        )
    overlap_values = [
        *report["resume_overlap"].values(),
        *report["job_overlap"].values(),
        *report["semantic_overlap"].values(),
    ]
    if any(overlap_values):
        raise GroupedValidationError("Grouped split contains cross-split source leakage.")
    return report
