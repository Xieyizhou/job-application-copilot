"""Build one training corpus from human annotations and reviewed consensus gold."""

from __future__ import annotations

from collections import Counter
import hashlib
from typing import Any, Iterable

from ml.annotation_generation import normalize_text


CORPUS_SCHEMA_VERSION = 1
ALLOWED_GOLD_SOURCES = {
    "blind_consensus",
    "human_annotation",
    "human_adjudication",
}
POSITIVE_SUPPORT_LABELS = {"Direct", "Partial"}


class EvidenceCorpusError(ValueError):
    """Raised when a reviewed source cannot safely enter the training corpus."""


def _pair_id(task_id: str, candidate_id: str, binary_label: int) -> str:
    material = f"{task_id}:{candidate_id}:{binary_label}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def gold_tasks_to_dataset(
    gold_tasks: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert candidate-level gold judgments into tasks and labeled pairs."""
    tasks: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for gold in gold_tasks:
        decision_source = str(gold.get("decision_source", ""))
        if decision_source not in ALLOWED_GOLD_SOURCES:
            raise EvidenceCorpusError(
                f"Unsupported gold decision source: {decision_source or 'missing'}"
            )
        if decision_source == "blind_consensus" and int(gold.get("reviewer_count", 0)) < 3:
            raise EvidenceCorpusError("Blind consensus gold requires at least three reviewers.")
        task_id = str(gold.get("gold_id", "")).strip()
        semantic_group = str(gold.get("semantic_case_group_id", "")).strip()
        support_label = str(gold.get("support_label", ""))
        candidates = gold.get("candidates")
        if not task_id or not semantic_group:
            raise EvidenceCorpusError("Gold tasks require gold_id and semantic_case_group_id.")
        if support_label not in {*POSITIVE_SUPPORT_LABELS, "No Support"}:
            raise EvidenceCorpusError(f"Unsupported gold label: {support_label or 'missing'}")
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise EvidenceCorpusError("Gold tasks require at least two candidates.")
        candidate_ids = {str(candidate.get("candidate_id", "")) for candidate in candidates}
        selected_id = gold.get("best_candidate_id")
        if support_label in POSITIVE_SUPPORT_LABELS and selected_id not in candidate_ids:
            raise EvidenceCorpusError("Supported gold tasks need valid best evidence.")
        if support_label == "No Support" and selected_id is not None:
            raise EvidenceCorpusError("No Support gold cannot select best evidence.")
        evaluation_group = f"semantic:{semantic_group}"
        normalized_candidates: list[dict[str, str]] = []
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id", "")).strip()
            evidence = str(candidate.get("evidence", "")).strip()
            candidate_label = str(candidate.get("support_label", ""))
            if not candidate_id or not evidence:
                raise EvidenceCorpusError("Gold candidates require id and evidence.")
            if candidate_label not in {*POSITIVE_SUPPORT_LABELS, "No Support"}:
                raise EvidenceCorpusError(
                    f"Unsupported candidate gold label: {candidate_label or 'missing'}"
                )
            normalized_candidates.append(
                {"candidate_id": candidate_id, "evidence": evidence}
            )
            binary_label = int(candidate_label in POSITIVE_SUPPORT_LABELS)
            pairs.append(
                {
                    "schema_version": CORPUS_SCHEMA_VERSION,
                    "pair_id": _pair_id(task_id, candidate_id, binary_label),
                    "task_id": task_id,
                    "role_family": str(gold.get("role_family", "unknown")),
                    "requirement": str(gold.get("requirement", "")),
                    "evidence": evidence,
                    "binary_label": binary_label,
                    "support_label": candidate_label,
                    "label_scope": "reviewed_candidate_judgment",
                    "source_resume_hash": "",
                    "source_job_hash": "",
                    "semantic_case_group_id": semantic_group,
                    "template_group": evaluation_group,
                    "evaluation_group": evaluation_group,
                    "review_source": decision_source,
                }
            )
        tasks.append(
            {
                "schema_version": CORPUS_SCHEMA_VERSION,
                "task_id": task_id,
                "role_family": str(gold.get("role_family", "unknown")),
                "requirement": str(gold.get("requirement", "")),
                "candidates": normalized_candidates,
                "selected_candidate_id": selected_id,
                "support_label": support_label,
                "cover_letter_safe": None,
                "source_dataset": "reviewed_consensus_gold",
                "source_resume_hash": "",
                "source_job_hash": "",
                "semantic_case_group_id": semantic_group,
                "template_group": evaluation_group,
                "evaluation_group": evaluation_group,
                "review_source": decision_source,
            }
        )
    return tasks, pairs


def _human_evaluation_group(task: dict[str, Any]) -> str:
    semantic_group = str(task.get("semantic_case_group_id", "")).strip()
    if semantic_group:
        return f"semantic:{semantic_group}"
    return f"template:{task.get('template_group', 'unknown')}"


def combine_reviewed_sources(
    human_tasks: Iterable[dict[str, Any]],
    human_pairs: Iterable[dict[str, Any]],
    gold_tasks: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Merge reviewed sources while preserving provenance and group boundaries."""
    normalized_human_tasks: list[dict[str, Any]] = []
    task_group: dict[str, str] = {}
    for task in human_tasks:
        row = dict(task)
        task_id = str(row["task_id"])
        group = _human_evaluation_group(row)
        row["evaluation_group"] = group
        row["review_source"] = "human_annotation"
        normalized_human_tasks.append(row)
        task_group[task_id] = group
    normalized_human_pairs: list[dict[str, Any]] = []
    for pair in human_pairs:
        row = dict(pair)
        task_id = str(row["task_id"])
        row["evaluation_group"] = task_group.get(
            task_id,
            f"template:{row.get('template_group', 'unknown')}",
        )
        row["review_source"] = "human_annotation"
        normalized_human_pairs.append(row)

    consensus_tasks, consensus_pairs = gold_tasks_to_dataset(gold_tasks)
    all_tasks = [*normalized_human_tasks, *consensus_tasks]
    all_pairs = [*normalized_human_pairs, *consensus_pairs]
    task_ids = [str(task["task_id"]) for task in all_tasks]
    if len(task_ids) != len(set(task_ids)):
        raise EvidenceCorpusError("Reviewed task identifiers must be unique.")

    deduplicated_pairs: list[dict[str, Any]] = []
    seen_pair_content: dict[tuple[str, str], int] = {}
    duplicate_pair_count = 0
    for pair in all_pairs:
        content_key = (
            normalize_text(str(pair["requirement"])),
            normalize_text(str(pair["evidence"])),
        )
        binary_label = int(pair["binary_label"])
        prior_label = seen_pair_content.get(content_key)
        if prior_label is not None:
            if prior_label != binary_label:
                raise EvidenceCorpusError(
                    "The same requirement/evidence content has conflicting reviewed labels."
                )
            duplicate_pair_count += 1
            continue
        seen_pair_content[content_key] = binary_label
        deduplicated_pairs.append(pair)

    manifest = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "dataset_name": "reviewed_evidence_training_v3",
        "task_count": len(all_tasks),
        "pair_count": len(deduplicated_pairs),
        "duplicate_pairs_removed": duplicate_pair_count,
        "task_source_counts": dict(
            Counter(str(task["review_source"]) for task in all_tasks)
        ),
        "pair_source_counts": dict(
            Counter(str(pair["review_source"]) for pair in deduplicated_pairs)
        ),
        "task_label_counts": dict(
            Counter(str(task["support_label"]) for task in all_tasks)
        ),
        "pair_label_counts": dict(
            Counter(str(pair["binary_label"]) for pair in deduplicated_pairs)
        ),
        "evaluation_groups": len(
            {str(task["evaluation_group"]) for task in all_tasks}
        ),
        "gold_policy": (
            "Human annotations and reviewed candidate judgments only; generator intent "
            "is never imported."
        ),
    }
    return all_tasks, deduplicated_pairs, manifest
