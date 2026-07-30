"""Validation helpers for complete candidate-level evidence judgments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


CANDIDATE_SUPPORT_LABELS = ("Direct", "Partial", "No Support")
POSITIVE_SUPPORT_LABELS = {"Direct", "Partial"}
UNLABELED_CANDIDATE_LABEL = "Unlabeled"
_LABEL_STRENGTH = {"No Support": 0, "Partial": 1, "Direct": 2}


class CandidateJudgmentError(ValueError):
    """Raised when candidate labels and the selected best evidence disagree."""


def normalized_candidate_labels(value: object) -> dict[str, str]:
    """Return a string mapping or an empty mapping for legacy events."""
    if not isinstance(value, Mapping):
        return {}
    return {str(candidate_id): str(label) for candidate_id, label in value.items()}


def validate_complete_candidate_labels(
    task: Mapping[str, Any],
    candidate_labels: Mapping[str, str],
    selected_candidate_id: str | None,
) -> tuple[str, dict[str, str]]:
    """Validate all candidate judgments and derive the task-level support label."""
    candidate_ids = {
        str(candidate["candidate_id"])
        for candidate in task.get("candidates", [])
        if isinstance(candidate, Mapping)
    }
    normalized = {
        str(candidate_id): str(label)
        for candidate_id, label in candidate_labels.items()
    }
    if set(normalized) != candidate_ids:
        missing = sorted(candidate_ids - set(normalized))
        extra = sorted(set(normalized) - candidate_ids)
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unknown: " + ", ".join(extra))
        raise CandidateJudgmentError(
            "Every candidate needs exactly one support label"
            + (f" ({'; '.join(details)})" if details else "")
            + "."
        )
    invalid = sorted(
        {
            label
            for label in normalized.values()
            if label not in CANDIDATE_SUPPORT_LABELS
        }
    )
    if invalid:
        raise CandidateJudgmentError(
            "Unsupported candidate labels: " + ", ".join(invalid)
        )

    supported = {
        candidate_id: label
        for candidate_id, label in normalized.items()
        if label in POSITIVE_SUPPORT_LABELS
    }
    if not supported:
        if selected_candidate_id is not None:
            raise CandidateJudgmentError(
                "Best evidence must be None when every candidate is No Support."
            )
        return "No Support", normalized
    if selected_candidate_id is None:
        raise CandidateJudgmentError(
            "Choose the best evidence from the candidates labeled Direct or Partial."
        )
    selected_id = str(selected_candidate_id)
    selected_label = normalized.get(selected_id)
    if selected_label not in POSITIVE_SUPPORT_LABELS:
        raise CandidateJudgmentError(
            "Best evidence must be labeled Direct or Partial."
        )
    strongest = max(_LABEL_STRENGTH[label] for label in supported.values())
    if _LABEL_STRENGTH[selected_label] != strongest:
        raise CandidateJudgmentError(
            "A Partial candidate cannot be best evidence while another candidate "
            "is labeled Direct."
        )
    return selected_label, normalized


def legacy_candidate_labels(
    task: Mapping[str, Any],
    state: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Return safe UI defaults without inventing labels for legacy positives."""
    candidate_ids = [
        str(candidate["candidate_id"])
        for candidate in task.get("candidates", [])
        if isinstance(candidate, Mapping)
    ]
    if not state:
        return {
            candidate_id: UNLABELED_CANDIDATE_LABEL
            for candidate_id in candidate_ids
        }
    stored = normalized_candidate_labels(state.get("candidate_labels"))
    if stored:
        return {
            candidate_id: stored.get(candidate_id, UNLABELED_CANDIDATE_LABEL)
            for candidate_id in candidate_ids
        }
    if state.get("support_label") == "No Support":
        return {candidate_id: "No Support" for candidate_id in candidate_ids}
    selected = state.get("selected_candidate_id")
    support_label = str(state.get("support_label", ""))
    return {
        candidate_id: (
            support_label
            if candidate_id == selected and support_label in POSITIVE_SUPPORT_LABELS
            else UNLABELED_CANDIDATE_LABEL
        )
        for candidate_id in candidate_ids
    }


def has_complete_candidate_labels(
    task: Mapping[str, Any],
    state: Mapping[str, Any] | None,
) -> bool:
    """Return whether an event explicitly labels every candidate."""
    if not state or state.get("action") != "label":
        return False
    labels = normalized_candidate_labels(state.get("candidate_labels"))
    candidate_ids = {
        str(candidate["candidate_id"])
        for candidate in task.get("candidates", [])
        if isinstance(candidate, Mapping)
    }
    return (
        bool(candidate_ids)
        and set(labels) == candidate_ids
        and all(label in CANDIDATE_SUPPORT_LABELS for label in labels.values())
    )
