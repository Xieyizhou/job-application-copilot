"""Versioned contracts for local reviewed-evidence model artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

from ml.annotation_generation import normalize_text


EVIDENCE_ARTIFACT_SCHEMA_VERSION = 2
SUPPORTED_ARTIFACT_MODEL_TYPES = frozenset(
    {
        "tfidf_cosine",
        "hybrid_lsa_reranker",
        "lexical_guarded_reranker",
        "pairwise_hybrid_reranker",
    }
)


class EvidenceArtifactError(ValueError):
    """Raised when a local model artifact is incompatible or stale."""


def training_corpus_fingerprint(
    tasks: Iterable[dict[str, Any]],
    pairs: Iterable[dict[str, Any]],
) -> str:
    """Return a stable fingerprint over every model-training input."""
    canonical_tasks: list[dict[str, Any]] = []
    for task in tasks:
        canonical_tasks.append(
            {
                "task_id": str(task.get("task_id", "")),
                "requirement": normalize_text(str(task.get("requirement", ""))),
                "support_label": str(task.get("support_label", "")),
                "selected_candidate_id": task.get("selected_candidate_id"),
                "evaluation_group": str(
                    task.get("evaluation_group") or task.get("template_group", "")
                ),
                "candidates": sorted(
                    (
                        {
                            "candidate_id": str(candidate.get("candidate_id", "")),
                            "evidence": normalize_text(
                                str(candidate.get("evidence", ""))
                            ),
                        }
                        for candidate in task.get("candidates", [])
                        if isinstance(candidate, dict)
                    ),
                    key=lambda candidate: candidate["candidate_id"],
                ),
            }
        )
    canonical_tasks.sort(key=lambda task: str(task["task_id"]))
    canonical_pairs: list[dict[str, Any]] = []
    for pair in pairs:
        canonical_pairs.append(
            {
                "pair_id": str(pair.get("pair_id", "")),
                "task_id": str(pair.get("task_id", "")),
                "requirement": normalize_text(str(pair.get("requirement", ""))),
                "evidence": normalize_text(str(pair.get("evidence", ""))),
                "binary_label": int(pair.get("binary_label", 0)),
                "support_label": str(pair.get("support_label", "")),
                "evaluation_group": str(
                    pair.get("evaluation_group") or pair.get("template_group", "")
                ),
            }
        )
    canonical_pairs.sort(
        key=lambda pair: (str(pair["pair_id"]), str(pair["task_id"]))
    )
    payload = json.dumps(
        {"tasks": canonical_tasks, "pairs": canonical_pairs},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_evidence_artifact(
    artifact: Any,
    *,
    tasks: Iterable[dict[str, Any]],
    pairs: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Return a compatible artifact or reject legacy and stale files."""
    if not isinstance(artifact, dict):
        raise EvidenceArtifactError("Evidence model artifact must be an object.")
    if int(artifact.get("schema_version", 0)) != EVIDENCE_ARTIFACT_SCHEMA_VERSION:
        raise EvidenceArtifactError(
            "Legacy evidence model artifact rejected; retrain with the current script."
        )
    model_type = str(artifact.get("model_type", ""))
    if model_type not in SUPPORTED_ARTIFACT_MODEL_TYPES:
        raise EvidenceArtifactError(
            f"Unsupported evidence model type: {model_type or 'missing'}"
        )
    model = artifact.get("model")
    if not callable(getattr(model, "predict_proba", None)):
        raise EvidenceArtifactError("Evidence model does not expose predict_proba.")
    try:
        threshold = float(artifact["threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceArtifactError("Evidence model threshold is invalid.") from error
    if not math.isfinite(threshold):
        raise EvidenceArtifactError("Evidence model threshold must be finite.")
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        raise EvidenceArtifactError("Evidence model metadata is missing.")
    selection = metadata.get("selection")
    if (
        not isinstance(selection, dict)
        or str(selection.get("selected_method", "")) != model_type
    ):
        raise EvidenceArtifactError(
            "Artifact model type does not match its recorded model selection."
        )
    expected_fingerprint = training_corpus_fingerprint(tasks, pairs)
    if metadata.get("training_corpus_fingerprint") != expected_fingerprint:
        raise EvidenceArtifactError(
            "Evidence model was trained on a different corpus; retrain before evaluation."
        )
    return artifact
