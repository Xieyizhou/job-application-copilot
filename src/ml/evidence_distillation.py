"""Offline-only contracts for teacher-scored evidence-ranking experiments."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
from sklearn.metrics import roc_auc_score


DISTILLATION_SCHEMA_VERSION = 1
ALLOWED_INPUT_ROLES = {"human_reviewed_training"}
FORBIDDEN_PATH_TOKENS = ("holdout", "reserve", "shadow", "v8", "successor_v")


class DistillationError(ValueError):
    """Raised when an offline distillation experiment violates its contract."""


@dataclass(frozen=True)
class TeacherSpec:
    """Pinned teacher identity without copying any teacher weights into Git."""

    teacher_id: str
    model: str
    source: str
    kind: str
    revision: str | None = None

    def manifest(self) -> dict[str, str | None]:
        """Return serializable teacher metadata."""
        return {
            "id": self.teacher_id,
            "kind": self.kind,
            "model": self.model,
            "revision": self.revision,
            "source": self.source,
        }


def load_distillation_protocol(path: str) -> dict[str, Any]:
    """Load and validate the committed, content-free distillation protocol."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DistillationError("Distillation protocol must be a JSON object.")
    validate_distillation_protocol(payload)
    return payload


def validate_distillation_protocol(protocol: Mapping[str, Any]) -> None:
    """Fail closed unless the experiment is isolated from product and holdouts."""
    if protocol.get("schema_version") != DISTILLATION_SCHEMA_VERSION:
        raise DistillationError("Unsupported distillation protocol schema.")
    if protocol.get("product_integration_allowed") is not False or protocol.get(
        "rollout_allowed"
    ) is not False:
        raise DistillationError("Distillation protocol must not authorize product use.")
    policy = protocol.get("dataset_policy")
    if not isinstance(policy, Mapping):
        raise DistillationError("Distillation protocol needs a dataset policy.")
    required_policy = {
        "allow_human_reviewed_training_only": True,
        "allow_unlabeled_source_pool": False,
        "allow_v4_v7_before_full_qa_and_human_labels": False,
        "allow_v6_sealed_holdout": False,
        "allow_frozen_reserve": False,
        "allow_shadow_diagnostics": False,
        "allow_v8_annotation_pilot": False,
    }
    if any(policy.get(key) is not value for key, value in required_policy.items()):
        raise DistillationError("Distillation dataset policy is unsafe.")
    teachers = protocol.get("teachers")
    if not isinstance(teachers, list) or not teachers:
        raise DistillationError("Distillation protocol needs at least one teacher.")
    ids = [str(teacher.get("id", "")) for teacher in teachers if isinstance(teacher, Mapping)]
    if len(ids) != len(teachers) or not all(ids) or len(ids) != len(set(ids)):
        raise DistillationError("Teacher identifiers must be unique and non-empty.")


def assert_allowed_input_path(path: str, *, input_role: str) -> None:
    """Reject sealed, diagnostic, or unreviewed source paths before scoring."""
    if input_role not in ALLOWED_INPUT_ROLES:
        raise DistillationError("Only human-reviewed training rows may be teacher-scored.")
    normalized = path.lower().replace("\\", "/")
    if any(token in normalized for token in FORBIDDEN_PATH_TOKENS):
        raise DistillationError("Teacher scoring may not read holdout, reserve, shadow, or successor data.")


def select_balanced_task_sample(
    pairs: Sequence[Mapping[str, Any]],
    *,
    max_tasks: int,
    random_state: int,
    require_binary_classes: bool = False,
) -> list[dict[str, Any]]:
    """Select complete task groups with deterministic role-family balancing.

    When requested, the selected candidate rows must contain both binary
    classes. This keeps small diagnostic samples valid without changing the
    task-group boundary or the deterministic full-sample selection.
    """
    if max_tasks <= 0:
        raise DistillationError("max_tasks must be positive.")
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    task_family: dict[str, str] = {}
    for pair in pairs:
        task_id = str(pair.get("task_id", ""))
        family = str(pair.get("role_family", ""))
        if not task_id or not family:
            raise DistillationError("Every pair needs task_id and role_family.")
        if task_id in task_family and task_family[task_id] != family:
            raise DistillationError("A task cannot span role families.")
        task_family[task_id] = family
        by_task[task_id].append(dict(pair))
    tasks_by_family: dict[str, list[str]] = defaultdict(list)
    for task_id, family in task_family.items():
        tasks_by_family[family].append(task_id)
    if max_tasks > len(by_task):
        raise DistillationError("max_tasks exceeds available complete task groups.")
    ordered_families = sorted(tasks_by_family)
    family_ranked = {
        family: sorted(
            task_ids,
            key=lambda task_id: hashlib.sha256(
                f"{random_state}:{family}:{task_id}".encode()
            ).hexdigest(),
        )
        for family, task_ids in tasks_by_family.items()
    }
    selected: list[str] = []
    cursor = 0
    while len(selected) < max_tasks:
        family = ordered_families[cursor % len(ordered_families)]
        if family_ranked[family]:
            selected.append(family_ranked[family].pop(0))
        elif all(not task_ids for task_ids in family_ranked.values()):
            break
        cursor += 1
    if len(selected) != max_tasks:
        raise DistillationError("Unable to build the requested balanced task sample.")
    if require_binary_classes:
        task_labels: dict[str, set[int]] = {}
        for task_id, task_pairs in by_task.items():
            labels = {int(pair.get("binary_label", -1)) for pair in task_pairs}
            if not labels <= {0, 1} or -1 in labels:
                raise DistillationError(
                    "Binary-class coverage requires binary_label on every pair."
                )
            task_labels[task_id] = labels
        if set().union(*task_labels.values()) != {0, 1}:
            raise DistillationError(
                "Binary-class coverage requires both classes in the source pairs."
            )
        selected_labels = set().union(*(task_labels[task_id] for task_id in selected))
        if selected_labels != {0, 1}:
            missing = ({0, 1} - selected_labels).pop()
            candidates = [
                task_id
                for task_id in by_task
                if task_id not in selected and missing in task_labels[task_id]
            ]
            candidates.sort(
                key=lambda task_id: (
                    task_family[task_id] not in {
                        task_family[selected_id] for selected_id in selected
                    },
                    hashlib.sha256(
                        f"{random_state}:binary-coverage:{task_id}".encode()
                    ).hexdigest(),
                )
            )
            replacement: tuple[int, str] | None = None
            for candidate in candidates:
                for index, selected_id in enumerate(selected):
                    remaining = selected[:index] + selected[index + 1 :] + [candidate]
                    if set().union(*(task_labels[task_id] for task_id in remaining)) == {
                        0,
                        1,
                    }:
                        same_family = (
                            task_family[candidate] == task_family[selected_id]
                        )
                        if same_family:
                            replacement = (index, candidate)
                            break
                        if replacement is None:
                            replacement = (index, candidate)
                if replacement is not None and (
                    task_family[candidate]
                    == task_family[selected[replacement[0]]]
                ):
                    break
            if replacement is None:
                raise DistillationError(
                    "Unable to preserve both binary classes in the requested sample."
                )
            selected[replacement[0]] = replacement[1]
    return [pair for task_id in selected for pair in by_task[task_id]]


def cosine_scores(
    requirement_embeddings: Sequence[Sequence[float]],
    evidence_embeddings: Sequence[Sequence[float]],
) -> list[float]:
    """Return finite cosine similarity scores for aligned embedding rows."""
    left = np.asarray(requirement_embeddings, dtype=np.float64)
    right = np.asarray(evidence_embeddings, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape or not len(left):
        raise DistillationError("Embedding matrices must be non-empty and aligned.")
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    if np.any(left_norm == 0) or np.any(right_norm == 0):
        raise DistillationError("Embedding vectors must be non-zero.")
    scores = np.sum(left * right, axis=1) / (left_norm * right_norm)
    if not np.all(np.isfinite(scores)):
        raise DistillationError("Embedding scores must be finite.")
    return scores.astype(float).tolist()


class OllamaEmbeddingTeacher:
    """Local-only Ollama embedding client used for a non-promoting baseline."""

    def __init__(
        self,
        model: str,
        *,
        endpoint: str = "http://127.0.0.1:11434/api/embed",
        request: Callable[[Request], bytes] | None = None,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self._request = request or self._post

    @staticmethod
    def _post(request: Request) -> bytes:
        try:
            with urlopen(request, timeout=120) as response:  # noqa: S310
                return response.read()
        except URLError as error:
            raise DistillationError(
                "Local Ollama is unavailable; start Ollama before scoring."
            ) from error

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed text locally; no request is sent beyond the loopback interface."""
        if not texts or any(not text.strip() for text in texts):
            raise DistillationError("Embedding input must contain non-empty text.")
        payload = json.dumps({"model": self.model, "input": list(texts)}).encode()
        request = Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            response = json.loads(self._request(request).decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DistillationError("Local Ollama returned invalid embedding JSON.") from error
        embeddings = response.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise DistillationError("Local Ollama returned misaligned embeddings.")
        try:
            return [[float(value) for value in row] for row in embeddings]
        except (TypeError, ValueError) as error:
            raise DistillationError("Local Ollama returned invalid embedding values.") from error


class LocalCrossEncoderTeacher:
    """Local-only cross-encoder wrapper for offline teacher diagnostics."""

    def __init__(
        self,
        model_path: str,
        *,
        instruction: str | None,
        batch_size: int = 4,
        device: str = "auto",
        predict: Callable[[Sequence[tuple[str, str]]], Sequence[float]] | None = None,
    ) -> None:
        if not Path(model_path).is_dir():
            raise DistillationError("Local reranker model directory does not exist.")
        if instruction is not None and not instruction.strip():
            raise DistillationError("Reranker instruction must be non-empty when provided.")
        if batch_size <= 0:
            raise DistillationError("Reranker batch size must be positive.")
        if device not in {"auto", "cpu", "mps"}:
            raise DistillationError("Reranker device must be auto, cpu, or mps.")
        self.model_path = model_path
        self.instruction = instruction
        self.batch_size = batch_size
        self.device = device
        self._predict = predict
        self._resolved_device = "injected"
        self._model_load_seconds = 0.0
        self._inference_seconds = 0.0

    def score(self, pairs: Sequence[Mapping[str, Any]]) -> list[float]:
        """Return finite support scores without sending source text off-device."""
        inputs = [
            (str(pair.get("requirement", "")), str(pair.get("evidence", "")))
            for pair in pairs
        ]
        if not inputs or any(not query.strip() or not document.strip() for query, document in inputs):
            raise DistillationError("Reranker inputs must contain requirement and evidence text.")
        if self._predict is None:
            import torch
            from sentence_transformers import CrossEncoder

            resolved_device = self.device
            if resolved_device == "auto":
                resolved_device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._resolved_device = resolved_device
            model_kwargs: dict[str, Any] = {
                "local_files_only": True,
                "device": resolved_device,
            }
            if self.instruction is not None:
                model_kwargs["prompts"] = {"evidence_support": self.instruction}
                model_kwargs["default_prompt_name"] = "evidence_support"
            load_started = time.perf_counter()
            model = CrossEncoder(
                self.model_path,
                **model_kwargs,
            )
            self._model_load_seconds = time.perf_counter() - load_started
            inference_started = time.perf_counter()
            values = model.predict(
                inputs,
                batch_size=self.batch_size,
                activation_fn=torch.nn.Sigmoid(),
                show_progress_bar=True,
            )
            self._inference_seconds = time.perf_counter() - inference_started
        else:
            inference_started = time.perf_counter()
            values = self._predict(inputs)
            self._inference_seconds = time.perf_counter() - inference_started
        scores = [float(value) for value in values]
        if len(scores) != len(inputs) or any(not math.isfinite(value) for value in scores):
            raise DistillationError("Local reranker returned invalid or misaligned scores.")
        return scores

    def efficiency_manifest(self, *, pairs: int) -> dict[str, Any]:
        """Return runtime diagnostics without persisting hardware identifiers."""
        if pairs <= 0 or self._inference_seconds <= 0:
            raise DistillationError("Efficiency metrics require a completed scoring run.")
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_bytes = int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)
        return {
            "device": self._resolved_device,
            "batch_size": self.batch_size,
            "model_load_seconds": self._model_load_seconds,
            "inference_seconds": self._inference_seconds,
            "pairs_per_second": pairs / self._inference_seconds,
            "peak_process_rss_bytes": peak_rss_bytes,
        }


def build_teacher_score_rows(
    pairs: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    teacher: TeacherSpec,
) -> list[dict[str, Any]]:
    """Persist only identifiers and scores, never requirement or evidence text."""
    if len(pairs) != len(scores):
        raise DistillationError("Teacher scores must align with input pairs.")
    rows: list[dict[str, Any]] = []
    for pair, score in zip(pairs, scores, strict=True):
        value = float(score)
        if not math.isfinite(value):
            raise DistillationError("Teacher scores must be finite.")
        pair_id = str(pair.get("pair_id", ""))
        task_id = str(pair.get("task_id", ""))
        if not pair_id or not task_id:
            raise DistillationError("Teacher score rows need pair_id and task_id.")
        rows.append(
            {
                "pair_id": pair_id,
                "role_family": str(pair["role_family"]),
                "score": value,
                "task_id": task_id,
                "teacher_id": teacher.teacher_id,
            }
        )
    return rows


def score_manifest(
    pairs: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    *,
    teacher: TeacherSpec,
    input_role: str,
) -> dict[str, Any]:
    """Return aggregate provenance without persisting source text or labels."""
    if input_role not in ALLOWED_INPUT_ROLES:
        raise DistillationError("Teacher score manifest has an unsafe input role.")
    payload = "\n".join(
        json.dumps(dict(pair), sort_keys=True, separators=(",", ":"))
        for pair in pairs
    ).encode()
    return {
        "schema_version": DISTILLATION_SCHEMA_VERSION,
        "experiment": f"offline_{teacher.kind}",
        "teacher": teacher.manifest(),
        "input_role": input_role,
        "input_pairs": len(pairs),
        "tasks": len({str(pair["task_id"]) for pair in pairs}),
        "score_rows": len(score_rows),
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "source_text_persisted": False,
        "teacher_output_is_gold": False,
        "model_selection_allowed": False,
        "holdout_accessed": False,
        "product_integration_allowed": False,
    }


def training_score_diagnostics(
    pairs: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
) -> dict[str, Any]:
    """Summarize reviewed-training scores without selecting a decision threshold."""
    if len(pairs) != len(scores) or not pairs:
        raise DistillationError("Diagnostic scores must align with non-empty input pairs.")
    labels = [int(pair["binary_label"]) for pair in pairs]
    if set(labels) != {0, 1}:
        raise DistillationError("Training diagnostics require both binary classes.")
    values = [float(score) for score in scores]
    if any(not math.isfinite(value) for value in values):
        raise DistillationError("Diagnostic scores must be finite.")

    label_scores: dict[str, list[float]] = defaultdict(list)
    tasks: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for pair, label, score in zip(pairs, labels, values, strict=True):
        support_label = str(pair["support_label"])
        label_scores[support_label].append(score)
        tasks[str(pair["task_id"])].append((label, score))

    supported_ranks: list[int] = []
    for candidates in tasks.values():
        if not any(label == 1 for label, _ in candidates):
            continue
        ordered = sorted(candidates, key=lambda item: item[1], reverse=True)
        supported_ranks.append(
            next(index for index, (label, _) in enumerate(ordered, start=1) if label == 1)
        )

    return {
        "diagnostic_scope": "human_reviewed_training_sample",
        "model_selection_allowed": False,
        "threshold_selected": False,
        "binary_auc": float(roc_auc_score(labels, values)),
        "mean_score_by_human_label": {
            label: float(np.mean(label_scores[label]))
            for label in ("Direct", "Partial", "No Support")
            if label_scores[label]
        },
        "supported_tasks": len(supported_ranks),
        "supported_task_top1": float(np.mean([rank == 1 for rank in supported_ranks])),
        "supported_task_mrr": float(np.mean([1.0 / rank for rank in supported_ranks])),
    }


def training_score_slices(
    pairs: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    slice_field: str,
) -> dict[str, dict[str, Any]]:
    """Compute descriptive training diagnostics for pre-existing metadata slices."""
    if len(pairs) != len(scores) or not pairs:
        raise DistillationError("Slice scores must align with non-empty input pairs.")
    grouped: dict[str, list[tuple[Mapping[str, Any], float]]] = defaultdict(list)
    for pair, score in zip(pairs, scores, strict=True):
        value = pair.get(slice_field)
        if value in (None, ""):
            raise DistillationError(f"Slice field {slice_field!r} is unavailable.")
        grouped[str(value)].append((pair, float(score)))

    report: dict[str, dict[str, Any]] = {}
    for name, rows in sorted(grouped.items()):
        slice_pairs = [pair for pair, _ in rows]
        slice_scores = [score for _, score in rows]
        labels = {int(pair["binary_label"]) for pair in slice_pairs}
        diagnostics: dict[str, Any] = {
            "pairs": len(rows),
            "tasks": len({str(pair["task_id"]) for pair in slice_pairs}),
            "mean_score": float(np.mean(slice_scores)),
            "binary_auc": (
                float(roc_auc_score(
                    [int(pair["binary_label"]) for pair in slice_pairs],
                    slice_scores,
                ))
                if labels == {0, 1}
                else None
            ),
        }
        task_ids = {str(pair["task_id"]) for pair in slice_pairs}
        task_complete = all(
            all(
                str(candidate["task_id"]) != task_id
                or str(candidate.get(slice_field)) == name
                for candidate in pairs
            )
            for task_id in task_ids
        )
        if labels == {0, 1} and task_complete:
            task_diagnostics = training_score_diagnostics(slice_pairs, slice_scores)
            diagnostics["supported_task_top1"] = task_diagnostics["supported_task_top1"]
            diagnostics["supported_task_mrr"] = task_diagnostics["supported_task_mrr"]
        else:
            diagnostics["supported_task_top1"] = None
            diagnostics["supported_task_mrr"] = None
        report[name] = diagnostics
    return report


def training_ranking_errors(
    pairs: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
) -> list[dict[str, Any]]:
    """Return identifier-only supported-task ranking errors for local review."""
    if len(pairs) != len(scores) or not pairs:
        raise DistillationError("Ranking-error scores must align with input pairs.")
    tasks: dict[str, list[tuple[Mapping[str, Any], float]]] = defaultdict(list)
    for pair, score in zip(pairs, scores, strict=True):
        tasks[str(pair["task_id"])].append((pair, float(score)))
    errors: list[dict[str, Any]] = []
    for task_id, candidates in sorted(tasks.items()):
        positives = [row for row in candidates if int(row[0]["binary_label"]) == 1]
        if not positives:
            continue
        ordered = sorted(candidates, key=lambda row: row[1], reverse=True)
        if int(ordered[0][0]["binary_label"]) == 1:
            continue
        best_positive = max(positives, key=lambda row: row[1])
        errors.append(
            {
                "task_id": task_id,
                "role_family": str(ordered[0][0]["role_family"]),
                "top_pair_id": str(ordered[0][0]["pair_id"]),
                "top_score": ordered[0][1],
                "best_supported_pair_id": str(best_positive[0]["pair_id"]),
                "best_supported_score": best_positive[1],
            }
        )
    return errors


def build_teacher_error_audit_queue(
    pairs: Sequence[Mapping[str, Any]],
    embedding_scores: Sequence[float],
    reranker_scores: Sequence[float],
    *,
    max_business_inversions: int = 8,
) -> list[dict[str, Any]]:
    """Build a local diagnostic queue without changing or hiding human labels."""
    if (
        len(pairs) != len(embedding_scores)
        or len(pairs) != len(reranker_scores)
        or not pairs
    ):
        raise DistillationError("Audit scores must align with non-empty input pairs.")
    if max_business_inversions < 0:
        raise DistillationError("Business inversion limit cannot be negative.")

    tasks: dict[str, list[tuple[Mapping[str, Any], float, float]]] = defaultdict(list)
    for pair, embedding, reranker in zip(
        pairs,
        embedding_scores,
        reranker_scores,
        strict=True,
    ):
        tasks[str(pair["task_id"])].append((pair, float(embedding), float(reranker)))

    reasons: dict[str, set[str]] = defaultdict(set)
    for error in training_ranking_errors(pairs, reranker_scores):
        reasons[str(error["task_id"])].add("qwen_top1_error")

    inversions: list[tuple[bool, float, str]] = []
    for task_id, candidates in tasks.items():
        if not candidates or str(candidates[0][0]["role_family"]) != "Business":
            continue
        positives = [row for row in candidates if int(row[0]["binary_label"]) == 1]
        negatives = [row for row in candidates if int(row[0]["binary_label"]) == 0]
        for positive in positives:
            for negative in negatives:
                reranker_margin = negative[2] - positive[2]
                if reranker_margin <= 0:
                    continue
                embedding_was_correct = positive[1] > negative[1]
                inversions.append((embedding_was_correct, reranker_margin, task_id))
    selected_business: list[str] = []
    for _, _, task_id in sorted(
        inversions,
        key=lambda row: (not row[0], -row[1], row[2]),
    ):
        if task_id not in selected_business:
            selected_business.append(task_id)
        if len(selected_business) == max_business_inversions:
            break
    for task_id in selected_business:
        reasons[task_id].add("business_pair_inversion")

    queue: list[dict[str, Any]] = []
    for task_id in sorted(reasons):
        candidates = tasks[task_id]
        queue.append(
            {
                "audit_id": hashlib.sha256(f"teacher-audit:{task_id}".encode()).hexdigest()[:16],
                "task_id": task_id,
                "role_family": str(candidates[0][0]["role_family"]),
                "audit_reasons": sorted(reasons[task_id]),
                "requirement": str(candidates[0][0]["requirement"]),
                "candidates": [
                    {
                        "pair_id": str(pair["pair_id"]),
                        "evidence": str(pair["evidence"]),
                        "human_label": str(pair["support_label"]),
                        "embedding_score": embedding,
                        "reranker_score": reranker,
                    }
                    for pair, embedding, reranker in sorted(
                        candidates,
                        key=lambda row: str(row[0]["pair_id"]),
                    )
                ],
                "diagnostic_only": True,
                "human_gold_changes_allowed": False,
                "model_selection_allowed": False,
                "holdout_accessed": False,
            }
        )
    return queue
