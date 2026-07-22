"""Privacy-preserving evaluation for real-derived local ML validation sets."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ml.evidence import score_evidence_pair
from ml.inference import DEFAULT_MODEL_PATH, predict_relevance_batch, suppress_collapsed_relevance_signals


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEMANTIC_MANIFEST = PROJECT_ROOT / "data" / "ml" / "validation" / "semantic_evidence_real_v1.json"
DEFAULT_RELEVANCE_MANIFEST = PROJECT_ROOT / "data" / "ml" / "validation" / "relevance_real_holdout_v1.json"
DEFAULT_REAL_PAIRS = PROJECT_ROOT / "data" / "ml" / "processed" / "canonical_pairs.parquet"
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RELEVANCE_LABELS = {"Good Fit", "Potential Fit", "No Fit"}


class ValidationManifestError(ValueError):
    """Raised when a validation manifest is malformed or privacy-unsafe."""


def load_manifest(path: Path) -> dict[str, Any]:
    """Load one versioned validation manifest."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValidationManifestError("validation manifest must use schema_version 1")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValidationManifestError("validation manifest must contain at least one case")
    case_ids = [str(case.get("case_id", "")) for case in cases if isinstance(case, dict)]
    if len(case_ids) != len(cases) or any(not case_id for case_id in case_ids):
        raise ValidationManifestError("every validation case needs a case_id")
    if len(case_ids) != len(set(case_ids)):
        raise ValidationManifestError("validation case_id values must be unique")
    return data


def validate_semantic_manifest(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate de-identified sentence-pair cases and return them."""
    cases = list(data["cases"])
    for case in cases:
        requirement = str(case.get("requirement", "")).strip()
        evidence = str(case.get("evidence", "")).strip()
        expected = case.get("expected_accepted")
        if not requirement or not evidence or not isinstance(expected, bool):
            raise ValidationManifestError(
                f"{case.get('case_id', 'unknown')} needs requirement, evidence, and expected_accepted"
            )
        if "@" in requirement or "@" in evidence or "http://" in evidence or "https://" in evidence:
            raise ValidationManifestError(f"{case['case_id']} contains contact or URL data")
    return cases


def evaluate_semantic_manifest(
    manifest_path: Path = DEFAULT_SEMANTIC_MANIFEST,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Evaluate evidence acceptance without exposing source documents."""
    data = load_manifest(manifest_path)
    cases = validate_semantic_manifest(data)
    rows: list[dict[str, Any]] = []
    true_positive = false_positive = true_negative = false_negative = 0
    for case in cases:
        score = score_evidence_pair(case["requirement"], case["evidence"], model_path=model_path)
        expected = bool(case["expected_accepted"])
        predicted = bool(score["accepted"])
        if expected and predicted:
            true_positive += 1
        elif expected:
            false_negative += 1
        elif predicted:
            false_positive += 1
        else:
            true_negative += 1
        rows.append(
            {
                "case_id": case["case_id"],
                "expected": expected,
                "predicted": predicted,
                "similarity": score["similarity"],
                "match_type": score["match_type"],
                "passed": expected == predicted,
            }
        )
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "dataset_id": data.get("dataset_id", ""),
        "status": "complete",
        "cases": len(cases),
        "accuracy": (true_positive + true_negative) / len(cases),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
        "failed_case_ids": [row["case_id"] for row in rows if not row["passed"]],
        "rows": rows,
    }


def validate_relevance_manifest(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Require hash-only real-data references and recognized weak labels."""
    cases = list(data["cases"])
    forbidden_fields = {"resume_text", "job_text", "name", "email", "phone", "employer"}
    for case in cases:
        if forbidden_fields & set(case):
            raise ValidationManifestError(f"{case.get('case_id', 'unknown')} contains raw or identifying fields")
        if not HASH_PATTERN.fullmatch(str(case.get("resume_sha256", ""))):
            raise ValidationManifestError(f"{case.get('case_id', 'unknown')} has an invalid resume hash")
        if not HASH_PATTERN.fullmatch(str(case.get("job_sha256", ""))):
            raise ValidationManifestError(f"{case.get('case_id', 'unknown')} has an invalid job hash")
        if case.get("source_label") not in RELEVANCE_LABELS:
            raise ValidationManifestError(f"{case.get('case_id', 'unknown')} has an invalid source label")
    return cases


def resolve_relevance_cases(cases: list[dict[str, Any]], pairs_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve hash references against the ignored local real-pair table."""
    try:
        import pandas as pd
    except ModuleNotFoundError as error:  # pragma: no cover - optional local evaluation dependency
        raise RuntimeError("pandas is required for real-pair evaluation") from error
    frame = pd.read_parquet(pairs_path)
    lookup = {
        (str(row["resume_hash"]), str(row["job_hash"])): row
        for row in frame.to_dict("records")
    }
    resolved: list[dict[str, Any]] = []
    missing: list[str] = []
    for case in cases:
        row = lookup.get((case["resume_sha256"], case["job_sha256"]))
        if row is None:
            missing.append(case["case_id"])
            continue
        resolved.append({"case": case, "resume_text": str(row["resume_text"]), "job_text": str(row["job_text"])})
    return resolved, missing


def evaluate_relevance_manifest(
    manifest_path: Path = DEFAULT_RELEVANCE_MANIFEST,
    *,
    pairs_path: Path = DEFAULT_REAL_PAIRS,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Evaluate aggregate domain-shift diagnostics from hash-only references."""
    data = load_manifest(manifest_path)
    cases = validate_relevance_manifest(data)
    if not pairs_path.is_file():
        return {"dataset_id": data.get("dataset_id", ""), "status": "pairs_unavailable", "cases": len(cases)}
    if not model_path.is_file():
        return {"dataset_id": data.get("dataset_id", ""), "status": "model_unavailable", "cases": len(cases)}
    resolved, missing = resolve_relevance_cases(cases, pairs_path)
    signals = predict_relevance_batch(
        [(item["resume_text"], item["job_text"]) for item in resolved],
        model_path=model_path,
    )
    guarded = suppress_collapsed_relevance_signals(signals)
    if any(not signal.get("available") for signal in signals):
        return {
            "dataset_id": data.get("dataset_id", ""),
            "status": "inference_unavailable",
            "cases": len(cases),
            "resolved": len(resolved),
            "missing_case_ids": missing,
        }
    probabilities = [float(signal["probability"]) for signal in signals]
    label_groups: dict[str, list[float]] = {}
    for item, probability in zip(resolved, probabilities):
        label_groups.setdefault(str(item["case"]["source_label"]), []).append(probability)
    threshold = float(signals[0]["threshold"]) if signals else 0.5
    binary_rows = [
        (item["case"]["source_label"] == "Good Fit", probability >= threshold)
        for item, probability in zip(resolved, probabilities)
        if item["case"]["source_label"] != "Potential Fit"
    ]
    binary_accuracy = (
        sum(expected == predicted for expected, predicted in binary_rows) / len(binary_rows)
        if binary_rows
        else 0.0
    )
    return {
        "dataset_id": data.get("dataset_id", ""),
        "status": "complete",
        "cases": len(cases),
        "resolved": len(resolved),
        "missing_case_ids": missing,
        "threshold": threshold,
        "probability_min": min(probabilities) if probabilities else None,
        "probability_max": max(probabilities) if probabilities else None,
        "probability_range": max(probabilities) - min(probabilities) if probabilities else None,
        "mean_probability_by_weak_label": {
            label: sum(values) / len(values)
            for label, values in sorted(label_groups.items())
        },
        "binary_accuracy_good_vs_no_fit": binary_accuracy,
        "displayable": all(bool(signal.get("displayable")) for signal in guarded),
        "collapsed": any(not bool(signal.get("displayable")) for signal in guarded),
        "weak_label_counts": dict(Counter(case["source_label"] for case in cases)),
    }
