"""Pure-Python loading and inference for the optional local relevance model."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence
import unicodedata

from ml.features import FEATURE_NAMES, pair_feature_vector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "ml" / "models" / "relevance_baseline.json"
MODEL_SCHEMA_VERSION = 1
SKLEARN_TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


def local_model_status(model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Return portable artifact availability without importing ML packages."""
    return {
        "available": model_path.is_file(),
        "path": str(model_path),
        "status": "artifact_found" if model_path.is_file() else "not_trained",
    }


@lru_cache(maxsize=4)
def _load_artifact(path: str, modified_ns: int) -> dict[str, Any]:
    _ = modified_ns
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise ValueError("model artifact must be an object")
    if artifact.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported model artifact schema")
    if artifact.get("model_type") != "portable_pair_relevance":
        raise ValueError("unsupported model artifact type")
    vectorizer = dict(artifact.get("vectorizer", {}))
    classifier = dict(artifact.get("classifier", {}))
    vocabulary = dict(vectorizer.get("vocabulary", {}))
    idf = list(vectorizer.get("idf", []))
    coefficients = list(classifier.get("coefficients", []))
    if not vocabulary or len(idf) != len(vocabulary):
        raise ValueError("invalid vectorizer parameters")
    if len(coefficients) != len(vocabulary) + len(FEATURE_NAMES):
        raise ValueError("invalid classifier coefficient count")
    return artifact


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "displayable": False,
        "probability": None,
        "label": "Unavailable",
        "threshold": None,
        "model_version": None,
        "reason": reason,
    }


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(character for character in normalized if not unicodedata.combining(character))


def _analyze(text: str) -> list[str]:
    tokens = SKLEARN_TOKEN_PATTERN.findall(_strip_accents(text.lower()))
    return [*tokens, *(f"{first} {second}" for first, second in zip(tokens, tokens[1:]))]


def _tfidf_vector(text: str, vocabulary: dict[str, int], idf: list[float]) -> dict[int, float]:
    counts = Counter(term for term in _analyze(text) if term in vocabulary)
    weighted = {
        vocabulary[term]: (1.0 + math.log(count)) * float(idf[vocabulary[term]])
        for term, count in counts.items()
    }
    norm = math.sqrt(sum(value * value for value in weighted.values()))
    if norm:
        return {index: value / norm for index, value in weighted.items()}
    return {}


def _probability(
    resume_text: str,
    job_text: str,
    artifact: dict[str, Any],
    vector_cache: dict[str, dict[int, float]],
) -> float:
    vectorizer = dict(artifact["vectorizer"])
    vocabulary = {str(term): int(index) for term, index in dict(vectorizer["vocabulary"]).items()}
    idf = [float(value) for value in vectorizer["idf"]]
    classifier = dict(artifact["classifier"])
    coefficients = [float(value) for value in classifier["coefficients"]]

    def vector(text: str) -> dict[int, float]:
        if text not in vector_cache:
            vector_cache[text] = _tfidf_vector(text, vocabulary, idf)
        return vector_cache[text]

    resume_vector = vector(resume_text)
    job_vector = vector(job_text)
    shared_score = sum(
        coefficients[index] * resume_value * job_vector[index]
        for index, resume_value in resume_vector.items()
        if index in job_vector
    )
    numeric_values = pair_feature_vector(resume_text, job_text)
    numeric_offset = len(vocabulary)
    numeric_score = sum(
        coefficients[numeric_offset + index] * value
        for index, value in enumerate(numeric_values)
    )
    decision = float(classifier["intercept"]) + shared_score + numeric_score
    if decision >= 0:
        return 1.0 / (1.0 + math.exp(-decision))
    exponential = math.exp(decision)
    return exponential / (1.0 + exponential)


def predict_relevance_batch(
    pairs: Sequence[tuple[str, str]],
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> list[dict[str, Any]]:
    """Predict aligned pairs without loading sklearn, scipy, pandas, or joblib."""
    if not pairs:
        return []
    if not model_path.is_file():
        return [_unavailable("Local relevance model has not been trained.") for _ in pairs]
    results = [_unavailable("Resume or job text is empty.") for _ in pairs]
    valid_indexes = [index for index, (resume, job) in enumerate(pairs) if resume.strip() and job.strip()]
    if not valid_indexes:
        return results
    try:
        artifact = _load_artifact(str(model_path), model_path.stat().st_mtime_ns)
        threshold = float(artifact.get("threshold", 0.5))
        metadata = dict(artifact.get("metadata", {}))
        version = str(metadata.get("model_version", "local-v1"))
        vector_cache: dict[str, dict[int, float]] = {}
        for index in valid_indexes:
            probability = _probability(*pairs[index], artifact, vector_cache)
            results[index] = {
                "available": True,
                "displayable": True,
                "probability": probability,
                "label": "Supporting" if probability >= threshold else "Weak",
                "threshold": threshold,
                "model_version": version,
                "reason": "Auxiliary estimate from the optional local relevance model.",
            }
        return results
    except Exception as error:  # noqa: BLE001 - optional artifacts must never break the dashboard
        return [_unavailable(f"Local model could not be loaded: {error}") for _ in pairs]


def suppress_collapsed_relevance_signals(
    signals: Sequence[dict[str, Any]],
    *,
    minimum_batch_size: int = 5,
    collapse_ceiling: float = 0.02,
    minimum_range: float = 0.01,
) -> list[dict[str, Any]]:
    """Hide a synthetic-model batch when its outputs collapse out of distribution.

    Raw probabilities remain in the returned diagnostic records.  The display flag
    prevents the product UI from presenting a collapsed batch as meaningful fit.
    """
    copied = [dict(signal) for signal in signals]
    probabilities = [
        float(signal["probability"])
        for signal in copied
        if signal.get("available") and signal.get("probability") is not None
    ]
    collapsed = (
        len(probabilities) >= minimum_batch_size
        and max(probabilities) < collapse_ceiling
        and (max(probabilities) - min(probabilities)) < minimum_range
    )
    if not collapsed:
        return copied
    for signal in copied:
        if not signal.get("available"):
            continue
        signal["displayable"] = False
        signal["reason"] = (
            "Synthetic relevance outputs collapsed for this real-world batch; "
            "the signal is hidden until it is validated on representative data."
        )
    return copied


def predict_relevance(
    resume_text: str,
    job_text: str,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Predict one pair through the batch-safe inference path."""
    return predict_relevance_batch([(resume_text, job_text)], model_path=model_path)[0]


def portable_text_similarity(
    first_text: str,
    second_text: str,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> float | None:
    """Return portable TF-IDF cosine similarity, or None without a valid model."""
    return portable_text_similarities([(first_text, second_text)], model_path=model_path)[0]


def portable_text_similarities(
    pairs: Sequence[tuple[str, str]],
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> list[float | None]:
    """Return TF-IDF cosine similarities with one artifact load and local vector cache."""
    if not pairs:
        return []
    if not model_path.is_file():
        return [None for _ in pairs]
    try:
        artifact = _load_artifact(str(model_path), model_path.stat().st_mtime_ns)
        vectorizer = dict(artifact["vectorizer"])
        vocabulary = {
            str(term): int(index)
            for term, index in dict(vectorizer["vocabulary"]).items()
        }
        idf = [float(value) for value in vectorizer["idf"]]
        vector_cache: dict[str, dict[int, float]] = {}

        def vector(text: str) -> dict[int, float]:
            if text not in vector_cache:
                vector_cache[text] = _tfidf_vector(text, vocabulary, idf)
            return vector_cache[text]

        similarities: list[float | None] = []
        for first_text, second_text in pairs:
            if not first_text.strip() or not second_text.strip():
                similarities.append(None)
                continue
            first = vector(first_text)
            second = vector(second_text)
            similarities.append(
                float(sum(value * second[index] for index, value in first.items() if index in second))
            )
        return similarities
    except Exception:  # noqa: BLE001 - similarity enhancement is optional
        return [None for _ in pairs]
