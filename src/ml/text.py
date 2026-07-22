"""Privacy-safe text normalization and hashing helpers."""

from __future__ import annotations

import hashlib
import re
import unicodedata


WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text_for_hash(text: str) -> str:
    """
    Normalize text conservatively for duplicate detection.

    This normalization:
    - converts Unicode compatibility variants
    - lowercases text
    - collapses repeated whitespace
    - removes surrounding whitespace

    It intentionally does not remove punctuation or words.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.lower()
    normalized = WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def sha256_text(text: str) -> str:
    """Return an exact SHA-256 hash for text."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_text_hash(text: str) -> str:
    """Return a SHA-256 hash after conservative normalization."""

    return sha256_text(normalize_text_for_hash(text))


def pair_hash(
    resume_text: str,
    job_text: str,
    *,
    normalized: bool,
) -> str:
    """Return a stable hash for an ordered resume-job pair."""

    if normalized:
        resume_value = normalize_text_for_hash(resume_text)
        job_value = normalize_text_for_hash(job_text)
    else:
        resume_value = resume_text
        job_value = job_text

    combined = (
        resume_value
        + "\0<RESUME_JOB_BOUNDARY>\0"
        + job_value
    )

    return sha256_text(combined)
