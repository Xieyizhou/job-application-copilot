"""Deterministic pair features for the local relevance model."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import TypeVar

TOKEN_PATTERN = re.compile(r"(?u)\b[\w+#.-]{2,}\b")
FEATURE_NAMES = (
    "token_jaccard",
    "job_token_recall",
    "resume_token_precision",
    "bigram_jaccard",
    "length_ratio",
    "log_resume_tokens",
    "log_job_tokens",
)

FeatureToken = TypeVar("FeatureToken")


def lexical_tokens(text: str) -> tuple[str, ...]:
    """Return normalized lexical tokens without retaining source text."""
    return tuple(TOKEN_PATTERN.findall(str(text).lower()))


def _jaccard(first: set[FeatureToken], second: set[FeatureToken]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def pair_feature_vector(resume_text: str, job_text: str) -> list[float]:
    """Create compact, explainable resume/job overlap features."""
    resume_tokens = lexical_tokens(resume_text)
    job_tokens = lexical_tokens(job_text)
    resume_set = set(resume_tokens)
    job_set = set(job_tokens)
    shared = resume_set & job_set
    resume_bigrams = set(zip(resume_tokens, resume_tokens[1:]))
    job_bigrams = set(zip(job_tokens, job_tokens[1:]))
    longest = max(len(resume_tokens), len(job_tokens), 1)
    shortest = min(len(resume_tokens), len(job_tokens))
    return [
        _jaccard(resume_set, job_set),
        len(shared) / len(job_set) if job_set else 0.0,
        len(shared) / len(resume_set) if resume_set else 0.0,
        _jaccard(resume_bigrams, job_bigrams),
        shortest / longest,
        math.log1p(len(resume_tokens)),
        math.log1p(len(job_tokens)),
    ]


def pair_feature_matrix(
    resume_texts: Sequence[str],
    job_texts: Sequence[str],
) -> object:
    """Create a numeric matrix for aligned resume/job inputs."""
    import numpy as np

    if len(resume_texts) != len(job_texts):
        raise ValueError("resume_texts and job_texts must have equal lengths")
    return np.asarray(
        [pair_feature_vector(resume, job) for resume, job in zip(resume_texts, job_texts)],
        dtype=np.float64,
    ).reshape(len(resume_texts), len(FEATURE_NAMES))
