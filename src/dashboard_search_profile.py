"""Local resume signals used to suggest a practical job-search query."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SearchProfile:
    """A concise, editable search suggestion derived from resume text."""

    query: str
    keywords: tuple[str, ...]
    source_ready: bool


ROLE_SIGNALS: tuple[tuple[str, tuple[tuple[str, int], ...]], ...] = (
    (
        "Machine Learning Engineer",
        (
            ("machine learning", 6),
            ("algorithm engineer", 5),
            ("neural network", 4),
            ("reinforcement learning", 4),
            ("ml pipeline", 4),
            ("scikit learn", 3),
            ("model evaluation", 3),
            ("convolutional neural network", 3),
            ("cnn", 2),
            ("python", 1),
        ),
    ),
    (
        "Robotics Engineer",
        (
            ("robotics", 6),
            ("autonomous systems", 5),
            ("path planning", 5),
            ("mavsdk", 4),
            ("gazebo", 4),
            ("px4", 4),
            ("uav", 3),
        ),
    ),
    (
        "Data Analyst",
        (
            ("data analyst", 7),
            ("data analysis", 5),
            ("dashboard", 4),
            ("sql", 3),
            ("data visualization", 3),
            ("excel", 2),
            ("reporting", 2),
        ),
    ),
    (
        "Data Scientist",
        (
            ("data scientist", 7),
            ("data science", 5),
            ("predictive model", 4),
            ("statistics", 3),
            ("classification", 3),
            ("econometrics", 3),
            ("pca", 2),
        ),
    ),
    (
        "Software Engineer",
        (
            ("software engineer", 7),
            ("backend", 4),
            ("frontend", 4),
            ("full stack", 4),
            ("software development", 3),
            ("api development", 3),
            ("c sharp", 2),
            ("java", 2),
        ),
    ),
)


def normalize_resume_text(text: str) -> str:
    """Normalize punctuation so resume signals match without fuzzy guesses."""
    return " ".join(re.sub(r"[^a-z0-9+#]+", " ", text.lower()).split())


def infer_search_profile(text: str) -> SearchProfile:
    """Return the strongest supported role and up to three visible keywords."""
    normalized = normalize_resume_text(text)
    if not normalized:
        return SearchProfile("Entry Level", (), False)

    scored: list[tuple[int, int, str, tuple[str, ...]]] = []
    for rank, (query, signals) in enumerate(ROLE_SIGNALS):
        matched = tuple(signal for signal, _weight in signals if signal in normalized)
        score = sum(weight for signal, weight in signals if signal in normalized)
        scored.append((score, -rank, query, matched))
    score, _rank, query, matched = max(scored)
    if score < 4:
        return SearchProfile("Entry Level", tuple(matched[:3]), True)
    return SearchProfile(query, tuple(matched[:3]), True)


def search_profile_from_path(path: Path | None) -> SearchProfile:
    """Read the canonical uploaded resume and safely infer its search profile."""
    if path is None or not path.is_file():
        return SearchProfile("Entry Level", (), False)
    try:
        return infer_search_profile(path.read_text(encoding="utf-8"))
    except OSError:
        return SearchProfile("Entry Level", (), False)
