"""Shared numeric reductions and pair alignment checks for evidence models."""

from collections.abc import Sequence

import numpy as np


def _mean(values: Sequence[bool | float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _validate_aligned(
    requirements: Sequence[str],
    evidence: Sequence[str],
) -> None:
    if len(requirements) != len(evidence):
        raise ValueError("requirements and evidence must have equal lengths")
    if not requirements:
        raise ValueError("at least one requirement/evidence pair is required")
