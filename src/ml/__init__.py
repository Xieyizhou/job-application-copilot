"""Offline machine-learning utilities for the local job toolkit."""

from ml.inference import (
    local_model_status,
    portable_text_similarity,
    portable_text_similarities,
    predict_relevance,
    predict_relevance_batch,
    suppress_collapsed_relevance_signals,
)
from ml.jd_quality import classify_jd_quality

__all__ = [
    "local_model_status",
    "portable_text_similarity",
    "portable_text_similarities",
    "predict_relevance",
    "predict_relevance_batch",
    "suppress_collapsed_relevance_signals",
    "classify_jd_quality",
]
