"""Offline machine-learning utilities for the local job toolkit."""

from ml.inference import (
    local_model_status,
    portable_text_similarity,
    portable_text_similarities,
    predict_relevance,
    predict_relevance_batch,
    suppress_collapsed_relevance_signals,
)
from ml.jd_quality import (
    JDQualityError,
    assert_cover_letter_jd_ready,
    classify_jd_quality,
    jd_quality_warning_messages,
)

__all__ = [
    "local_model_status",
    "portable_text_similarity",
    "portable_text_similarities",
    "predict_relevance",
    "predict_relevance_batch",
    "suppress_collapsed_relevance_signals",
    "classify_jd_quality",
    "assert_cover_letter_jd_ready",
    "jd_quality_warning_messages",
    "JDQualityError",
]
