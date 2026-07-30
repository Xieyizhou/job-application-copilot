from __future__ import annotations

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation_translation import (
    AnnotationTranslationError,
    candidate_translation,
    index_translation_rows,
    requirement_translation,
    verified_task_translation,
)


TASK = {
    "task_id": "task-1",
    "requirement": "Build recurring SQL reports.",
    "candidates": [
        {"candidate_id": "candidate-b", "evidence": "Prepared meeting notes."},
        {"candidate_id": "candidate-a", "evidence": "Automated monthly SQL reports."},
    ],
}
TRANSLATION = {
    "task_id": "task-1",
    "requirement": {
        "source": "Build recurring SQL reports.",
        "zh": "构建定期 SQL 报表。",
    },
    "candidates": [
        {
            "candidate_id": "candidate-a",
            "source": "Automated monthly SQL reports.",
            "zh": "自动生成每月 SQL 报表。",
        },
        {
            "candidate_id": "candidate-b",
            "source": "Prepared meeting notes.",
            "zh": "整理会议记录。",
        },
    ],
}


def test_translation_sidecar_is_candidate_order_independent() -> None:
    indexed = index_translation_rows([TRANSLATION])
    verified = verified_task_translation(TASK, indexed)

    assert requirement_translation(verified) == "构建定期 SQL 报表。"
    assert candidate_translation(verified, "candidate-a") == "自动生成每月 SQL 报表。"


def test_translation_is_hidden_when_english_source_changes() -> None:
    changed = {**TASK, "requirement": "Build daily SQL reports."}

    assert verified_task_translation(
        changed,
        index_translation_rows([TRANSLATION]),
    ) is None


def test_translation_sidecar_rejects_duplicate_task_ids() -> None:
    with pytest.raises(AnnotationTranslationError, match="Duplicate"):
        index_translation_rows([TRANSLATION, TRANSLATION])
