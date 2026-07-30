"""Tests for the sentence-embedding candidate freeze script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "ml"
    / "freeze_sentence_embedding_candidate.py"
)


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("candidate_freeze_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_freeze_requires_stable_development_gate(tmp_path: Path) -> None:
    module = _load_script()
    report_path = tmp_path / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    training_dir = tmp_path / "training"
    training_dir.mkdir()

    with pytest.raises(SystemExit, match="Development gate is not stable"):
        module.build_specification(
            {"development_gate": {"stable_improvement": False}},
            report_path=report_path,
            training_dir=training_dir,
        )
