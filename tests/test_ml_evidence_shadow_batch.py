"""Tests for balanced shadow batch construction helpers."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_shadow_batch_output_must_stay_private(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts.ml import run_evidence_shadow_batch as script

    private = tmp_path / "shadow"
    monkeypatch.setattr(script, "SHADOW_ROOT", private)

    assert script._private_new_directory(private / "batch") == (
        private / "batch"
    ).resolve()
    with pytest.raises(SystemExit, match="must stay under"):
        script._private_new_directory(tmp_path / "public")
