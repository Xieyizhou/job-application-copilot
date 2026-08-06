"""Contracts for failed-development and sealed-holdout source lifecycle."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from ml.source_lifecycle import (
    DEVELOPMENT_CANDIDATE_STATE,
    MATERIALIZED_DEVELOPMENT_STATE,
    validate_local_partition_integrity,
    load_source_lifecycle,
    validate_source_lifecycle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = PROJECT_ROOT / "config" / "ml_source_lifecycle.json"


def test_public_lifecycle_freezes_v4_through_v9_without_content() -> None:
    ledger = load_source_lifecycle(LEDGER_PATH)
    rows = ledger["source_partitions"]

    assert [row["version"] for row in rows] == [
        "successor_v4",
        "successor_v5",
        "successor_v6",
        "successor_v7",
        "successor_v9",
    ]
    assert all(
        row["development"]["state"] == DEVELOPMENT_CANDIDATE_STATE
        for row in rows[:4]
    )
    assert rows[4]["development"]["state"] == (
        MATERIALIZED_DEVELOPMENT_STATE
    )
    assert rows[4]["development"]["content_materialized"] is True
    assert rows[4]["development"]["quality_audit_completed"] is True
    assert rows[4]["development"]["automated_quality_gate_passed"] is True
    assert all(
        row["development"]["human_labels_available"] is False
        for row in rows
    )
    assert all(row["holdout"]["content_opened"] is False for row in rows)
    assert all(row["holdout"]["labels_opened"] is False for row in rows)
    assert rows[2]["holdout"]["state"] == (
        "sealed_operational_holdout_candidate"
    )
    assert rows[4]["holdout"]["state"] == (
        "sealed_operational_holdout_candidate"
    )
    batch = ledger["development_batches"][0]
    assert batch["version"] == "successor_v8"
    assert batch["assignments"] == 24
    assert batch["resume_groups"] == 4
    assert batch["new_holdout_created"] is False
    assert batch["existing_holdout_partition_id"] == (
        "successor_v6_source_pool"
    )
    assert batch["human_labels_available"] is False


def test_local_lifecycle_integrity_includes_v9_assignments() -> None:
    ledger = load_source_lifecycle(LEDGER_PATH)

    result = validate_local_partition_integrity(
        ledger,
        PROJECT_ROOT / "data" / "ml" / "source_partitions",
    )

    assert result == {
        "partitions_checked": 5,
        "development_batches_checked": 1,
        "holdouts_opened": 0,
    }


def test_lifecycle_rejects_direct_training_and_open_holdout_claims() -> None:
    ledger = load_source_lifecycle(LEDGER_PATH)
    unsafe_training = deepcopy(ledger)
    unsafe_training["global_policy"][
        "failed_development_direct_training_allowed"
    ] = True
    with pytest.raises(ValueError, match="global policy is unsafe"):
        validate_source_lifecycle(unsafe_training)

    opened = deepcopy(ledger)
    opened["source_partitions"][2]["holdout"]["content_opened"] = True
    with pytest.raises(ValueError, match="cannot mark holdout content opened"):
        validate_source_lifecycle(opened)
