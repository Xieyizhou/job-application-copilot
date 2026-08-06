"""Validate content-free lifecycle commitments for successor source pools."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from ml.source_pool_partition import assignment_sha256


DEVELOPMENT_CANDIDATE_STATE = (
    "training_candidate_requires_full_qa_and_human_labels"
)
FROZEN_DEVELOPMENT_STATE = (
    "source_assignment_frozen_awaiting_quality_audit"
)
MATERIALIZED_DEVELOPMENT_STATE = (
    "development_materialized_awaiting_manual_quality_audit"
)
DEVELOPMENT_STATES = {
    DEVELOPMENT_CANDIDATE_STATE,
    FROZEN_DEVELOPMENT_STATE,
    MATERIALIZED_DEVELOPMENT_STATE,
}
SEALED_HOLDOUT_STATES = {
    "sealed_retired_unopened",
    "sealed_operational_holdout_candidate",
}
DEVELOPMENT_BATCH_STATE = "awaiting_manual_stratified_quality_audit"


def load_source_lifecycle(path: Path) -> dict[str, Any]:
    """Load and validate a public, content-free lifecycle ledger."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Source lifecycle ledger must be an object.")
    validate_source_lifecycle(payload)
    return payload


def validate_source_lifecycle(payload: Mapping[str, Any]) -> None:
    """Reject lifecycle states that could leak failed data into evaluation."""
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported source lifecycle schema.")
    global_policy = payload.get("global_policy")
    if not isinstance(global_policy, Mapping) or any(
        global_policy.get(field) is not False
        for field in (
            "development_model_selection_allowed",
            "failed_development_direct_training_allowed",
            "holdout_content_access_before_model_freeze",
            "product_integration_allowed",
        )
    ):
        raise ValueError("Source lifecycle global policy is unsafe.")
    rows = payload.get("source_partitions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Source lifecycle needs partition rows.")
    ids: set[str] = set()
    versions: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Lifecycle partition row must be an object.")
        partition_id = str(row.get("partition_id", ""))
        version = str(row.get("version", ""))
        if not partition_id or partition_id in ids or version in versions:
            raise ValueError("Lifecycle partition identities must be unique.")
        ids.add(partition_id)
        versions.add(version)
        development = row.get("development")
        holdout = row.get("holdout")
        if not isinstance(development, Mapping) or not isinstance(
            holdout, Mapping
        ):
            raise ValueError("Lifecycle row needs development and holdout states.")
        if development.get("state") not in DEVELOPMENT_STATES:
            raise ValueError("Unknown development lifecycle state.")
        if development.get("human_labels_available") is not False:
            raise ValueError("Unreviewed development cannot claim human labels.")
        if development.get("state") == FROZEN_DEVELOPMENT_STATE and (
            development.get("content_materialized") is not False
            or development.get("quality_audit_completed") is not False
        ):
            raise ValueError("Frozen development cannot claim unopened work completed.")
        if development.get("state") == MATERIALIZED_DEVELOPMENT_STATE and (
            development.get("content_materialized") is not True
            or development.get("quality_audit_completed") is not True
            or development.get("automated_quality_gate_passed") is not True
        ):
            raise ValueError(
                "Materialized development must record its automated quality gate."
            )
        if holdout.get("state") not in SEALED_HOLDOUT_STATES:
            raise ValueError("Unknown sealed holdout lifecycle state.")
        if holdout.get("content_opened") is not False or holdout.get(
            "labels_opened"
        ) is not False:
            raise ValueError("Lifecycle ledger cannot mark holdout content opened.")
    development_batches = payload.get("development_batches", [])
    if not isinstance(development_batches, list):
        raise ValueError("Development batches must be a list.")
    for batch in development_batches:
        if not isinstance(batch, Mapping):
            raise ValueError("Development batch must be an object.")
        if batch.get("state") != DEVELOPMENT_BATCH_STATE:
            raise ValueError("Unknown development-batch state.")
        if (
            batch.get("automated_quality_gate_passed") is not True
            or batch.get("manual_quality_gate_passed") is not False
            or batch.get("human_labels_available") is not False
            or batch.get("new_holdout_created") is not False
            or batch.get("product_integration_allowed") is not False
        ):
            raise ValueError("Development batch lifecycle is unsafe.")
        if batch.get("existing_holdout_partition_id") != (
            "successor_v6_source_pool"
        ):
            raise ValueError("Development batch must preserve the v6 holdout.")


def validate_local_partition_integrity(
    ledger: Mapping[str, Any],
    source_partition_root: Path,
) -> dict[str, int]:
    """Verify local assignment checksums without reading any source content."""
    checked = 0
    holdouts_opened = 0
    for row in ledger["source_partitions"]:
        directory = source_partition_root / str(row["version"])
        manifest = json.loads(
            (directory / "partition_manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("partition_id") != row["partition_id"]:
            raise ValueError("Local partition id differs from lifecycle ledger.")
        for side, filename, checksum_field in (
            (
                "development",
                "development_source_assignments.json",
                "development_assignment_sha256",
            ),
            (
                "holdout",
                "sealed_holdout_source_assignments.json",
                "sealed_holdout_assignment_sha256",
            ),
        ):
            payload = json.loads(
                (directory / filename).read_text(encoding="utf-8")
            )
            assignments = payload.get("assignments")
            if not isinstance(assignments, list):
                raise ValueError("Local lifecycle assignment is malformed.")
            checksum = assignment_sha256(assignments)
            if checksum != row[side]["assignment_sha256"] or checksum != manifest.get(
                checksum_field
            ):
                raise ValueError("Local lifecycle assignment checksum changed.")
            if any(
                bool(payload.get(field))
                for field in (
                    "content_included",
                    "labels_included",
                    "model_signals_included",
                )
            ):
                raise ValueError("Lifecycle assignment exposes restricted content.")
        if manifest.get("holdout_content_opened") is not False or manifest.get(
            "holdout_labels_opened"
        ) is not False:
            holdouts_opened += 1
        checked += 1
    if holdouts_opened:
        raise ValueError("A sealed lifecycle holdout has been opened.")
    development_batches_checked = 0
    for batch in ledger.get("development_batches", []):
        directory = source_partition_root / str(batch["version"])
        if (directory / "sealed_holdout_source_assignments.json").exists():
            raise ValueError("Development-only batch created a new holdout.")
        manifest = json.loads(
            (directory / "partition_manifest.json").read_text(encoding="utf-8")
        )
        payload = json.loads(
            (directory / "development_source_assignments.json").read_text(
                encoding="utf-8"
            )
        )
        assignments = payload.get("assignments")
        if not isinstance(assignments, list):
            raise ValueError("Development-batch assignment is malformed.")
        checksum = assignment_sha256(assignments)
        if (
            checksum != batch["assignment_sha256"]
            or checksum != manifest.get("development_assignment_sha256")
        ):
            raise ValueError("Development-batch assignment checksum changed.")
        if (
            manifest.get("new_holdout_created") is not False
            or manifest.get("existing_holdout_partition_id")
            != batch["existing_holdout_partition_id"]
        ):
            raise ValueError("Development batch changed its holdout boundary.")
        development_batches_checked += 1
    return {
        "partitions_checked": checked,
        "development_batches_checked": development_batches_checked,
        "holdouts_opened": holdouts_opened,
    }
