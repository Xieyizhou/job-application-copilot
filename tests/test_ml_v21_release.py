"""Tests for the content-free MiniLM v21 release contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ml.validate_minilm_v21_release import (
    DEFAULT_CONTRACT,
    ReleaseContractError,
    load_contract,
    validate_local_bundle,
)


def test_public_contract_matches_final_v21_and_blocks_product_rollout() -> None:
    contract = load_contract(DEFAULT_CONTRACT)

    assert contract["training"]["tasks"] == 3024
    assert contract["training"]["pairs"] == 12096
    assert contract["frozen_evaluation"]["task_agreement"] == 0.90625
    assert contract["authorization"]["web_shadow_allowed"] is True
    assert contract["authorization"]["product_integration_allowed"] is False


def test_contract_rejects_teacher_proxy_as_human_gold(tmp_path: Path) -> None:
    contract = load_contract(DEFAULT_CONTRACT)
    contract["data_contract"]["teacher_labels_are_human_gold"] = True
    path = tmp_path / "unsafe.json"
    path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ReleaseContractError, match="Unsafe"):
        load_contract(path)


def test_local_bundle_validation_checks_manifest_hash(tmp_path: Path) -> None:
    contract = load_contract(DEFAULT_CONTRACT)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ReleaseContractError, match="manifest hash"):
        validate_local_bundle(contract, bundle)
