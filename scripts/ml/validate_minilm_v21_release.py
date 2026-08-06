"""Validate the public v21 release contract and optional ignored local bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = PROJECT_ROOT / "config" / "ml_minilm_v21_release.json"
DEFAULT_BUNDLE = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "distillation"
    / "web_replacement_v1"
    / "web_candidate_minilm_v21_e15"
)


class ReleaseContractError(ValueError):
    """Raised when the public release contract or local artifact is inconsistent."""


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for one local artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    """Load and validate the content-free public v21 contract."""
    contract = json.loads(path.read_text(encoding="utf-8"))
    training = contract.get("training", {})
    evaluation = contract.get("frozen_evaluation", {})
    data = contract.get("data_contract", {})
    authorization = contract.get("authorization", {})
    if (
        contract.get("schema_version") != 1
        or contract.get("release_id") != "minilm-v21-possessive-ranking-e15"
        or training.get("tasks") != 3024
        or training.get("pairs") != 12096
        or evaluation.get("dataset_id")
        != "teacher_eval_e15_frozen_shuffled_agency_proxy"
        or evaluation.get("tasks") != 96
        or evaluation.get("pairs") != 384
        or data.get("teacher_labels_are_human_gold") is not False
        or data.get("shadow_or_holdout_used_for_training") is not False
        or authorization.get("web_shadow_allowed") is not True
        or authorization.get("product_integration_allowed") is not False
        or authorization.get("demo_integration_allowed") is not False
        or authorization.get("rollout_allowed") is not False
    ):
        raise ReleaseContractError("Unsafe or inconsistent MiniLM v21 release contract.")
    return contract


def validate_local_bundle(contract: dict[str, Any], bundle_dir: Path) -> dict[str, Any]:
    """Validate an optional ignored bundle without exposing model or teacher content."""
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ReleaseContractError("Local v21 bundle manifest is missing.")
    expected_manifest = str(contract["bundle"]["manifest_sha256"])
    if sha256(manifest_path) != expected_manifest:
        raise ReleaseContractError("Local v21 bundle manifest hash does not match.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_path = bundle_dir / "model_state.pt"
    if (
        manifest.get("model_version") != contract["release_id"]
        or manifest.get("model_type") != contract["bundle"]["model_type"]
        or manifest.get("decision_contract")
        != contract["bundle"]["decision_contract"]
        or manifest.get("product_integration_allowed") is not False
        or manifest.get("demo_integration_allowed") is not False
        or not model_path.is_file()
        or sha256(model_path) != contract["training"]["model_state_sha256"]
    ):
        raise ReleaseContractError("Local v21 bundle content does not match the contract.")
    return {
        "bundle_present": True,
        "bundle_manifest_sha256": expected_manifest,
        "model_state_sha256": contract["training"]["model_state_sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--require-local-artifacts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = load_contract(args.contract)
    result: dict[str, Any] = {
        "release_id": contract["release_id"],
        "public_contract_valid": True,
        "row_level_data_read": False,
        "product_integration_allowed": False,
    }
    if args.bundle_dir.is_dir():
        result.update(validate_local_bundle(contract, args.bundle_dir))
    elif args.require_local_artifacts:
        raise SystemExit("Local v21 artifacts are required but unavailable.")
    else:
        result["bundle_present"] = False
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
