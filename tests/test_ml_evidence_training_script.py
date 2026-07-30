from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import joblib
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "ml" / "train_evidence_reranker.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_train_evidence_reranker_script",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load evidence training script.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dataset(dataset_dir: Path) -> None:
    tasks = [
        {
            "task_id": "task-sql",
            "requirement": "Build recurring SQL reports.",
            "support_label": "Direct",
            "selected_candidate_id": "candidate-sql",
            "evaluation_group": "semantic:sql",
            "candidates": [
                {
                    "candidate_id": "candidate-sql",
                    "evidence": "Automated monthly reports with SQL.",
                }
            ],
        },
        {
            "task_id": "task-api",
            "requirement": "Deploy predictive services.",
            "support_label": "No Support",
            "selected_candidate_id": None,
            "evaluation_group": "semantic:api",
            "candidates": [
                {
                    "candidate_id": "candidate-api",
                    "evidence": "Prepared stakeholder meeting notes.",
                }
            ],
        },
    ]
    pairs = [
        {
            "pair_id": "pair-sql",
            "task_id": "task-sql",
            "requirement": "Build recurring SQL reports.",
            "evidence": "Automated monthly reports with SQL.",
            "binary_label": 1,
            "support_label": "Direct",
            "evaluation_group": "semantic:sql",
        },
        {
            "pair_id": "pair-api",
            "task_id": "task-api",
            "requirement": "Deploy predictive services.",
            "evidence": "Prepared stakeholder meeting notes.",
            "binary_label": 0,
            "support_label": "No Support",
            "evaluation_group": "semantic:api",
        },
    ]
    dataset_dir.mkdir()
    (dataset_dir / "annotated_tasks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in tasks),
        encoding="utf-8",
    )
    (dataset_dir / "training_pairs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in pairs),
        encoding="utf-8",
    )


def _report(selected_method: str) -> dict[str, object]:
    return {
        "experiment": "test",
        "evaluation_protocol": "test grouped protocol",
        "model_selection": {
            "selected_method": selected_method,
            "promotion_status": "blocked_until_fixed_real_holdout",
        },
        "method_threshold_medians": {
            "hybrid_lsa_reranker": 0.5,
        },
        "methods": {},
    }


def test_training_script_fits_selected_hybrid_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    dataset_dir = tmp_path / "dataset"
    _dataset(dataset_dir)
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        module,
        "run_annotation_experiment",
        lambda tasks, pairs, random_state: _report("hybrid_lsa_reranker"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--dataset-dir",
            str(dataset_dir),
            "--model-path",
            str(model_path),
            "--report-path",
            str(report_path),
        ],
    )

    module.main()

    artifact = joblib.load(model_path)
    assert artifact["schema_version"] == 2
    assert artifact["model_type"] == "hybrid_lsa_reranker"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["artifact_generation"]["status"] == "saved_experimental_artifact"


def test_training_script_saves_report_before_unsupported_winner_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script()
    dataset_dir = tmp_path / "dataset"
    _dataset(dataset_dir)
    model_path = tmp_path / "model.joblib"
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        module,
        "run_annotation_experiment",
        lambda tasks, pairs, random_state: _report("concept_lexical_rule"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--dataset-dir",
            str(dataset_dir),
            "--model-path",
            str(model_path),
            "--report-path",
            str(report_path),
        ],
    )

    with pytest.raises(SystemExit, match="comparison report was saved"):
        module.main()

    assert not model_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["artifact_generation"]["status"] == "comparison_complete_not_fitted"
