"""Tests for local relevance model training and inference."""

from __future__ import annotations

import tempfile
from pathlib import Path
import json
import unittest

from ml.inference import (
    predict_relevance,
    predict_relevance_batch,
    suppress_collapsed_relevance_signals,
)
from ml.relevance import PairRelevanceModel


def fitted_model() -> PairRelevanceModel:
    resumes = [
        "python sql analytics dashboard",
        "python pandas reporting",
        "java spring backend services",
        "java microservices kubernetes",
        "python statistics machine learning",
        "sales account management crm",
    ]
    jobs = [
        "python sql data analyst",
        "python reporting data analyst",
        "python sql data analyst",
        "python data scientist",
        "machine learning python scientist",
        "python machine learning engineer",
    ]
    labels = [1, 1, 0, 0, 1, 0]
    return PairRelevanceModel(max_features=100).fit(resumes, jobs, labels)


class RelevanceModelTests(unittest.TestCase):
    @staticmethod
    def write_portable_model(model_path: Path) -> PairRelevanceModel:
        model = fitted_model()
        artifact = model.export_portable(
            threshold=0.5,
            metadata={"model_version": "test-v1"},
        )
        model_path.write_text(json.dumps(artifact), encoding="utf-8")
        return model

    def test_fit_and_predict(self) -> None:
        model = fitted_model()
        probabilities = model.predict_proba(
            ["python sql analytics", "java spring"],
            ["python sql analyst", "python data analyst"],
        )
        self.assertEqual(len(probabilities), 2)
        self.assertGreater(probabilities[0], probabilities[1])

    def test_untrained_model_rejects_prediction(self) -> None:
        with self.assertRaises(RuntimeError):
            PairRelevanceModel().predict_proba(["resume"], ["job"])

    def test_missing_artifact_degrades_safely(self) -> None:
        result = predict_relevance("resume", "job", model_path=Path("/missing/model.joblib"))
        self.assertFalse(result["available"])
        self.assertIsNone(result["probability"])

    def test_saved_artifact_can_be_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.json"
            model = self.write_portable_model(model_path)
            results = predict_relevance_batch(
                [("python sql", "python sql analyst")],
                model_path=model_path,
            )
            self.assertTrue(results[0]["available"])
            self.assertEqual(results[0]["model_version"], "test-v1")
            expected = model.predict_proba(["python sql"], ["python sql analyst"])[0]
            self.assertAlmostEqual(results[0]["probability"], expected, places=12)

    def test_empty_pair_does_not_disable_valid_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.json"
            self.write_portable_model(model_path)
            results = predict_relevance_batch(
                [("", "job"), ("python sql", "python sql analyst")],
                model_path=model_path,
            )
            self.assertFalse(results[0]["available"])
            self.assertTrue(results[1]["available"])

    def test_collapsed_real_world_batch_is_hidden_not_shown_as_zero(self) -> None:
        signals = [
            {"available": True, "displayable": True, "probability": value, "reason": "raw"}
            for value in (0.0001, 0.0002, 0.0004, 0.0008, 0.0012)
        ]
        guarded = suppress_collapsed_relevance_signals(signals)
        self.assertTrue(all(signal["available"] for signal in guarded))
        self.assertTrue(all(not signal["displayable"] for signal in guarded))
        self.assertTrue(all("collapsed" in signal["reason"] for signal in guarded))
        self.assertTrue(all(signal["displayable"] for signal in signals))

    def test_well_spread_batch_remains_displayable(self) -> None:
        signals = [
            {"available": True, "displayable": True, "probability": value, "reason": "raw"}
            for value in (0.02, 0.1, 0.3, 0.6, 0.9)
        ]
        guarded = suppress_collapsed_relevance_signals(signals)
        self.assertTrue(all(signal["displayable"] for signal in guarded))


if __name__ == "__main__":
    unittest.main()
