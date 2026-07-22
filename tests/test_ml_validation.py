"""Tests for anonymous real-derived ML validation manifests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ml.validation import (
    DEFAULT_RELEVANCE_MANIFEST,
    DEFAULT_SEMANTIC_MANIFEST,
    ValidationManifestError,
    evaluate_relevance_manifest,
    evaluate_semantic_manifest,
    load_manifest,
    validate_relevance_manifest,
)


NO_MODEL = Path("/missing/local-model.json")


class AnonymousValidationTests(unittest.TestCase):
    def test_semantic_manifest_is_deidentified_and_evaluable(self) -> None:
        report = evaluate_semantic_manifest(DEFAULT_SEMANTIC_MANIFEST, model_path=NO_MODEL)
        self.assertEqual(report["cases"], 24)
        self.assertGreaterEqual(report["f1"], 0.85)
        manifest_text = DEFAULT_SEMANTIC_MANIFEST.read_text(encoding="utf-8")
        self.assertNotIn("@", manifest_text)
        self.assertNotIn("http://", manifest_text)
        self.assertNotIn("https://", manifest_text)

    def test_relevance_manifest_contains_hashes_not_source_text(self) -> None:
        data = load_manifest(DEFAULT_RELEVANCE_MANIFEST)
        cases = validate_relevance_manifest(data)
        self.assertEqual(len(cases), 22)
        self.assertEqual(len({case["resume_sha256"] for case in cases}), 22)
        self.assertEqual(len({case["job_sha256"] for case in cases}), 22)

    def test_relevance_validation_degrades_when_local_pairs_are_absent(self) -> None:
        report = evaluate_relevance_manifest(
            DEFAULT_RELEVANCE_MANIFEST,
            pairs_path=Path("/missing/canonical_pairs.parquet"),
            model_path=NO_MODEL,
        )
        self.assertEqual(report["status"], "pairs_unavailable")

    def test_raw_text_fields_are_rejected_from_hash_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.json"
            source = load_manifest(DEFAULT_RELEVANCE_MANIFEST)
            source["cases"][0]["resume_text"] = "private source text"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValidationManifestError, "raw or identifying"):
                validate_relevance_manifest(load_manifest(path))


if __name__ == "__main__":
    unittest.main()
