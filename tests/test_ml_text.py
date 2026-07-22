"""Tests for ML text normalization and hashing."""

from __future__ import annotations

import unittest

from ml.text import (
    normalize_text_for_hash,
    normalized_text_hash,
    pair_hash,
    sha256_text,
)


class NormalizeTextTests(unittest.TestCase):
    def test_lowercases_and_collapses_whitespace(self) -> None:
        text = "  Python\n\nSQL\tExperience  "

        result = normalize_text_for_hash(text)

        self.assertEqual(
            result,
            "python sql experience",
        )

    def test_unicode_compatibility_normalization(self) -> None:
        full_width = "Ｐｙｔｈｏｎ"
        regular = "Python"

        self.assertEqual(
            normalize_text_for_hash(full_width),
            normalize_text_for_hash(regular),
        )

    def test_punctuation_is_preserved(self) -> None:
        result = normalize_text_for_hash(
            "Python, SQL, and C++."
        )

        self.assertEqual(
            result,
            "python, sql, and c++.",
        )


class HashTests(unittest.TestCase):
    def test_exact_hash_is_stable(self) -> None:
        self.assertEqual(
            sha256_text("example"),
            sha256_text("example"),
        )

    def test_exact_hash_detects_case_difference(self) -> None:
        self.assertNotEqual(
            sha256_text("Python"),
            sha256_text("python"),
        )

    def test_normalized_hash_ignores_case_and_whitespace(self) -> None:
        first = "Python   SQL"
        second = " python sql "

        self.assertEqual(
            normalized_text_hash(first),
            normalized_text_hash(second),
        )

    def test_pair_hash_is_order_sensitive(self) -> None:
        first = pair_hash(
            "Resume A",
            "Job B",
            normalized=True,
        )
        second = pair_hash(
            "Job B",
            "Resume A",
            normalized=True,
        )

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
