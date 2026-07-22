"""Tests for deterministic ML pair features."""

from __future__ import annotations

import unittest

from ml.features import FEATURE_NAMES, pair_feature_matrix, pair_feature_vector


class PairFeatureTests(unittest.TestCase):
    def test_identical_text_has_full_overlap(self) -> None:
        values = pair_feature_vector("Python SQL analytics", "Python SQL analytics")
        self.assertEqual(values[0], 1.0)
        self.assertEqual(values[1], 1.0)
        self.assertEqual(values[2], 1.0)

    def test_disjoint_text_has_no_token_overlap(self) -> None:
        values = pair_feature_vector("python pandas", "java spring")
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[1], 0.0)

    def test_matrix_validates_aligned_inputs(self) -> None:
        with self.assertRaises(ValueError):
            pair_feature_matrix(["resume"], ["job", "other"])

    def test_matrix_shape_is_stable(self) -> None:
        matrix = pair_feature_matrix(["python"], ["python role"])
        self.assertEqual(matrix.shape, (1, len(FEATURE_NAMES)))


if __name__ == "__main__":
    unittest.main()
