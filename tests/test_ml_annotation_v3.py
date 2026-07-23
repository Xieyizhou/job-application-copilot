"""Tests for the expanded, style-balanced v3 annotation queue."""

from __future__ import annotations

from collections import Counter
import unittest

from ml.annotation_generation import build_tasks_from_records
from ml.annotation_scenarios_v3 import expanded_fictional_records, scenario_design_audit


class AnnotationV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = expanded_fictional_records(random_state=73)

    def test_design_balances_roles_relations_and_surface_styles(self) -> None:
        audit = scenario_design_audit(self.records)
        self.assertEqual(audit["records"], 160)
        self.assertEqual(
            audit["expected_relation_counts"],
            {"Direct": 56, "Partial": 56, "No Support": 48},
        )
        self.assertEqual(audit["requirement_style_count"], 26)
        self.assertEqual(audit["candidate_surface_style_count"], 16)
        self.assertLess(audit["largest_style_share_within_semantic_role"], 0.1)
        openings = Counter(
            " ".join(sentence.lower().split()[:4])
            for record in self.records
            for sentence in record["evidence_sentences"]
        )
        self.assertGreaterEqual(len(openings), 120)
        self.assertLess(max(openings.values()) / 640, 0.08)

    def test_queue_has_exact_role_and_preferred_position_balance(self) -> None:
        tasks = build_tasks_from_records(
            self.records,
            unique_count=160,
            blind_repeat_fraction=0.1,
            random_state=73,
        )
        unique = [task for task in tasks if not task["blind_duplicate_of"]]
        self.assertEqual(len(tasks), 176)
        self.assertEqual(len(unique), 160)
        self.assertEqual(
            Counter(task["role_family"] for task in unique),
            {"Data": 40, "ML": 40, "Software": 40, "Business": 40},
        )
        record_by_requirement = {
            record["requirement_sentences"][0]: record
            for record in self.records
        }
        positions: Counter[str] = Counter()
        for task in unique:
            record = record_by_requirement[task["requirement"]]
            preferred = record["preferred_evidence"]
            if preferred:
                index = next(
                    index
                    for index, candidate in enumerate(task["candidates"])
                    if candidate["evidence"] == preferred
                )
                positions[chr(65 + index)] += 1
            self.assertEqual(len(task["candidates"]), 4)
            self.assertEqual(
                len({candidate["candidate_id"] for candidate in task["candidates"]}),
                4,
            )
        self.assertEqual(positions, {"A": 28, "B": 28, "C": 28, "D": 28})

    def test_queue_does_not_expose_generation_hints(self) -> None:
        tasks = build_tasks_from_records(
            self.records,
            unique_count=160,
            blind_repeat_fraction=0,
            random_state=73,
        )
        hidden_fields = {
            "sampling_stratum",
            "expected_relation",
            "requirement_style",
            "candidate_surface_styles",
            "candidate_semantic_roles",
            "preferred_evidence",
        }
        self.assertTrue(all(not hidden_fields.intersection(task) for task in tasks))
        self.assertTrue(
            all(
                set(candidate) == {"candidate_id", "evidence"}
                for task in tasks
                for candidate in task["candidates"]
            )
        )


if __name__ == "__main__":
    unittest.main()
