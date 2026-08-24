"""Tests for the locked-pipeline arithmetic in tools/borealtc/_common.py.

Pure standard-library tests: the augmentation plan and fold-block inference
are exercised against hand-computed values, including the paper Table I
partition counts (expected values are comparison configuration and may
appear in tests per the stage contract).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    from tools.borealtc import _common
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.borealtc import _common


class AugmentationPlanTests(unittest.TestCase):
    def test_plan_for_paper_table1_counts(self):
        plan = _common.expected_augmentation_plan(_common.EXPECTED_PARTITION_COUNTS)
        self.assertEqual(plan["pw_len"], 500)
        self.assertEqual(plan["mw_len"], 170)
        self.assertEqual(plan["st_len"], 10)
        self.assertEqual(plan["n_strides_part"], 33)
        self.assertEqual(plan["min_count"], 111)
        self.assertEqual(
            plan["n_slides"],
            {"ASPHALT": 33, "FLOORING": 8, "ICE": 8, "SANDY_LOAM": 29, "SNOW": 13},
        )
        self.assertEqual(
            plan["aug_stride"],
            {"ASPHALT": 10, "FLOORING": 41, "ICE": 41, "SANDY_LOAM": 11, "SNOW": 25},
        )
        self.assertEqual(
            plan["windows"],
            {
                "ASPHALT": 3663,
                "FLOORING": 3384,
                "ICE": 3600,
                "SANDY_LOAM": 3654,
                "SNOW": 3653,
            },
        )
        self.assertEqual(plan["total_windows"], 17954)
        self.assertTrue(plan["windows_stay_within_partition"])

    def test_plan_small_synthetic(self):
        plan = _common.expected_augmentation_plan({"A": 2, "B": 4})
        # n_strides_part = 33, strides_min = 66 -> A: 33 slides, B: 16 slides.
        self.assertEqual(plan["n_slides"], {"A": 33, "B": 16})
        self.assertEqual(plan["aug_stride"], {"A": 10, "B": 20})
        self.assertEqual(plan["windows"], {"A": 66, "B": 64})
        self.assertTrue(plan["windows_stay_within_partition"])


class FoldBlockInferenceTests(unittest.TestCase):
    def test_recovers_block_lengths(self):
        labels: list[int] = []
        expected = []
        for fold in range(_common.N_FOLDS):
            lengths = [fold + 2, 3, 4, 5, 6]
            expected.append(lengths)
            for cls, n in enumerate(lengths):
                labels.extend([cls] * n)
        self.assertEqual(_common.infer_fold_blocks(labels), expected)

    def test_rejects_wrong_block_count(self):
        with self.assertRaises(_common.AuditError):
            _common.infer_fold_blocks([0, 1, 2, 3, 4])

    def test_rejects_non_ascending_blocks(self):
        labels = []
        for _ in range(_common.N_FOLDS):
            for cls in (0, 2, 1, 3, 4):
                labels.extend([cls] * 2)
        with self.assertRaises(_common.AuditError):
            _common.infer_fold_blocks(labels)


class ComparisonConfigTests(unittest.TestCase):
    def test_official_classes_sorted_and_exclude_mixed(self):
        self.assertEqual(
            list(_common.OFFICIAL_CLASSES), sorted(_common.OFFICIAL_CLASSES)
        )
        self.assertNotIn(_common.MIXED_CLASS, _common.OFFICIAL_CLASSES)
        self.assertEqual(len(_common.DAT_CLASS_PREFIXES), 5)

    def test_sandy_loam_display_name(self):
        self.assertEqual(_common.PAPER_DISPLAY_NAMES["SANDY_LOAM"], "SILTY LOAM")

    def test_table1_total(self):
        self.assertEqual(
            sum(_common.EXPECTED_PARTITION_COUNTS.values()),
            _common.EXPECTED_TOTAL_PARTITIONS,
        )

    def test_natural_weighted_accuracy(self):
        recalls = {"A": 100.0, "B": 50.0}
        counts = {"A": 1, "B": 3}
        self.assertAlmostEqual(
            _common.natural_weighted_accuracy(recalls, counts), 62.5
        )


if __name__ == "__main__":
    unittest.main()
