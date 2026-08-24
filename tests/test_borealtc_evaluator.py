"""Tests for tools/borealtc/evaluate_committed_results.py.

The .dat parser and tolerance logic are standard-library tests; the metric
computation test uses scikit-learn on synthetic arrays and is skipped when
scikit-learn/numpy are unavailable (e.g. under the system interpreter).
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

try:
    from tools.borealtc import _common
    from tools.borealtc import evaluate_committed_results as evaluator
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.borealtc import _common
    from tools.borealtc import evaluate_committed_results as evaluator

HAVE_SKLEARN = (
    importlib.util.find_spec("sklearn") is not None
    and importlib.util.find_spec("numpy") is not None
)


class DatParserTests(unittest.TestCase):
    def test_parses_key_value_lines(self):
        parsed = _common.parse_metric_dat("acc = 93.96\np-ASP = 92.98\n\n")
        self.assertEqual(parsed, {"acc": 93.96, "p-ASP": 92.98})

    def test_rejects_malformed_lines(self):
        with self.assertRaises(_common.AuditError):
            _common.parse_metric_dat("accuracy 93.96\n")


class ToleranceTests(unittest.TestCase):
    def test_two_decimal_tolerance(self):
        self.assertTrue(_common.within_tolerance(93.96, 93.96))
        self.assertTrue(_common.within_tolerance(93.96, 93.97))
        self.assertTrue(_common.within_tolerance(93.96, 93.95))
        self.assertFalse(_common.within_tolerance(93.96, 93.98))
        self.assertFalse(_common.within_tolerance(93.96, 93.94))


@unittest.skipUnless(HAVE_SKLEARN, "scikit-learn/numpy not installed")
class EvaluateModelTests(unittest.TestCase):
    def _synthetic_values(self):
        import numpy as np

        # Five folds, each with ascending class blocks of 4 test windows.
        true = []
        for _ in range(_common.N_FOLDS):
            for cls in range(5):
                true.extend([cls] * 4)
        true = np.array(true, dtype=np.int64)
        pred = true.copy()
        pred[0] = 1  # one ASPHALT window misclassified as FLOORING
        return {
            "terrains": list(_common.OFFICIAL_CLASSES),
            "pred": pred,
            "true": true,
            "ftime": np.full(true.shape, 0.5),
        }

    def test_metrics_and_fold_inference(self):
        values = self._synthetic_values()
        result = evaluator.evaluate_model(
            "CNN", values, dict(_common.EXPECTED_PARTITION_COUNTS)
        )
        self.assertEqual(result["n_samples"], 100)
        self.assertAlmostEqual(result["accuracy_pct"], 99.0)
        self.assertAlmostEqual(
            result["per_class"]["ASPHALT"]["recall_pct"], 95.0
        )
        self.assertEqual(result["per_class"]["ASPHALT"]["support"], 20)
        self.assertEqual(len(result["fold_test_window_counts_inferred"]), 5)
        self.assertEqual(
            result["fold_test_window_counts_inferred"][0],
            dict.fromkeys(_common.OFFICIAL_CLASSES, 4),
        )
        self.assertEqual(result["ftime_summary_s"]["count"], 100)
        self.assertEqual(result["ptime_summary_s"], "absent")
        cm = result["confusion_matrix"]
        self.assertEqual(cm[0][0], 19)
        self.assertEqual(cm[0][1], 1)

    def test_rejects_wrong_terrain_order(self):
        values = self._synthetic_values()
        values["terrains"] = list(reversed(_common.OFFICIAL_CLASSES))
        with self.assertRaises(_common.AuditError):
            evaluator.evaluate_model(
                "CNN", values, dict(_common.EXPECTED_PARTITION_COUNTS)
            )


if __name__ == "__main__":
    unittest.main()
