"""Tests for tools/borealtc/compare_runs.py (R1 corrective rerun comparison).

Standard-library only: synthetic run directories with minimal JSON artifacts.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from tools.borealtc import compare_runs
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.borealtc import compare_runs


def synthetic_artifacts() -> dict[str, dict]:
    per_class = {
        "ASPHALT": {"precision_pct": 92.98, "recall_pct": 83.89, "f1_pct": 88.2, "support": 3663},
        "SNOW": {"precision_pct": 86.84, "recall_pct": 92.31, "f1_pct": 89.49, "support": 3653},
    }
    model = {
        "n_samples": 17954,
        "artifact": {"sha256": "aa" * 32},
        "confusion_matrix": [[10, 2], [1, 12]],
        "per_class": per_class,
        "fold_test_window_counts_inferred": [{"ASPHALT": 4, "SNOW": 5}] * 5,
        "accuracy_pct": 93.96,
        "macro_f1_pct": 94.0,
        "weighted_f1_pct": 93.92,
        "balanced_accuracy_pct": 94.05,
        "natural_weighted_accuracy_pct_diagnostic": 95.9,
        "upstream_ap_diagnostic": {"reproduced_value_pct": 18.61, "exception": None},
    }
    return {
        "pipeline_reconstruction.json": {
            "sampling_freq_hz": {"imu": 100.0, "pro": 6.5},
            "pre_split_partition_counts": {"ASPHALT": 111, "SNOW": 281},
            "pre_split_order_sha256": "c" * 64,
            "split_fold_hashes": [{"train": "t" * 64, "test": "e" * 64}] * 5,
            "split_deterministic_on_rerun": True,
            "fold_test_partition_counts": [{"ASPHALT": 22, "SNOW": 56}] * 5,
            "fold_test_window_counts": [{"ASPHALT": 726, "SNOW": 728}] * 5,
            "total_test_windows_all_folds": 17954,
            "augmentation_plan_independent": {
                "n_slides": {"ASPHALT": 33, "SNOW": 13},
                "aug_stride": {"ASPHALT": 10, "SNOW": 25},
            },
        },
        "split_audit.json": {
            "fold_summaries": [
                {
                    "fold": k + 1,
                    "run_overlap": {
                        "runs_in_both": 56,
                        "pct_test_runs_also_in_train": 100.0,
                        "pct_test_partitions_from_train_runs": 100.0,
                    },
                    "adjacent_train_test_pairs": 421,
                    "duplicate_partitions": 0,
                }
                for k in range(5)
            ]
        },
        "evaluator_report.json": {
            "models": {"CNN": model, "Mamba": copy.deepcopy(model)},
            "upstream_metric_smoke": {
                "exception": None,
                "compared": {
                    "CNN": {"identical_text": True, "identical_parsed": True},
                    "Mamba": {"identical_text": True, "identical_parsed": True},
                },
            },
        },
    }


def write_run(run_dir: Path, artifacts: dict[str, dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in artifacts.items():
        (run_dir / name).write_text(json.dumps(payload), encoding="utf-8")


class CompareRunsTests(unittest.TestCase):
    def _compare(self, mutate=None):
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = Path(tmp) / "old"
            new_dir = Path(tmp) / "new"
            write_run(old_dir, synthetic_artifacts())
            new_artifacts = synthetic_artifacts()
            if mutate:
                mutate(new_artifacts)
            write_run(new_dir, new_artifacts)
            return compare_runs.compare(old_dir, new_dir)

    def test_identical_runs_match(self):
        result = self._compare()
        self.assertTrue(result["identity_match"])
        self.assertTrue(result["metrics_within_tolerance"])
        self.assertEqual(
            result["verdict"], "IDENTITY_MATCH_AND_METRICS_WITHIN_TOLERANCE"
        )
        self.assertEqual(result["max_metric_abs_delta_pp"], 0.0)

    def test_membership_hash_divergence_fails(self):
        def mutate(artifacts):
            artifacts["pipeline_reconstruction.json"]["split_fold_hashes"] = [
                {"train": "x" * 64, "test": "e" * 64}
            ] * 5

        result = self._compare(mutate)
        self.assertFalse(result["identity_match"])
        self.assertEqual(result["verdict"], "DIVERGENCE")
        self.assertFalse(result["identity"]["split_fold_hashes"]["match"])

    def test_metric_within_tolerance_still_passes(self):
        def mutate(artifacts):
            artifacts["evaluator_report.json"]["models"]["CNN"]["accuracy_pct"] = (
                93.96 + 0.005
            )

        result = self._compare(mutate)
        self.assertTrue(result["metrics_within_tolerance"])
        self.assertEqual(
            result["verdict"], "IDENTITY_MATCH_AND_METRICS_WITHIN_TOLERANCE"
        )

    def test_metric_beyond_tolerance_fails(self):
        def mutate(artifacts):
            artifacts["evaluator_report.json"]["models"]["Mamba"]["accuracy_pct"] = (
                93.68 + 0.02
            )

        result = self._compare(mutate)
        self.assertFalse(result["metrics_within_tolerance"])
        self.assertEqual(result["verdict"], "DIVERGENCE")
        self.assertFalse(
            result["metrics"]["Mamba"]["accuracy_pct"]["within_tolerance"]
        )

    def test_smoke_regression_is_identity_divergence(self):
        def mutate(artifacts):
            artifacts["evaluator_report.json"]["upstream_metric_smoke"]["compared"][
                "CNN"
            ]["identical_text"] = False

        result = self._compare(mutate)
        self.assertFalse(result["identity_match"])
        self.assertFalse(result["identity"]["upstream_smoke_identity"]["match"])


if __name__ == "__main__":
    unittest.main()
