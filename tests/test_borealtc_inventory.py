"""Tests for tools/borealtc/inventory.py pairing and partition semantics.

Standard-library unittest, pytest-compatible. Uses synthetic data only; the
pinned upstream checkout and heavy packages are not required (pandas/numpy
dependent cases are skipped cleanly when unavailable).
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from tools.borealtc import inventory
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.borealtc import inventory

HAVE_PANDAS = importlib.util.find_spec("pandas") is not None
HAVE_NUMPY = importlib.util.find_spec("numpy") is not None


class PairRunsTests(unittest.TestCase):
    def test_pairs_and_orphans(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for name in ("imu_00.csv", "pro_00.csv", "imu_01.csv", "notes.txt"):
                (d / name).write_text("time\n0\n", encoding="utf-8")
            pairs, orphans = inventory.pair_runs(d)
            self.assertEqual(sorted(pairs), ["00"])
            self.assertEqual(orphans, ["pro_01"])
            self.assertEqual(pairs["00"]["imu"].name, "imu_00.csv")

    def test_ignores_unrelated_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "extra_00.csv").write_text("time\n0\n", encoding="utf-8")
            pairs, orphans = inventory.pair_runs(d)
            self.assertEqual(pairs, {})
            self.assertEqual(orphans, [])


class PartitionReconstructionTests(unittest.TestCase):
    def test_complete_partitions_only(self):
        # wind_len = int(5 s * 100 Hz) = 500 samples; incomplete tails drop.
        self.assertEqual(inventory.reconstruct_partitions(499), 0)
        self.assertEqual(inventory.reconstruct_partitions(500), 1)
        self.assertEqual(inventory.reconstruct_partitions(999), 1)
        self.assertEqual(inventory.reconstruct_partitions(1000), 2)
        self.assertEqual(inventory.reconstruct_partitions(0), 0)


@unittest.skipUnless(HAVE_PANDAS, "pandas not installed")
class AuditFileTests(unittest.TestCase):
    def test_audit_file_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "imu_00.csv"
            rows = ["time,wx,wy,wz,ax,ay,az"]
            rows += [f"{i / 100.0},0,0,0,0,0,9.8" for i in range(1000)]
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            info = inventory.audit_file(path, "imu")
            self.assertEqual(info["rows"], 1000)
            self.assertTrue(info["schema_ok"])
            self.assertTrue(info["time_strictly_increasing"])
            self.assertEqual(info["n_duplicate_times"], 0)
            self.assertEqual(info["n_missing_values"], 0)
            self.assertEqual(info["freq_from_median_hz"], 100.0)
            self.assertEqual(info["freq_upstream_rule_hz"], 100.0)
            self.assertAlmostEqual(info["duration_s"], 9.99, places=6)

    def test_audit_file_flags_nans_and_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pro_00.csv"
            path.write_text(
                "time,curL,curR,velL\n0.0,1,2,3\n0.5,,2,3\n", encoding="utf-8"
            )
            info = inventory.audit_file(path, "pro")
            self.assertFalse(info["schema_ok"])
            self.assertEqual(info["n_missing_values"], 1)


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class ProEndShiftTests(unittest.TestCase):
    def test_end_shift_detection(self):
        import numpy as np

        # One run: 1000 imu samples at 100 Hz -> 2 partitions. The pro stream
        # stops at 9 s (59 samples), so the second partition's 32-sample pro
        # window overruns the end and needs the upstream `overwin` shift.
        imu_time = np.arange(1000) / 100.0
        pro_time = np.arange(0, 9.0, 1 / 6.5)
        shifted, infeasible = inventory.count_pro_end_shifts(imu_time, pro_time, 2)
        self.assertEqual(infeasible, 0)
        self.assertEqual(shifted, 1)

    def test_infeasible_short_pro(self):
        import numpy as np

        imu_time = np.arange(500) / 100.0
        pro_time = np.arange(10) / 6.5
        _, infeasible = inventory.count_pro_end_shifts(imu_time, pro_time, 1)
        self.assertEqual(infeasible, 1)


if __name__ == "__main__":
    unittest.main()
