"""Tests for tools/borealtc/audit_splits.py adjacency and overlap logic.

Standard-library only; synthetic partition identities.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

try:
    from tools.borealtc import audit_splits
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.borealtc import audit_splits


class AdjacencyTests(unittest.TestCase):
    def test_counts_boundaries_in_both_directions(self):
        train = [("ASPHALT", 0, 0), ("ASPHALT", 0, 2)]
        test = [("ASPHALT", 0, 1)]
        # Test partition 1 touches train partitions 0 and 2.
        self.assertEqual(
            audit_splits.adjacency_pairs(train, test), {"ASPHALT": 2}
        )

    def test_no_adjacency_across_runs_or_classes(self):
        train = [("ASPHALT", 0, 0), ("SNOW", 0, 1)]
        test = [("ASPHALT", 1, 1), ("SNOW", 1, 0)]
        self.assertEqual(audit_splits.adjacency_pairs(train, test), {})

    def test_non_adjacent_windows_do_not_count(self):
        train = [("ICE", 3, 0)]
        test = [("ICE", 3, 2)]
        self.assertEqual(audit_splits.adjacency_pairs(train, test), {})


class RunOverlapTests(unittest.TestCase):
    def test_full_overlap(self):
        train = [("ICE", 1, 0), ("ICE", 2, 0)]
        test = [("ICE", 1, 1), ("ICE", 2, 1)]
        overlap = audit_splits.run_overlap(train, test)
        self.assertEqual(overlap["runs_in_both"], 2)
        self.assertEqual(overlap["pct_test_runs_also_in_train"], 100.0)
        self.assertEqual(overlap["pct_test_partitions_from_train_runs"], 100.0)

    def test_disjoint_runs(self):
        train = [("ICE", 1, 0)]
        test = [("ICE", 2, 0), ("ICE", 3, 0)]
        overlap = audit_splits.run_overlap(train, test)
        self.assertEqual(overlap["runs_in_both"], 0)
        self.assertEqual(overlap["pct_test_runs_also_in_train"], 0.0)
        self.assertEqual(overlap["pct_test_partitions_from_train_runs"], 0.0)

    def test_partial_overlap_partition_percentage(self):
        train = [("ICE", 1, 0)]
        test = [("ICE", 1, 1), ("ICE", 2, 0), ("ICE", 2, 1), ("ICE", 2, 2)]
        overlap = audit_splits.run_overlap(train, test)
        self.assertEqual(overlap["runs_in_both"], 1)
        self.assertEqual(overlap["pct_test_runs_also_in_train"], 50.0)
        self.assertEqual(overlap["pct_test_partitions_from_train_runs"], 25.0)


if __name__ == "__main__":
    unittest.main()
