"""Stage 0B.1 split/run/overlap audit of the locked BorealTC protocol.

Reconstructs the official StratifiedKFold(RANDOM_STATE=21) split with the
unchanged upstream code and quantifies, without altering it:

- original run IDs represented in train and test of every fold, and the
  fraction of test partitions whose source run also appears in train;
- temporal adjacency (0 s boundary gap) between train and test 5 s
  partitions from the same run;
- exact partition duplicates across train/test (must be zero);
- time-support overlap of 1.7 s augmented windows across train/test
  (bounded analytically from the augmentation geometry);
- class counts before and after balancing for train and test;
- how the balancing parameters depend on the full label distribution
  (which includes each fold's test labels), quantified per fold;
- how test balancing reweights the aggregate accuracy relative to the
  natural partition distribution;
- determinism of the split under repeated execution;
- a grouped-by-run split feasibility DIAGNOSTIC (no training, no metrics).

The official protocol is documented, not judged: it estimates within-run
generalization to unseen 5 s partitions; it does not estimate leave-run-out
or leave-site-out generalization.

Usage: python -m tools.borealtc.audit_splits --run-dir <run_dir>
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from tools.borealtc import _common
except ImportError:  # executed as a loose script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.borealtc import _common


def adjacency_pairs(
    train: list[tuple[str, int, int]], test: list[tuple[str, int, int]]
) -> dict[str, int]:
    """Count cross-set pairs of temporally adjacent partitions (win, win+1)
    from the same (terrain, run). Adjacent 5 s partitions are contiguous in
    time, so every pair marks a 0 s train/test boundary gap.
    """
    train_wins: dict[tuple[str, int], set[int]] = defaultdict(set)
    test_wins: dict[tuple[str, int], set[int]] = defaultdict(set)
    for terr, run, win in train:
        train_wins[(terr, run)].add(win)
    for terr, run, win in test:
        test_wins[(terr, run)].add(win)
    per_class: dict[str, int] = Counter()
    for key in set(train_wins) | set(test_wins):
        terr = key[0]
        for w in train_wins.get(key, ()):
            if (w + 1) in test_wins.get(key, ()):
                per_class[terr] += 1
            if (w - 1) in test_wins.get(key, ()):
                per_class[terr] += 1
    return dict(per_class)


def run_overlap(
    train: list[tuple[str, int, int]], test: list[tuple[str, int, int]]
) -> dict:
    train_runs = {(t, r) for t, r, _ in train}
    test_runs = {(t, r) for t, r, _ in test}
    both = train_runs & test_runs
    test_parts_run_in_train = sum(1 for t, r, _ in test if (t, r) in train_runs)
    return {
        "train_runs": len(train_runs),
        "test_runs": len(test_runs),
        "runs_in_both": len(both),
        "pct_test_runs_also_in_train": round(
            100.0 * len(both) / len(test_runs), 2
        )
        if test_runs
        else 0.0,
        "pct_test_partitions_from_train_runs": round(
            100.0 * test_parts_run_in_train / len(test), 2
        )
        if test
        else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()

    identity = _common.verify_upstream_identity()
    print(f"upstream OK: {identity['commit']}")

    unified, labels, summary, pre = _common.build_unified()
    sampling_freq = {
        str(k): float(v) for k, v in summary["sampling_freq"].to_dict().items()
    }
    if sampling_freq != {"imu": 100.0, "pro": 6.5}:
        raise _common.AuditError(f"unexpected sampling frequencies {sampling_freq}")
    global_counts = dict(sorted(Counter(str(v) for v in labels).items()))
    plan = _common.expected_augmentation_plan(global_counts)

    train_folds, test_folds = _common.split_folds(unified, labels, pre)
    train_folds_b, test_folds_b = _common.split_folds(unified, labels, pre)

    report: dict = {
        "upstream": identity,
        "random_state": _common.RANDOM_STATE,
        "global_partition_counts": global_counts,
        "augmentation_plan": plan,
    }
    failures: list[str] = []
    split_rows: list[list[object]] = []
    overlap_rows: list[list[object]] = []
    fold_summaries = []

    deterministic = True
    for k in range(_common.N_FOLDS):
        train_m = _common.fold_membership(train_folds[k])
        test_m = _common.fold_membership(test_folds[k])
        deterministic &= _common.membership_hash(train_m) == _common.membership_hash(
            _common.fold_membership(train_folds_b[k])
        ) and _common.membership_hash(test_m) == _common.membership_hash(
            _common.fold_membership(test_folds_b[k])
        )

        duplicates = set(train_m) & set(test_m)
        if duplicates:
            failures.append(f"fold {k + 1}: {len(duplicates)} duplicated partitions")

        overlap = run_overlap(train_m, test_m)
        adj = adjacency_pairs(train_m, test_m)
        train_counts = Counter(t for t, _, _ in train_m)
        test_counts = Counter(t for t, _, _ in test_m)

        # Balancing parameters recomputed from this fold's train labels only,
        # to quantify the dependence on the full (train+test) distribution.
        plan_train_only = _common.expected_augmentation_plan(dict(train_counts))
        n_slides_diff = {
            c: plan_train_only["n_slides"][c] - plan["n_slides"][c]
            for c in _common.OFFICIAL_CLASSES
        }

        fold_summaries.append(
            {
                "fold": k + 1,
                "run_overlap": overlap,
                "adjacent_train_test_pairs": sum(adj.values()),
                "duplicate_partitions": len(duplicates),
                "n_slides_if_train_only": plan_train_only["n_slides"],
                "n_slides_delta_vs_official": n_slides_diff,
            }
        )

        for cls in _common.OFFICIAL_CLASSES:
            split_rows.append(
                [
                    k + 1,
                    cls,
                    train_counts.get(cls, 0),
                    test_counts.get(cls, 0),
                    train_counts.get(cls, 0) * plan["n_slides"][cls],
                    test_counts.get(cls, 0) * plan["n_slides"][cls],
                    len({r for t, r, _ in train_m if t == cls}),
                    len({r for t, r, _ in test_m if t == cls}),
                    len(
                        {r for t, r, _ in train_m if t == cls}
                        & {r for t, r, _ in test_m if t == cls}
                    ),
                    adj.get(cls, 0),
                ]
            )

    report["split_deterministic_on_rerun"] = bool(deterministic)
    if not deterministic:
        failures.append("split is not deterministic across reruns")
    report["fold_summaries"] = fold_summaries

    # Window-level time-support overlap across train/test, from geometry:
    # augmented windows never leave their 5 s partition (max end <= pw_len)
    # and partitions are disjoint between train and test, so cross-set window
    # overlap is exactly zero when both conditions hold.
    natural_total = sum(global_counts.values())
    aug_total = plan["total_windows"]
    for cls in _common.OFFICIAL_CLASSES:
        within = plan["max_window_end"][cls] <= plan["pw_len"]
        if not within:
            failures.append(f"{cls}: augmented windows escape the 5 s partition")
        overlap_rows.append(
            [
                cls,
                plan["n_slides"][cls],
                plan["aug_stride"][cls],
                plan["max_window_end"][cls],
                plan["pw_len"],
                within,
                0 if within else "UNBOUNDED",
                round(100.0 * global_counts[cls] / natural_total, 2),
                round(100.0 * plan["windows"][cls] / aug_total, 2),
            ]
        )
    report["cross_set_window_overlap"] = (
        "0 windows (windows are confined to their source partition and "
        "train/test partitions are disjoint)"
    )
    report["balancing_note"] = (
        "n_slides per class is computed once from the union of fold-1 train "
        "and test labels, i.e. the full 1391-partition distribution; the same "
        "balancing is applied to every fold's train AND test set, so test "
        "class support is near-uniform rather than natural."
    )

    # Grouped-by-run diagnostic (no training).
    grouped_rows: list[list[object]] = []
    parts_per_run: dict[str, list[int]] = {}
    counter: dict[tuple[str, int], int] = Counter()
    for i in range(unified["imu"].shape[0]):
        terr = str(unified["imu"][i, 0, 0])
        run = int(unified["imu"][i, 0, 2])
        counter[(terr, run)] += 1
    for cls in _common.OFFICIAL_CLASSES:
        sizes = sorted(v for (t, _), v in counter.items() if t == cls)
        parts_per_run[cls] = sizes
        grouped_rows.append(
            [
                cls,
                len(sizes),
                sum(sizes),
                min(sizes),
                statistics.median(sizes),
                max(sizes),
                round(100.0 * max(sizes) / sum(sizes), 2),
                len(sizes) >= _common.N_FOLDS,
            ]
        )
    report["grouped_split_diagnostic"] = {
        "partitions_per_run": parts_per_run,
        "note": (
            "Diagnostic only. A grouped-by-run 5-fold split is structurally "
            "feasible for every class (>= 5 runs per class), with fold-size "
            "imbalance driven by the largest runs. No grouped-run training "
            "or performance is produced in Stage 0B.1."
        ),
    }

    _common.write_csv(
        run_dir / "split_audit.csv",
        [
            "fold",
            "class",
            "train_partitions",
            "test_partitions",
            "train_windows_after_balancing",
            "test_windows_after_balancing",
            "train_runs",
            "test_runs",
            "runs_in_both",
            "adjacent_train_test_pairs",
        ],
        split_rows,
    )
    _common.write_csv(
        run_dir / "window_overlap_audit.csv",
        [
            "class",
            "n_slides_per_partition",
            "aug_stride_samples",
            "max_window_end_sample",
            "partition_len_samples",
            "windows_confined_to_partition",
            "cross_set_overlapping_windows",
            "natural_partition_share_pct",
            "balanced_window_share_pct",
        ],
        overlap_rows,
    )
    _common.write_csv(
        run_dir / "grouped_split_diagnostic.csv",
        [
            "class",
            "runs",
            "partitions",
            "min_partitions_per_run",
            "median_partitions_per_run",
            "max_partitions_per_run",
            "largest_run_share_pct",
            "grouped_5fold_feasible",
        ],
        grouped_rows,
    )
    report["failures"] = failures
    _common.write_json(run_dir / "split_audit.json", report)

    for line in failures:
        print(f"PROBLEM: {line}")
    for s in fold_summaries:
        print(
            f"fold {s['fold']}: runs_in_both={s['run_overlap']['runs_in_both']} "
            f"({s['run_overlap']['pct_test_runs_also_in_train']}% of test runs), "
            f"adjacent pairs={s['adjacent_train_test_pairs']}, "
            f"duplicates={s['duplicate_partitions']}"
        )
    print("SPLIT_AUDIT_OK" if not failures else "SPLIT_AUDIT_FAIL")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
