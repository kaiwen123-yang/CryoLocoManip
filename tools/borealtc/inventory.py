"""Stage 0B.1 dataset and schema audit for the pinned BorealTC checkout.

Read-only. For every class directory (official five classes plus MIXED):

- pair `imu_XX.csv` with `pro_XX.csv` and report missing/orphan files;
- record per-file rows, columns, dtypes, byte size, SHA256, NaNs,
  time monotonicity, duplicate timestamps, start/end/duration, and
  sample-interval statistics (median/min/max, robust inferred frequency,
  and the upstream `round(1 / min(diff))` frequency);
- reconstruct complete 5 s partition counts from the CSVs using the locked
  pipeline semantics (int(5 * hf) samples per partition, incomplete tails
  dropped) and reconcile them against the committed summary file;
- quantify wheel/proprioception partition end-shifts (`overwin` semantics);
- inventory MIXED separately from the official five-class contract.

Usage: python -m tools.borealtc.inventory --run-dir <run_dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tools.borealtc import _common
except ImportError:  # executed as a loose script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.borealtc import _common

EXPECTED_SCHEMAS = {
    "imu": ["time", "wx", "wy", "wz", "ax", "ay", "az"],
    "pro": ["time", "curL", "curR", "velL", "velR"],
}


def pair_runs(class_dir: Path) -> tuple[dict[str, dict[str, Path]], list[str]]:
    """Pair imu_XX/pro_XX files by run id; return (pairs, orphan_names)."""
    files: dict[str, dict[str, Path]] = {}
    for csv_path in sorted(class_dir.glob("*.csv")):
        stem = csv_path.stem
        if "_" not in stem:
            continue
        kind, _, run_id = stem.partition("_")
        if kind not in ("imu", "pro"):
            continue
        files.setdefault(run_id, {})[kind] = csv_path
    orphans = [
        f"{kind}_{run_id}"
        for run_id, kinds in files.items()
        for kind in ("imu", "pro")
        if kind not in kinds
    ]
    pairs = {r: k for r, k in files.items() if "imu" in k and "pro" in k}
    return pairs, sorted(orphans)


def audit_file(csv_path: Path, kind: str) -> dict:
    import numpy as np
    import pandas as pd

    df = pd.read_csv(csv_path)
    columns = df.columns.tolist()
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    t = df["time"].to_numpy() if "time" in df.columns else np.array([])
    dt = np.diff(t) if t.size > 1 else np.array([])
    info = {
        "file": csv_path.name,
        "kind": kind,
        "rows": int(len(df)),
        "columns": ";".join(columns),
        "schema_ok": columns == EXPECTED_SCHEMAS[kind],
        "non_numeric_columns": ";".join(non_numeric),
        "n_missing_values": int(df.isna().sum().sum()),
        "bytes": csv_path.stat().st_size,
        "sha256": _common.sha256_file(csv_path),
        "time_start_s": float(t[0]) if t.size else float("nan"),
        "time_end_s": float(t[-1]) if t.size else float("nan"),
        "duration_s": float(t[-1] - t[0]) if t.size else 0.0,
        "time_strictly_increasing": bool((dt > 0).all()) if dt.size else True,
        "n_duplicate_times": int((dt == 0).sum()) if dt.size else 0,
        "dt_median_s": float(np.median(dt)) if dt.size else float("nan"),
        "dt_min_s": float(dt.min()) if dt.size else float("nan"),
        "dt_max_s": float(dt.max()) if dt.size else float("nan"),
    }
    info["freq_from_median_hz"] = (
        round(1.0 / info["dt_median_s"], 1) if dt.size else float("nan")
    )
    # Upstream get_recordings computes round(1 / min(diff(time)), 1).
    info["freq_upstream_rule_hz"] = (
        round(1.0 / info["dt_min_s"], 1) if dt.size else float("nan")
    )
    info["_time"] = t
    return info


def reconstruct_partitions(
    imu_rows: int, hf_hz: float = _common.IMU_NOMINAL_HZ
) -> int:
    """Complete 5 s partitions per the locked semantics: starts at every
    wind_len samples, keep only starts with start + wind_len <= rows.
    """
    wind_len = int(_common.PART_WINDOW_S * hf_hz)
    return max(0, imu_rows // wind_len)


def count_pro_end_shifts(
    imu_time, pro_time, n_partitions: int
) -> tuple[int, int]:
    """Count partitions whose pro slice needed the upstream `overwin` end
    shift, and runs whose pro stream cannot fill one partition window.
    """
    import numpy as np

    lf_winlen = int(_common.PART_WINDOW_S * _common.PRO_NOMINAL_HZ)
    wind_len = int(_common.PART_WINDOW_S * _common.IMU_NOMINAL_HZ)
    shifted = 0
    infeasible = 1 if len(pro_time) < lf_winlen else 0
    for w in range(n_partitions):
        t_start = imu_time[w * wind_len]
        idx = int(np.abs(pro_time - t_start).argmin())
        if idx + lf_winlen > len(pro_time):
            shifted += 1
    return shifted, infeasible


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()

    identity = _common.verify_upstream_identity()
    data_dir = _common.upstream_dir() / _common.DATA_SUBDIR
    print(f"upstream OK: {identity['commit']} ({identity['path']})")

    class_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    found_classes = [d.name for d in class_dirs]
    expected_classes = sorted([*_common.OFFICIAL_CLASSES, _common.MIXED_CLASS])
    if found_classes != expected_classes:
        raise _common.AuditError(
            f"class directories {found_classes} != expected {expected_classes}"
        )

    inventory_rows: list[list[object]] = []
    class_rows: list[list[object]] = []
    partition_rows: list[list[object]] = []
    reconstructed: dict[str, int] = {}
    problems: list[str] = []

    committed_summary = _common.parse_metric_dat(
        (_common.upstream_dir() / _common.UPSTREAM_SUMMARY_DAT).read_text(
            encoding="utf-8"
        )
    )

    for class_dir in class_dirs:
        cls = class_dir.name
        pairs, orphans = pair_runs(class_dir)
        if orphans:
            problems.append(f"{cls}: orphan files {orphans}")
        n_rows = {"imu": 0, "pro": 0}
        raw_duration = 0.0
        n_partitions = 0
        n_shifted = 0
        n_infeasible = 0
        imu_freqs: set[float] = set()
        pro_freqs: set[float] = set()
        for run_id in sorted(pairs):
            per_kind = {}
            for kind in ("imu", "pro"):
                info = audit_file(pairs[run_id][kind], kind)
                per_kind[kind] = info
                n_rows[kind] += info["rows"]
                if not info["schema_ok"]:
                    problems.append(f"{cls}/{info['file']}: schema {info['columns']}")
                if info["n_missing_values"]:
                    problems.append(
                        f"{cls}/{info['file']}: {info['n_missing_values']} NaNs"
                    )
                if not info["time_strictly_increasing"]:
                    problems.append(f"{cls}/{info['file']}: time not increasing")
                (imu_freqs if kind == "imu" else pro_freqs).add(
                    info["freq_upstream_rule_hz"]
                )
                inventory_rows.append(
                    [cls, run_id]
                    + [
                        info[k]
                        for k in (
                            "file",
                            "kind",
                            "rows",
                            "columns",
                            "schema_ok",
                            "non_numeric_columns",
                            "n_missing_values",
                            "bytes",
                            "sha256",
                            "time_start_s",
                            "time_end_s",
                            "duration_s",
                            "time_strictly_increasing",
                            "n_duplicate_times",
                            "dt_median_s",
                            "dt_min_s",
                            "dt_max_s",
                            "freq_from_median_hz",
                            "freq_upstream_rule_hz",
                        )
                    ]
                )
            raw_duration += per_kind["imu"]["duration_s"]
            parts = reconstruct_partitions(per_kind["imu"]["rows"])
            n_partitions += parts
            shifted, infeasible = count_pro_end_shifts(
                per_kind["imu"]["_time"], per_kind["pro"]["_time"], parts
            )
            n_shifted += shifted
            n_infeasible += infeasible

        # Partition reconstruction assumes the upstream frequency rule yields
        # exactly 100 Hz / 6.5 Hz for every file; anything else would make
        # get_recordings' setdefault-based frequency order-dependent.
        if imu_freqs and imu_freqs != {_common.IMU_NOMINAL_HZ}:
            problems.append(f"{cls}: imu upstream-rule freqs {sorted(imu_freqs)} != 100.0")
        if pro_freqs and pro_freqs != {_common.PRO_NOMINAL_HZ}:
            problems.append(f"{cls}: pro upstream-rule freqs {sorted(pro_freqs)} != 6.5")

        reconstructed[cls] = n_partitions
        retained = n_partitions * _common.PART_WINDOW_S
        class_rows.append(
            [
                cls,
                _common.PAPER_DISPLAY_NAMES.get(cls, cls),
                cls in _common.OFFICIAL_CLASSES,
                len(pairs),
                len(orphans),
                n_rows["imu"],
                n_rows["pro"],
                round(raw_duration, 2),
                n_partitions,
                round(retained, 1),
                round(100.0 * retained / raw_duration, 2) if raw_duration else 0.0,
                n_shifted,
                n_infeasible,
                ";".join(str(f) for f in sorted(imu_freqs)),
                ";".join(str(f) for f in sorted(pro_freqs)),
            ]
        )

    official_total = 0
    for cls in sorted(reconstructed):
        official = cls in _common.OFFICIAL_CLASSES
        expected = _common.EXPECTED_PARTITION_COUNTS.get(cls, "")
        committed = committed_summary.get(cls, "")
        match_expected = (reconstructed[cls] == expected) if official else ""
        match_committed = (
            (reconstructed[cls] == int(committed)) if committed != "" else ""
        )
        if official:
            official_total += reconstructed[cls]
            if reconstructed[cls] != expected:
                problems.append(
                    f"{cls}: reconstructed {reconstructed[cls]} != Table I {expected}"
                )
        partition_rows.append(
            [
                cls,
                _common.PAPER_DISPLAY_NAMES.get(cls, cls),
                official,
                reconstructed[cls],
                expected,
                committed,
                match_expected,
                match_committed,
            ]
        )
    partition_rows.append(
        [
            "TOTAL_OFFICIAL",
            "",
            True,
            official_total,
            _common.EXPECTED_TOTAL_PARTITIONS,
            "",
            official_total == _common.EXPECTED_TOTAL_PARTITIONS,
            "",
        ]
    )

    _common.write_csv(
        run_dir / "data_inventory.csv",
        [
            "class",
            "run_id",
            "file",
            "kind",
            "rows",
            "columns",
            "schema_ok",
            "non_numeric_columns",
            "n_missing_values",
            "bytes",
            "sha256",
            "time_start_s",
            "time_end_s",
            "duration_s",
            "time_strictly_increasing",
            "n_duplicate_times",
            "dt_median_s",
            "dt_min_s",
            "dt_max_s",
            "freq_from_median_hz",
            "freq_upstream_rule_hz",
        ],
        inventory_rows,
    )
    _common.write_csv(
        run_dir / "class_summary.csv",
        [
            "class",
            "paper_display_name",
            "official_contract",
            "run_pairs",
            "orphans",
            "imu_rows",
            "pro_rows",
            "raw_imu_duration_s",
            "partitions_5s",
            "retained_duration_s",
            "retained_pct",
            "pro_windows_end_shifted",
            "pro_runs_below_one_window",
            "imu_freqs_upstream_rule_hz",
            "pro_freqs_upstream_rule_hz",
        ],
        class_rows,
    )
    _common.write_csv(
        run_dir / "partition_counts.csv",
        [
            "class",
            "paper_display_name",
            "official_contract",
            "reconstructed_partitions",
            "expected_table1",
            "committed_summary_dat",
            "match_table1",
            "match_summary_dat",
        ],
        partition_rows,
    )

    for line in problems:
        print(f"PROBLEM: {line}")
    ok = official_total == _common.EXPECTED_TOTAL_PARTITIONS and not any(
        p.startswith(tuple(_common.OFFICIAL_CLASSES)) for p in problems
    )
    print(
        f"reconstructed official partitions: {official_total} "
        f"(expected {_common.EXPECTED_TOTAL_PARTITIONS})"
    )
    print("INVENTORY_OK" if ok else "INVENTORY_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
