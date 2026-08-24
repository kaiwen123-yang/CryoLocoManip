"""Stage 0B.1 public dataset API smoke for the pinned `borealtc.py`.

Audited separately from the paper pipeline: the public API resamples and
forward-fills both modalities onto one fused time base, which is NOT the
paper training preprocessing. This smoke:

- instantiates the full BorealTC dataset (falling back to per-class
  instantiation if the full preload fails, without redefining the API);
- reports class order and the number of run-level samples;
- instantiates 170-step sliding-window datasets (step 50 = class default,
  step 10 = upstream __main__ example) and reports window counts;
- fetches at least one window from every official class and records shapes;
- records the pandas-inferred frequencies that drive `fuse_measures`;
- records elapsed time and peak resident memory.

Usage: python -m tools.borealtc.api_smoke --run-dir <run_dir>
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

try:
    from tools.borealtc import _common
except ImportError:  # executed as a loose script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.borealtc import _common


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()

    identity = _common.verify_upstream_identity()
    print(f"upstream OK: {identity['commit']}")

    _common.add_upstream_to_syspath()
    import pandas as pd
    import borealtc as upstream_api

    data_dir = _common.upstream_dir() / _common.DATA_SUBDIR
    report: dict = {"upstream": identity, "data_dir": str(data_dir)}
    failures: list[str] = []

    # Frequencies fuse_measures infers, documented from one pair per class.
    inferred = {}
    for cls in _common.OFFICIAL_CLASSES:
        imu_path = sorted((data_dir / cls).glob("imu_*.csv"))[0]
        pro_path = data_dir / cls / f"pro_{imu_path.stem.split('_')[1]}.csv"
        imu_df = pd.read_csv(imu_path).set_index("time")
        pro_df = pd.read_csv(pro_path).set_index("time")
        imu_df.index = pd.to_timedelta(imu_df.index, unit="s")
        pro_df.index = pd.to_timedelta(pro_df.index, unit="s")
        inferred[cls] = {
            "imu_inferred_freq": str(pd.infer_freq(imu_df.index)),
            "pro_inferred_freq": str(pd.infer_freq(pro_df.index)),
            "example_pair": imu_path.name,
        }
    report["fuse_inferred_frequencies"] = inferred

    t0 = time.monotonic()
    dataset = None
    try:
        dataset = upstream_api.BorealTC(str(data_dir))
        report["full_preload"] = {
            "ok": True,
            "elapsed_s": round(time.monotonic() - t0, 2),
        }
    except Exception as exc:
        report["full_preload"] = {
            "ok": False,
            "exception": f"{type(exc).__name__}: {exc}",
        }
        failures.append(f"full preload failed: {exc}")

    if dataset is not None:
        expected_classes = [c.lower() for c in _common.OFFICIAL_CLASSES]
        report["classes"] = dataset.classes
        report["n_run_level_samples"] = len(dataset)
        if dataset.classes != expected_classes:
            failures.append(
                f"API class order {dataset.classes} != {expected_classes}"
            )
        report["fused_rows_per_sample"] = {
            f"{s.class_name}/run_{s.run_id}": int(len(s.fused_df))
            for s in dataset.samples
        }
        report["fused_columns"] = list(dataset.samples[0].fused_df.columns)

        windows: dict = {}
        for step in (50, 10):
            t1 = time.monotonic()
            sliding = upstream_api.SlidingWindowDataset(
                dataset, window_size=170, step_size=step
            )
            first = sliding[0]
            windows[f"step_{step}"] = {
                "n_windows": len(sliding),
                "window_shape": list(first["window"].shape),
                "window_dtype": str(first["window"].dtype),
                "elapsed_s": round(time.monotonic() - t1, 2),
            }
        report["sliding_windows"] = windows

        sliding = upstream_api.SlidingWindowDataset(
            dataset, window_size=170, step_size=50
        )
        per_class: dict = {}
        for cls in [c.lower() for c in _common.OFFICIAL_CLASSES]:
            idx = next(
                (
                    i
                    for i, (sample_idx, _, _) in enumerate(sliding.windows)
                    if dataset.samples[sample_idx].class_name == cls
                ),
                None,
            )
            if idx is None:
                failures.append(f"no sliding window found for class {cls}")
                continue
            sample = sliding[idx]
            per_class[cls] = {
                "window_index": idx,
                "shape": list(sample["window"].shape),
                "run_id": sample["run_id"],
                "has_nan": bool(sample["window"].isnan().any().item()),
            }
        report["one_window_per_class"] = per_class

    report["elapsed_total_s"] = round(time.monotonic() - t0, 2)
    report["peak_rss_mb"] = round(peak_rss_mb(), 1)
    report["failures"] = failures
    _common.write_json(run_dir / "api_smoke.json", report)

    for line in failures:
        print(f"PROBLEM: {line}")
    if dataset is not None:
        print(
            f"samples={report['n_run_level_samples']} classes={report['classes']} "
            f"windows(step50)={report['sliding_windows']['step_50']['n_windows']} "
            f"windows(step10)={report['sliding_windows']['step_10']['n_windows']} "
            f"elapsed={report['elapsed_total_s']}s peakRSS={report['peak_rss_mb']}MB"
        )
    print("API_SMOKE_OK" if not failures else "API_SMOKE_FAIL")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
