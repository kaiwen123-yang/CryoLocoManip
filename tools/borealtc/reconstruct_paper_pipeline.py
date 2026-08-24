"""Stage 0B.1 paper-pipeline reconstruction on the pinned BorealTC checkout.

Executes the unchanged upstream preprocessing chain exactly as the locked
`mamba_train.py` does (get_recordings -> partition_data -> kfold_splits ->
augment_data -> cleanup_data -> normalize_data) and records:

- class/modality sampling frequencies as computed by upstream;
- partition tensor shapes before the split;
- per-fold train/test partition counts by class;
- per-fold post-augmentation window counts by class and modality, with an
  independent arithmetic cross-check (`expected_augmentation_plan`);
- IMU and wheel-service model input shapes after cleanup/normalization;
- normalization statistics and their train-fold-only provenance (verified
  numerically, not assumed);
- deterministic identity hashes of the reconstructed split under
  RANDOM_STATE=21 (kfold executed twice and compared).

The public `borealtc.py` API is audited separately (api_smoke.py); nothing
here treats it as the paper pipeline.

Usage: python -m tools.borealtc.reconstruct_paper_pipeline --run-dir <run_dir>
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

try:
    from tools.borealtc import _common
except ImportError:  # executed as a loose script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.borealtc import _common


def class_counts(fold_data) -> dict[str, int]:
    labels = fold_data["imu"][:, 0, 0]
    return dict(sorted(Counter(str(v) for v in labels).items()))


def main() -> int:
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()

    identity = _common.verify_upstream_identity()
    print(f"upstream OK: {identity['commit']}")

    report: dict = {"upstream": identity}
    failures: list[str] = []

    # ---- get_recordings + partition_data (pre-split tensors) ----
    unified, labels, summary, pre = _common.build_unified()
    sampling_freq = {
        str(k): float(v) for k, v in summary["sampling_freq"].to_dict().items()
    }
    report["sampling_freq_hz"] = sampling_freq
    if sampling_freq != {"imu": 100.0, "pro": 6.5}:
        failures.append(f"unexpected upstream sampling frequencies: {sampling_freq}")

    report["unified_shapes"] = {
        sens: list(arr.shape) for sens, arr in unified.items()
    }
    report["unified_channel_layout"] = "terrain, terr_idx, run_idx, win_idx, time, <sensor channels>"
    pre_split_counts = dict(sorted(Counter(str(v) for v in labels).items()))
    report["pre_split_partition_counts"] = pre_split_counts
    total = sum(pre_split_counts.values())
    if total != _common.EXPECTED_TOTAL_PARTITIONS:
        failures.append(
            f"pre-split partitions {total} != {_common.EXPECTED_TOTAL_PARTITIONS}"
        )
    order_hash = _common.sha256_text(
        ";".join(
            f"{unified['imu'][i, 0, 0]}|{int(unified['imu'][i, 0, 2])}|{int(unified['imu'][i, 0, 3])}"
            for i in range(unified["imu"].shape[0])
        )
    )
    report["pre_split_order_sha256"] = order_hash

    # ---- kfold_splits determinism (executed twice, compared) ----
    train_a, test_a = _common.split_folds(unified, labels, pre)
    train_b, test_b = _common.split_folds(unified, labels, pre)
    fold_hashes = []
    deterministic = True
    for k in range(_common.N_FOLDS):
        ha = {
            "train": _common.membership_hash(_common.fold_membership(train_a[k])),
            "test": _common.membership_hash(_common.fold_membership(test_a[k])),
        }
        hb = {
            "train": _common.membership_hash(_common.fold_membership(train_b[k])),
            "test": _common.membership_hash(_common.fold_membership(test_b[k])),
        }
        deterministic &= ha == hb
        fold_hashes.append(ha)
    report["split_random_state"] = _common.RANDOM_STATE
    report["split_fold_hashes"] = fold_hashes
    report["split_deterministic_on_rerun"] = bool(deterministic)
    if not deterministic:
        failures.append("kfold_splits is not deterministic across reruns")

    fold_partition_rows: list[list[object]] = []
    fold_test_partition_counts: list[dict[str, int]] = []
    for k in range(_common.N_FOLDS):
        tr = class_counts(train_a[k])
        te = class_counts(test_a[k])
        fold_test_partition_counts.append(te)
        for cls in _common.OFFICIAL_CLASSES:
            fold_partition_rows.append(
                [k + 1, cls, tr.get(cls, 0), te.get(cls, 0)]
            )
    report["fold_test_partition_counts"] = fold_test_partition_counts

    # ---- augment_data (unchanged upstream) + arithmetic cross-check ----
    plan = _common.expected_augmentation_plan(pre_split_counts)
    report["augmentation_plan_independent"] = plan
    aug_train, aug_test = pre.augment_data(
        train_a,
        test_a,
        summary,
        moving_window=_common.MOVING_WINDOW_S,
        stride=_common.STRIDE_S,
        homogeneous=_common.HOMOGENEOUS_AUGMENTATION,
    )
    aug_rows: list[list[object]] = []
    fold_test_window_counts: list[dict[str, int]] = []
    for k in range(_common.N_FOLDS):
        for split_name, part, aug in (
            ("train", class_counts(train_a[k]), aug_train[k]),
            ("test", class_counts(test_a[k]), aug_test[k]),
        ):
            wc = class_counts(aug)
            if split_name == "test":
                fold_test_window_counts.append(wc)
            for cls in _common.OFFICIAL_CLASSES:
                expected_windows = part.get(cls, 0) * plan["n_slides"][cls]
                actual = wc.get(cls, 0)
                if actual != expected_windows:
                    failures.append(
                        f"fold {k + 1} {split_name} {cls}: windows {actual} != "
                        f"partitions*n_slides {expected_windows}"
                    )
                aug_rows.append(
                    [
                        k + 1,
                        split_name,
                        cls,
                        part.get(cls, 0),
                        plan["n_slides"][cls],
                        actual,
                        expected_windows,
                        list(aug["imu"].shape[1:]),
                        list(aug["pro"].shape[1:]),
                    ]
                )
    report["fold_test_window_counts"] = fold_test_window_counts
    report["total_test_windows_all_folds"] = int(
        sum(sum(c.values()) for c in fold_test_window_counts)
    )

    # ---- cleanup + normalization on fold 1 (as mamba_train.py does) ----
    clean_train, clean_test = pre.cleanup_data(aug_train[0], aug_test[0])
    norm_train, norm_test = pre.normalize_data(clean_train, clean_test)
    report["model_input_shapes"] = {
        "imu_train": list(norm_train["imu"].shape),
        "pro_train": list(norm_train["pro"].shape),
        "imu_test": list(norm_test["imu"].shape),
        "pro_test": list(norm_test["pro"].shape),
        "dtype": str(norm_train["imu"].dtype),
    }
    mean_imu = np.mean(clean_train["imu"], axis=(0, 1))
    std_imu = np.std(clean_train["imu"], axis=(0, 1))
    mean_pro = np.mean(clean_train["pro"], axis=(0, 1))
    std_pro = np.std(clean_train["pro"], axis=(0, 1))
    provenance_ok = bool(
        np.allclose(
            norm_test["imu"],
            (clean_test["imu"] - mean_imu) / std_imu,
            atol=1e-5,
        )
        and np.allclose(
            norm_test["pro"],
            (clean_test["pro"] - mean_pro) / std_pro,
            atol=1e-5,
        )
    )
    report["normalization"] = {
        "provenance": "train-fold statistics only, applied to train and test",
        "verified_numerically": provenance_ok,
        "fold1_imu_mean": [round(float(v), 6) for v in mean_imu],
        "fold1_imu_std": [round(float(v), 6) for v in std_imu],
        "fold1_pro_mean": [round(float(v), 6) for v in mean_pro],
        "fold1_pro_std": [round(float(v), 6) for v in std_pro],
        "imu_channels": list(_common.IMU_CHANNELS),
        "pro_channels": list(_common.PRO_CHANNELS),
    }
    if not provenance_ok:
        failures.append("normalization provenance check failed")
    labels_train = sorted(set(int(v) for v in norm_train["labels"]))
    report["label_domain_after_cleanup"] = labels_train
    if labels_train != list(range(len(_common.OFFICIAL_CLASSES))):
        failures.append(f"unexpected label domain {labels_train}")

    _common.write_csv(
        run_dir / "fold_partition_counts.csv",
        ["fold", "class", "train_partitions", "test_partitions"],
        fold_partition_rows,
    )
    _common.write_csv(
        run_dir / "augmented_window_counts.csv",
        [
            "fold",
            "split",
            "class",
            "partitions",
            "n_slides_per_partition",
            "windows",
            "expected_windows",
            "imu_window_shape",
            "pro_window_shape",
        ],
        aug_rows,
    )
    report["failures"] = failures
    _common.write_json(run_dir / "pipeline_reconstruction.json", report)

    for line in failures:
        print(f"PROBLEM: {line}")
    print(f"total test windows across folds: {report['total_test_windows_all_folds']}")
    print("RECONSTRUCTION_OK" if not failures else "RECONSTRUCTION_FAIL")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
