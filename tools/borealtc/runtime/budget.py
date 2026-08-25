"""Projected Stage 0B.2B full-training budget from smoke measurements (§8).

Usage:

    python -m tools.borealtc.runtime.budget --cnn-smoke <cnn_smoke.json> \
        [--mamba-smoke <mamba_smoke.json>] --out <stage0b2b_budget.json>

All projections derive from measured one-batch and preprocessing timings of
the SAME device the smoke ran on; no GPU-to-CPU extrapolation is performed.
Lower/central/conservative epoch-count assumptions are declared explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

try:
    from tools.borealtc import _common
    from tools.borealtc.runtime import _rt
except ImportError:  # executed as a loose script
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tools.borealtc import _common
    from tools.borealtc.runtime import _rt

SEEDS_TOTAL = 3  # canonical seed 21 + at least two predeclared stability seeds
RESTART_OVERHEAD_FACTOR = 1.15  # declared assumption: 15% failure/restart slack

# Declared epoch-count assumptions per scenario. max_epochs values come from
# the released configurations; early-stopping behavior is unknown before
# 0B.2B, so the conservative bound always uses max_epochs.
EPOCH_SCENARIOS = {
    "cnn": {"lower": 25, "central": 60, "conservative": 150},
    "mamba": {"lower": 15, "central": 35, "conservative": 60},
}
MAX_EPOCHS = {"cnn": 150, "mamba": 60}
BATCH_SIZE = {"cnn": 10, "mamba": 16}


def steps_per_epoch(train_len: int, batch_size: int) -> int:
    return math.ceil(train_len / batch_size)


def fold_epoch_seconds(
    train_len: int,
    val_len: int,
    batch_size: int,
    step_s: float,
    val_forward_s: float,
) -> float:
    """One epoch = all train steps + one validation sweep (released
    val_check_interval covers the epoch at least once)."""
    train_steps = steps_per_epoch(train_len, batch_size)
    val_steps = steps_per_epoch(val_len, batch_size)
    return train_steps * step_s + val_steps * val_forward_s


def project_model(smoke: dict, model_kind: str) -> dict:
    timings = smoke["timings_s"]
    step_s = (
        timings.get("forward_s", 0.0)
        + timings.get("backward_s", 0.0)
        + timings.get("optimizer_step_s", 0.0)
    )
    val_s = timings.get("val_forward_s", 0.0)
    train_len = smoke["train_dataset_len"]
    val_len = smoke["val_dataset_len"]
    batch = BATCH_SIZE[model_kind]
    preprocessing_s = sum(
        timings.get(k, 0.0)
        for k in (
            "get_recordings_s",
            "partition_data_s",
            "augment_data_s",
            "cleanup_normalize_s",
            "multichannel_spectrogram_all_folds_s",
            "global_min_max_s",
        )
    )
    epoch_s = fold_epoch_seconds(train_len, val_len, batch, step_s, val_s)
    scenarios = {}
    for name, epochs in EPOCH_SCENARIOS[model_kind].items():
        fold_s = epochs * epoch_s
        five_fold_s = 5 * fold_s + preprocessing_s
        scenarios[name] = {
            "assumed_epochs_per_fold": epochs,
            "epoch_seconds": round(epoch_s, 2),
            "single_fold_hours": round(fold_s / 3600, 3),
            "five_fold_canonical_seed_hours": round(five_fold_s / 3600, 3),
            "five_fold_three_seeds_hours": round(
                (5 * SEEDS_TOTAL * fold_s + preprocessing_s) / 3600, 3
            ),
            "with_restart_overhead_hours": round(
                (5 * SEEDS_TOTAL * fold_s + preprocessing_s)
                * RESTART_OVERHEAD_FACTOR
                / 3600,
                3,
            ),
        }
    return {
        "device": smoke.get("device"),
        "measured_step_seconds": round(step_s, 4),
        "measured_val_forward_seconds": round(val_s, 4),
        "measured_preprocessing_seconds": round(preprocessing_s, 2),
        "train_dataset_len_fold1": train_len,
        "val_dataset_len_fold1": val_len,
        "batch_size": batch,
        "steps_per_epoch": steps_per_epoch(train_len, batch),
        "max_epochs_released": MAX_EPOCHS[model_kind],
        "peak_cpu_rss_mib": smoke.get("peak_cpu_rss_mib"),
        "peak_gpu_mem_allocated_mib": smoke.get("peak_gpu_mem_allocated_mib"),
        "peak_gpu_mem_reserved_mib": smoke.get("peak_gpu_mem_reserved_mib"),
        "scenarios": scenarios,
        "extrapolation_note": (
            "projected for the measured device only; CPU training was not "
            "measured and is not extrapolated"
        ),
    }


def storage_projection(cnn_smoke: dict, mamba_smoke: dict | None) -> dict:
    ckpt_bytes_mamba = 350_000  # released mamba_borealtc.ckpt scale (~0.33 MiB)
    cnn_params = cnn_smoke.get("model_parameter_count", 0)
    # Lightning checkpoint ~= params * 4 bytes * 3 (weights + Adam moments)
    # plus metadata; save_top_k=1 + save_last=True keeps 2 checkpoints.
    cnn_ckpt_bytes = cnn_params * 4 * 3 + 1_000_000
    per_model_runs = 5 * SEEDS_TOTAL
    spect_cache = cnn_smoke.get("spectrogram_cache_bytes_all_folds", 0)
    ckpt_total = per_model_runs * 2 * cnn_ckpt_bytes + per_model_runs * 2 * (
        ckpt_bytes_mamba * 4  # training ckpt adds optimizer state
    )
    logs_total = per_model_runs * 2 * 50_000_000  # tensorboard + stdout, generous
    total = ckpt_total + logs_total + spect_cache
    return {
        "cnn_checkpoint_bytes_each_est": cnn_ckpt_bytes,
        "mamba_checkpoint_bytes_each_est": ckpt_bytes_mamba * 4,
        "checkpoints_total_bytes_est": ckpt_total,
        "logs_total_bytes_est": logs_total,
        "spectrogram_cache_bytes_measured": spect_cache,
        "total_bytes_est": total,
        "total_gib_est": _rt.bytes_to_gib(total),
        "fits_25gib_cap": total < 25 * _rt.GIB,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-smoke", required=True)
    parser.add_argument("--mamba-smoke", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cnn = json.loads(Path(args.cnn_smoke).read_text(encoding="utf-8"))
    mamba = (
        json.loads(Path(args.mamba_smoke).read_text(encoding="utf-8"))
        if args.mamba_smoke
        else None
    )
    budget = {
        "stage": "stage0b2b_projection",
        "timestamp_utc": _common.utc_iso(),
        "assumptions": {
            "seeds_total": SEEDS_TOTAL,
            "restart_overhead_factor": RESTART_OVERHEAD_FACTOR,
            "epoch_scenarios": EPOCH_SCENARIOS,
            "note": (
                "lower/central epoch counts are declared assumptions about "
                "early stopping; conservative uses released max_epochs"
            ),
        },
        "cnn": project_model(cnn, "cnn"),
        "mamba": (
            project_model(mamba, "mamba")
            if mamba and mamba.get("status") == "PASS"
            else {
                "status": "BLOCKED_OR_ABSENT",
                "note": "no faithful Mamba smoke measurement is available",
            }
        ),
        "storage": storage_projection(cnn, mamba),
    }
    _common.write_json(Path(args.out), budget)
    print(f"BUDGET_WRITTEN: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
