"""Tests for the budget estimator and the storage-gate arithmetic."""

import csv

from tools.borealtc.runtime import _rt, budget


def _smoke_fixture() -> dict:
    return {
        "device": "cuda",
        "timings_s": {
            "get_recordings_s": 30.0,
            "partition_data_s": 10.0,
            "augment_data_s": 60.0,
            "cleanup_normalize_s": 5.0,
            "forward_s": 0.010,
            "backward_s": 0.015,
            "optimizer_step_s": 0.005,
            "val_forward_s": 0.008,
        },
        "train_dataset_len": 12904,
        "val_dataset_len": 1433,
        "test_dataset_len": 3617,
        "model_parameter_count": 100_000,
        "peak_cpu_rss_mib": 4000.0,
        "peak_gpu_mem_allocated_mib": 500.0,
        "spectrogram_cache_bytes_all_folds": 2_000_000_000,
        "status": "PASS",
    }


def test_steps_per_epoch_ceils():
    assert budget.steps_per_epoch(12904, 16) == 807
    assert budget.steps_per_epoch(12904, 10) == 1291
    assert budget.steps_per_epoch(10, 10) == 1


def test_fold_epoch_seconds_formula():
    got = budget.fold_epoch_seconds(100, 10, 10, step_s=2.0, val_forward_s=1.0)
    assert got == 10 * 2.0 + 1 * 1.0


def test_project_model_scenarios_ordered():
    proj = budget.project_model(_smoke_fixture(), "mamba")
    sc = proj["scenarios"]
    assert (
        sc["lower"]["five_fold_canonical_seed_hours"]
        < sc["central"]["five_fold_canonical_seed_hours"]
        < sc["conservative"]["five_fold_canonical_seed_hours"]
    )
    assert proj["steps_per_epoch"] == 807
    for name in sc:
        assert (
            sc[name]["with_restart_overhead_hours"]
            > sc[name]["five_fold_three_seeds_hours"]
        )
    assert sc["conservative"]["assumed_epochs_per_fold"] == 60


def test_storage_projection_reports_cap_fit():
    out = budget.storage_projection(_smoke_fixture(), None)
    assert out["total_bytes_est"] > 0
    assert out["fits_25gib_cap"] is True
    assert out["spectrogram_cache_bytes_measured"] == 2_000_000_000


def test_evaluate_storage_cap_and_reserve():
    baseline = {"cache/venvs/a": 0, "cache/pip": 1_000}
    now_ok = {"cache/venvs/a": 5 * _rt.GIB, "cache/pip": 1_000}
    ev = _rt.evaluate_storage(baseline, now_ok, now_free_bytes=100 * _rt.GIB)
    assert ev["cap_ok"] and ev["reserve_ok"]
    assert ev["added_since_baseline_bytes"] == 5 * _rt.GIB

    now_over = {"cache/venvs/a": 26 * _rt.GIB, "cache/pip": 1_000}
    ev2 = _rt.evaluate_storage(baseline, now_over, now_free_bytes=100 * _rt.GIB)
    assert not ev2["cap_ok"]

    ev3 = _rt.evaluate_storage(baseline, now_ok, now_free_bytes=74 * _rt.GIB)
    assert not ev3["reserve_ok"]


def test_failed_attempts_append_and_header(tmp_path):
    path = tmp_path / "failed_attempts.csv"
    row = {
        "attempt_id": "probe_mamba_build_1",
        "timestamp_utc": "2026-08-25T00:00:00Z",
        "phase": "declared_probe",
        "environment_label": "EXACT_DECLARED_STACK",
        "command_or_step": "pip install mamba-ssm==1.2.0.post1",
        "outcome": "FAILED",
        "error_class": "CalledProcessError",
        "error_snippet": "nvcc fatal",
        "log_path": "logs/x.log",
        "log_sha256": "ab" * 32,
    }
    _rt.append_failed_attempt(path, row)
    _rt.append_failed_attempt(path, dict(row, attempt_id="probe_mamba_build_2"))
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["attempt_id"] for r in rows] == [
        "probe_mamba_build_1",
        "probe_mamba_build_2",
    ]
    assert list(rows[0].keys()) == _rt.FAILED_ATTEMPTS_HEADER

    import pytest

    with pytest.raises(Exception):
        _rt.append_failed_attempt(path, {"attempt_id": "incomplete"})
