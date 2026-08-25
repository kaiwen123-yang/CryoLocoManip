"""Real-data one-batch training smokes for Stage 0B.2A (prompt §7).

Usage:

    python -m tools.borealtc.runtime.smoke --model cnn   --env-label <label> \
        --out <cnn_smoke.json>   [--device cuda] [--fold 1]
    python -m tools.borealtc.runtime.smoke --model mamba --env-label <label> \
        --out <mamba_smoke.json> [--device cuda] [--fold 1] \
        [--dump-test-batch <npz>]

Executes, with the unmodified upstream pipeline and model classes on real
BorealTC data for the selected official fold: exactly one training batch
(forward, backward, optimizer step) and one validation forward batch. The
Lightning Trainer is deliberately NOT used, so no epoch loop, checkpoint
write, or logger write can occur; optimizer, loss, gradient clipping, and
batch construction follow the released configuration. A one-batch smoke
establishes runtime feasibility only — its loss/accuracy values are not a
reproduction result.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
import traceback
from pathlib import Path

try:
    from tools.borealtc import _common
    from tools.borealtc.runtime import _rt
except ImportError:  # executed as a loose script
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tools.borealtc import _common
    from tools.borealtc.runtime import _rt

# Released configurations, transcribed from the pinned mamba_train.py /
# cnn_train.py (verified against runtime_source_semantics.json by tests).
MAMBA_SSM_CFG_IMU = {"d_state": 16, "d_conv": 4, "expand": 4}
MAMBA_SSM_CFG_PRO = {"d_state": 16, "d_conv": 3, "expand": 6}
MAMBA_TRAIN_OPT = {
    "d_model_imu": 32,
    "d_model_pro": 8,
    "norm_epsilon": 6.3e-6,
    "valid_perc": 0.1,
    "init_learn_rate": 1.5e-3,
    "learn_drop_factor": 0.25,
    "reduce_lr_patience": 4,
    "max_epochs": 60,
    "minibatch_size": 16,
    "valid_patience": 8,
    "gradient_threshold": None,
    "focal_loss": True,
    "focal_loss_alpha": 0.75,
    "focal_loss_gamma": 2.25,
    "num_classes": 5,
    "out_method": "last_state",
}
CNN_PAR = {
    "num_classes": 5,
    "time_window": 0.4,
    "time_overlap": 0.2,
    "filter_size": [3, 3],
    "num_filters": 32,
}
CNN_TRAIN_OPT = {
    "hamming": True,
    "valid_perc": 0.1,
    "init_learn_rate": 0.005,
    "learn_drop_factor": 0.1,
    "max_epochs": 150,
    "minibatch_size": 10,
    "valid_patience": 8,
    "scheduler": "plateau",
    "reduce_lr_patience": 4,
    "gradient_threshold": 6,
    "focal_loss": False,
    "dropout": 0.0,
}


def _rss_mib() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


class _Timer:
    def __init__(self, sync):
        self._sync = sync
        self.laps: dict[str, float] = {}

    def lap(self, name: str, start: float) -> float:
        self._sync()
        end = time.perf_counter()
        self.laps[name] = round(end - start, 4)
        return end


def _build_fold_data(model_kind: str, fold_index: int, timer: _Timer) -> dict:
    """Run the released preprocessing chain and return fold tensors."""
    t0 = time.perf_counter()
    pre = _common.load_upstream_preprocessing()
    summary = _common.make_upstream_summary()
    csv_dir = _common.upstream_dir() / _common.DATA_SUBDIR
    terr_dfs = pre.get_recordings(csv_dir, summary)
    t1 = timer.lap("get_recordings_s", t0)
    train_folds, test_folds = pre.partition_data(
        terr_dfs,
        summary,
        int(_common.PART_WINDOW_S),
        _common.N_FOLDS,
        random_state=_common.RANDOM_STATE,
    )
    t2 = timer.lap("partition_data_s", t1)
    aug_train, aug_test = pre.augment_data(
        train_folds,
        test_folds,
        summary,
        moving_window=_common.MOVING_WINDOW_S,
        stride=_common.STRIDE_S,
        homogeneous=_common.HOMOGENEOUS_AUGMENTATION,
    )
    t3 = timer.lap("augment_data_s", t2)

    if model_kind == "mamba":
        train_fold, test_fold = pre.cleanup_data(
            aug_train[fold_index], aug_test[fold_index]
        )
        train_fold, test_fold = pre.normalize_data(train_fold, test_fold)
        timer.lap("cleanup_normalize_s", t3)
        return {"train": train_fold, "test": test_fold, "summary": summary}

    import einops as ein  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    train_mcs_folds, test_mcs_folds = pre.apply_multichannel_spectogram(
        aug_train,
        aug_test,
        summary,
        _common.MOVING_WINDOW_S,
        CNN_PAR["time_window"],
        CNN_PAR["time_overlap"],
        hamming=CNN_TRAIN_OPT["hamming"],
    )
    t4 = timer.lap("multichannel_spectrogram_all_folds_s", t3)
    maxs = np.stack(
        [
            ein.rearrange(train_mcs_folds[idx]["data"], "a b c d -> (a b c) d").max(axis=0)
            for idx in range(_common.N_FOLDS)
        ]
    ).max(axis=0)
    mins = np.stack(
        [
            ein.rearrange(train_mcs_folds[idx]["data"], "a b c d -> (a b c) d").min(axis=0)
            for idx in range(_common.N_FOLDS)
        ]
    ).min(axis=0)
    timer.lap("global_min_max_s", t4)
    spect_bytes = int(
        sum(train_mcs_folds[i]["data"].nbytes for i in range(_common.N_FOLDS))
        + sum(test_mcs_folds[i]["data"].nbytes for i in range(_common.N_FOLDS))
    )
    return {
        "train": train_mcs_folds[fold_index],
        "test": test_mcs_folds[fold_index],
        "mins": mins,
        "maxs": maxs,
        "summary": summary,
        "spectrogram_cache_bytes_all_folds": spect_bytes,
    }


def _one_batch_cycle(model, batch_to_device, train_batch, val_batch, grad_clip, timer):
    """One faithful train step + one validation forward. Returns measurements."""
    import torch

    device_params_before = [p.detach().clone() for p in model.parameters()]
    cfg = model.configure_optimizers()
    optimizer = cfg["optimizer"]
    scheduler_name = type(cfg["lr_scheduler"]["scheduler"]).__name__

    x, target = batch_to_device(train_batch)
    t0 = time.perf_counter()
    model.train()
    pred = model(x)
    logits = pred[0] if isinstance(pred, tuple) else pred
    loss = model.loss(logits, target)
    t1 = timer.lap("forward_s", t0)
    loss.backward()
    t2 = timer.lap("backward_s", t1)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    grad_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(g.detach()) for g in grads])
    ).item()
    grads_all_finite = all(bool(torch.isfinite(g).all().item()) for g in grads)
    if grad_clip is not None:
        # lightning.Trainer(gradient_clip_val=...) defaults to norm clipping.
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
    optimizer.step()
    optimizer.zero_grad()
    timer.lap("optimizer_step_s", t2)

    update_sq = 0.0
    params_finite = True
    for before, after in zip(device_params_before, model.parameters()):
        delta = after.detach() - before
        update_sq += float(torch.sum(delta * delta).item())
        params_finite = params_finite and bool(torch.isfinite(after).all().item())
    param_update_norm = update_sq**0.5

    xv, tv = batch_to_device(val_batch)
    t3 = time.perf_counter()
    model.eval()
    with torch.no_grad():
        pred_v = model(xv)
        logits_v = pred_v[0] if isinstance(pred_v, tuple) else pred_v
        val_loss = model.loss(logits_v, tv)
    timer.lap("val_forward_s", t3)

    return {
        "train_loss": float(loss.item()),
        "val_loss": float(val_loss.item()),
        "grad_global_norm_preclip": grad_norm,
        "grad_global_norm": grad_norm,
        "grads_all_finite": grads_all_finite,
        "param_update_norm": param_update_norm,
        "params_finite_after_step": params_finite,
        "optimizer_steps": 1,
        "backward_calls": 1,
        "optimizer_class": type(optimizer).__name__,
        "optimizer_lr": optimizer.param_groups[0]["lr"],
        "scheduler_class": scheduler_name,
        "gradient_clip_val": grad_clip,
        "logits_finite": bool(torch.isfinite(logits).all().item()),
        "train_logits_shape": list(logits.shape),
        "val_logits_shape": list(logits_v.shape),
    }


def _smoke_mamba(args, timer, result):
    import lightning as L
    import torch

    _common.add_upstream_to_syspath()
    from utils.datamodule import MambaDataModule  # noqa: PLC0415
    from utils.models import MambaTerrain  # noqa: PLC0415

    fold = _build_fold_data("mamba", args.fold - 1, timer)
    L.seed_everything(_common.RANDOM_STATE)
    t0 = time.perf_counter()
    # Loader I/O deviation (recorded in the result JSON): the released
    # mamba_network uses num_workers=8 / persistent_workers=True, but
    # fork-based DataLoader workers deadlock on this WSL2 host even when the
    # parent process has never initialized CUDA (evidence:
    # mamba_smoke_pretest_hang_evidence.txt). num_workers=0 changes only who
    # executes __getitem__; the datamodule split, seeded sampler order, batch
    # composition, and all numerical semantics are unchanged. The released
    # worker configuration is a recorded Stage 0B.2B execution risk.
    datamodule = MambaDataModule(
        fold["train"],
        fold["test"],
        train_transform=None,
        test_transform=None,
        train_data_augmentation=None,
        valid_percent=MAMBA_TRAIN_OPT["valid_perc"],
        batch_size=MAMBA_TRAIN_OPT["minibatch_size"],
        num_workers=0,
        persistent_workers=False,
    )
    result["loader_deviation"] = (
        "released num_workers=8/persistent_workers=True replaced by "
        "num_workers=0 for this smoke: fork-based workers deadlock on WSL2 "
        "(host limitation, recorded as 0B.2B risk); batch composition and "
        "numerical semantics unchanged"
    )
    t1 = timer.lap("datamodule_init_s", t0)

    # Batches are still fetched before the first CUDA call so the harness
    # itself never mixes CUDA initialization into data loading.
    train_batch = next(iter(datamodule.train_dataloader()))
    val_batch = next(iter(datamodule.val_dataloader()))
    t2 = timer.lap("first_batch_fetch_s", t1)

    if args.dump_test_batch:
        import numpy as np  # noqa: PLC0415

        test_batch = next(iter(datamodule.test_dataloader()))
        xb, yb = test_batch
        np.savez(
            args.dump_test_batch,
            imu=xb["imu"].numpy(),
            pro=xb["pro"].numpy(),
            labels=yb.numpy(),
        )
        result["test_batch_dumped_to"] = args.dump_test_batch

    model = MambaTerrain(
        d_model_imu=MAMBA_TRAIN_OPT["d_model_imu"],
        d_model_pro=MAMBA_TRAIN_OPT["d_model_pro"],
        norm_epsilon=MAMBA_TRAIN_OPT["norm_epsilon"],
        ssm_cfg_imu=MAMBA_SSM_CFG_IMU,
        ssm_cfg_pro=MAMBA_SSM_CFG_PRO,
        out_method=MAMBA_TRAIN_OPT["out_method"],
        num_classes=MAMBA_TRAIN_OPT["num_classes"],
        lr=MAMBA_TRAIN_OPT["init_learn_rate"],
        learning_rate_factor=MAMBA_TRAIN_OPT["learn_drop_factor"],
        reduce_lr_patience=MAMBA_TRAIN_OPT["reduce_lr_patience"],
        class_weights=None,
        focal_loss=MAMBA_TRAIN_OPT["focal_loss"],
        focal_loss_alpha=MAMBA_TRAIN_OPT["focal_loss_alpha"],
        focal_loss_gamma=MAMBA_TRAIN_OPT["focal_loss_gamma"],
    ).to(args.device)
    timer.lap("model_init_s", t2)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def to_device(batch):
        x, y = batch
        return (
            {"imu": x["imu"].to(args.device), "pro": x["pro"].to(args.device)},
            y.to(args.device),
        )

    x, y = train_batch
    result["input_shapes"] = {
        "imu": list(x["imu"].shape),
        "pro": list(x["pro"].shape),
    }
    result["input_dtypes"] = {"imu": str(x["imu"].dtype), "pro": str(x["pro"].dtype)}
    result["label_dtype"] = str(y.dtype)
    result["label_values"] = sorted(set(int(v) for v in y.tolist()))
    result["labels_in_domain"] = _rt.labels_in_domain(y.tolist())
    result["inputs_finite"] = bool(
        torch.isfinite(x["imu"]).all().item() and torch.isfinite(x["pro"]).all().item()
    )
    result["train_dataset_len"] = len(datamodule.train_dataset)
    result["val_dataset_len"] = len(datamodule.val_dataset)
    result["test_dataset_len"] = len(datamodule.test_dataset)
    result["model_parameter_count"] = sum(p.numel() for p in model.parameters())

    cycle = _one_batch_cycle(
        model, to_device, train_batch, val_batch, MAMBA_TRAIN_OPT["gradient_threshold"], timer
    )
    result.update(cycle)


def _smoke_cnn(args, timer, result):
    import lightning as L
    import numpy as np
    import pipeline as pp
    import torch

    _common.add_upstream_to_syspath()
    from utils.augmentations import NormalizeMCS  # noqa: PLC0415
    from utils.datamodule import MCSDataModule  # noqa: PLC0415
    from utils.models import CNNTerrain  # noqa: PLC0415

    fold = _build_fold_data("cnn", args.fold - 1, timer)
    result["spectrogram_cache_bytes_all_folds"] = fold["spectrogram_cache_bytes_all_folds"]
    L.seed_everything(_common.RANDOM_STATE)

    def to_f32(arr):
        return arr.astype(np.float32)

    def transpose(arr):
        return np.transpose(arr, (2, 0, 1))

    train_transform = pp.Bifunctor(
        pp.Compose([transpose, NormalizeMCS(fold["mins"], fold["maxs"]), to_f32]),
        pp.Identity(),
    )
    test_transform = pp.Bifunctor(
        pp.Compose([transpose, NormalizeMCS(fold["mins"], fold["maxs"]), to_f32]),
        pp.Identity(),
    )
    t0 = time.perf_counter()
    datamodule = MCSDataModule(
        fold["train"],
        fold["test"],
        train_transform,
        test_transform,
        pp.Identity(),
        valid_percent=CNN_TRAIN_OPT["valid_perc"],
        batch_size=CNN_TRAIN_OPT["minibatch_size"],
        num_workers=0,
        persistent_workers=False,
    )
    t1 = timer.lap("datamodule_init_s", t0)

    # Released CNN loaders use num_workers=0 (no fork), but batches are still
    # fetched before the first CUDA call for symmetry with the Mamba smoke.
    train_batch = next(iter(datamodule.train_dataloader()))
    val_batch = next(iter(datamodule.val_dataloader()))
    t1 = timer.lap("first_batch_fetch_s", t1)

    n_samples, n_freq, n_wind, in_size = fold["train"]["data"].shape
    model = CNNTerrain(
        in_size=in_size,
        num_filters=CNN_PAR["num_filters"],
        filter_size=CNN_PAR["filter_size"],
        num_classes=CNN_PAR["num_classes"],
        n_wind=n_wind,
        n_freq=n_freq,
        lr=CNN_TRAIN_OPT["init_learn_rate"],
        learning_rate_factor=CNN_TRAIN_OPT["learn_drop_factor"],
        reduce_lr_patience=CNN_TRAIN_OPT["reduce_lr_patience"],
        focal_loss=CNN_TRAIN_OPT["focal_loss"],
        scheduler=CNN_TRAIN_OPT["scheduler"],
        dropout=CNN_TRAIN_OPT["dropout"],
    ).to(args.device)
    timer.lap("model_init_s", t1)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def to_device(batch):
        x, y = batch
        return x.to(args.device), y.to(args.device)

    x, y = train_batch
    result["input_shapes"] = {"mcs": list(x.shape)}
    result["input_dtypes"] = {"mcs": str(x.dtype)}
    result["mcs_fold_shape"] = [int(n_samples), int(n_freq), int(n_wind), int(in_size)]
    result["label_dtype"] = str(y.dtype)
    result["label_values"] = sorted(set(int(v) for v in y.tolist()))
    result["labels_in_domain"] = _rt.labels_in_domain(y.tolist())
    result["inputs_finite"] = bool(torch.isfinite(x).all().item())
    result["train_dataset_len"] = len(datamodule.train_dataset)
    result["val_dataset_len"] = len(datamodule.val_dataset)
    result["test_dataset_len"] = len(datamodule.test_dataset)
    result["model_parameter_count"] = sum(p.numel() for p in model.parameters())

    cycle = _one_batch_cycle(
        model, to_device, train_batch, val_batch, CNN_TRAIN_OPT["gradient_threshold"], timer
    )
    result.update(cycle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("cnn", "mamba"), required=True)
    parser.add_argument("--env-label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--dump-test-batch", default=None)
    args = parser.parse_args()

    upstream_before = _common.verify_upstream_identity(require_clean=True)

    result: dict = {
        "model": args.model,
        "env_label": args.env_label,
        "device": args.device,
        "fold": args.fold,
        "seed": _common.RANDOM_STATE,
        "started_utc": _common.utc_iso(),
        "upstream_commit": upstream_before["commit"],
        "note": (
            "one-batch runtime-feasibility smoke; no full epoch or fold was "
            "trained and no accuracy from this run is a reproduction result"
        ),
    }

    import torch

    def sync():
        # Never the first CUDA call: synchronizing only an already-initialized
        # context keeps the pre-fork phase CUDA-clean (see WSL2 fork deadlock
        # note in the smoke functions).
        if args.device.startswith("cuda") and torch.cuda.is_initialized():
            torch.cuda.synchronize()

    timer = _Timer(sync)
    result["torch_version"] = torch.__version__
    result["torch_cuda_version"] = torch.version.cuda

    status = "PASS"
    try:
        if args.model == "mamba":
            _smoke_mamba(args, timer, result)
        else:
            _smoke_cnn(args, timer, result)
        result["torch_initial_seed"] = int(torch.initial_seed())
        status = _rt.smoke_status(result)
    except Exception as exc:  # noqa: BLE001 - recorded, never suppressed
        status = "FAIL_EXCEPTION"
        result["error_class"] = type(exc).__name__
        result["error_message"] = str(exc)[:2000]
        result["traceback"] = traceback.format_exc()[-4000:]

    result["timings_s"] = timer.laps
    result["peak_cpu_rss_mib"] = _rss_mib()
    if args.device.startswith("cuda") and torch.cuda.is_initialized():
        result["cuda_available"] = True
        result["gpu_name"] = torch.cuda.get_device_name(0)
        result["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        result["peak_gpu_mem_allocated_mib"] = round(
            torch.cuda.max_memory_allocated() / (1024**2), 1
        )
        result["peak_gpu_mem_reserved_mib"] = round(
            torch.cuda.max_memory_reserved() / (1024**2), 1
        )

    upstream_after = _common.verify_upstream_identity(require_clean=True)
    result["upstream_clean_after"] = upstream_after["worktree_clean"]
    result["status"] = status
    result["ended_utc"] = _common.utc_iso()
    _common.write_json(Path(args.out), result)
    print(
        f"SMOKE[{args.model}/{args.env_label}]: status={status} "
        f"train_loss={result.get('train_loss')} val_loss={result.get('val_loss')} "
        f"vram_mib={result.get('peak_gpu_mem_allocated_mib')}"
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
