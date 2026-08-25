"""Released Mamba checkpoint audit for Stage 0B.2A (prompt §6).

Usage:

    python -m tools.borealtc.runtime.checkpoint_audit inventory \
        --out-dir <run_dir>
    python -m tools.borealtc.runtime.checkpoint_audit analyze \
        --out-dir <run_dir> --env-label <label> [--semantics <json>] \
        [--with-model] [--device cpu|cuda] [--forward-batch <npz>]

`inventory` is standard-library only (hashes, git identity, zip structure).
`analyze` needs torch; it records every load attempt (weights_only=True first,
then the hash-justified weights_only=False load), the state-dict inventory,
the inferred constructor arguments, and — only when a faithful runtime exists —
a real-batch forward. torch.load is applied exclusively to the pinned,
hash-recorded upstream checkpoint; no untrusted pickle is ever loaded.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

try:
    from tools.borealtc import _common
except ImportError:  # executed as a loose script
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tools.borealtc import _common

CKPT_REL = "checkpoints/mamba_borealtc.ckpt"
LOAD_ATTEMPTS_HEADER = [
    "attempt_id",
    "timestamp_utc",
    "env_label",
    "torch_version",
    "weights_only",
    "map_location",
    "outcome",
    "error_class",
    "error_snippet",
    "duration_s",
]

# MambaTerrain.__init__ kwargs expected in hyper_parameters (utils/models.py).
EXPECTED_HPARAM_KEYS = (
    "d_model_imu",
    "d_model_pro",
    "norm_epsilon",
    "ssm_cfg_imu",
    "ssm_cfg_pro",
    "out_method",
    "num_classes",
    "lr",
)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def infer_constructor_args(hparams: dict) -> dict:
    """Restrict stored hyperparameters to the released constructor surface."""
    return {k: to_jsonable(hparams.get(k, "__missing__")) for k in EXPECTED_HPARAM_KEYS}


def param_count(shapes: dict[str, tuple[int, ...]]) -> int:
    total = 0
    for shape in shapes.values():
        n = 1
        for dim in shape:
            n *= dim
        total += n
    return total


def prefix_summary(shapes: dict[str, tuple[int, ...]]) -> dict[str, int]:
    """Parameter count per top-level module prefix."""
    out: dict[str, int] = {}
    for key, shape in shapes.items():
        prefix = key.split(".", 1)[0]
        n = 1
        for dim in shape:
            n *= dim
        out[prefix] = out.get(prefix, 0) + n
    return out


def fold_evidence(strings: list[str]) -> dict:
    """Collect fold-identity evidence from strings found in the checkpoint."""
    hits = sorted({s for s in strings if "fold" in s.lower()})
    return {
        "fold_marker_strings": hits,
        "verdict": (
            "FOLD_MARKER_PRESENT" if hits else "NO_FOLD_MARKER_IN_CHECKPOINT"
        ),
    }


def compare_expected(ctor: dict, semantics: dict | None) -> dict:
    """Compare inferred constructor args against mamba_train.py literals."""
    if not semantics:
        return {"available": False}
    cfg = semantics.get("training_configs", {}).get("mamba_train.py", {})
    opt = cfg.get("mamba_train_opt", {})
    expected = {
        "d_model_imu": opt.get("d_model_imu"),
        "d_model_pro": opt.get("d_model_pro"),
        "norm_epsilon": opt.get("norm_epsilon"),
        "ssm_cfg_imu": cfg.get("ssm_cfg_imu"),
        "ssm_cfg_pro": cfg.get("ssm_cfg_pro"),
        "out_method": opt.get("out_method"),
        "lr": opt.get("init_learn_rate"),
    }
    rows = {}
    for key, exp in expected.items():
        got = ctor.get(key)
        rows[key] = {"expected_from_source": exp, "checkpoint": got, "match": got == exp}
    return {"available": True, "fields": rows}


def _string_scan(obj, out: list[str]) -> None:
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _string_scan(k, out)
            _string_scan(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _string_scan(v, out)


def _merge_inventory(path: Path, update: dict) -> None:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.update(update)
    _common.write_json(path, data)


def cmd_inventory(out_dir: Path) -> int:
    upstream = _common.upstream_dir()
    ckpt = upstream / CKPT_REL
    proc = subprocess.run(
        ["git", "-C", str(upstream), "ls-tree", "HEAD", CKPT_REL],
        capture_output=True,
        text=True,
        timeout=60,
    )
    parts = proc.stdout.split()
    blob = parts[2] if len(parts) >= 3 else "unavailable"
    with zipfile.ZipFile(ckpt) as zf:
        entries = [
            {"name": i.filename, "size": i.file_size, "compressed": i.compress_size}
            for i in zf.infolist()
        ]
    inventory = {
        "file": CKPT_REL,
        "size_bytes": ckpt.stat().st_size,
        "sha256": _common.sha256_file(ckpt),
        "git_blob_id": blob,
        "is_zip_torch_serialization": True,
        "zip_entry_count": len(entries),
        "zip_entries_first_20": entries[:20],
        "inventory_timestamp_utc": _common.utc_iso(),
    }
    _merge_inventory(out_dir / "checkpoint_inventory.json", inventory)
    print(
        f"CHECKPOINT_INVENTORY_WRITTEN: sha256={inventory['sha256'][:16]}... "
        f"size={inventory['size_bytes']} git_blob={blob[:12]}"
    )
    return 0


def _record_attempt(csv_path: Path, row: dict) -> None:
    import csv as _csv

    new = not csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=LOAD_ATTEMPTS_HEADER)
        if new:
            writer.writeheader()
        writer.writerow(row)


def cmd_analyze(args) -> int:
    import torch

    out_dir = Path(args.out_dir)
    upstream = _common.upstream_dir()
    ckpt_path = upstream / CKPT_REL
    attempts_csv = out_dir / "checkpoint_load_attempts.csv"
    sha = _common.sha256_file(ckpt_path)

    semantics = None
    if args.semantics:
        semantics = json.loads(Path(args.semantics).read_text(encoding="utf-8"))

    def attempt(attempt_id: str, weights_only: bool):
        start = time.perf_counter()
        row = {
            "attempt_id": attempt_id,
            "timestamp_utc": _common.utc_iso(),
            "env_label": args.env_label,
            "torch_version": torch.__version__,
            "weights_only": weights_only,
            "map_location": "cpu",
            "outcome": "",
            "error_class": "",
            "error_snippet": "",
            "duration_s": 0.0,
        }
        try:
            obj = torch.load(ckpt_path, map_location="cpu", weights_only=weights_only)
            row["outcome"] = "LOADED"
            return obj, row
        except Exception as exc:  # noqa: BLE001 - recorded, never suppressed
            row["outcome"] = "FAILED"
            row["error_class"] = type(exc).__name__
            row["error_snippet"] = str(exc)[:400].replace("\n", " | ")
            return None, row
        finally:
            row["duration_s"] = round(time.perf_counter() - start, 4)

    obj_wo, row_wo = attempt(f"{args.env_label}_weights_only_true", True)
    _record_attempt(attempts_csv, row_wo)
    loaded = obj_wo
    if loaded is None:
        # The artifact is the pinned, hash-recorded upstream checkpoint
        # (sha256 recorded above); a full unpickling load is justified for it
        # and for it only.
        obj_full, row_full = attempt(f"{args.env_label}_weights_only_false", False)
        _record_attempt(attempts_csv, row_full)
        loaded = obj_full
    if loaded is None:
        print("CHECKPOINT_LOAD_FAILED: see checkpoint_load_attempts.csv")
        return 1

    top_keys = sorted(loaded.keys()) if isinstance(loaded, dict) else []
    state_dict = loaded.get("state_dict", {})
    shapes = {k: tuple(v.shape) for k, v in state_dict.items()}
    dtypes = {k: str(v.dtype) for k, v in state_dict.items()}
    rows = [
        [k, "x".join(map(str, shapes[k])) or "scalar", dtypes[k], int(state_dict[k].numel())]
        for k in state_dict
    ]
    _common.write_csv(
        out_dir / "checkpoint_state_dict.csv",
        ["key", "shape", "dtype", "numel"],
        rows,
    )

    hparams = to_jsonable(dict(loaded.get("hyper_parameters", {})))
    ctor = infer_constructor_args(hparams)
    strings: list[str] = []
    _string_scan(to_jsonable({k: v for k, v in loaded.items() if k != "state_dict"}), strings)

    analysis = {
        "env_label": args.env_label,
        "torch_version": torch.__version__,
        "sha256_at_analysis": sha,
        "top_level_keys": top_keys,
        "lightning_version": loaded.get("pytorch-lightning_version", "absent"),
        "epoch": loaded.get("epoch", "absent"),
        "global_step": loaded.get("global_step", "absent"),
        "has_optimizer_states": "optimizer_states" in loaded,
        "has_lr_schedulers": "lr_schedulers" in loaded,
        "has_loops_state": "loops" in loaded,
        "callback_keys": sorted(loaded.get("callbacks", {}).keys())
        if isinstance(loaded.get("callbacks"), dict)
        else "absent",
        "hyper_parameters": hparams,
        "inferred_constructor_args": ctor,
        "constructor_vs_source": compare_expected(ctor, semantics),
        "state_dict_tensor_count": len(shapes),
        "parameter_count_total": param_count(shapes),
        "parameter_count_by_module": prefix_summary(shapes),
        "torch_cuda_metadata_in_checkpoint": "none stored by Lightning format",
        "fold_identity": fold_evidence(strings),
        "table3_equivalence": {
            "verdict": "NOT_DEMONSTRATED",
            "reason": (
                "Table III is the concatenation of five per-fold test "
                "predictions; the release ships exactly one BorealTC Mamba "
                "checkpoint with no fold marker, so equivalence to the "
                "five-fold ensemble cannot be established from released "
                "evidence"
            ),
        },
    }

    if args.with_model:
        analysis["model_load"] = _model_load(
            loaded, state_dict, args.device, args.forward_batch
        )

    _merge_inventory(
        out_dir / "checkpoint_inventory.json", {f"analysis_{args.env_label}": analysis}
    )
    print(
        f"CHECKPOINT_ANALYZED[{args.env_label}]: params="
        f"{analysis['parameter_count_total']} keys={len(shapes)} "
        f"fold={analysis['fold_identity']['verdict']}"
    )
    return 0


def _model_load(loaded, state_dict, device: str, forward_batch: str | None) -> dict:
    """Construct the released MambaTerrain and load the state dict strictly."""
    import torch

    result: dict = {"device": device}
    try:
        _common.add_upstream_to_syspath()
        from utils.models import MambaTerrain  # noqa: PLC0415

        hparams = dict(loaded.get("hyper_parameters", {}))
        model = MambaTerrain(**hparams)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        result["constructed"] = True
        result["missing_keys"] = list(missing)
        result["unexpected_keys"] = list(unexpected)
        result["strict_load_ok"] = not missing and not unexpected
        result["model_parameter_count"] = sum(p.numel() for p in model.parameters())
    except Exception as exc:  # noqa: BLE001 - recorded, never suppressed
        result["constructed"] = False
        result["error_class"] = type(exc).__name__
        result["error_snippet"] = str(exc)[:400]
        return result

    if forward_batch:
        try:
            import numpy as np  # noqa: PLC0415

            batch = np.load(forward_batch)
            x = {
                "imu": torch.from_numpy(batch["imu"]).to(device),
                "pro": torch.from_numpy(batch["pro"]).to(device),
            }
            model = model.to(device).eval()
            with torch.no_grad():
                logits = model(x)
            probs = torch.softmax(logits, dim=1)
            result["forward"] = {
                "input_imu_shape": list(batch["imu"].shape),
                "input_pro_shape": list(batch["pro"].shape),
                "logits_shape": list(logits.shape),
                "logits_finite": bool(torch.isfinite(logits).all().item()),
                "pred_classes": probs.argmax(dim=1).tolist(),
                "true_labels": [int(v) for v in batch["labels"].tolist()],
                "outcome": "FORWARD_OK",
            }
        except Exception as exc:  # noqa: BLE001 - recorded, never suppressed
            result["forward"] = {
                "outcome": "FORWARD_FAILED",
                "error_class": type(exc).__name__,
                "error_snippet": str(exc)[:400],
            }
    else:
        result["forward"] = {"outcome": "SKIPPED_NO_FAITHFUL_RUNTIME_BATCH"}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p_inv = sub.add_parser("inventory")
    p_inv.add_argument("--out-dir", required=True)
    p_an = sub.add_parser("analyze")
    p_an.add_argument("--out-dir", required=True)
    p_an.add_argument("--env-label", required=True)
    p_an.add_argument("--semantics", default=None)
    p_an.add_argument("--with-model", action="store_true")
    p_an.add_argument("--device", default="cpu")
    p_an.add_argument("--forward-batch", default=None)
    args = parser.parse_args()
    if args.action == "inventory":
        return cmd_inventory(Path(args.out_dir))
    return cmd_analyze(args)


if __name__ == "__main__":
    sys.exit(main())
