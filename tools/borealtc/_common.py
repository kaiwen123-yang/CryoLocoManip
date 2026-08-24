"""Shared identities, comparison configuration, and helpers for Stage 0B.1.

Design rules:

- Module import stays standard-library only; NumPy/pandas and the upstream
  BorealTC modules are imported lazily inside functions so that pure helpers
  remain testable without the scientific environment.
- Expected paper/repository values live here as *comparison configuration*
  only. Tools must compute their outputs from data and then compare; they
  must never emit these constants as computed results.
- The pinned upstream checkout is never modified; importing upstream modules
  disables bytecode generation so the worktree stays clean.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# Locked identities (docs/STAGE0B1_BOREALTC_SOURCE_CLOSURE.md)
# --------------------------------------------------------------------------

CANONICAL_PROJECT_ROOT = "/mnt/g/CryoLocoManip"

UPSTREAM_REMOTE = "https://github.com/norlab-ulaval/BorealTC.git"
UPSTREAM_BRANCH = "python"
UPSTREAM_COMMIT = "0146dcd9fa08c34a9075a80448ac04c0a947b568"
UPSTREAM_COMMIT_SUBJECT = "preload all samples (#7)"
UPSTREAM_CODE_LICENSE = "MIT"
UPSTREAM_DATA_LICENSE = "CC0-1.0"

PAPER_TITLE = "Proprioception Is All You Need: Terrain Classification for Boreal Forests"
PAPER_DOI = "10.1109/IROS58592.2024.10801407"
PAPER_ARXIV = "2403.16877v2"
PAPER_VENUE = "IEEE/RSJ IROS 2024"

# Repository label -> paper display label. Raw data labels are never renamed.
OFFICIAL_CLASSES = ("ASPHALT", "FLOORING", "ICE", "SANDY_LOAM", "SNOW")
MIXED_CLASS = "MIXED"
PAPER_DISPLAY_NAMES = {
    "ASPHALT": "ASPHALT",
    "FLOORING": "FLOORING",
    "ICE": "ICE",
    "SANDY_LOAM": "SILTY LOAM",
    "SNOW": "SNOW",
}
DAT_CLASS_PREFIXES = {c[:3]: c for c in OFFICIAL_CLASSES}  # ASP/FLO/ICE/SAN/SNO

# Locked paper-pipeline parameters (mamba_train.py at the pinned commit).
RANDOM_STATE = 21
N_FOLDS = 5
PART_WINDOW_S = 5.0
MOVING_WINDOW_S = 1.7
STRIDE_S = 0.1
HOMOGENEOUS_AUGMENTATION = True
IMU_CHANNELS = ("wx", "wy", "wz", "ax", "ay", "az")
PRO_CHANNELS = ("velL", "velR", "curL", "curR")
UPSTREAM_COLUMNS = {
    "imu": {c: True for c in IMU_CHANNELS},
    "pro": {c: True for c in PRO_CHANNELS},
}

# Canonical committed artifacts (relative to the upstream checkout).
CANONICAL_RESULTS = {
    "CNN": "results/husky/results_CNN_hamming_mw_1.7.npy",
    "Mamba": "results/husky/results_mamba_optim2_mw_1.7.npy",
}
CANONICAL_METRIC_FILES = {
    "CNN": "metrics/husky/CNN-1700-hamming.dat",
    "Mamba": "metrics/husky/mamba-1700-optim2.dat",
}
UPSTREAM_SUMMARY_DAT = "summary/terrains.dat"
UPSTREAM_METRIC_SCRIPT = "compile_metrics.py"
DATA_SUBDIR = "data/borealtc"

# --------------------------------------------------------------------------
# Comparison configuration (expected values; tests and comparisons only)
# --------------------------------------------------------------------------

# Paper Table I: complete 5 s partitions per class.
EXPECTED_PARTITION_COUNTS = {
    "ASPHALT": 111,
    "FLOORING": 423,
    "ICE": 450,
    "SANDY_LOAM": 126,
    "SNOW": 281,
}
EXPECTED_TOTAL_PARTITIONS = 1391

# Paper Table III (percent), keyed by repository class label.
EXPECTED_TABLE3 = {
    "CNN": {
        "accuracy": 93.96,
        "per_class": {
            "ASPHALT": {"precision": 92.98, "recall": 83.89, "f1": 88.20},
            "FLOORING": {"precision": 97.29, "recall": 98.70, "f1": 97.99},
            "ICE": {"precision": 97.25, "recall": 98.11, "f1": 97.68},
            "SANDY_LOAM": {"precision": 96.00, "recall": 97.24, "f1": 96.61},
            "SNOW": {"precision": 86.84, "recall": 92.31, "f1": 89.49},
        },
    },
    "Mamba": {
        "accuracy": 93.68,
        "per_class": {
            "ASPHALT": {"precision": 91.90, "recall": 85.50, "f1": 88.59},
            "FLOORING": {"precision": 95.46, "recall": 98.17, "f1": 96.79},
            "ICE": {"precision": 97.12, "recall": 97.36, "f1": 97.24},
            "SANDY_LOAM": {"precision": 95.39, "recall": 96.20, "f1": 95.79},
            "SNOW": {"precision": 88.68, "recall": 91.57, "f1": 90.10},
        },
    },
}
METRIC_TOLERANCE_PP = 0.01  # percentage points, for two-decimal metrics

IMU_NOMINAL_HZ = 100.0
PRO_NOMINAL_HZ = 6.5


class AuditError(RuntimeError):
    """Raised on identity, schema, or semantic mismatches. Always fatal."""


# --------------------------------------------------------------------------
# Paths and identity
# --------------------------------------------------------------------------


def repo_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if str(root) != CANONICAL_PROJECT_ROOT:
        raise AuditError(
            f"tools must run from the canonical checkout {CANONICAL_PROJECT_ROOT}, "
            f"found {root}"
        )
    return root


def upstream_dir() -> Path:
    return repo_root() / "third_party" / "BorealTC"


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True, timeout=120
    )
    if proc.returncode != 0:
        raise AuditError(f"git {' '.join(args)} failed in {cwd}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def verify_upstream_identity(require_clean: bool = True) -> dict:
    """Verify the pinned upstream checkout; raise AuditError on any mismatch."""
    up = upstream_dir()
    if not up.is_dir():
        raise AuditError(f"upstream checkout missing: {up}")
    remote = _git(up, "remote", "get-url", "origin")
    if remote != UPSTREAM_REMOTE:
        raise AuditError(f"upstream remote must be {UPSTREAM_REMOTE}, found {remote}")
    head = _git(up, "rev-parse", "HEAD")
    if head != UPSTREAM_COMMIT:
        raise AuditError(f"upstream HEAD must be {UPSTREAM_COMMIT}, found {head}")
    subject = _git(up, "show", "-s", "--format=%s", "HEAD")
    if subject != UPSTREAM_COMMIT_SUBJECT:
        raise AuditError(
            f"upstream commit subject mismatch: expected "
            f"{UPSTREAM_COMMIT_SUBJECT!r}, found {subject!r}"
        )
    dirty = _git(up, "status", "--porcelain")
    if require_clean and dirty:
        raise AuditError(f"upstream worktree is not clean:\n{dirty}")
    return {
        "remote": remote,
        "commit": head,
        "commit_subject": subject,
        "commit_date": _git(up, "show", "-s", "--format=%cI", "HEAD"),
        "worktree_clean": not dirty,
        "path": str(up),
    }


def project_git_state() -> dict:
    root = repo_root()
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "short_commit": _git(root, "rev-parse", "--short", "HEAD"),
        "branch": _git(root, "branch", "--show-current"),
        "dirty": bool(_git(root, "status", "--porcelain")),
    }


def add_upstream_to_syspath() -> Path:
    """Make upstream modules importable without writing bytecode into it."""
    up = upstream_dir()
    sys.dont_write_bytecode = True
    if str(up) not in sys.path:
        sys.path.insert(0, str(up))
    return up


# --------------------------------------------------------------------------
# Small deterministic utilities (standard library only)
# --------------------------------------------------------------------------


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def within_tolerance(a: float, b: float, tol: float = METRIC_TOLERANCE_PP) -> bool:
    return abs(a - b) <= tol + 1e-9


def parse_metric_dat(text: str) -> dict[str, float]:
    """Parse upstream `key = value` metric files (e.g. metrics/husky/*.dat)."""
    out: dict[str, float] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if "=" not in line:
            raise AuditError(f".dat line {lineno} is not 'key = value': {raw!r}")
        key, _, value = line.partition("=")
        out[key.strip()] = float(value.strip())
    return out


def expected_augmentation_plan(
    partition_counts: dict[str, int],
    hf_hz: float = IMU_NOMINAL_HZ,
    part_window_s: float = PART_WINDOW_S,
    moving_window_s: float = MOVING_WINDOW_S,
    stride_s: float = STRIDE_S,
) -> dict:
    """Pure reimplementation of the homogeneous-augmentation arithmetic in
    upstream `utils/preprocessing.augment_data` (locked commit), used as an
    independent cross-check of the traced upstream execution.
    """
    pw_len = int(part_window_s * hf_hz)
    mw_len = int(moving_window_s * hf_hz)
    st_len = int(stride_s * hf_hz)
    n_strides_part = (pw_len - mw_len) // st_len
    classes = sorted(partition_counts)
    min_count = min(partition_counts.values())
    strides_min = n_strides_part * min_count
    n_slides = {c: int(strides_min / partition_counts[c]) for c in classes}
    aug_stride = {c: (pw_len - mw_len) // n_slides[c] for c in classes}
    windows = {c: partition_counts[c] * n_slides[c] for c in classes}
    max_window_end = {c: aug_stride[c] * (n_slides[c] - 1) + mw_len for c in classes}
    return {
        "pw_len": pw_len,
        "mw_len": mw_len,
        "st_len": st_len,
        "n_strides_part": n_strides_part,
        "min_count": min_count,
        "n_slides": n_slides,
        "aug_stride": aug_stride,
        "windows": windows,
        "total_windows": sum(windows.values()),
        "max_window_end": max_window_end,
        "windows_stay_within_partition": all(
            end <= pw_len for end in max_window_end.values()
        ),
    }


def infer_fold_blocks(
    true_labels: "list[int]",
    n_classes: int = len(OFFICIAL_CLASSES),
    n_folds: int = N_FOLDS,
) -> list[list[int]]:
    """Recover per-fold, per-class test-window counts from a fold-concatenated
    true-label array.

    Upstream test windows are emitted per fold as consecutive class blocks in
    ascending class order (augment_data stacks terrains alphabetically), so
    the concatenated array must run-length decode into exactly
    n_folds x n_classes ascending blocks. Raises AuditError otherwise.
    """
    blocks: list[tuple[int, int]] = []  # (label, length)
    for label in true_labels:
        label = int(label)
        if blocks and blocks[-1][0] == label:
            blocks[-1] = (label, blocks[-1][1] + 1)
        else:
            blocks.append((label, 1))
    if len(blocks) != n_folds * n_classes:
        raise AuditError(
            f"expected {n_folds * n_classes} label blocks, found {len(blocks)}"
        )
    folds: list[list[int]] = []
    for k in range(n_folds):
        fold = blocks[k * n_classes : (k + 1) * n_classes]
        if [b[0] for b in fold] != list(range(n_classes)):
            raise AuditError(
                f"fold {k + 1} block labels are not ascending 0..{n_classes - 1}: "
                f"{[b[0] for b in fold]}"
            )
        folds.append([b[1] for b in fold])
    return folds


def natural_weighted_accuracy(
    recalls_pct: dict[str, float], partition_counts: dict[str, int]
) -> float:
    """Diagnostic: aggregate accuracy reweighted to the natural (pre-balancing)
    class distribution, computed as the partition-share-weighted mean recall.
    """
    total = sum(partition_counts.values())
    return sum(
        recalls_pct[c] * partition_counts[c] / total for c in partition_counts
    )


# --------------------------------------------------------------------------
# Upstream pipeline access (heavy imports; venv only)
# --------------------------------------------------------------------------


def load_upstream_preprocessing():
    add_upstream_to_syspath()
    import utils.preprocessing as upstream_preprocessing  # noqa: PLC0415

    return upstream_preprocessing


def make_upstream_summary():
    """Reproduce mamba_train.py's summary DataFrame construction exactly."""
    import pandas as pd  # noqa: PLC0415

    return pd.DataFrame({"columns": pd.Series({k: dict(v) for k, v in UPSTREAM_COLUMNS.items()})})


def build_unified(data_dir: Path | None = None):
    """Run upstream get_recordings + partition_data with n_splits=None.

    Returns (unified, labels, summary, upstream_module). `unified` is the
    pre-split partition tensor dict; `labels` the per-partition terrain
    strings, exactly as fed to kfold_splits by the locked pipeline.
    """
    pre = load_upstream_preprocessing()
    if data_dir is None:
        data_dir = upstream_dir() / DATA_SUBDIR
    summary = make_upstream_summary()
    terr_dfs = pre.get_recordings(data_dir, summary)
    unified = pre.partition_data(
        terr_dfs, summary, int(PART_WINDOW_S), n_splits=None, random_state=RANDOM_STATE
    )
    labels = unified["imu"][:, 0, 0]
    return unified, labels, summary, pre


def split_folds(unified, labels, pre):
    """Run upstream kfold_splits with the locked RANDOM_STATE."""
    return pre.kfold_splits(unified, labels, N_FOLDS, RANDOM_STATE)


def fold_membership(fold_data) -> list[tuple[str, int, int]]:
    """Extract sorted (terrain, run_idx, win_idx) partition identities from a
    fold's imu tensor (channels: terrain, terr_idx, run_idx, win_idx, time).
    """
    imu = fold_data["imu"]
    out = [
        (str(imu[i, 0, 0]), int(imu[i, 0, 2]), int(imu[i, 0, 3]))
        for i in range(imu.shape[0])
    ]
    return sorted(out)


def membership_hash(membership: list[tuple[str, int, int]]) -> str:
    return sha256_text(json.dumps(membership, sort_keys=True))
