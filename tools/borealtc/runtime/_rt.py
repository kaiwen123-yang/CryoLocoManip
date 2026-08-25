"""Shared Stage 0B.2A runtime-feasibility helpers.

Standard-library only at import time. Pure decision logic lives here so it is
unit-testable without the scientific environments.
"""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

from tools.borealtc import _common

GIB = 1024**3

# Hard resource rules (prompts/CODEX_02A_BOREALTC_RUNTIME_FEASIBILITY.md §3).
MAX_NEW_PERSISTENT_GIB = 25.0
MIN_FREE_RESERVE_GIB = 75.0

# Environment labels (docs/STAGE0B2_BOREALTC_TRAINING_AND_GENERALIZATION.md §3.2
# and prompt §5.1).
ENV_EXACT = "EXACT_DECLARED_STACK"
ENV_COMPAT = "FAITHFUL_COMPATIBILITY_STACK"
DECLARED_CLASSIFICATIONS = (
    "EXACT_DECLARED_STACK",
    "DECLARED_STACK_UNDERDETERMINED",
    "DECLARED_STACK_INSTALLABLE_BUT_GPU_UNSUPPORTED",
    "DECLARED_STACK_BUILD_BLOCKED",
    "DECLARED_STACK_RUNTIME_BLOCKED",
)

# Terminal states (prompt §1).
TERMINAL_PASS = "PASS_BOREALTC_RUNTIME_FEASIBILITY"
TERMINAL_PASS_BLOCKED_MAMBA = "PASS_WITH_BLOCKED_MAMBA_BOREALTC_RUNTIME_FEASIBILITY"

FAILED_ATTEMPTS_HEADER = [
    "attempt_id",
    "timestamp_utc",
    "phase",
    "environment_label",
    "command_or_step",
    "outcome",
    "error_class",
    "error_snippet",
    "log_path",
    "log_sha256",
]


def append_failed_attempt(csv_path: Path, row: dict) -> None:
    """Append one attempt record (failed OR succeeded-after-failure context).

    The file is created with the canonical header on first use. Every
    install/build/import/load/smoke attempt that did not succeed on the
    happy path must be recorded here; successful attempts may be recorded
    with outcome=SUCCEEDED for continuity of numbering.
    """
    missing = [k for k in FAILED_ATTEMPTS_HEADER if k not in row]
    if missing:
        raise _common.AuditError(f"failed_attempts row missing fields: {missing}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FAILED_ATTEMPTS_HEADER)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row[k] for k in FAILED_ATTEMPTS_HEADER})


def du_bytes(path: Path) -> int:
    """Recursive apparent size in bytes; 0 for a missing path."""
    if not path.exists():
        return 0
    proc = subprocess.run(
        ["du", "-sb", str(path)], capture_output=True, text=True, timeout=1800
    )
    if proc.returncode != 0:
        raise _common.AuditError(f"du failed for {path}: {proc.stderr.strip()}")
    return int(proc.stdout.split()[0])


def free_bytes(mount: str = "/mnt/g") -> int:
    st = os.statvfs(mount)
    return st.f_bavail * st.f_frsize


def bytes_to_gib(n: int) -> float:
    return round(n / GIB, 3)


def evaluate_storage(
    baseline_tracked: dict[str, int],
    now_tracked: dict[str, int],
    now_free_bytes: int,
    cap_gib: float = MAX_NEW_PERSISTENT_GIB,
    reserve_gib: float = MIN_FREE_RESERVE_GIB,
) -> dict:
    """Pure storage-gate evaluation.

    added_since_baseline counts only growth of tracked roots (missing roots
    count as 0 at baseline). cap_ok enforces the 25 GiB new-persistent-asset
    cap; reserve_ok enforces the 75 GiB end-state free reserve.
    """
    added = 0
    per_root = {}
    for root, now in sorted(now_tracked.items()):
        base = baseline_tracked.get(root, 0)
        delta = now - base
        per_root[root] = {
            "baseline_bytes": base,
            "now_bytes": now,
            "delta_bytes": delta,
        }
        added += delta
    added_gib = added / GIB
    free_gib = now_free_bytes / GIB
    return {
        "per_root": per_root,
        "added_since_baseline_bytes": added,
        "added_since_baseline_gib": round(added_gib, 3),
        "free_bytes": now_free_bytes,
        "free_gib": round(free_gib, 3),
        "cap_gib": cap_gib,
        "reserve_gib": reserve_gib,
        "cap_ok": added_gib <= cap_gib,
        "reserve_ok": free_gib >= reserve_gib,
    }


def finite(value) -> bool:
    """True only for real finite numbers (rejects None/NaN/inf/bool)."""
    if isinstance(value, bool) or value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return v == v and v not in (float("inf"), float("-inf"))


def smoke_status(measurements: dict) -> str:
    """Classify a one-batch smoke measurement dict as PASS or FAIL.

    Requirements (prompt §7.1): finite loss/grad-norm/update-norm, a strictly
    positive parameter update, exactly one optimizer step, labels inside the
    official class domain, and finite forward outputs.
    """
    required_finite = (
        "train_loss",
        "grad_global_norm",
        "param_update_norm",
        "val_loss",
    )
    for key in required_finite:
        if not finite(measurements.get(key)):
            return f"FAIL_NONFINITE_{key.upper()}"
    if measurements.get("optimizer_steps") != 1:
        return "FAIL_OPTIMIZER_STEP_COUNT"
    if measurements.get("backward_calls") != 1:
        return "FAIL_BACKWARD_CALL_COUNT"
    if float(measurements["param_update_norm"]) <= 0.0:
        return "FAIL_NO_PARAMETER_UPDATE"
    if not measurements.get("labels_in_domain", False):
        return "FAIL_LABEL_DOMAIN"
    if not measurements.get("logits_finite", False):
        return "FAIL_NONFINITE_LOGITS"
    if not measurements.get("inputs_finite", False):
        return "FAIL_NONFINITE_INPUTS"
    if not measurements.get("grads_all_finite", False):
        return "FAIL_NONFINITE_GRADIENTS"
    if not measurements.get("params_finite_after_step", False):
        return "FAIL_NONFINITE_PARAMETERS"
    return "PASS"


def labels_in_domain(labels, n_classes: int = len(_common.OFFICIAL_CLASSES)) -> bool:
    """All labels must be integral values inside 0..n_classes-1."""
    seen = list(labels)
    if not seen:
        return False
    for lab in seen:
        f = float(lab)
        if f != int(f):
            return False
        if not 0 <= int(f) < n_classes:
            return False
    return True


def terminal_state(cnn_status: str, mamba_status: str, mamba_blocker_complete: bool) -> str:
    """Stage terminal-state logic (prompt §1).

    A Mamba block may coexist with a pass only when the CNN smoke passed and
    the blocker record is complete.
    """
    if cnn_status == "PASS" and mamba_status == "PASS":
        return TERMINAL_PASS
    if cnn_status == "PASS" and mamba_status != "PASS" and mamba_blocker_complete:
        return TERMINAL_PASS_BLOCKED_MAMBA
    if cnn_status != "PASS":
        return "FAIL_BOREALTC_RUNTIME_FEASIBILITY_CNN_SMOKE"
    return "FAIL_BOREALTC_RUNTIME_FEASIBILITY_MAMBA_UNRESOLVED"
