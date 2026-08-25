"""Assemble the mamba_blocker.json record required by prompt §7.2.

Usage:

    python -m tools.borealtc.runtime.mamba_blocker --out <mamba_blocker.json> \
        --env-label <label> --env-python <path> --command "<exact command>" \
        --root-cause "<first root-cause error>" \
        --dependency-chain "<relevant dependency chain>" \
        --workaround-analysis "<why a workaround would (not) preserve semantics>" \
        --cpu-analysis "<CPU-only possibility and practicality for 0B.2B>" \
        --substatus <recommended terminal substatus> \
        [--log <path>]...

Every referenced log is hashed and must exist. `is_blocker_complete` is the
pure completeness check used by the terminal-state logic.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from tools.borealtc import _common
except ImportError:  # executed as a loose script
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tools.borealtc import _common

REQUIRED_FIELDS = (
    "failed_environment_label",
    "exact_command",
    "compiler_identity",
    "cuda_identity",
    "torch_identity",
    "gpu_identity",
    "retained_logs",
    "first_root_cause_error",
    "dependency_chain",
    "workaround_semantics_analysis",
    "cpu_only_analysis",
    "recommended_terminal_substatus",
)


def is_blocker_complete(blocker: dict) -> bool:
    for field in REQUIRED_FIELDS:
        value = blocker.get(field)
        if value is None or value == "" or value == []:
            return False
    return all(
        entry.get("sha256") and entry.get("path") for entry in blocker["retained_logs"]
    )


def _env_query(python: str, code: str) -> str:
    try:
        proc = subprocess.run(
            [python, "-c", code], capture_output=True, text=True, timeout=120
        )
    except OSError as exc:
        return f"unavailable ({exc})"
    out = proc.stdout.strip()
    return out if proc.returncode == 0 and out else f"unavailable ({proc.stderr.strip()[:200]})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--env-label", required=True)
    parser.add_argument("--env-python", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--root-cause", required=True)
    parser.add_argument("--dependency-chain", required=True)
    parser.add_argument("--workaround-analysis", required=True)
    parser.add_argument("--cpu-analysis", required=True)
    parser.add_argument("--substatus", required=True)
    parser.add_argument("--log", action="append", default=[])
    args = parser.parse_args()

    logs = []
    for log in args.log:
        path = Path(log)
        if not path.is_file():
            raise _common.AuditError(f"referenced log missing: {path}")
        logs.append(
            {
                "path": str(path),
                "sha256": _common.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()

    blocker = {
        "timestamp_utc": _common.utc_iso(),
        "failed_environment_label": args.env_label,
        "exact_command": args.command,
        "compiler_identity": subprocess.run(
            ["gcc", "--version"], capture_output=True, text=True, timeout=30
        ).stdout.splitlines()[0],
        "cuda_identity": subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=30
        ).stdout.strip().splitlines()[-1],
        "torch_identity": _env_query(
            args.env_python,
            "import torch; print(torch.__version__, torch.version.cuda)",
        ),
        "gpu_identity": gpu,
        "retained_logs": logs,
        "first_root_cause_error": args.root_cause,
        "dependency_chain": args.dependency_chain,
        "workaround_semantics_analysis": args.workaround_analysis,
        "cpu_only_analysis": args.cpu_analysis,
        "recommended_terminal_substatus": args.substatus,
    }
    blocker["complete"] = is_blocker_complete(blocker)
    _common.write_json(Path(args.out), blocker)
    print(f"MAMBA_BLOCKER_WRITTEN: {args.out} complete={blocker['complete']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
