"""Create/finalize the Stage 0B.2A run manifest.

Usage (from the repository root):

    python -m tools.borealtc.runtime.manifest0b2a create --run-dir <dir> \
        --command "<cmd>"
    python -m tools.borealtc.runtime.manifest0b2a finalize --run-dir <dir> \
        --exit-code 0

Same schema contract as tools/borealtc/manifest.py (Stage 0B.1), with the
stage/method identity of the runtime-feasibility gate.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

try:
    from tools.borealtc import _common
except ImportError:  # executed as a loose script
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tools.borealtc import _common

STAGE = "stage0b2a"
METHOD = "borealtc_runtime_feasibility"


def _first_line(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except OSError:
        return "unavailable"
    if proc.returncode != 0:
        return "unavailable"
    out = proc.stdout.strip().splitlines()
    return out[0] if out else "unavailable"


def _nvcc_release() -> str:
    try:
        proc = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=30
        )
    except OSError:
        return "unavailable"
    for line in proc.stdout.splitlines():
        if "release" in line:
            return line.strip()
    return "unavailable"


def build_manifest(run_dir: Path, command: str) -> dict:
    git = _common.project_git_state()
    upstream = _common.verify_upstream_identity(require_clean=False)
    return {
        "stage": STAGE,
        "method": METHOD,
        "run_id": run_dir.name,
        "git": {
            "commit": git["commit"],
            "dirty": git["dirty"],
            "branch": git["branch"],
        },
        "command": command,
        "working_directory": str(_common.repo_root()),
        "machine": {"hostname_or_id": socket.gethostname()},
        "versions": {
            "os": f"{platform.system()} {platform.release()}",
            "kernel": platform.version(),
            "python": platform.python_version(),
            "compiler": _first_line(["gcc", "--version"]),
            "cuda": _nvcc_release(),
            "gpu_driver": _first_line(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
            ),
            "ros": os.environ.get("ROS_DISTRO", "unavailable"),
            "simulator": "unavailable",
        },
        "inputs": {
            "dataset_identity": (
                f"BorealTC data/borealtc @ {upstream['commit']} "
                f"({_common.UPSTREAM_DATA_LICENSE})"
            ),
            "config_identity": (
                "released mamba_train.py/cnn_train.py configurations at the "
                "pinned commit; one-batch smokes only, no epoch/fold training"
            ),
            "evaluator_identity": (
                f"tools/borealtc/runtime @ CryoLocoManip {git['commit']} "
                "(runtime/checkpoint/feasibility audit only)"
            ),
        },
        "seeds": [_common.RANDOM_STATE],
        "started_at_utc": _common.utc_iso(),
        "ended_at_utc": None,
        "exit_code": None,
        "exclusions": [],
        "parent_run_id": None,
    }


def _assert_schema_conformance(manifest: dict) -> None:
    schema_path = _common.repo_root() / "schemas" / "run_manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    missing = [k for k in schema.get("required", []) if k not in manifest]
    if missing:
        raise _common.AuditError(f"manifest missing required fields: {missing}")
    for key in ("git", "machine", "versions", "inputs"):
        sub_required = schema["properties"][key].get("required", [])
        sub_missing = [k for k in sub_required if k not in manifest[key]]
        if sub_missing:
            raise _common.AuditError(f"manifest.{key} missing fields: {sub_missing}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p_create = sub.add_parser("create")
    p_create.add_argument("--run-dir", required=True)
    p_create.add_argument("--command", required=True)
    p_final = sub.add_parser("finalize")
    p_final.add_argument("--run-dir", required=True)
    p_final.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "manifest.json"
    if args.action == "create":
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = build_manifest(run_dir, args.command)
        _assert_schema_conformance(manifest)
        _common.write_json(manifest_path, manifest)
        print(f"MANIFEST_CREATED: {manifest_path}")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["ended_at_utc"] = _common.utc_iso()
        manifest["exit_code"] = args.exit_code
        _assert_schema_conformance(manifest)
        _common.write_json(manifest_path, manifest)
        print(f"MANIFEST_FINALIZED: {manifest_path} exit_code={args.exit_code}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
