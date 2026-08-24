"""Create and finalize the Stage 0B.1 run manifest.

Usage (from the repository root):

    python -m tools.borealtc.manifest create --run-dir <dir> --command "<cmd>"
    python -m tools.borealtc.manifest finalize --run-dir <dir> --exit-code 0

The manifest conforms to schemas/run_manifest.schema.json; conformance of the
required top-level keys is asserted here with the standard library so the
check works without jsonschema installed.
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.borealtc import _common


def _first_line(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except OSError:
        return "unavailable"
    if proc.returncode != 0:
        return "unavailable"
    out = proc.stdout.strip().splitlines()
    return out[0] if out else "unavailable"


def _nvidia_driver() -> str:
    line = _first_line(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    return line


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
        "stage": "stage0b1",
        "method": "borealtc_source_closure",
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
            "gpu_driver": _nvidia_driver(),
            "ros": os.environ.get("ROS_DISTRO", "unavailable"),
            "simulator": "unavailable",
        },
        "inputs": {
            "dataset_identity": (
                f"BorealTC data/borealtc @ {upstream['commit']} "
                f"({_common.UPSTREAM_DATA_LICENSE})"
            ),
            "config_identity": (
                "locked mamba_train.py params: PART_WINDOW=5s, MOVING_WINDOW=1.7s, "
                "STRIDE=0.1s, N_FOLDS=5, HOMOGENEOUS_AUGMENTATION=True, "
                f"RANDOM_STATE={_common.RANDOM_STATE}"
            ),
            "evaluator_identity": (
                f"tools/borealtc @ CryoLocoManip {git['commit']} "
                "(no training, audit only)"
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
