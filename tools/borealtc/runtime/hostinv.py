"""Host/GPU/driver/toolchain inventory for Stage 0B.2A.

Usage:

    python -m tools.borealtc.runtime.hostinv --out <host_gpu_inventory.json>

Standard-library only; every external identity comes from a read-only command.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from tools.borealtc import _common
    from tools.borealtc.runtime import _rt
except ImportError:  # executed as a loose script
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tools.borealtc import _common
    from tools.borealtc.runtime import _rt


def _run(cmd: list[str], timeout: int = 60) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _first_line(text: str) -> str:
    lines = text.splitlines()
    return lines[0] if lines else ""


def gpu_inventory() -> dict:
    code, out, err = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ]
    )
    gpus = []
    if code == 0:
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                gpus.append(
                    {
                        "name": parts[0],
                        "driver_version": parts[1],
                        "memory_total": parts[2],
                        "compute_capability": parts[3],
                    }
                )
    code2, out2, _ = _run(["nvidia-smi"])
    cuda_level = ""
    if code2 == 0:
        m = re.search(r"CUDA Version:\s*([0-9.]+)", out2)
        cuda_level = m.group(1) if m else ""
    return {
        "gpus": gpus,
        "nvidia_smi_available": code == 0,
        "driver_cuda_compat_level": cuda_level,
        "query_error": err if code != 0 else "",
    }


def nvcc_inventory() -> dict:
    path = shutil.which("nvcc") or ""
    code, out, _ = _run(["nvcc", "--version"]) if path else (127, "", "")
    release = ""
    if code == 0:
        m = re.search(r"release ([0-9.]+), (V[0-9.]+)", out)
        release = f"{m.group(1)} ({m.group(2)})" if m else _first_line(out)
    return {"path": path, "release": release, "available": code == 0}


def docker_inventory() -> dict:
    code, ver, _ = _run(["docker", "--version"])
    info = {
        "client_version": ver if code == 0 else "unavailable",
        "server_version": "unavailable",
        "root_dir": "unavailable",
        "root_dir_filesystem": "unavailable",
        "root_dir_on_wsl_linux_vhd": None,
        "image_pull_allowed_by_policy": False,
    }
    code2, out2, err2 = _run(
        ["docker", "info", "--format", "{{.DockerRootDir}}|{{.ServerVersion}}"],
        timeout=30,
    )
    if code2 == 0 and "|" in out2:
        root_dir, server = out2.split("|", 1)
        info["root_dir"] = root_dir
        info["server_version"] = server
        code3, out3, _ = _run(["findmnt", "-T", root_dir, "-n", "-o", "SOURCE,FSTYPE"])
        if code3 == 0:
            info["root_dir_filesystem"] = out3
        on_vhd = not root_dir.startswith("/mnt/")
        info["root_dir_on_wsl_linux_vhd"] = on_vhd
        # Prompt §3: images may only be pulled/built when their storage is
        # proven to stay off the WSL Linux virtual disk.
        info["image_pull_allowed_by_policy"] = not on_vhd
    else:
        info["server_error"] = err2
    return info


def _meminfo_kib(key: str) -> int:
    text = Path("/proc/meminfo").read_text(encoding="utf-8")
    m = re.search(rf"^{key}:\s+(\d+)\s+kB", text, re.MULTILINE)
    return int(m.group(1)) if m else 0


def collect() -> dict:
    cpu_model = ""
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    gcc = _first_line(_run(["gcc", "--version"])[1])
    gxx = _first_line(_run(["g++", "--version"])[1])
    ninja = _first_line(_run(["ninja", "--version"])[1])
    py310 = _first_line(_run(["python3.10", "--version"])[1])
    return {
        "timestamp_utc": _common.utc_iso(),
        "hostname": platform.node(),
        "kernel": platform.release(),
        "os": f"{platform.system()} {platform.version()}",
        "wsl_distro": os.environ.get("WSL_DISTRO_NAME", "unknown"),
        "cpu": {"model": cpu_model, "logical_cores": os.cpu_count()},
        "ram": {
            "total_kib": _meminfo_kib("MemTotal"),
            "available_kib": _meminfo_kib("MemAvailable"),
        },
        "python": {
            "interpreter": sys.executable,
            "version": platform.python_version(),
            "python3_10": py310,
        },
        "compilers": {"gcc": gcc, "gxx": gxx, "ninja": ninja},
        "gpu": gpu_inventory(),
        "nvcc": nvcc_inventory(),
        "docker": docker_inventory(),
        "gdrive_free_bytes": _rt.free_bytes("/mnt/g"),
        "gdrive_free_gib": _rt.bytes_to_gib(_rt.free_bytes("/mnt/g")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    inv = collect()
    _common.write_json(Path(args.out), inv)
    gpus = inv["gpu"]["gpus"]
    label = gpus[0]["name"] if gpus else "no-gpu"
    cc = gpus[0]["compute_capability"] if gpus else "n/a"
    print(f"HOST_INVENTORY_WRITTEN: {args.out} gpu={label!r} compute_cap={cc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
