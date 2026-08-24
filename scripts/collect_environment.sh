#!/usr/bin/env bash
# Collect a timestamped environment inventory for reproducibility audits.
# Non-destructive: the only write is one report file below ignored
# artifacts/runtime/. Missing optional tools are reported as "unavailable"
# and are never fatal.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/artifacts/runtime"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT="${OUT_DIR}/environment_${STAMP}.txt"

mkdir -p "${OUT_DIR}"

have() { command -v "$1" >/dev/null 2>&1; }

section() { printf '\n== %s ==\n' "$1"; }

# capture <label> <command...>: full indented output, or "unavailable".
capture() {
  local label="$1"
  shift
  if have "$1"; then
    printf '%s:\n' "${label}"
    "$@" 2>&1 | sed 's/^/  /' || true
  else
    printf '%s: unavailable (%s not found)\n' "${label}" "$1"
  fi
}

# capture1 <label> <command...>: first output line only, or "unavailable".
capture1() {
  local label="$1"
  shift
  if have "$1"; then
    printf '%s: %s\n' "${label}" "$("$@" 2>&1 | head -n 1)"
  else
    printf '%s: unavailable (%s not found)\n' "${label}" "$1"
  fi
}

{
  printf 'CryoLocoManip environment inventory\n'
  printf 'timestamp_utc=%s\n' "${STAMP}"

  section "OS and kernel"
  capture1 "uname" uname -a
  if [[ -r /etc/os-release ]]; then
    printf 'os_release:\n'
    sed 's/^/  /' /etc/os-release
  else
    printf 'os_release: unavailable\n'
  fi

  section "WSL"
  printf 'WSL_DISTRO_NAME=%s\n' "${WSL_DISTRO_NAME:-unavailable}"
  if [[ -r /proc/version ]]; then
    printf 'proc_version: %s\n' "$(cat /proc/version)"
  else
    printf 'proc_version: unavailable\n'
  fi

  section "G-drive mount"
  capture "findmnt_mnt_g" findmnt -T /mnt/g
  capture "disk_usage_mnt_g" df -h /mnt/g
  capture1 "fs_type_mnt_g" stat -f -c '%T' /mnt/g

  section "Git and repository state"
  capture1 "git_version" git --version
  if have git && git -C "${REPO_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
    printf 'repo_root=%s\n' "${REPO_ROOT}"
    printf 'origin=%s\n' "$(git -C "${REPO_ROOT}" remote get-url origin 2>/dev/null || echo unavailable)"
    printf 'head=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unavailable)"
    printf 'branch=%s\n' "$(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || echo unavailable)"
    printf 'dirty_files=%s\n' "$(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null | wc -l)"
  else
    printf 'repository_state: unavailable\n'
  fi

  section "Python and Conda"
  capture1 "python3" python3 --version
  printf 'python3_path: %s\n' "$(command -v python3 || echo unavailable)"
  capture1 "pip3" pip3 --version
  capture1 "conda" conda --version

  section "Compilers and build tools"
  capture1 "gcc" gcc --version
  capture1 "g++" g++ --version
  capture1 "clang" clang --version
  capture1 "cmake" cmake --version
  capture1 "make" make --version

  section "NVIDIA GPU and CUDA"
  if have nvidia-smi; then
    printf 'nvidia_smi:\n'
    nvidia-smi 2>&1 | sed 's/^/  /' || true
  else
    printf 'nvidia_smi: unavailable\n'
  fi
  capture "nvcc" nvcc --version
  printf 'cuda_installations: %s\n' "$(ls -d /usr/local/cuda* 2>/dev/null | tr '\n' ' ' || true)"

  section "Docker"
  if have docker; then
    printf 'docker_client: %s\n' "$(docker --version 2>&1)"
    printf 'docker_server: %s\n' "$(timeout 10 docker version --format '{{.Server.Version}}' 2>/dev/null || echo unavailable)"
  else
    printf 'docker_client: unavailable\n'
  fi

  section "ROS"
  printf 'ROS_DISTRO=%s\n' "${ROS_DISTRO:-unavailable}"
  if [[ -d /opt/ros ]]; then
    printf 'ros_installations: %s\n' "$(ls /opt/ros 2>/dev/null | tr '\n' ' ')"
  else
    printf 'ros_installations: unavailable\n'
  fi
  if have ros2; then
    printf 'ros2_package_count: %s\n' "$(timeout 60 ros2 pkg list 2>/dev/null | wc -l || echo unavailable)"
  else
    printf 'ros2_package_count: unavailable (ros2 not on PATH)\n'
  fi

  section "CPU and memory"
  capture1 "nproc" nproc
  if have lscpu; then
    printf 'lscpu_summary:\n'
    lscpu 2>/dev/null | grep -E '^(Architecture|CPU\(s\)|Model name|Vendor ID|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\))' | sed 's/^/  /' || true
  else
    printf 'lscpu_summary: unavailable\n'
  fi
  capture "memory" free -h
} | tee "${REPORT}"

printf '\nENVIRONMENT_INVENTORY_OK\n'
printf 'Report: %s\n' "${REPORT}"
exit 0
