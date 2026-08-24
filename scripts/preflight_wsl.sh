#!/usr/bin/env bash
set -Eeuo pipefail

EXPECTED_ROOT="/mnt/g/CryoLocoManip"
REMOTE_SSH="git@github.com:kaiwen123-yang/CryoLocoManip.git"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

fail() {
  printf 'PRECHECK_FAIL: %s\n' "$*" >&2
  exit 1
}

command -v git >/dev/null 2>&1 || fail "git is not installed"
[[ -d /mnt/g ]] || fail "/mnt/g is not mounted"
[[ -w /mnt/g ]] || fail "/mnt/g is not writable"

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || fail "run this script from inside the cloned repository"
ROOT_REAL="$(realpath "$ROOT")"
EXPECTED_REAL="$(realpath -m "$EXPECTED_ROOT")"
[[ "$ROOT_REAL" == "$EXPECTED_REAL" ]] || fail "canonical checkout must be $EXPECTED_REAL, found $ROOT_REAL"
[[ -w "$ROOT_REAL" ]] || fail "repository is not writable"

ORIGIN="$(git remote get-url origin 2>/dev/null || true)"
[[ "$ORIGIN" == "$REMOTE_SSH" ]] || fail "origin must be $REMOTE_SSH, found ${ORIGIN:-<missing>}"

FS_INFO="$(findmnt -T "$ROOT_REAL" -n -o SOURCE,FSTYPE,TARGET 2>/dev/null || true)"
[[ -n "$FS_INFO" ]] || fail "cannot resolve filesystem information for $ROOT_REAL"

mkdir -p \
  "$ROOT_REAL/data" \
  "$ROOT_REAL/third_party" \
  "$ROOT_REAL/simulators" \
  "$ROOT_REAL/checkpoints" \
  "$ROOT_REAL/cache" \
  "$ROOT_REAL/runs" \
  "$ROOT_REAL/outputs" \
  "$ROOT_REAL/configs/local" \
  "$ROOT_REAL/artifacts/runtime" \
  "$ROOT_REAL/logs"

LOCAL_PATHS="$ROOT_REAL/configs/local/paths.local.yaml"
if [[ ! -f "$LOCAL_PATHS" ]]; then
  cp "$ROOT_REAL/configs/paths.local.example.yaml" "$LOCAL_PATHS"
fi

REPORT="$ROOT_REAL/artifacts/runtime/preflight_${STAMP}.txt"
{
  echo "status=PASS"
  echo "timestamp_utc=$STAMP"
  echo "project_root=$ROOT_REAL"
  echo "origin=$ORIGIN"
  echo "filesystem=$FS_INFO"
  echo "kernel=$(uname -srmo)"
  echo "wsl_distro=${WSL_DISTRO_NAME:-unknown}"
  echo "git=$(git --version)"
  echo "head=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "dirty_count=$(git status --porcelain | wc -l)"
  echo "python=$(python3 --version 2>&1 || true)"
  echo "docker=$(docker --version 2>&1 || true)"
  echo "nvidia_smi=$(command -v nvidia-smi || true)"
  echo "nvcc=$(command -v nvcc || true)"
  echo "ros_distro=${ROS_DISTRO:-unset}"
  echo "disk=$(df -h "$ROOT_REAL" | tail -n 1)"
} | tee "$REPORT"

DUPLICATES="$(find "${HOME:-/home}" -maxdepth 4 -type d -name CryoLocoManip -print 2>/dev/null | grep -v "^$ROOT_REAL$" || true)"
if [[ -n "$DUPLICATES" ]]; then
  printf 'PRECHECK_WARNING: possible non-canonical checkout(s):\n%s\n' "$DUPLICATES" >&2
fi

printf 'PRECHECK_PASS: canonical workspace is %s\n' "$ROOT_REAL"
printf 'Local report: %s\n' "$REPORT"
