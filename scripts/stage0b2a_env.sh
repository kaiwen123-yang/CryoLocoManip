#!/usr/bin/env bash
# Shared Stage 0B.2A environment redirections. Source this before any pip
# install, build, or Python execution so that every cache, temp file, and
# build artifact stays below /mnt/g/CryoLocoManip (prompt §2/§3).

export REPO=/mnt/g/CryoLocoManip
export PIP_CACHE_DIR="$REPO/cache/pip"
export XDG_CACHE_HOME="$REPO/cache/xdg"
export TMPDIR="$REPO/cache/tmp"
export TORCH_EXTENSIONS_DIR="$REPO/cache/torch_extensions"
export PYTHONDONTWRITEBYTECODE=1
# The host shell leaks ROS Humble onto PYTHONPATH; clear it for venv work.
export PYTHONPATH=

mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR" "$TORCH_EXTENSIONS_DIR" \
  "$REPO/logs/stage0b2a/envbuild"

# drvfs refuses the lib64 -> lib symlink that the venv module creates on
# Linux (EPERM aborts `python -m venv`). Pre-creating lib64 as a real
# directory makes venv skip the symlink (it only links when the path is
# absent) — the documented drvfs workaround. Call BEFORE `python -m venv`.
prepare_venv_lib64() {
  local venv="$1"
  if [ -L "$venv/lib64" ]; then
    rm "$venv/lib64"
  fi
  mkdir -p "$venv/lib64"
}
