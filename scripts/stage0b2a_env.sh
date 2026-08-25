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

# drvfs cannot hold the lib64 -> lib symlink that venv creates on Linux;
# replace a broken symlink with a real directory (documented workaround).
fix_venv_lib64() {
  local venv="$1"
  if [ -L "$venv/lib64" ] && [ ! -e "$venv/lib64" ]; then
    rm "$venv/lib64"
  fi
  [ -e "$venv/lib64" ] || mkdir -p "$venv/lib64"
}
