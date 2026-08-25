#!/usr/bin/env bash
# Source this file before any CryoLocoManip environment build or formal run.
# It redirects project caches and enforces that pip runs only inside an
# isolated environment. It never edits shell startup files or global config.

set -o nounset

export CRYO_ROOT="/mnt/g/CryoLocoManip"

if [ ! -d "$CRYO_ROOT" ]; then
  echo "CRYO_ENV_ERROR: missing canonical root $CRYO_ROOT" >&2
  return 2 2>/dev/null || exit 2
fi

export PIP_CACHE_DIR="$CRYO_ROOT/cache/pip"
export XDG_CACHE_HOME="$CRYO_ROOT/cache/xdg"
export TMPDIR="$CRYO_ROOT/cache/tmp"
export TORCH_EXTENSIONS_DIR="$CRYO_ROOT/cache/torch_extensions"
export CUDA_CACHE_PATH="$CRYO_ROOT/cache/cuda"
export TRITON_CACHE_DIR="$CRYO_ROOT/cache/triton"
export HF_HOME="$CRYO_ROOT/cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export CONDA_ENVS_PATH="$CRYO_ROOT/cache/conda/envs"
export CONDA_PKGS_DIRS="$CRYO_ROOT/cache/conda/pkgs"
export PYTHONNOUSERSITE=1
export PIP_REQUIRE_VIRTUALENV=1
export PIP_USER=0
export PYTHONDONTWRITEBYTECODE=1

mkdir -p \
  "$PIP_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$TMPDIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$CUDA_CACHE_PATH" \
  "$TRITON_CACHE_DIR" \
  "$HF_HOME" \
  "$TRANSFORMERS_CACHE" \
  "$CONDA_ENVS_PATH" \
  "$CONDA_PKGS_DIRS"

cryo_assert_python_isolated() {
  local py="${1:-python}"
  local prefix
  local base_prefix
  local user_site

  if ! command -v "$py" >/dev/null 2>&1; then
    echo "CRYO_ENV_ERROR: Python command not found: $py" >&2
    return 3
  fi

  prefix="$($py -c 'import sys; print(sys.prefix)')"
  base_prefix="$($py -c 'import sys; print(sys.base_prefix)')"
  user_site="$($py -c 'import site; print(site.ENABLE_USER_SITE)')"

  if [ "$prefix" = "$base_prefix" ]; then
    echo "CRYO_ENV_ERROR: $py is not inside venv/Conda; prefix=$prefix" >&2
    return 4
  fi

  case "$prefix" in
    "$CRYO_ROOT"/cache/venvs/*|"$CRYO_ROOT"/cache/conda/envs/*)
      ;;
    *)
      echo "CRYO_ENV_ERROR: active environment is outside project root: $prefix" >&2
      return 5
      ;;
  esac

  if [ "$user_site" != "False" ]; then
    echo "CRYO_ENV_ERROR: user site-packages are enabled in formal environment" >&2
    return 6
  fi

  echo "CRYO_ENV_OK prefix=$prefix user_site=$user_site"
}

cryo_print_environment_routes() {
  cat <<EOF
CRYO_ROOT=$CRYO_ROOT
PIP_CACHE_DIR=$PIP_CACHE_DIR
XDG_CACHE_HOME=$XDG_CACHE_HOME
TMPDIR=$TMPDIR
TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR
CUDA_CACHE_PATH=$CUDA_CACHE_PATH
TRITON_CACHE_DIR=$TRITON_CACHE_DIR
HF_HOME=$HF_HOME
CONDA_ENVS_PATH=$CONDA_ENVS_PATH
CONDA_PKGS_DIRS=$CONDA_PKGS_DIRS
PYTHONNOUSERSITE=$PYTHONNOUSERSITE
PIP_REQUIRE_VIRTUALENV=$PIP_REQUIRE_VIRTUALENV
EOF
}
