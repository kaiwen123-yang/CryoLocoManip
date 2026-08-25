#!/usr/bin/env bash
# Stage 0B.2A faithful-compatibility environment (prompt §5.2).
#
# Only the torch runtime family moves (torch 2.7.1+cu128 / torchvision
# 0.22.1+cu128) so that official wheels support the host's Blackwell sm_120
# GPU; the scientific core keeps the exact upstream pins, and mamba-ssm /
# causal_conv1d keep their exact released versions, rebuilt from the
# unmodified sdists with an additional sm_120 code target appended through
# nvcc's documented NVCC_APPEND_FLAGS environment mechanism (no source is
# patched). Every deviation is enumerated in environment_matrix.csv.
#
# Usage:
#   bash scripts/stage0b2a_build_compat_env.sh core <pipeline-git-ref>
#   bash scripts/stage0b2a_build_compat_env.sh ssm <N>   # extension attempt N
set -Eeuo pipefail
source "$(dirname "$0")/stage0b2a_env.sh"

VENV="$REPO/cache/venvs/borealtc-runtime-compat"
PY="$VENV/bin/python"
LOGDIR="$REPO/logs/stage0b2a/envbuild"
STEP="${1:?usage: core <pipeline-ref> | ssm <attempt-number>}"

if [ "$STEP" = "core" ]; then
  PIPELINE_REF="${2:?pipeline git ref (commit) required for reproducibility}"
  prepare_venv_lib64 "$VENV"
  python3.10 -m venv --copies "$VENV"
  "$PY" -m pip install --upgrade pip 2>&1 | tee "$LOGDIR/compat_00_pip.log"
  "$PY" -m pip install --no-cache-dir torch==2.7.1+cu128 \
    torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128 \
    2>&1 | tee "$LOGDIR/compat_01_torch_cu128.log"
  "$PY" -m pip install packaging==23.2 wheel numpy==1.26.4 pandas==2.2.0 \
    pyarrow==15.0.0 scipy==1.12.0 scikit-learn==1.4.0 lightning==2.2.0 \
    tensorboard==2.15.2 einops==0.7.0 tqdm==4.66.2 2>&1 \
    | tee "$LOGDIR/compat_02_core.log"
  "$PY" -m pip install \
    "pipeline @ git+https://github.com/willGuimont/pipeline@${PIPELINE_REF}" \
    2>&1 | tee "$LOGDIR/compat_03_pipeline.log"
  "$PY" -m pip freeze 2>/dev/null > "$LOGDIR/requirements_compat.freeze.txt"
  echo "COMPAT_CORE_DONE"
elif [ "$STEP" = "ssm" ]; then
  ATTEMPT="${2:?attempt number required}"
  # Released 1.2.0.post1 sdists have no prebuilt wheel for torch 2.7, so a
  # local CUDA build is forced explicitly. The hardcoded sm_70/80/90 gencode
  # list gains one additional sm_120 target via nvcc's documented
  # NVCC_APPEND_FLAGS; sources are byte-identical to the released sdists.
  export CAUSAL_CONV1D_FORCE_BUILD=TRUE
  export MAMBA_FORCE_BUILD=TRUE
  export NVCC_APPEND_FLAGS="-gencode arch=compute_120,code=sm_120"
  export MAX_JOBS=4
  "$PY" -m pip install -v --no-build-isolation causal_conv1d==1.2.0.post1 2>&1 \
    | tee "$LOGDIR/compat_causal_conv1d_attempt${ATTEMPT}.log" | tail -5
  "$PY" -m pip install -v --no-build-isolation mamba-ssm==1.2.0.post1 2>&1 \
    | tee "$LOGDIR/compat_mamba_ssm_attempt${ATTEMPT}.log" | tail -5
  "$PY" -m pip freeze 2>/dev/null > "$LOGDIR/requirements_compat.freeze.txt"
  echo "COMPAT_SSM_DONE attempt=$ATTEMPT"
else
  echo "unknown step: $STEP" >&2
  exit 2
fi
