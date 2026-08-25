#!/usr/bin/env bash
# Stage 0B.2A declared-stack probe environment (prompt §5.1).
#
# Installs the material training-path dependencies exactly as the upstream
# requirements-lock.txt pins them, letting pip resolve the torch version that
# the upstream files imply (bare `torch` constrained by torchvision==0.17.0)
# from the default index, without choosing a newer model stack. Dev/notebook
# tooling from the lock (jupyterlab, ipython, matplotlib, plotly, black,
# pylint, ipykernel, pre-commit) is deliberately not installed; it is not
# imported by the released training path and the omission is recorded as a
# deviation in the runtime semantics audit.
#
# Usage:
#   bash scripts/stage0b2a_build_declared_probe_env.sh core     # venv + core
#   bash scripts/stage0b2a_build_declared_probe_env.sh ssm <N>  # extension
#                                                              # attempt N
set -Eeuo pipefail
source "$(dirname "$0")/stage0b2a_env.sh"

VENV="$REPO/cache/venvs/borealtc-runtime-declared-probe"
PY="$VENV/bin/python"
LOGDIR="$REPO/logs/stage0b2a/envbuild"
STEP="${1:?usage: core | ssm <attempt-number>}"

if [ "$STEP" = "core" ]; then
  python3.10 -m venv --copies "$VENV"
  fix_venv_lib64 "$VENV"
  "$PY" -m pip install --upgrade pip 2>&1 | tee "$LOGDIR/declared_00_pip.log"
  # Mirror the Dockerfile install order: packaging/torch/torchvision/wheel
  # first (torch deliberately left unpinned so the resolver documents the
  # implied version), then the pinned scientific core.
  "$PY" -m pip install --no-cache-dir packaging==23.2 wheel torch \
    torchvision==0.17.0 2>&1 | tee "$LOGDIR/declared_01_torch_resolution.log"
  "$PY" -m pip install numpy==1.26.4 pandas==2.2.0 pyarrow==15.0.0 \
    scipy==1.12.0 scikit-learn==1.4.0 lightning==2.2.0 tensorboard==2.15.2 \
    einops==0.7.0 tqdm==4.66.2 2>&1 | tee "$LOGDIR/declared_02_core.log"
  # Declared but never imported by the code (the code imports torchmetrics,
  # which lightning provides); installed --no-deps for lock fidelity.
  "$PY" -m pip install --no-deps torch-metrics==1.1.7 2>&1 \
    | tee "$LOGDIR/declared_03_torch_metrics.log"
  "$PY" -m pip install "pipeline @ git+https://github.com/willGuimont/pipeline" \
    2>&1 | tee "$LOGDIR/declared_04_pipeline.log"
  "$PY" -m pip freeze 2>/dev/null > "$LOGDIR/requirements_declared_probe.freeze.txt"
  echo "DECLARED_CORE_DONE"
elif [ "$STEP" = "ssm" ]; then
  ATTEMPT="${2:?attempt number required}"
  # Default upstream behavior: setup.py first tries the prebuilt GitHub
  # release wheel matching (torch, CUDA, cp310, cxx11abi), else compiles.
  "$PY" -m pip install --no-build-isolation causal_conv1d==1.2.0.post1 2>&1 \
    | tee "$LOGDIR/declared_causal_conv1d_attempt${ATTEMPT}.log"
  "$PY" -m pip install --no-build-isolation mamba-ssm==1.2.0.post1 2>&1 \
    | tee "$LOGDIR/declared_mamba_ssm_attempt${ATTEMPT}.log"
  "$PY" -m pip freeze 2>/dev/null > "$LOGDIR/requirements_declared_probe.freeze.txt"
  echo "DECLARED_SSM_DONE attempt=$ATTEMPT"
else
  echo "unknown step: $STEP" >&2
  exit 2
fi
