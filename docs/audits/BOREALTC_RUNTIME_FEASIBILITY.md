# BorealTC Runtime, Checkpoint, and Training-Feasibility Closure (Stage 0B.2A)

Terminal state: **PASS_BOREALTC_RUNTIME_FEASIBILITY**

Executed against `docs/STAGE0B2_BOREALTC_TRAINING_AND_GENERALIZATION.md` (§4)
and `prompts/CODEX_02A_BOREALTC_RUNTIME_FEASIBILITY.md`. All numbers were
produced by the tracked tools in `tools/borealtc/runtime/` (with the tracked
compatibility adapter in `tools/borealtc/compat/`) from the pinned upstream
checkout; nothing in `third_party/BorealTC` was modified, and no full epoch or
fold was trained.

Authoritative formal run (clean commit `9155960`, `manifest.git.dirty=false`):
`runs/stage0b2a/borealtc_runtime_feasibility/20260825T124621Z_9155960_borealtc-runtime-feas_21/`
(SHA256 manifest included; every failed attempt is in `failed_attempts.csv`
with hashed logs under `env_build_logs/`; three earlier aborted formal-run
directories are retained beside it and recorded as failed attempts).

## 1. Host and GPU inventory

| Item | Value |
| --- | --- |
| Host | WSL2 Ubuntu 22.04 (kernel 6.6.87.2-microsoft-standard-WSL2), LAPTOP-ATOM6ID4 |
| CPU / RAM | Intel Core Ultra 9 275HX (24 logical) / 23 GiB |
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU, 16303 MiB, **compute capability 12.0 (sm_120, Blackwell)** |
| Driver | 592.01 (driver CUDA compat level 13.1) |
| Local toolkit | nvcc 12.8 (V12.8.93) at /usr/local/cuda; gcc/g++ 11.4.0; ninja 1.13.0 |
| Python | 3.10.12 |
| Docker | client/server 29.4.0, root dir `/var/lib/docker` → **inside the WSL Linux vhd**, so image pulls/builds are prohibited by the storage policy and were never used |

## 2. Source-level runtime ambiguity (runtime_source_semantics.json)

The upstream lock pins the scientific core exactly (22 `==` pins) but the
runtime stack is **not uniquely determined** by released sources:

1. `torch` is declared bare; only `torchvision==0.17.0` constrains it. pip
   resolution (recorded live) selects **torch 2.2.0 with cu121-series nvidia
   wheels from default PyPI**; no index/backend is declared anywhere.
2. `git+https://github.com/willGuimont/pipeline` is unpinned (resolved at
   install time to `4b2dfed7…`; imported by models/datamodule/dataset).
3. The lock pins `torch-metrics==1.1.7` — an unrelated PyPI package; the
   `torchmetrics` actually imported by `utils/models.py` arrives only as an
   unpinned transitive dependency of lightning (resolved 1.9.0).
4. `transformers` (transitive via mamba-ssm) is unpinned; 2026 resolution
   installs a major version that removed `GreedySearchDecoderOnlyOutput` and
   requires torch>=2.5, breaking `import mamba_ssm` under the declared stack.
   An era pin `transformers==4.38.2` (2024-03) restores importability.
5. Dockerfile: `nvidia/cuda:12.2.0-devel-ubuntu22.04`, `TORCH_CUDA_ARCH_LIST`
   **ends at 8.6+PTX** (Ampere); README references a `DockerfileGPU` that does
   not exist at the locked commit and installs the un-locked
   `requirements.txt`.
6. `utils/models.py` imports `mamba_ssm` at module top level, so **the CNN
   path cannot even be imported without an installed mamba-ssm/causal_conv1d
   pair**.
7. The PyPI sdists of `mamba-ssm`/`causal-conv1d` 1.2.0.post1 **omit `csrc/`
   entirely** (hashes and listings in
   `env_build_logs/pypi_sdist_missing_csrc_evidence.txt`): real CUDA builds
   from PyPI are structurally impossible; only the prebuilt CI wheels
   (cu118/cu122 × torch≤2.3) or the GitHub release tags provide the kernels.

Because of 1–4 the declared environment cannot be labeled
`EXACT_DECLARED_STACK` on source evidence alone; the checkpoint audit (§4)
independently confirms Lightning 2.2.0.post0 as the training-era version.

## 3. Environment matrix (environment_matrix.csv)

### 3.1 Declared-stack probe — `DECLARED_STACK_INSTALLABLE_BUT_GPU_UNSUPPORTED`

`cache/venvs/borealtc-runtime-declared-probe` (freeze retained in the run):
Python 3.10.12, torch 2.2.0+cu121, torchvision 0.17.0, lightning 2.2.0, the
exact scientific-core pins, SSM extensions = upstream prebuilt
`cu122torch2.2cxx11abiFALSE` CI wheels (GCC 9.4/Ubuntu 20.04 build
fingerprint; embedded SASS sm_70/80/90 only, no PTX — verified with
cuobjdump).

- Installable: yes (with the recorded `transformers==4.38.2` era pin).
- GPU: **unusable** — torch itself warns sm_120 is unsupported (wheel arch
  list ends at sm_90) and every kernel launch fails with `no kernel image is
  available for execution on the device` (matmul and Mamba-block probes).
- CPU: the released Mamba path hard-requires CUDA
  (`Expected x.is_cuda() to be true`), so CPU-only execution is impossible
  without modifying code — recorded, not attempted.
- The venv was removed at the end of the formal run after all its audits
  completed (freeze, build logs, and attempt records retained; rebuildable
  by `scripts/stage0b2a_build_declared_probe_env.sh`).

### 3.2 Faithful compatibility stack — `FAITHFUL_COMPATIBILITY_STACK`

`cache/venvs/borealtc-runtime-compat` (freeze in the run): torch
**2.7.1+cu128** / torchvision 0.22.1+cu128 (first stable line with sm_120
wheels); **exact upstream pins preserved** for numpy 1.26.4, pandas 2.2.0,
pyarrow 15.0.0, scipy 1.12.0, scikit-learn 1.4.0, einops 0.7.0, tqdm 4.66.2,
tensorboard 2.15.2 and **lightning 2.2.0**; pipeline pinned to the same
`4b2dfed7…` commit; mamba-ssm and causal_conv1d **1.2.0.post1 rebuilt
unmodified** from the GitHub release tags `v1.2.0.post1`
(causal-conv1d `02e84be`, mamba `34076d6`) with
`NVCC_APPEND_FLAGS="-gencode arch=compute_120,code=sm_120"` (documented nvcc
env mechanism; embedded SASS verified {70,80,90,120}, local GCC 11.4
fingerprint).

Enumerated deviations (all in `environment_matrix.csv`):

| # | Deviation | Why | Semantics impact |
| --- | --- | --- | --- |
| 1 | torch 2.2.0→2.7.1+cu128, torchvision 0.17.0→0.22.1+cu128 | only official wheel line supporting sm_120 | runtime packages only; precision-32 training semantics unchanged |
| 2 | SSM extensions rebuilt from release tags, sm_120 gencode appended via `NVCC_APPEND_FLAGS` | PyPI sdists lack `csrc/`; CI wheels lack sm_120 | identical released sources, one added GPU code target |
| 3 | `transformers==4.38.2` era pin | unpinned transitive dep resolves to an incompatible 2026 major | import glue only (never used by the terrain models) |
| 4 | Tracked adapter drops the removed print-only `verbose` kwarg of `ReduceLROnPlateau` (`tools/borealtc/compat/torch_api.py`, feature-detected) | torch 2.7 removed the kwarg; the released call site raises TypeError | zero numerical effect; released mode/factor/patience unchanged |
| 5 | Smoke loaders run `num_workers=0` (released Mamba loader: 8/persistent) | fork-based DataLoader workers deadlock on this WSL2 host, reproduced even from a CUDA-clean parent (twice; evidence retained) | I/O worker placement only; split, seeded sampler order, and batch composition unchanged |

## 4. Released Mamba checkpoint audit (checkpoint_inventory.json)

`third_party/BorealTC/checkpoints/mamba_borealtc.ckpt`
(328078 B, sha256 `7f1374a25554d952…`, git blob `a4d35d19…`):

- Lightning **2.2.0.post0** training checkpoint: epoch 27, global_step 22652,
  optimizer/lr-scheduler/loops/callback states present.
- `weights_only=True` loading fails in both environments (Lightning
  AttributeDict global; exact errors in `checkpoint_load_attempts.csv`); the
  full `torch.load` was applied only to this pinned, hash-recorded artifact.
- Stored hyperparameters equal the released `mamba_train.py` configuration on
  **every** compared field; strict `load_state_dict` into the released
  `MambaTerrain` reconstructs with **0 missing / 0 unexpected keys**;
  24229 parameters in 28 tensors, equal to the constructed model.
- **Fold identity resolved from internal evidence**: the embedded
  ModelCheckpoint state's `best_model_path` is
  `…/terrain_classification_mamba_mw_1.7_fold_2_dataset_husky-epoch=19-val_loss=0.006191.ckpt`,
  so the artifact is the **fold-2 (husky, mw=1.7) run's early-stopped final
  state** (best epoch 19 + patience 8 = epoch 27; the run executed inside a
  `Vulpi2021-terrain-deep-learning` checkout on the authors' host).
- **Table III equivalence: NOT demonstrated** — Table III concatenates five
  per-fold test predictions; the release ships exactly one BorealTC Mamba
  checkpoint (one fold's model), so it cannot regenerate the published
  aggregate and is treated as a single-fold reference artifact.
- GPU forward on the faithful stack with the real fold-1 test batch (16
  samples): **FORWARD_OK**, logits shape (16, 5), all finite.

## 5. Real-data one-batch smokes (fold 1, official pipeline, GPU)

Both smokes run the unmodified upstream chain (`get_recordings →
partition_data(5 s, 5 folds, seed 21) → augment_data(1.7 s, 0.1 s,
homogeneous) → cleanup/normalize`; CNN additionally
`apply_multichannel_spectogram` over all five folds with the released
cross-fold global min/max) and the released model classes, loss, optimizer,
and gradient-clip configuration; exactly one training batch (forward,
backward, optimizer step) and one validation forward each. `L.seed_everything(21)`
is captured in both runs. A one-batch smoke establishes runtime feasibility
only — its loss values are not a reproduction result.

| Measurement | CNN | Mamba |
| --- | --- | --- |
| Status | **PASS** | **PASS** |
| Input batch | (10, 10, 21, 7) float32 spectrograms | imu (16, 170, 6) + pro (16, 11, 4) float32 |
| Label domain | int64 ∈ {0..4} | int64 ∈ {0..4} |
| Model parameters | 26 631 | 24 229 |
| Loss (train / val) | 1.5931 / 1.5865 (CE; untrained ≈ ln 5) | 0.0542 / 0.0518 (released sigmoid focal, α .75 γ 2.25) |
| Grad global norm | 10.39 (then released clip 6.0) | 0.0177 (clip None) |
| Param update norm | 0.742 (finite, > 0) | 0.213 (finite, > 0) |
| Optimizer / scheduler | AdamW / ReduceLROnPlateau (released) | AdamW / ReduceLROnPlateau mode=max (released) |
| Forward / backward / step (first call) | 0.76 / 0.14 / 0.24 s | 0.38 / 0.24 / 0.14 s |
| Peak VRAM (alloc) | 18.3 MiB | 34.0 MiB |
| Peak CPU RSS | 4.24 GiB | 2.97 GiB |
| Preprocessing (warm) | 23.0 s total, of which all-fold spectrograms 15.3 s (cache 1.06 GB in RAM) | 6.6 s |
| Upstream worktree after | clean | clean |

Fold-1 dataset sizes under the released 10 % validation split: 12 904 train /
1 433 val / 3 617 test windows (matches the Stage 0B.1 reconstruction).

## 6. Projected Stage 0B.2B budget (stage0b2b_budget.json)

Measured one-batch step times include first-call CUDA kernel loading, so the
projections are upper-leaning; steps/epoch: CNN 1 291 (batch 10), Mamba 807
(batch 16). Scenario epochs are declared assumptions (conservative = released
max_epochs); "3 seeds" = canonical seed 21 + two stability seeds; restart
overhead factor 1.15.

| Model | Scenario (epochs/fold) | 5-fold canonical | 5-fold × 3 seeds | + restart overhead |
| --- | --- | ---: | ---: | ---: |
| CNN | lower (25) | 48.7 h | 146.1 h | 168.0 h |
| CNN | central (60) | 116.8 h | 350.5 h | 403.1 h |
| CNN | conservative (150) | 292.1 h | 876.3 h | 1 007.8 h |
| Mamba | lower (15) | 12.6 h | 37.8 h | 43.5 h |
| Mamba | central (35) | 29.4 h | 88.2 h | 101.4 h |
| Mamba | conservative (60) | 50.4 h | 151.1 h | 173.8 h |

The released mamba_borealtc training run early-stopped at epoch 27 of 60,
which sits between the lower and central Mamba scenarios. Storage projection
for 0B.2B: ≈ 2.46 GiB (checkpoints + logs + measured 1.06 GB spectrogram
cache), comfortably inside the cap. Peak measured memory: ≤ 4.3 GiB RSS,
≤ 34 MiB VRAM at smoke batch sizes. GPU-measured timings are not
extrapolated to CPU. These wall-clock bounds are wide; a 0B.2B pre-run
steady-state step measurement should tighten them before committing to the
full matrix.

## 7. Failed attempts (failed_attempts.csv — 16 rows, hashed logs)

1. venv creation EPERM on the drvfs `lib64` symlink → pre-create `lib64`.
2. Declared `import mamba_ssm` broken by 2026 `transformers` resolution →
   era pin 4.38.2.
3. Declared GPU matmul and Mamba forward: `no kernel image` on sm_120
   (classification evidence, expected).
4. Declared Mamba CPU forward: hard CUDA requirement (expected).
5. Compat SSM attempt 1: pip's shared built-wheel cache silently installed
   the torch-2.2-ABI binaries (undefined `c10::cuda::SetDevice` symbol) →
   `--no-cache-dir`.
6. Compat causal_conv1d build from the PyPI sdist: `csrc/` missing → GitHub
   release tags.
7. Mamba smoke hang #1 (~9 h) and #2 (~7 min, CUDA-clean parent): WSL2
   fork-worker deadlock → smoke loaders `num_workers=0`; released
   `num_workers=8` recorded as a 0B.2B host risk.
8. Mamba smoke TypeError: removed `ReduceLROnPlateau(verbose=)` kwarg →
   tracked feature-detected adapter.
9. Formal-run attempts 1–3 aborted by harness issues (pytest capture temp
   files on drvfs → `-s`; an over-eager mid-run storage gate → non-gating;
   an env-log copy glob missing `.json`); all three directories retained.

## 8. Storage (storage_budget.json)

Baseline free 103.03 GiB (before any installation); cap 25 GiB new
persistent assets; reserve ≥ 75 GiB at completion. Peak tracked additions
reached 13.17 GiB (both environments); after the documented end-of-run
reclamation (declared-probe venv removal + pip cache purge) the final state
is **+6.89 GiB tracked additions and 85.54 GiB free — WITHIN_BUDGET**, with
every invariant root (upstream checkout, both Stage 0B.1 venvs, Stage 0B.1
runs) byte-identical to baseline in every event. Docker was never used.

## 9. Stage 0B.2B execution risks recorded for planning

1. Upstream `mamba_network` hardcodes `num_workers=8` /
   `persistent_workers=True`; fork-based workers deadlock on this WSL2 host,
   so 0B.2B needs a loader-worker adaptation (API-glue level) or a
   native-Linux host.
2. Upstream `configure_optimizers` needs the tracked `verbose` adapter under
   torch 2.7.1 (already in `tools/borealtc/compat/`).
3. Lightning Trainer writes `tb_logs/` and `checkpoints/` relative to the
   working directory; 0B.2B must run with CWD outside `third_party/BorealTC`
   to keep the upstream worktree clean (this stage never invoked the
   Trainer).
4. `mamba_train.py` writes `results/` under CWD (same containment
   requirement).

## 10. Evidence boundary

This pass establishes only that the released CNN and Mamba code paths can be
loaded, executed forward/backward on real official-pipeline data, and
prepared for Stage 0B.2B training reproduction on this host under the
enumerated FAITHFUL_COMPATIBILITY_STACK deviations. It does **not**
establish: a regenerated training result; group-disjoint terrain
generalization; confidence calibration; quadruped transfer; foot–snow
interaction; action-conditioned outcomes; support–manipulation coupling;
whole-body-control improvement; VLA necessity; or thesis novelty.
