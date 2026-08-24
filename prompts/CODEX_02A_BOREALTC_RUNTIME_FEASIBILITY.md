# CODEX_02A — BorealTC Runtime, Checkpoint, and Training-Feasibility Closure

## Role

You are the execution agent for Stage 0B.2A of CryoLocoManip. Your task is to determine, with auditable evidence, whether the released BorealTC CNN and Mamba code paths can be prepared for formal training reproduction on the current WSL/GPU host.

You are **not** being asked to run full five-fold training, improve BorealTC, design a new model, or validate the CryoLocoManip thesis.

Read and treat as authoritative, in this order:

1. `docs/STAGE0B2_BOREALTC_TRAINING_AND_GENERALIZATION.md`
2. `docs/REPRODUCIBILITY_CONTRACT.md`
3. `docs/audits/BOREALTC_SOURCE_CLOSURE.md`
4. `docs/STAGE0B1_BOREALTC_SOURCE_CLOSURE.md`
5. this prompt

Do not invent a compatibility policy merely to make the stage pass.

---

## 1. Terminal states

Return exactly one:

- `PASS_BOREALTC_RUNTIME_FEASIBILITY`
- `PASS_WITH_BLOCKED_MAMBA_BOREALTC_RUNTIME_FEASIBILITY`
- `BLOCKED_BOREALTC_RUNTIME_FEASIBILITY_<REASON>`
- `FAIL_BOREALTC_RUNTIME_FEASIBILITY_<REASON>`

A Mamba block may coexist with a pass only when CNN is faithfully executable, Mamba failure is supported by retained build/runtime evidence, and no substitute architecture is presented as Mamba.

---

## 2. Canonical storage and Git rules

Canonical project:

```text
/mnt/g/CryoLocoManip
```

Remote:

```text
git@github.com:kaiwen123-yang/CryoLocoManip.git
```

All project environments, pip/Conda caches, source builds, temporary files, logs, and run outputs must remain below `/mnt/g/CryoLocoManip`.

Do not create or retain project assets under `/home`, `~`, `/root`, `/opt`, `/tmp`, or another WSL Linux-filesystem path. System tools may execute from their installed locations; project artifacts may not be stored there.

Start from the merged `main`:

```bash
cd /mnt/g/CryoLocoManip
git fetch origin --prune
git switch main
git pull --ff-only origin main
git switch -c stage0/borealtc-runtime-feasibility
```

Record start HEAD. Do not create another worktree or clone of CryoLocoManip.

Pinned upstream remains:

```text
/mnt/g/CryoLocoManip/third_party/BorealTC
0146dcd9fa08c34a9075a80448ac04c0a947b568
```

It must remain detached and clean. Never edit it, install files into it, or allow Python bytecode/log/checkpoint writes inside it.

Do not merge, tag, force-push, or rewrite history.

---

## 3. Resource and storage gate

Before any package installation:

1. run `scripts/preflight_wsl.sh` and `scripts/verify_workspace.py`;
2. inventory `/mnt/g` free space;
3. inventory the size of existing BorealTC environments, caches, upstream checkout, and Stage 0B.1 runs;
4. record GPU, driver, compute capability, CPU, RAM, Python, compiler, `nvcc`, Docker client/server, and Docker root directory;
5. redirect `TMPDIR`, `XDG_CACHE_HOME`, `PIP_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`, and any build cache below `/mnt/g/CryoLocoManip/cache/`.

Hard resource rules:

- Stage 0B.2A may add at most 25 GiB of persistent assets without returning `BLOCKED_BOREALTC_RUNTIME_FEASIBILITY_STORAGE`.
- At completion, `/mnt/g` should retain at least 75 GiB free. If that cannot be met without deleting prior evidence, stop and report a storage block.
- Do not delete Stage 0B.1 authoritative or preliminary runs.
- Do not pull/build an upstream Docker image unless Docker's image and layer storage is first proven not to consume the WSL Linux virtual disk and the projected size fits the cap.
- Do not duplicate the BorealTC dataset or upstream repository.

Create a machine-readable `storage_budget.json` before installation and update it after every environment attempt.

---

## 4. Source-level runtime ambiguity audit

Before installing dependencies, audit and record:

- `requirements-lock.txt`;
- `requirements.txt`;
- `Dockerfile` and any GPU Dockerfile at the locked commit;
- `README.md` installation statements;
- CNN/Mamba training scripts;
- model constructors and checkpoint loading code;
- SLURM scripts.

Explicitly determine:

1. which package versions are fully pinned;
2. which packages are under-specified, especially `torch`;
3. which CUDA/PyTorch wheel source or ABI is not specified;
4. what GPU architecture list is encoded by the Dockerfile;
5. whether the released checkpoint records Lightning, Torch, Mamba, or CUDA metadata;
6. whether the checkpoint has a traceable fold/training identity.

Write `runtime_source_semantics.json`.

Do not label an environment `EXACT_DECLARED_STACK` if source artifacts do not uniquely determine it. An underdetermined source stack must be labeled and explained.

---

## 5. Environment matrix

Use separate G-drive environments and never mutate the Stage 0B.1 environments.

Suggested roots:

```text
cache/venvs/borealtc-runtime-declared-probe
cache/venvs/borealtc-runtime-compat
```

Use Python 3.10, `venv --copies`, and the existing drvfs `lib64` workaround where required.

### 5.1 Declared-stack probe

The purpose is to determine install/build/runtime feasibility, not to force success.

- Install the exact pinned scientific-core versions.
- Resolve the Torch/Torchvision pair implied by the upstream files without silently choosing a newer model stack.
- Record resolver reports, wheel identities, indexes, hashes where available, and all deviations.
- Probe `mamba-ssm==1.2.0.post1` and `causal_conv1d==1.2.0.post1` in a bounded manner.
- Retain full build logs.
- At most two materially different build attempts are allowed for each CUDA extension. A repeated identical build is forbidden.
- Do not patch upstream source or third-party package source to obtain a pass.

Classify this environment as one of:

- `EXACT_DECLARED_STACK`;
- `DECLARED_STACK_UNDERDETERMINED`;
- `DECLARED_STACK_INSTALLABLE_BUT_GPU_UNSUPPORTED`;
- `DECLARED_STACK_BUILD_BLOCKED`;
- `DECLARED_STACK_RUNTIME_BLOCKED`.

### 5.2 Faithful compatibility stack

Only if the declared stack is blocked or cannot use the installed GPU, create a modern compatibility environment that supports the host.

Requirements:

- preserve the released CNN and Mamba architectures;
- preserve preprocessing, loss, optimizer, scheduler, early stopping, and output semantics;
- enumerate every package/version/API difference;
- place compatibility imports/adapters in tracked CryoLocoManip code, never upstream;
- demonstrate that any adapter changes API glue only, not numerical architecture or training objective;
- do not replace Mamba with another sequence model.

Label all resulting evidence `FAITHFUL_COMPATIBILITY_STACK`.

Write `environment_matrix.csv` and one freeze file per environment.

---

## 6. Released Mamba checkpoint audit

Checkpoint:

```text
third_party/BorealTC/checkpoints/mamba_borealtc.ckpt
```

Required outputs:

- file SHA256, Git object id, size;
- top-level checkpoint keys;
- stored hyperparameters and metadata;
- state-dict key/shape inventory;
- inferred model parameter count;
- expected constructor arguments;
- missing/unexpected keys under attempted load;
- evidence for or against a fold identity;
- evidence for or against equivalence to the five-fold Table III ensemble/results;
- CPU-load result;
- GPU-forward result when a faithful runtime exists.

Use `torch.load` only for the pinned, hash-recorded checkpoint. Record `weights_only` behavior and never load untrusted arbitrary pickles.

Do not claim the checkpoint reproduces Table III unless released evidence proves it represents all five fold models. An unresolved checkpoint role is an acceptable audited finding.

Write:

```text
checkpoint_inventory.json
checkpoint_state_dict.csv
checkpoint_load_attempts.csv
```

---

## 7. Real-data one-batch training smokes

No full epoch or full fold training is allowed.

Use the Stage 0B.1 paper pipeline and fixed split membership. Select fold 1 and construct real train/validation batches.

### 7.1 CNN smoke

Required:

- released Hamming/spectrogram input construction;
- released CNN architecture and loss;
- one forward pass;
- one backward pass;
- one optimizer step;
- one validation forward pass;
- finite input, logits, loss, gradients, parameters, and update;
- input/output shapes and label domains;
- deterministic seed capture;
- peak CPU RSS and GPU VRAM;
- initialization, preprocessing, forward, backward, and optimizer-step time.

### 7.2 Mamba smoke

Execute the same checks only if a faithful Mamba environment exists.

If Mamba cannot run, produce a `mamba_blocker.json` containing:

- failed environment label;
- exact command;
- compiler/CUDA/Torch/GPU identities;
- retained log paths and hashes;
- first root-cause error and relevant dependency chain;
- why a proposed workaround would or would not preserve semantics;
- whether CPU-only execution is technically possible and whether it is practical for 0B.2B;
- recommended terminal substatus.

### 7.3 No result inflation

A one-batch smoke establishes runtime feasibility only. Do not report accuracy as a reproduction result.

---

## 8. Projected full-reproduction budget

Using measured one-batch and preprocessing timings, estimate separately for CNN and Mamba:

- five-fold canonical-seed time;
- five folds × three model seeds;
- checkpoint and log storage;
- preprocessed/spectrogram cache storage;
- peak host RAM and GPU VRAM;
- expected failure/restart overhead.

Report lower, central, and conservative estimates; state assumptions. Do not extrapolate from a GPU smoke to CPU training without a separate measurement.

Write `stage0b2b_budget.json` and a human-readable table in the audit report.

---

## 9. CryoLocoManip implementation and tests

Add tracked code only under appropriate project paths, such as:

```text
tools/borealtc/runtime/
tests/test_borealtc_runtime_*.py
docs/audits/BOREALTC_RUNTIME_FEASIBILITY.md
```

Minimum tracked capabilities:

- host/GPU inventory;
- source-runtime semantic extraction;
- checkpoint inventory independent of model import where possible;
- environment/result manifest helpers;
- CNN/Mamba smoke wrappers;
- budget estimator;
- validation that the upstream worktree remains clean;
- tests for parser, identity, shape, finite-value, and status logic.

If new code is needed:

1. implement and test it;
2. commit and push it;
3. confirm a clean worktree;
4. start the formal run from that clean commit;
5. make a later documentation-only commit that references the run.

Never claim a formal run from uncommitted code.

---

## 10. Formal run contract

Run root:

```text
runs/stage0b2a/borealtc_runtime_feasibility/<run_id>/
```

Required contents:

```text
manifest.json
git_state.txt
command.sh
status.json
environment.txt
host_gpu_inventory.json
storage_budget.json
runtime_source_semantics.json
environment_matrix.csv
requirements_declared_probe.freeze.txt
requirements_compat.freeze.txt              # if applicable
checkpoint_inventory.json
checkpoint_state_dict.csv
checkpoint_load_attempts.csv
cnn_smoke.json
mamba_smoke.json                             # if executed
mamba_blocker.json                           # if blocked
stage0b2b_budget.json
failed_attempts.csv
stdout.log
stderr.log
sha256_manifest.txt
```

Every failed install/build/import/load/smoke attempt must be represented in `failed_attempts.csv`; do not show only the successful path.

`manifest.git.dirty` must be `false` at formal run start.

---

## 11. Acceptance checks

Before terminal closure, verify:

- workspace preflight and validator pass;
- upstream commit and cleanliness pass before and after every probe;
- project formal run started from clean committed code;
- storage cap and reserve are satisfied;
- environment labels match actual deviations;
- CNN real-data forward/backward smoke passes;
- Mamba either passes faithfully or has a complete blocker record;
- checkpoint role is not overstated;
- full-training budget is bounded;
- no full epoch/fold training occurred;
- no new model or thesis claim was introduced;
- all unit tests and compile checks pass;
- final branch is pushed and clean.

---

## 12. Evidence boundary

A Stage 0B.2A pass establishes only runtime and artifact feasibility for later training reproduction. It does not establish:

- the published training result has been regenerated;
- group-disjoint terrain generalization;
- confidence calibration;
- quadruped transfer;
- foot–snow interaction;
- action-conditioned outcomes;
- support–manipulation coupling;
- whole-body-control improvement;
- VLA necessity;
- thesis novelty.

---

## 13. Final report

Return:

1. terminal state;
2. start HEAD, clean execution commit, and final documentation commit;
3. branch and push state;
4. canonical/upstream storage confirmation;
5. host GPU/driver/CUDA/compute-capability inventory;
6. source-runtime ambiguity findings;
7. declared-stack feasibility result;
8. compatibility-stack differences, if used;
9. Mamba checkpoint identity and role conclusion;
10. CNN smoke result and resource measurements;
11. Mamba smoke result or blocker evidence;
12. projected Stage 0B.2B time/storage budget;
13. failed attempts and retained logs;
14. tests and commands;
15. final disk usage and worktree cleanliness;
16. exact evidence boundary.

Push only:

```text
origin/stage0/borealtc-runtime-feasibility
```

Do not open a PR, merge, or tag.