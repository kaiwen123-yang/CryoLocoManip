# CODEX_01 — BorealTC Source, Data, Split, and Evaluator Closure

## Role

You are the execution agent for Stage 0B.1 of CryoLocoManip. Your job is to reconstruct and audit an existing public baseline exactly enough to establish its evidence boundary. You are **not** being asked to improve BorealTC, train a new model, or validate the CryoLocoManip thesis.

Read and treat as authoritative, in this order:

1. `docs/STAGE0B1_BOREALTC_SOURCE_CLOSURE.md`
2. `docs/REPRODUCIBILITY_CONTRACT.md`
3. `docs/PROJECT_CONTRACT.md`
4. `docs/STAGE0_PLAN.md`
5. `docs/REPOSITORY_LAYOUT.md`

If this prompt conflicts with the Stage 0B.1 scientific contract, the scientific contract wins. Do not invent a replacement policy to make the stage pass.

## Terminal states

Return exactly one terminal state:

- `PASS_BOREALTC_SOURCE_CLOSURE`
- `PASS_WITH_WARNINGS_BOREALTC_SOURCE_CLOSURE`
- `BLOCKED_BOREALTC_SOURCE_CLOSURE_<REASON>`
- `FAIL_BOREALTC_SOURCE_CLOSURE_<REASON>`

Never use `PASS` when a mandatory identity, split semantic, canonical result, or evaluator result is unresolved.

---

## 0. Non-negotiable storage and Git rules

### Canonical project checkout

```text
/mnt/g/CryoLocoManip
```

The only project remote is:

```text
git@github.com:kaiwen123-yang/CryoLocoManip.git
```

Do not create a maintained project checkout, environment, data cache, temporary run, or model artifact under `/home`, `~`, `/root`, `/opt`, or the WSL Linux filesystem.

### Allowed G-drive locations

```text
/mnt/g/CryoLocoManip/third_party/BorealTC
/mnt/g/CryoLocoManip/cache/venvs/borealtc-source-closure
/mnt/g/CryoLocoManip/cache/pip
/mnt/g/CryoLocoManip/cache/tmp
/mnt/g/CryoLocoManip/runs/stage0b1/borealtc_source_closure
/mnt/g/CryoLocoManip/outputs
/mnt/g/CryoLocoManip/artifacts/runtime
```

Before invoking Python or pip, export:

```bash
export PIP_CACHE_DIR=/mnt/g/CryoLocoManip/cache/pip
export XDG_CACHE_HOME=/mnt/g/CryoLocoManip/cache/xdg
export TMPDIR=/mnt/g/CryoLocoManip/cache/tmp
mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR"
```

### Git policy

1. Start from a clean, current `main`.
2. Use only fast-forward pulls.
3. Create branch:

```text
stage0/borealtc-source-closure
```

4. Do not rewrite history, force-push, merge, or tag.
5. Commit only CryoLocoManip source, tests, documentation, and small evidence summaries.
6. Do not commit upstream code, datasets, checkpoints, result arrays, generated figures, environments, or run directories.
7. Push the branch at the end only after validations pass or a blocker is documented.

If the canonical checkout is dirty before this task, stop and report the exact files. Do not stash or discard unknown user work.

---

## 1. Preflight and start anchor

Run and save the exact output of:

```bash
cd /mnt/g/CryoLocoManip
git fetch origin --prune
git switch main
git pull --ff-only origin main
bash scripts/preflight_wsl.sh
python3 scripts/verify_workspace.py
python3 -m unittest discover -s tests -v
python3 -m compileall src scripts tests
```

Record:

- start `HEAD`;
- branch and status;
- `/mnt/g` filesystem and available capacity;
- Python, Git, GPU, CUDA, ROS, Docker, compiler, CPU, and memory inventory using `scripts/collect_environment.sh`;
- confirmation that no second maintained `CryoLocoManip` checkout exists under the Linux filesystem.

Then create the stage branch.

---

## 2. Locked upstream source

Use exactly:

```text
Repository: https://github.com/norlab-ulaval/BorealTC.git
Branch identity: python
Commit: 0146dcd9fa08c34a9075a80448ac04c0a947b568
Commit subject: preload all samples (#7)
```

Clone or reuse only:

```text
/mnt/g/CryoLocoManip/third_party/BorealTC
```

If the directory does not exist:

```bash
git clone --branch python --single-branch https://github.com/norlab-ulaval/BorealTC.git \
  /mnt/g/CryoLocoManip/third_party/BorealTC
```

Then fetch and detach at the locked commit. If it already exists, first verify its remote and clean state. Never delete, reset, clean, or overwrite an unexpected existing checkout without reporting the conflict.

Mandatory checks:

```bash
git -C /mnt/g/CryoLocoManip/third_party/BorealTC remote -v
git -C /mnt/g/CryoLocoManip/third_party/BorealTC rev-parse HEAD
git -C /mnt/g/CryoLocoManip/third_party/BorealTC show -s --format='%H%n%s%n%cI' HEAD
git -C /mnt/g/CryoLocoManip/third_party/BorealTC status --porcelain
```

Keep the upstream checkout clean for the entire task. Any upstream script that writes files must be executed only in a disposable copy below the stage run directory.

---

## 3. Environment policy

Create a dedicated environment only below:

```text
/mnt/g/CryoLocoManip/cache/venvs/borealtc-source-closure
```

Use `python3 -m venv --copies` if drvfs symlink behavior requires it. Install only dependencies needed for source/data/evaluator closure, such as:

- NumPy
- pandas
- SciPy
- scikit-learn
- tqdm
- PyTorch only if required for the public `borealtc.py` API smoke and not already usable

Do **not** install:

- `mamba-ssm`
- `causal-conv1d`
- `tsnecuda`
- Lightning for training
- Optuna
- simulators
- ROS packages
- VLA packages
- CUDA extensions

Do not train CNN, Mamba, LSTM, SVM, or any new model in this stage.

Freeze the actual environment to the run directory. Record package versions and all install commands. If standard PyTorch cannot be made available without installing a CUDA extension stack, complete the core source/evaluator closure and use the warning terminal state only if the scientific contract permits it.

---

## 4. Run identity and immutable artifacts

Create one canonical run:

```text
runs/stage0b1/borealtc_source_closure/<run_id>/
```

The run ID must include UTC time and the CryoLocoManip short SHA. Create `manifest.json` before substantive execution and update its end time and exit code at completion.

All raw logs, inventories, copied upstream artifacts, intermediate arrays, and evaluator results stay in the ignored run directory.

Generate SHA256 for:

- every BorealTC CSV;
- the two canonical result arrays;
- the corresponding committed metric files;
- the locked paper/source identity files used in the audit;
- every final run artifact.

---

## 5. Dataset and schema audit

Implement `tools/borealtc/inventory.py` and tests.

For each class directory and each run pair:

1. pair `imu_XX.csv` with `pro_XX.csv`;
2. report missing/orphan files;
3. record row counts, columns, numeric dtypes, byte size, SHA256;
4. check time monotonicity, duplicates, missing values, start/end/duration;
5. calculate median/min/max sample interval and robust inferred frequency;
6. identify malformed rows or schema differences;
7. inventory `MIXED` separately and exclude it from the official five-class result contract;
8. preserve raw label `SANDY_LOAM`, while recording its paper display name `SILTY LOAM`.

Reconcile the official five-class 5 s partition counts:

```text
ASPHALT 111
FLOORING 423
ICE 450
SANDY_LOAM / SILTY LOAM 126
SNOW 281
TOTAL 1391
```

Do not infer these counts from the committed summary file alone. Reconstruct them from the CSVs using the locked preprocessing semantics.

Explain raw recorded duration versus retained complete 5 s partition duration.

---

## 6. Paper-pipeline reconstruction

Implement `tools/borealtc/reconstruct_paper_pipeline.py` and tests.

Trace the locked upstream implementation without changing it:

- `utils/preprocessing.get_recordings`
- `partition_data`
- `kfold_splits`
- `augment_data`
- `cleanup_data`
- `normalize_data`
- the result-concatenation logic in `mamba_train.py`

Required outputs:

- class and modality sampling frequencies;
- partition tensor shapes before split;
- exact fold train/test partition counts by class;
- exact post-augmentation window counts by fold/class/modality;
- IMU and wheel-service model input shapes;
- normalization statistics provenance;
- deterministic hash/identity of the reconstructed split under `RANDOM_STATE=21`.

The committed public dataset API in `borealtc.py` must be audited separately. Do not treat its resample-and-fuse implementation as the paper training pipeline unless exact equivalence is demonstrated.

---

## 7. Split, run, and overlap audit

Implement `tools/borealtc/audit_splits.py` and tests.

For each official fold, quantify:

- original run IDs in train and test;
- count and percentage of run IDs appearing in both;
- temporal adjacency between train/test 5 s partitions from the same run;
- exact duplicates across train/test;
- overlapping time support across train/test after 1.7 s augmentation;
- train and test class counts before and after oversampling;
- whether any test-label counts influence oversampling parameters;
- whether test oversampling changes the weighting of the reported aggregate accuracy;
- deterministic reproduction under repeated execution.

Also construct a grouped-by-run split **diagnostic description** and fold-feasibility table, but do not train or publish grouped-run performance in Stage 0B.1.

Use neutral language. Do not call the official protocol invalid merely because runs cross folds; state exactly which generalization question it estimates and which it does not.

---

## 8. Independent evaluation of committed results

Implement `tools/borealtc/evaluate_committed_results.py` and tests.

Use exactly these result arrays:

```text
results/husky/results_CNN_hamming_mw_1.7.npy
results/husky/results_mamba_optim2_mw_1.7.npy
```

Do not substitute `mamba-1700-optim` for `mamba-1700-optim2`.

The evaluator must independently validate and compute:

- sample counts;
- class order and label domain;
- accuracy;
- per-class precision, recall, F1, and support;
- macro-F1;
- weighted-F1;
- balanced accuracy;
- confusion matrix;
- inference and preprocessing timing summaries when present.

Required paper accuracy targets:

```text
CNN    93.96%
Mamba  93.68%
```

Required per-class targets are listed in `docs/STAGE0B1_BOREALTC_SOURCE_CLOSURE.md`.

Compare independent outputs against:

```text
metrics/husky/CNN-1700-hamming.dat
metrics/husky/mamba-1700-optim2.dat
```

Use tolerance `±0.01` percentage points for two-decimal metrics. Report every mismatch, including class-name mapping and rounding.

Audit the upstream `compile_metrics.py` AP calculation separately. Do not promote its scalar AP to a standard multiclass average-precision claim unless its mathematical semantics are justified. The paper Table III closure does not depend on AP.

---

## 9. Upstream metric smoke without modifying upstream

If lightweight dependencies permit, create a disposable minimal copy under the run directory containing the unchanged upstream `compile_metrics.py` and required committed result artifacts, then run it there.

Requirements:

- preserve the exact SHA256 of the upstream script;
- do not patch it;
- do not write into `/mnt/g/CryoLocoManip/third_party/BorealTC`;
- compare its generated metric text against committed upstream metrics and the independent evaluator.

A failure caused only by a non-core optional dependency may become a warning only after the independent paper-result reconstruction succeeds.

---

## 10. Public dataset API smoke

Using the locked `borealtc.py`:

- instantiate the full BorealTC dataset if memory permits;
- report class order and number of run-level samples;
- instantiate a 170-step sliding-window dataset;
- report number and shape of windows;
- inspect at least one sample from every official class;
- record elapsed time and peak resident memory if practical.

If full preload is not feasible, demonstrate the failure and run a class-bounded smoke. Do not silently redefine the public API.

---

## 11. Evidence matrix and audit note

Create the first verified literature row in:

```text
literature/evidence_matrix.csv
```

and a detailed note:

```text
literature/notes/larocque2024_borealtc.md
```

The note must distinguish:

- paper claims;
- repository claims;
- independently reproduced committed-result metrics;
- split diagnostics;
- unsupported transfer claims;
- direct relevance and non-relevance to CryoLocoManip.

Do not copy the paper text or commit the PDF.

---

## 12. Required tracked implementation

Minimum tracked files:

```text
tools/borealtc/__init__.py
tools/borealtc/inventory.py
tools/borealtc/reconstruct_paper_pipeline.py
tools/borealtc/evaluate_committed_results.py
tools/borealtc/audit_splits.py
tests/test_borealtc_inventory.py
tests/test_borealtc_pipeline.py
tests/test_borealtc_evaluator.py
tests/test_borealtc_split_audit.py
docs/audits/BOREALTC_SOURCE_CLOSURE.md
literature/evidence_matrix.csv
literature/notes/larocque2024_borealtc.md
```

Code must be typed where practical, deterministic, and fail loudly on identity/schema mismatches. Do not hard-code the expected metrics as output; expected values may appear only in tests or comparison configuration.

---

## 13. Required validations

At completion, run at minimum:

```bash
bash scripts/preflight_wsl.sh
python3 scripts/verify_workspace.py
python3 -m unittest discover -s tests -v
python3 -m compileall src scripts tests tools
```

Also execute every new audit command on the canonical pinned upstream checkout and save stdout/stderr in the run directory.

Verify:

- CryoLocoManip tracked worktree contains only intended changes;
- upstream BorealTC worktree remains clean;
- no project data/environment was written to the WSL Linux filesystem;
- no file larger than 10 MB is staged for commit unless it is a small, justified text/CSV evidence artifact;
- the run manifest and SHA256 manifest are complete.

---

## 14. Commit and push

Use one or a small number of logically coherent commits. Suggested final subject:

```text
Close BorealTC source and evaluator semantics
```

Push:

```text
origin/stage0/borealtc-source-closure
```

Do not open or merge a PR unless explicitly instructed by the user in the Codex session. Do not tag.

---

## 15. Final terminal report format

Return a complete report with these sections:

1. terminal state;
2. start and end Git identities;
3. canonical storage and duplicate-check result;
4. upstream repository/commit/license identities;
5. environment and installed dependencies;
6. dataset/run/schema inventory;
7. reconstructed Table I partition counts;
8. reconstructed official split/window semantics;
9. numerical split/run/overlap findings;
10. independent CNN/Mamba metrics and comparison to Table III;
11. upstream evaluator comparison;
12. public dataset API smoke result;
13. tests and commands;
14. tracked files changed;
15. run artifact path and SHA256 manifest;
16. warnings, blockers, and exclusions;
17. explicit evidence-boundary statement.

The evidence-boundary statement must say, in substance, that BorealTC closes a wheeled-UGV terrain-classification baseline but does not validate quadruped contact, action-conditioned outcomes, support–manipulation coupling, whole-body control, VLA necessity, or thesis novelty.
