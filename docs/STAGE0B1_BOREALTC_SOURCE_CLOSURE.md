# Stage 0B.1 — BorealTC Source, Data, and Evaluator Closure

## Status

Pre-execution scientific contract. This stage is an audit and reconstruction stage, not a model-training stage and not evidence that the CryoLocoManip thesis hypothesis is valid.

## Decision question

Can the public BorealTC paper, repository, dataset, preprocessing contract, split semantics, and committed result artifacts be reconstructed without inventing policy, and can the exact boundary between terrain classification and action-conditioned support/manipulation prediction be stated from auditable evidence?

## Why Stage 0B is split

The upstream repository mixes several distinct artifacts:

- a simple public PyTorch dataset API in `borealtc.py`;
- the preprocessing/training pipeline used by the paper in `utils/preprocessing.py` and the training scripts;
- committed checkpoints and result arrays;
- committed metric text files;
- BorealTC and converted Vulpi data.

These are not assumed to have identical semantics. Stage 0B.1 closes source, data, split, and evaluator semantics before any Mamba/CNN retraining or new method implementation.

## Locked upstream identities

### Paper

- Title: **Proprioception Is All You Need: Terrain Classification for Boreal Forests**
- Venue: IEEE/RSJ IROS 2024
- DOI: `10.1109/IROS58592.2024.10801407`
- arXiv: `2403.16877v2`
- Paper-reported BorealTC results, Table III:
  - CNN accuracy: `93.96%`
  - Mamba accuracy: `93.68%`

### Code and data

- Repository: `https://github.com/norlab-ulaval/BorealTC.git`
- Default branch: `python`
- Locked commit: `0146dcd9fa08c34a9075a80448ac04c0a947b568`
- Locked commit subject: `preload all samples (#7)`
- Code license: MIT
- BorealTC data license: CC0 1.0
- Upstream repository must remain unmodified and detached/pinned at the locked commit.

### Paper data contract to reconstruct

- Platform: Clearpath Husky A200 wheeled skid-steer UGV.
- Modalities:
  - IMU: 3-axis angular velocity + 3-axis linear acceleration, nominally 100 Hz;
  - wheel service/proprioception: left/right wheel velocity + left/right motor current, nominally 6.5 Hz.
- Official BorealTC classes and paper 5 s partition counts:
  - ASPHALT: 111
  - FLOORING: 423
  - ICE: 450
  - SILTY LOAM in the paper / `SANDY_LOAM` in the repository: 126
  - SNOW: 281
  - total: 1391 partitions = 6955 s ≈ 115.92 min.
- Paper protocol:
  - non-overlapping 5 s partitions;
  - 5-fold cross-validation;
  - 1.7 s model samples;
  - class rebalancing by oversampling;
  - 10% of training data used for validation.

## Canonical committed result artifacts

The paper Table III numbers must be reconstructed from the committed result arrays, not copied from text files:

- CNN: `results/husky/results_CNN_hamming_mw_1.7.npy`
- Mamba: `results/husky/results_mamba_optim2_mw_1.7.npy`

Expected independent metrics, after mapping repository terrain names to the paper display names:

| Model | Terrain | Precision (%) | Recall (%) | F1 (%) |
| --- | --- | ---: | ---: | ---: |
| CNN | ASPHALT | 92.98 | 83.89 | 88.20 |
| CNN | FLOORING | 97.29 | 98.70 | 97.99 |
| CNN | ICE | 97.25 | 98.11 | 97.68 |
| CNN | SILTY/SANDY LOAM | 96.00 | 97.24 | 96.61 |
| CNN | SNOW | 86.84 | 92.31 | 89.49 |
| Mamba | ASPHALT | 91.90 | 85.50 | 88.59 |
| Mamba | FLOORING | 95.46 | 98.17 | 96.79 |
| Mamba | ICE | 97.12 | 97.36 | 97.24 |
| Mamba | SILTY/SANDY LOAM | 95.39 | 96.20 | 95.79 |
| Mamba | SNOW | 88.68 | 91.57 | 90.10 |

Tolerance for values rounded to two decimal places: `±0.01 percentage points`, unless an exact floating-point reconstruction demonstrates a justified edge case.

## Mandatory audits

### A. Repository and artifact integrity

1. Clone only to `/mnt/g/CryoLocoManip/third_party/BorealTC`.
2. Verify remote, branch history, locked commit, clean worktree, code license, and data license.
3. Record Git object identity and SHA256 for every upstream file directly used in a claim.
4. Inventory repository size and confirm no files are stored under the WSL Linux filesystem.

### B. Dataset integrity

For every BorealTC class and run:

- pair `imu_XX.csv` with `pro_XX.csv`;
- record row counts, columns, numeric dtypes, time range, monotonicity, duplicate timestamps, missing values, and inferred sampling statistics;
- verify the expected run pairs and report missing/orphan files;
- reconstruct exact 5 s partition counts using the paper training pipeline;
- reconcile repository `SANDY_LOAM` with paper `SILTY LOAM` without silently renaming raw data;
- exclude `MIXED` from the official five-class paper evaluation while inventorying it separately;
- verify the total duration claim and explain any difference between raw duration and retained full 5 s partitions.

### C. Pipeline semantics

Independently trace and document:

1. `utils/preprocessing.get_recordings`;
2. `partition_data` and `kfold_splits`;
3. `augment_data` class balancing and sliding-window generation;
4. `cleanup_data` and normalization;
5. model input shapes for IMU and wheel-service branches;
6. result concatenation over folds;
7. official metric generation.

The public `borealtc.py` / `SlidingWindowDataset` API must be audited separately. Do not assume it reproduces the paper preprocessing pipeline.

### D. Split and leakage audit

Reconstruct the official split exactly, then quantify without changing it:

- number of 5 s partitions per fold and class;
- number of original run IDs represented in both train and test within each fold;
- temporal adjacency of train/test partitions from the same run;
- duplicate or overlapping 1.7 s windows across train and test;
- effect of test-set oversampling on class support and aggregate accuracy;
- whether class-balancing parameters use any test-label counts;
- whether the same split is deterministic under `RANDOM_STATE=21`.

This audit does **not** declare the paper invalid. It identifies what the reported cross-validation estimates and what it does not estimate. A grouped-by-run alternative is diagnostic only in Stage 0B.1; no retraining is allowed.

### E. Independent evaluator reconstruction

Create a standard-library/NumPy/scikit-learn evaluator independent of upstream `compile_metrics.py` that:

- reads the two canonical `.npy` result artifacts;
- validates array lengths, label domains, terrain ordering, and fold concatenation assumptions;
- calculates accuracy, per-class precision, recall, F1, macro-F1, weighted-F1, balanced accuracy, confusion matrix, and class support;
- reproduces Table III;
- compares against upstream committed metric files;
- records the upstream `average_precision_score` implementation as diagnostic only unless its multiclass meaning is formally justified.

### F. Lightweight execution smoke

Using an environment stored below `/mnt/g/CryoLocoManip/cache/venvs/`:

- import and instantiate the public BorealTC dataset API;
- instantiate at least one sliding-window dataset;
- run upstream metric compilation if its lightweight dependencies permit;
- run all CryoLocoManip audit scripts and tests.

No Mamba/CUDA extension installation, model training, Optuna search, simulator installation, or checkpoint fine-tuning is allowed in this stage.

## Required CryoLocoManip outputs

Tracked code and documentation:

- `tools/borealtc/inventory.py`
- `tools/borealtc/reconstruct_paper_pipeline.py`
- `tools/borealtc/evaluate_committed_results.py`
- `tools/borealtc/audit_splits.py`
- tests for each audit module;
- `docs/audits/BOREALTC_SOURCE_CLOSURE.md`
- `literature/evidence_matrix.csv` with the verified BorealTC paper entry;
- `literature/notes/larocque2024_borealtc.md`.

Ignored run artifact root:

`runs/stage0b1/borealtc_source_closure/<run_id>/`

Minimum run contents:

- `manifest.json` conforming to `schemas/run_manifest.schema.json`;
- `environment.txt`;
- `upstream_identity.json`;
- `data_inventory.csv`;
- `class_summary.csv`;
- `partition_counts.csv`;
- `split_audit.csv`;
- `window_overlap_audit.csv`;
- `independent_metrics.csv`;
- `confusion_cnn.csv`;
- `confusion_mamba.csv`;
- `paper_result_comparison.csv`;
- `limitations.json`;
- full stdout/stderr logs;
- SHA256 manifest for the run outputs.

## Acceptance criteria

All of the following are mandatory for `PASS_BOREALTC_SOURCE_CLOSURE`:

1. canonical workspace preflight and workspace validator pass;
2. upstream source is clean and exactly pinned to the locked commit;
3. all official BorealTC run pairs and schemas are inventoried;
4. 5 s partition counts reconcile with the paper Table I or every discrepancy is traced to a specific source semantic;
5. independent evaluator reproduces the CNN and Mamba Table III values within tolerance;
6. upstream and independent evaluators agree on the canonical committed results;
7. split/run/overlap semantics are numerically documented;
8. the public dataset API smoke passes;
9. no heavy dependency or training occurred;
10. all tracked tests and compile checks pass;
11. branch is pushed but not merged or tagged;
12. the terminal report states exactly what BorealTC can and cannot support for CryoLocoManip.

## Evidence boundary

A pass establishes only that:

- BorealTC is a reproducible public baseline for wheeled-UGV proprioceptive terrain classification under its published split and sample contract;
- the committed paper-result artifacts are auditable;
- its transfer and leakage boundaries are understood.

A pass does **not** establish:

- quadruped foot–snow contact estimation;
- action-conditioned outcome prediction;
- support–manipulation coupling;
- whole-body control improvement;
- VLA necessity;
- cross-season action-validity memory;
- thesis novelty.

## Terminal states

Use exactly one:

- `PASS_BOREALTC_SOURCE_CLOSURE`
- `PASS_WITH_WARNINGS_BOREALTC_SOURCE_CLOSURE`
- `BLOCKED_BOREALTC_SOURCE_CLOSURE_<REASON>`
- `FAIL_BOREALTC_SOURCE_CLOSURE_<REASON>`

A warning state is allowed only when all paper metrics and dataset identities close, but a non-core lightweight tool cannot execute without changing upstream policy. Missing core semantics, invented substitutions, or inability to reconstruct Table III are blocking/failing conditions, not warnings.
