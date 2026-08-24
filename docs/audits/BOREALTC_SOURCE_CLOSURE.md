# BorealTC Source, Data, Split, and Evaluator Closure (Stage 0B.1)

Terminal state: **PASS_BOREALTC_SOURCE_CLOSURE**

Executed against `docs/STAGE0B1_BOREALTC_SOURCE_CLOSURE.md`. All numbers in
this document were computed by the tracked tools in `tools/borealtc/` from
the pinned upstream checkout; expected paper/contract values appear only as
comparison references. The full machine-readable evidence lives in the
ignored run directory
`runs/stage0b1/borealtc_source_closure/20260824T095031Z_767f1c5_borealtc-src-closure_21/`
(SHA256 manifest included).

## 1. Locked identities

| Item | Value |
| --- | --- |
| Paper | *Proprioception Is All You Need: Terrain Classification for Boreal Forests*, IEEE/RSJ IROS 2024, DOI `10.1109/IROS58592.2024.10801407`, arXiv `2403.16877v2` |
| Repository | `https://github.com/norlab-ulaval/BorealTC.git`, branch `python` |
| Commit | `0146dcd9fa08c34a9075a80448ac04c0a947b568` — "preload all samples (#7)" (2025-04-14) |
| Licenses | Code MIT; BorealTC data CC0 1.0 |
| Checkout | `third_party/BorealTC`, detached at the locked commit, worktree clean before and after every audit |
| Canonical result arrays | `results/husky/results_CNN_hamming_mw_1.7.npy` (sha256 `cc1f732b…`), `results/husky/results_mamba_optim2_mw_1.7.npy` (sha256 `a437fac4…`) |
| Canonical metric files | `metrics/husky/CNN-1700-hamming.dat`, `metrics/husky/mamba-1700-optim2.dat` |
| Audit environment | Python 3.10.12 venv on G-drive; numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, scikit-learn 1.7.2, tqdm 4.70.0, torch 2.13.0+cpu (upstream lock pinned numpy 1.26.4 / pandas 2.2.0 / scipy 1.12.0 / scikit-learn 1.4.0; see §8) |

## 2. Dataset inventory (data/borealtc)

72 `imu_XX.csv`/`pro_XX.csv` run pairs, 0 orphans, all schemas
`time,wx,wy,wz,ax,ay,az` / `time,curL,curR,velL,velR`, all strictly
increasing timestamps, no duplicate timestamps, no NaNs — except MIXED (see
below). The upstream frequency rule `round(1/min(diff(time)),1)` yields
exactly 100.0 Hz for every IMU file and 6.5 Hz for every pro file, so
`get_recordings`' first-file `setdefault` is order-independent.

| Class (paper name) | Runs | IMU rows | Raw IMU duration s | 5 s partitions | Retained s | Retained % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ASPHALT | 7 | 57 010 | 570.03 | **111** | 555 | 97.36 |
| FLOORING | 10 | 214 113 | 2141.03 | **423** | 2115 | 98.78 |
| ICE | 13 | 228 730 | 2287.17 | **450** | 2250 | 98.37 |
| SANDY_LOAM (SILTY LOAM) | 14 | 66 275 | 662.61 | **126** | 630 | 95.08 |
| SNOW | 27 | 147 506 | 1474.79 | **281** | 1405 | 95.27 |
| **Official total** | **71** | 713 634 | **7135.63** | **1391** | **6955** | **97.47** |
| MIXED (excluded) | 1 | 114 924 | 1149.23 | 229 | 1145 | 99.63 |

- Partition counts were reconstructed from the CSVs with the locked
  semantics (`int(5 s × 100 Hz) = 500` IMU samples per partition, incomplete
  tails dropped) and reconcile **exactly** with paper Table I and with the
  committed `summary/terrains.dat` (111/423/450/126/281, total 1391 = 6955 s
  ≈ 115.9 min). Raw-vs-retained difference (180.6 s official) is entirely the
  dropped incomplete tails.
- Raw label `SANDY_LOAM` is preserved everywhere; the paper display name
  SILTY LOAM is recorded alongside, never substituted into raw data.
- MIXED is inventoried separately and excluded from the official five-class
  contract; its two CSVs carry an extra trailing `terrain` column and NaN
  cells (67 997 IMU / 4 436 pro), unlike every official file.
- Wheel/pro partitions: 2 of 1391 partitions (1 ASPHALT, 1 ICE) required the
  upstream `overwin` end-shift; no run is shorter than one 32-sample pro
  window.

## 3. Paper-pipeline reconstruction (unchanged upstream code)

Chain executed exactly as the locked `mamba_train.py`: `get_recordings` →
`partition_data(PART_WINDOW=5, N_FOLDS=5, RANDOM_STATE=21)` →
`augment_data(mw=1.7, stride=0.1, homogeneous=True)` → `cleanup_data` →
`normalize_data`.

- Sampling summary: `{imu: 100.0 Hz, pro: 6.5 Hz}`.
- Pre-split tensors: IMU `(1391, 500, 11)`, pro `(1391, 32, 9)`; channel
  layout `terrain, terr_idx, run_idx, win_idx, time, <sensor channels>`.
- 5-fold StratifiedKFold on terrain labels, `shuffle=True`,
  `RandomState(21)`: test partitions per fold `[279, 278, 278, 278, 278]`;
  split identical across repeated executions (SHA256 fold-membership hashes
  recorded).
- Homogeneous augmentation (33 strides/partition available; min class 111):
  per-class slides/stride `ASPHALT 33/10`, `FLOORING 8/41`, `ICE 8/41`,
  `SANDY_LOAM 29/11`, `SNOW 13/25` — reproduced by the unchanged upstream
  code **and** by an independent arithmetic reimplementation; every
  fold/class window count equals partitions × slides.
- Post-augmentation test windows per fold `[3617, 3579, 3579, 3579, 3600]`,
  total **17 954**; per-class totals ASPHALT 3663, FLOORING 3384, ICE 3600,
  SANDY_LOAM 3654, SNOW 3653.
- Model inputs after cleanup/normalize (fold 1): IMU `(14337, 170, 6)` train
  / `(3617, 170, 6)` test, wheel-service `(N, 11, 4)`, float32. The CNN
  variant consumes multichannel spectrograms of these same windows
  (`hamming`); spectrogram internals are out of scope here.
- Normalization statistics come from the training fold only and are applied
  to train and test; verified numerically, not assumed.
- Source quirk: `utils/preprocessing.py` defines `partition_data`,
  `kfold_splits`, and `augment_data_ablation` **twice**; the later,
  semantically identical definitions are the effective ones.
- Windows never cross a partition boundary (max window end ≤ 500 samples for
  every class), so 1.7 s augmentation cannot create cross-partition overlap.

## 4. Split, run, and overlap audit (official protocol, unmodified)

| Fold | Test partitions | Test runs | Runs also in train | % test partitions from train-runs | Adjacent train/test pairs | Duplicates |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 279 | 56 | 56 (100%) | 100.0 | 421 | 0 |
| 2 | 278 | 58 | 58 (100%) | 100.0 | 420 | 0 |
| 3 | 278 | 60 | 59 (98.3%) | 99.64 | 451 | 0 |
| 4 | 278 | 59 | 59 (100%) | 100.0 | 433 | 0 |
| 5 | 278 | 61 | 60 (98.4%) | 99.64 | 419 | 0 |

- 2 144 train/test partition pairs across folds are temporally adjacent
  (0 s boundary gap) within the same run; exact duplicates and cross-set
  1.7 s window time-support overlap are both **zero**.
- Balancing parameters (`n_slides` per class) are computed once from the
  full 1391-partition label distribution (fold-1 train ∪ test), which
  includes every fold's test labels, and the same balancing is applied to
  every fold's train **and** test set. Recomputed from fold-1 train labels
  alone they would differ (SANDY_LOAM 28 vs 29, SNOW 12 vs 13).
- Test balancing changes aggregate weighting: natural partition shares are
  8.0/30.4/32.4/9.1/20.2 %, balanced window shares 20.4/18.8/20.1/20.4/20.3 %.
  Reweighting per-class recalls to the natural distribution gives a
  diagnostic accuracy of 95.90 % (CNN) / 95.38 % (Mamba) versus the reported
  93.96 / 93.68 % — the balanced protocol gives more weight to the harder
  minority classes.
- Scope statement (neutral): the official protocol estimates within-run
  generalization to unseen 5 s partitions under a class-balanced test
  distribution. It does not estimate leave-run-out or leave-site-out
  generalization. This is a property, not an invalidity claim.
- Grouped-by-run 5-fold split is structurally feasible (≥ 5 runs per class:
  7/10/13/14/27) but would be strongly imbalanced — the largest single run
  holds 63.3 % of ICE partitions (285/450) and 45.9 % of ASPHALT (51/111).
  Diagnostic only; no grouped-run training or performance in Stage 0B.1.

## 5. Independent evaluation of the committed result arrays

Both canonical arrays hold int64 index arrays `pred`/`true` over terrain
order `ASPHALT, FLOORING, ICE, SANDY_LOAM, SNOW`, length 17 954 each — equal
to the reconstructed fold-concatenated test-window total, with per-class
support exactly `3663/3384/3600/3654/3653`. The per-fold per-class block
structure recovered from the label layout matches the reconstructed
StratifiedKFold(21) fold counts for both models, fold by fold.

Independent metrics (scikit-learn, no upstream evaluator code), rounded to
2 dp — every value equals paper Table III and the committed `.dat` files
within ±0.01 pp (34/34 comparisons, zero mismatches):

| Model | Accuracy | Macro-F1 | Weighted-F1 | Balanced acc. |
| --- | ---: | ---: | ---: | ---: |
| CNN | **93.96** | 94.00 | 93.92 | 94.05 |
| Mamba | **93.68** | 93.70 | 93.64 | 93.76 |

Per-class precision/recall/F1 (all matching Table III / `.dat`):

| Model | ASPHALT | FLOORING | ICE | SILTY (SANDY) LOAM | SNOW |
| --- | --- | --- | --- | --- | --- |
| CNN | 92.98 / 83.89 / 88.20 | 97.29 / 98.70 / 97.99 | 97.25 / 98.11 / 97.68 | 96.00 / 97.24 / 96.61 | 86.84 / 92.31 / 89.49 |
| Mamba | 91.90 / 85.50 / 88.59 | 95.46 / 98.17 / 96.79 | 97.12 / 97.36 / 97.24 | 95.39 / 96.20 / 95.79 | 88.68 / 91.57 / 90.10 |

- Confusion matrices: run artifacts `confusion_cnn.csv` / `confusion_mamba.csv`.
- `ftime`/`ptime` keys exist in both arrays but are **empty**, so no
  inference/preprocessing timing closure is possible from committed
  artifacts.
- Upstream evaluator smoke: the unchanged `compile_metrics.py`
  (sha256-verified copy, executed in a disposable run-directory copy, never
  inside `third_party/`) regenerates both committed `.dat` files
  **byte-identically**, and agrees with the independent evaluator.
- Upstream `average_precision_score` scalar reproduces exactly (18.61 CNN,
  18.35 Mamba) but feeds integer class indices as scores of one binary
  column; it is not a standard multiclass average precision and is recorded
  as **diagnostic only**. Table III closure does not depend on it.

## 6. Public dataset API smoke (`borealtc.py`, audited separately)

- Full preload: 71 run-level fused samples, classes
  `['asphalt', 'flooring', 'ice', 'sandy_loam', 'snow']` (MIXED excluded by
  the API), peak RSS 335 MB.
- `pd.infer_freq` returns `10ms` for IMU and **None** for the irregular
  6.5 Hz pro stream, so `fuse_measures` forward-fills both modalities onto
  the IMU 10 ms grid: fused windows are `(170, 10)` at 100 Hz — structurally
  different from the paper pipeline's separate `(170, 6)` IMU + `(11, 4)`
  wheel branches. The public API is therefore NOT the paper preprocessing,
  and no equivalence is claimed.
- `SlidingWindowDataset(window_size=170)`: 14 154 windows at step 50
  (class default), 70 615 at step 10 (upstream `__main__` example); one
  window per official class fetched, all `(170, 10)` float32, no NaNs.

## 7. What this stage does and does not establish (evidence boundary)

Established: BorealTC is a reproducible public baseline for **wheeled-UGV
(Husky A200) proprioceptive terrain classification** under its published
partition/split/balancing contract; its committed paper-result artifacts are
auditable and its split semantics (within-run, partition-level,
class-balanced) are now numerically documented.

Not established — and not evidence for the CryoLocoManip thesis:

- quadruped foot–snow contact estimation (no legged platform, no contact
  or arm reaction forces in the data);
- action-conditioned outcome prediction (labels are terrain classes, not
  outcomes of candidate actions; no paired counterfactual actions);
- support–manipulation coupling, whole-body control improvement, VLA
  necessity, cross-season action-validity memory, or thesis novelty.

## 8. Warnings and open items

1. Audit ran on newer NumPy/pandas/scikit-learn than the upstream lock;
   every closure target still reproduced (fold fingerprints and all Table
   III metrics). Fold *identity* beyond class counts is not recoverable
   from committed artifacts.
2. `mamba-ssm`, `causal-conv1d`, `tsnecuda`, Lightning, Optuna were NOT
   installed; CNN/Mamba/LSTM/SVM training and checkpoint evaluation are
   deferred to Stage 0B.2 (upstream checkpoints exist but were not
   executed here).
3. Full limitations ledger: `limitations.json` in the run directory.
