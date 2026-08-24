# LaRocque et al. 2024 — BorealTC (IROS 2024)

Citekey `larocque2024_borealtc` · DOI `10.1109/IROS58592.2024.10801407` ·
arXiv `2403.16877` · code/data
`https://github.com/norlab-ulaval/BorealTC` @
`0146dcd9fa08c34a9075a80448ac04c0a947b568` (code MIT, BorealTC data CC0 1.0).

Audit evidence: `docs/audits/BOREALTC_SOURCE_CLOSURE.md` and the
authoritative R1 formal run (clean commit `ac4fb7a`, `dirty=false`, exact
upstream-lock core environment)
`runs/stage0b1/borealtc_source_closure/20260824T103047Z_ac4fb7a_borealtc-src-closure-r1_21/`;
the preliminary run
`20260824T095031Z_767f1c5_borealtc-src-closure_21/` is preserved as its
parent. No paper text is copied here; no PDF is committed.

## 1. Paper claims (CLAIMED_BY_SOURCE unless stated otherwise)

- Proprioceptive (IMU + wheel velocity/current) terrain classification on a
  Husky A200 wheeled UGV over five terrains (ASPHALT, FLOORING, ICE, SILTY
  LOAM, SNOW), 5 s partitions, 5-fold CV, 1.7 s model samples, oversampling
  class balance, 10 % of train used for validation.
- BorealTC Table III headline: CNN 93.96 % accuracy, Mamba 93.68 %.
- Dataset contribution: 116 min Husky proprioception over boreal terrains.

## 2. Repository claims (verified at the locked commit)

- Data: 71 official run pairs + 1 MIXED pair, CSV schemas
  `time,wx,wy,wz,ax,ay,az` / `time,curL,curR,velL,velR`, 100 Hz / 6.5 Hz.
- Committed artifacts: checkpoints, result arrays (`results/husky/…npy`),
  metric files (`metrics/husky/…dat`), partition summary
  (`summary/terrains.dat`).
- Pipeline: `utils/preprocessing.py` + `mamba_train.py` with
  `RANDOM_STATE=21`; public `borealtc.py` dataset API is a separate,
  resample-and-ffill implementation.

## 3. Independently REPRODUCED from released artifacts (Stage 0B.1)

- Table I partition counts reconstructed from raw CSVs: 111/423/450/126/281
  (total 1391 = 6955 s); exact match to paper and committed summary.
- Table III: all 32 values (2 accuracies + 30 per-class P/R/F1) recomputed
  from the two canonical result arrays with an independent scikit-learn
  evaluator; every value within ±0.01 pp of the paper and the committed
  `.dat` files. Extra aggregates: macro-F1 94.00/93.70, weighted-F1
  93.92/93.64, balanced accuracy 94.05/93.76 (CNN/Mamba).
- Unchanged upstream `compile_metrics.py`, run in a disposable copy,
  regenerates both committed `.dat` files byte-identically.
- Split reconstruction: StratifiedKFold(21) fold sizes [279,278,278,278,278];
  per-fold per-class test-window counts recovered from the committed arrays
  match the reconstruction exactly for both models (17 954 windows; class
  support 3663/3384/3600/3654/3653).
- Upstream AP scalar (18.61/18.35) reproduces but is a non-standard
  construction (class indices fed as scores); DIAGNOSTIC_ONLY.
- R1 exact-version closure: the identical committed audit code re-executed
  under the upstream lock stack (numpy 1.26.4, pandas 2.2.0, scipy 1.12.0,
  scikit-learn 1.4.0) reproduces every split membership SHA256, count,
  overlap statistic, confusion matrix, and metric with zero delta versus
  the scikit-learn 1.7.2 preliminary run
  (`version_pin_comparison.json`, verdict
  IDENTITY_MATCH_AND_METRICS_WITHIN_TOLERANCE).

Training itself was NOT rerun; end-to-end training closure is Stage 0B.2.
Overall row claim level: **AUDITED** (metric recomputation from released
artifacts is REPRODUCED-grade; the trained models are not).

## 4. Split diagnostics (facts, not invalidity claims)

- The official CV splits 5 s partitions, not runs: per fold, 98.3–100 % of
  test runs also contribute training partitions; 99.6–100 % of test
  partitions come from runs seen in training; 2 144 train/test partition
  pairs are temporally adjacent (0 s gap). Exact duplicates and cross-set
  1.7 s window time-support overlap are zero.
- Class balancing parameters derive from the full 1391-partition label
  distribution (includes every fold's test labels) and are also applied to
  the test sets; natural-weighted diagnostic accuracy is 95.90 % (CNN) /
  95.38 % (Mamba) versus reported 93.96/93.68 %.
- The protocol therefore estimates within-run generalization to unseen
  partitions under a balanced test distribution — not leave-run-out /
  leave-site-out transfer. A grouped-by-run split is feasible (≥ 5 runs per
  class) but imbalanced (largest ICE run holds 63.3 % of ICE partitions).

## 5. Transfer claims NOT supported by this source (INFERENCE guard)

BorealTC provides no evidence for: quadruped foot–snow contact estimation;
action-conditioned outcome prediction (slip/sinkage/crust failure under
candidate actions); support–manipulation coupling; whole-body control;
manipulation forces or payload effects; VLA necessity; seasonal
action-validity memory. Terrain-class accuracy must not be cited as action
safety evidence (docs/PROJECT_CONTRACT.md prohibited claims).

## 6. Relevance to CryoLocoManip

- Direct (medium): the Stage 0B baseline; establishes what proprioceptive
  terrain classification already solves on snow/ice-adjacent terrain, the
  gap between terrain labels and action outcomes, and an auditable
  reproduction target for Stage 0B.2 retraining.
- Not relevant to the thesis core: no legged embodiment, no arm, no contact
  wrench data, no candidate-action conditioning, no uncertainty gating.
