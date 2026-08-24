# Stage 0B.2 — BorealTC Formal Training Reproduction and Run-Level Generalization

## Status

Pre-execution scientific contract.

Stage 0B.1 closed the released source, raw data, preprocessing, split, committed-result, evaluator, and public-API semantics for BorealTC. Stage 0B.2 asks a different set of questions:

1. can the released training and checkpoint stack be executed faithfully on the available host without silently changing model semantics;
2. can the published CNN and Mamba results be regenerated from training rather than only recomputed from released prediction arrays;
3. how much performance remains when the evaluation unit is an unseen acquisition run rather than an unseen 5 s partition from a run already represented in training;
4. how reliable are the resulting confidence scores under natural class frequencies and run-level distribution shift.

This stage still does **not** develop a CryoLocoManip method or validate the thesis hypothesis.

---

## 1. Why Stage 0B.2 is split

A single task combining CUDA compatibility work, checkpoint loading, full five-fold training, grouped-run evaluation, calibration, and statistical analysis would conflate infrastructure failures with scientific findings. Stage 0B.2 is therefore divided into three gates.

### Stage 0B.2A — Runtime, checkpoint, and training-feasibility closure

Determine whether the declared upstream environment, released Mamba checkpoint, CNN pipeline, and Mamba pipeline can execute on the available WSL GPU host without modifying the pinned upstream checkout or inventing model substitutions.

No full five-fold training is allowed in 0B.2A.

### Stage 0B.2B — Official-protocol training reproduction

Train the released CNN and, if 0B.2A establishes a faithful runtime, Mamba under the Stage 0B.1 official preprocessing and split contract. This track reproduces the paper's within-run, class-balanced protocol and is not a cross-run generalization claim.

### Stage 0B.2C — Run-level generalization and calibration stress test

Train and evaluate declared model families under a group-disjoint protocol in which no original acquisition run occurs in both training and test. This is a new diagnostic protocol, not a reproduction of the paper.

The three tracks must remain distinguishable in paths, manifests, tables, and prose.

---

## 2. Locked identities inherited from Stage 0B.1

### Paper and upstream

- Paper: **Proprioception Is All You Need: Terrain Classification for Boreal Forests**
- Venue: IEEE/RSJ IROS 2024
- DOI: `10.1109/IROS58592.2024.10801407`
- arXiv: `2403.16877v2`
- Upstream repository: `https://github.com/norlab-ulaval/BorealTC.git`
- Locked upstream commit: `0146dcd9fa08c34a9075a80448ac04c0a947b568`
- Upstream checkout: `/mnt/g/CryoLocoManip/third_party/BorealTC`
- Upstream worktree must remain detached, unmodified, and clean.

### Dataset and official preprocessing

- Official classes: `ASPHALT`, `FLOORING`, `ICE`, `SANDY_LOAM`, `SNOW`.
- `MIXED` remains outside the official five-class paper protocol.
- 1391 complete non-overlapping 5 s partitions.
- IMU: 100 Hz, six channels.
- Wheel service: nominal 6.5 Hz, four channels.
- Official split: five-fold `StratifiedKFold`, `shuffle=True`, `RANDOM_STATE=21` at the 5 s partition level.
- Model sample duration: 1.7 s.
- Official homogeneous augmentation and class balancing are preserved only in the official-protocol track.
- Public `borealtc.py` semantics remain separate from the paper preprocessing pipeline.

### Released reference results

- CNN accuracy: `93.96%`.
- Mamba accuracy: `93.68%`.
- The exact per-class metrics and committed result identities are those closed in Stage 0B.1.

---

## 3. Environment and compatibility policy

### 3.1 Exact declared upstream stack

The locked upstream environment declares, among other packages:

```text
numpy==1.26.4
pandas==2.2.0
pyarrow==15.0.0
scipy==1.12.0
scikit-learn==1.4.0
torchvision==0.17.0
lightning==2.2.0
mamba-ssm==1.2.0.post1
causal_conv1d==1.2.0.post1
```

The upstream Dockerfile uses CUDA 12.2-era assumptions and an architecture list ending at Ampere-class compute capabilities. The current host must not be assumed compatible merely because `nvidia-smi` works.

### 3.2 Three environment labels

Every result must use one of the following labels.

- `EXACT_DECLARED_STACK`: all material versions and build semantics match upstream.
- `FAITHFUL_COMPATIBILITY_STACK`: model architecture, data pipeline, loss, optimizer, scheduler, split, and evaluation semantics are unchanged, but runtime packages are updated solely for host/GPU compatibility. Every difference must be enumerated.
- `DIAGNOSTIC_MODERN_BASELINE`: a modern reimplementation or replacement model used only for diagnostic comparison. It is not an upstream reproduction.

No result from a compatibility stack may be described as exact-environment reproduction.

### 3.3 No silent upstream patching

- Never modify `third_party/BorealTC`.
- Compatibility code belongs under tracked `tools/borealtc/compat/` or in a hash-recorded disposable run copy.
- A compatibility adapter may alter import/API glue, device selection, checkpoint key mapping, or deterministic-runtime settings only when the numerical model and training semantics remain unchanged.
- Replacing Mamba with an LSTM, Transformer, newer Mamba architecture, or arbitrary state-space model is not a compatibility adapter.
- If released semantics cannot execute faithfully, close the corresponding item as `BLOCKED`; do not force a pass.

---

## 4. Stage 0B.2A — Runtime, checkpoint, and feasibility closure

### Decision question

Can the released CNN and Mamba code paths be loaded, run forward/backward, and prepared for full reproduction on the current host, and what exact compatibility deviations are required?

### Mandatory host inventory

Record:

- WSL distribution and kernel;
- GPU model, driver, memory, and compute capability;
- `nvidia-smi` CUDA compatibility level;
- installed `nvcc` and toolkit identities, if any;
- Python and compiler identities;
- free disk space before and after;
- Docker availability and Docker storage root, without pulling an image unless its storage location is proven to remain outside the WSL Linux filesystem.

### Exact-stack feasibility probe

For each material dependency, record whether an installable artifact exists and whether it builds/runs on the host. A failed exact-stack probe must retain full logs. Do not repeatedly rebuild the same failed extension without changing a documented hypothesis.

### Released checkpoint audit

For `checkpoints/mamba_borealtc.ckpt`:

- record Git object id, SHA256, file size, Lightning/checkpoint metadata, hyperparameters, state-dict key inventory, parameter count, and expected architecture;
- determine whether the checkpoint represents a single fold, a transfer model, a final model, or an unresolved artifact;
- do not assign a fold identity without source evidence;
- load it and execute inference only if the architecture and environment can be reconstructed faithfully;
- a checkpoint smoke cannot reproduce Table III unless the released artifact demonstrably represents all five folds.

### CNN and Mamba training smoke

Using real BorealTC data and the official paper pipeline:

- construct one official fold;
- run at least one training batch and one validation batch for CNN;
- run at least one training batch and one validation batch for Mamba if a faithful runtime exists;
- verify finite loss, finite gradients, optimizer update, deterministic seed capture, input shape, label domain, and model output shape;
- record peak CPU RAM, GPU VRAM, initialization time, per-step time, and projected five-fold storage/time;
- do not run full epochs or a full fold.

### Stage 0B.2A pass conditions

A full pass requires:

1. exact host/GPU/runtime inventory;
2. clean, pinned upstream throughout;
3. checkpoint identity and semantics audited to the maximum permitted by released evidence;
4. CNN faithful forward/backward smoke;
5. Mamba faithful forward/backward smoke, **or** an explicit `BLOCKED_MAMBA` substatus supported by build/runtime evidence;
6. no substitute architecture presented as Mamba;
7. a concrete, bounded Stage 0B.2B execution plan with estimated runtime and storage;
8. all runs start from clean committed CryoLocoManip code and contain full manifests.

A blocked Mamba runtime does not invalidate the CNN or strict-generalization diagnostics, but it prevents claiming complete CNN+Mamba training reproduction.

---

## 5. Stage 0B.2B — Official-protocol training reproduction

### Purpose

Regenerate model predictions from training under the exact Stage 0B.1 official protocol.

### Required models

- CNN with the released Hamming/spectrogram configuration.
- Mamba with the released `optim2` architecture and training configuration, only if 0B.2A established a faithful runtime.

A computationally cheap SVM or LSTM may be added as a diagnostic anchor, but cannot replace either required paper model.

### Split and seed policy

- Split membership remains fixed at `RANDOM_STATE=21` and must match Stage 0B.1 hashes.
- Canonical model seed: `21`.
- Stability seeds: at least two additional predeclared seeds, while preserving the same split.
- Hyperparameters are locked before the final runs.
- No Optuna search or test-set-driven selection is permitted.

### Validation and test separation

- Validation samples are drawn from training partitions only, following released semantics for the official track.
- Test data may not determine early stopping, checkpoint choice, calibration, or model selection beyond the upstream protocol already being reproduced.
- Any known upstream test-label dependence in balancing remains documented as part of the official protocol and is removed only in the separate diagnostic track.

### Reproduction equivalence

Because training is stochastic, exact equality to released prediction arrays is not required. For each model report:

- canonical-seed accuracy and per-class metrics;
- three-seed median, range, and confidence interval where meaningful;
- difference from the paper/released result;
- fold-level metrics and failed folds;
- training duration, peak memory, and checkpoint identities.

A model is `REPRODUCED_TRAINING` when:

- all five folds train and evaluate without semantic substitutions;
- the canonical-seed aggregate accuracy is within 2.0 percentage points of the released value;
- the median across declared seeds is within 2.0 percentage points of the released value;
- no single class F1 differs by more than 5.0 percentage points without a traced explanation;
- the released value is compatible with the observed seed/fold variability rather than reached by post-hoc seed selection.

If these conditions are not met, report the numerical gap and close as partial or failed; do not tune against the released test result.

---

## 6. Stage 0B.2C — Group-disjoint generalization and calibration

### Purpose

Measure what BorealTC models retain when the test unit is an unseen acquisition run. This protocol is diagnostic and must never be described as the paper protocol.

### Primary group-disjoint protocol

- Group identity: `(terrain_class, raw_run_id)`.
- Primary split: deterministic five-fold `StratifiedGroupKFold`, `shuffle=True`, `random_state=21`, or a documented equivalent if the exact scikit-learn API cannot satisfy the class/run structure.
- No group may occur in more than one of train, validation, or test for a fold.
- Preserve natural test frequencies; no test oversampling.
- Training-only augmentation and class balancing may be used, with all parameters computed from the training groups only.
- Validation groups must be disjoint from both train and test groups.
- The exact assignment and its imbalance must be committed before model training.

Because run sizes are highly unequal, no claim of perfect fold stratification is allowed. Report per-fold class support and largest-run influence.

### Robustness protocol

At least for a computationally cheap baseline, repeat group-disjoint evaluation across multiple fixed split seeds or perform leave-one-run-out aggregation. This quantifies split sensitivity caused by the small number and unequal size of runs.

### Required model comparisons

At minimum:

- SVM or another lightweight reproducible baseline;
- CNN;
- Mamba if faithfully available;
- IMU-only, wheel-service-only, and fused observations for at least one trainable model.

The official-protocol model and group-disjoint model are retrained separately. Applying an official model to a different split without retraining is not a valid grouped-run result.

### Required metrics

Report at both window and raw-run levels:

- accuracy;
- macro-F1, weighted-F1, balanced accuracy;
- per-class precision, recall, F1, and support;
- confusion matrices;
- negative log-likelihood;
- multiclass Brier score;
- expected calibration error with declared bins;
- maximum calibration error;
- risk–coverage/selective-classification curves;
- class-conditional calibration;
- inference latency and memory;
- all failed or excluded runs.

Run-level aggregation must give each raw acquisition run an explicit, declared weight. Provide both equal-run and sample-weighted summaries.

### Calibration policy

- Any temperature scaling or calibration model is fitted on group-disjoint validation data only.
- Report pre- and post-calibration metrics.
- Calibration must not alter the predicted class when evaluating pure temperature scaling.
- A confidence threshold selected using the final test set is forbidden.

### Statistical policy

- Use raw run as the primary bootstrap/resampling unit.
- Preserve paired run identities when comparing models trained/evaluated on the same split.
- Report confidence intervals for primary metrics and for the drop from official-protocol performance.
- Do not treat thousands of overlapping 1.7 s windows as independent trials.

### Interpretation boundary

A large official-to-grouped performance drop would show that run-level generalization is not closed by the released dataset/protocol. It would **not** by itself prove a new architecture is required, nor validate action-conditioned terrain interaction, quadruped transfer, support–manipulation coupling, or thesis novelty.

---

## 7. Required run structure

Use separate roots:

```text
runs/stage0b2a/borealtc_runtime_feasibility/<run_id>/
runs/stage0b2b/borealtc_official_training/<run_id>/
runs/stage0b2c/borealtc_grouped_generalization/<run_id>/
```

Each formal run follows `docs/REPRODUCIBILITY_CONTRACT.md` and includes, where applicable:

```text
manifest.json
git_state.txt
command.sh
status.json
environment.txt
requirements.freeze.txt
config.resolved.yaml
stdout.log
stderr.log
metrics.raw.csv
failed_cases.csv
sha256_manifest.txt
```

Training runs additionally include fold/seed checkpoint identities and immutable raw predictions. Aggregation occurs only through committed evaluators.

---

## 8. Storage and host policy

All WSL assets remain under:

```text
/mnt/g/CryoLocoManip
```

Before installing or training, Stage 0B.2A must produce a storage budget covering:

- environments;
- pip/Conda caches;
- checkpoints;
- logs;
- spectrogram/preprocessed caches;
- all fold/seed runs;
- a minimum free-space reserve.

Do not use Docker when its image/storage root would silently consume the WSL Linux virtual disk. Do not duplicate the BorealTC dataset.

Isaac Lab/Isaac Sim and native-Ubuntu multi-host execution are outside Stage 0B.2. They will receive a separate contract before Stage 0D. MuJoCo availability in WSL does not affect this terrain-classification stage.

---

## 9. Terminal states

### Stage 0B.2A

- `PASS_BOREALTC_RUNTIME_FEASIBILITY`
- `PASS_WITH_BLOCKED_MAMBA_BOREALTC_RUNTIME_FEASIBILITY`
- `BLOCKED_BOREALTC_RUNTIME_FEASIBILITY_<REASON>`
- `FAIL_BOREALTC_RUNTIME_FEASIBILITY_<REASON>`

### Stage 0B.2B

- `PASS_BOREALTC_OFFICIAL_TRAINING_REPRODUCTION`
- `PASS_WITH_BLOCKED_MAMBA_BOREALTC_OFFICIAL_TRAINING_REPRODUCTION`
- `PARTIAL_BOREALTC_OFFICIAL_TRAINING_REPRODUCTION`
- `BLOCKED_BOREALTC_OFFICIAL_TRAINING_REPRODUCTION_<REASON>`
- `FAIL_BOREALTC_OFFICIAL_TRAINING_REPRODUCTION_<REASON>`

### Stage 0B.2C

- `PASS_BOREALTC_GROUPED_GENERALIZATION_DIAGNOSTIC`
- `PARTIAL_DIAGNOSTIC_ONLY_BOREALTC_GROUPED_GENERALIZATION`
- `BLOCKED_BOREALTC_GROUPED_GENERALIZATION_<REASON>`
- `FAIL_BOREALTC_GROUPED_GENERALIZATION_<REASON>`

---

## 10. Stage-level evidence boundary

Completion of Stage 0B.2 may establish:

- the reproducibility or incompatibility of released BorealTC training artifacts on the available host;
- from-scratch reproduction of released CNN/Mamba training under a declared runtime;
- the magnitude and uncertainty of the within-run versus group-disjoint performance gap;
- confidence calibration and selective-classification behavior for a wheeled Husky terrain-label task.

It cannot establish:

- quadruped foot–snow contact estimation;
- action-conditioned slip, sinkage, or support prediction;
- mechanical-arm reaction-force coupling;
- whole-body-control benefit;
- VLA necessity;
- cross-season action-validity memory;
- general polar deployment readiness;
- thesis novelty.

No new CryoLocoManip model may be introduced before Stage 0B.2 is closed or explicitly terminated.