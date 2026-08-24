# Stage 0 Plan — Thesis-Spine Falsification and Baseline Closure

## Objective

Stage 0 determines whether CryoLocoManip should continue as a legged mobile-manipulation thesis branch. It does not optimize a new method before the strongest relevant public baselines are understood.

## Stage 0A — Workspace and evidence preflight

Deliverables:

- canonical checkout at `/mnt/g/CryoLocoManip` only;
- WSL, filesystem, Git, GPU, driver, CUDA, Python, ROS, Docker, and storage inventory;
- local path configuration under ignored `configs/local/`;
- reproducibility scripts and a clean initial commit;
- literature inventory schema and source-quality fields.

Exit criteria:

- G-drive mount is real and writable;
- no second maintained checkout exists in the Linux filesystem;
- SSH fetch/push works;
- all large-output paths resolve below `/mnt/g/CryoLocoManip` or another explicitly approved G-drive path.

## Stage 0B — BorealTC formal reproduction

Purpose: establish the public proprioceptive terrain-classification baseline and document the gap between terrain labels and action outcomes.

Required closure:

- paper/code/data version lock;
- complete dataset inventory and hashes where practical;
- official preprocessing/window contract;
- official split reconstruction;
- checkpoint evaluation;
- at least two trainable baselines from the repository;
- class-level confusion, calibration, cross-platform/combined-data behavior, runtime, and failed-run records;
- comparison against reported paper/repository numbers;
- explicit statement of what the data cannot support: no quadruped contact, no arm reaction force, no paired counterfactual candidate actions, and no seasonal revisit contract.

Decision: this stage cannot validate the thesis; it only closes an existing public capability.

## Stage 0C — Action-conditioned terrain-interaction baseline

Reproduce one sufficiently open baseline such as FusionForce, MonoForce, or an equivalent forward-dynamics/terrain-interaction model.

Required tests:

- official checkpoint or principal result reproduction;
- unseen control sequence;
- unseen terrain condition;
- action ranking, not only trajectory MSE;
- pure learned versus physics-residual or hybrid model where supported;
- uncertainty and failure-mode audit.

Gate:

- if candidate actions cannot be ranked more reliably than terrain class plus a strong conservative controller, the action-conditioned prediction branch is weakened.

## Stage 0D — Legged mobile-manipulation baseline

Reproduce one open quadruped-manipulator stack such as RAMBO or a comparable implementation.

Required tasks:

- standing manipulation;
- moving reach;
- push/pull;
- payload carriage;
- external disturbance;
- friction and payload sweeps;
- arm-only/decoupled, nominal whole-body, and robust baseline comparisons where implementable.

Gate:

- establish what current robust WBC/RL already solves on rigid or simplified support before introducing snow–ice models.

## Stage 0E — Support–manipulation coupling falsification benchmark

Only starts after 0B–0D are auditable.

Minimum factors:

- spatially varying friction;
- compliance and sinkage;
- brittle crust or support-capacity threshold;
- asymmetric support;
- slope;
- payload;
- end-effector force direction and magnitude;
- sensing noise/delay;
- structural model mismatch.

Minimum methods:

1. arm-only or decoupled control;
2. nominal WBC;
3. robust WBC;
4. terrain-estimate-conditioned WBC;
5. action-outcome prediction plus WBC;
6. privileged-parameter oracle.

Primary metrics:

- progress–risk Pareto frontier;
- tail failure/CVaR;
- slip, sinkage, crust failure, and support-margin violations;
- end-effector force/pose error;
- energy;
- unsafe authorization and unnecessary rejection;
- method ordering across simulators/contact models.

Hard gate:

- the snow–ice-aware branch must outperform strong robust WBC at matched progress, sensing, and compute budgets. Average success alone is insufficient.

## Stage 0F — VLA necessity test

This is conditional and cannot block the physical thesis branch.

Compare, with the same low-level skills and safety controller:

- finite-state or manually specified skill routing;
- symbolic planner plus open-vocabulary perception;
- frozen or lightly adapted VLM/VLA high-level interface.

Evaluate:

- unseen objects;
- paraphrased instructions;
- task recombination;
- multi-stage natural-language tasks;
- semantic ambiguity;
- latency and unsafe/invalid skill proposals.

Gate:

- if VLA does not materially improve open-vocabulary or compositional generalization, remove it from the core innovation claims.

## Stage 0G — bounded seasonal audit

Use FoMo/Boreas or another repeated-season resource only after interaction and control contracts are stable.

Question:

> Does an old map or prior interaction memory remain valid for current action selection, not merely current localization?

This is a later extension, not the first paper.

## Formal terminal states

Each stage must end as one of:

- `PASS` — all mandatory contracts and evidence satisfied;
- `NO_GO` — the hypothesis or method does not survive the declared comparison;
- `BLOCKED` — required data, code, hardware, or semantics are unavailable without inventing a replacement;
- `PARTIAL_DIAGNOSTIC_ONLY` — useful evidence obtained, but not sufficient for a method or thesis claim.