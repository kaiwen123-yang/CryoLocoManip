# CryoLocoManip

**Action-Conditioned Support–Manipulation Interaction Modeling and Trustworthy Whole-Body Control in Snow–Ice Environments**

CryoLocoManip is a reproducible research program for investigating how uncertain, deformable, and potentially brittle snow–ice support changes the feasible whole-body actions of legged mobile manipulators.

> Predict before contact, verify before action, and coordinate the whole body on uncertain snow–ice terrain.

## Current status

**Stage 0 — thesis-spine falsification and baseline closure.**

The repository does **not** yet claim that a snow–ice VLA, a new world model, or a terrain-aware whole-body controller is necessary. Stage 0 must first establish whether snow–ice support–manipulation coupling creates a scientifically distinct problem that strong robust whole-body control cannot already solve.

## Core candidate question

Given robot history `h_t`, candidate locomotion and manipulation actions `a_{t:t+H}`, embodiment `e`, and uncertain snow–ice support, can a robot:

1. predict action-conditioned interaction outcomes such as slip, sinkage, crust failure, support-margin loss, end-effector error, and energy cost;
2. determine when those predictions are sufficiently reliable to authorize the action;
3. coordinate base, feet, arm, and contact forces to improve the progress–risk Pareto frontier;
4. acquire additional evidence only when it can change the control decision?

## Stage 0 decision branches

- **Branch A — Coupling confirmed, VLA necessary:** support–manipulation prediction + trustworthy WBC + open-vocabulary task interface.
- **Branch B — Coupling confirmed, VLA unnecessary:** support–manipulation prediction + trustworthy WBC.
- **Branch C — Coupling not distinct or robust WBC is sufficient:** stop the mobile-manipulation thesis branch and return to action-conditioned terrain prediction, active traversability, seasonal interaction memory, or polar navigation integrity.

## Repository and storage contract

The canonical local checkout must be:

```text
/mnt/g/CryoLocoManip
```

which corresponds to:

```text
G:\CryoLocoManip
```

Do not create or maintain another project checkout under the WSL Linux filesystem, such as `~/CryoLocoManip`, `/home/<user>/CryoLocoManip`, or `/opt/CryoLocoManip`.

Large datasets, simulator assets, checkpoints, caches, and run outputs must also remain on the G drive and are excluded from Git. Only source code, compact configuration, contracts, small audit tables, and final reproducibility metadata should be committed.

## Initial evidence ladder

Stage 0 starts from existing public capabilities rather than a new method:

1. **BorealTC** — reproduce proprioceptive terrain-classification baselines and audit what terrain labels do *not* reveal about action outcomes.
2. **Action-conditioned terrain prediction** — reproduce one open baseline such as FusionForce/MonoForce/FDM and evaluate unseen actions and terrain shifts.
3. **Legged mobile-manipulation control** — reproduce one open WBC/RL baseline such as RAMBO or a comparable Go2 manipulator stack.
4. **Seasonal perception/localization** — reproduce a bounded FoMo/Boreas experiment after the interaction and control contracts are stable.
5. **Support–manipulation coupling benchmark** — only after the first three baselines are closed.
6. **VLA necessity test** — only after the physical coupling branch is validated.

## Non-goals for Stage 0

- training a foundation VLA from scratch;
- claiming a universal polar world model;
- unifying quadrupeds, humanoids, tracked vehicles, and drones in one controller;
- treating terrain classification accuracy as evidence of safe action selection;
- treating a custom low-fidelity simulator as proof of real snow–ice performance;
- selecting a research conclusion by tuning against the final evaluation set.

## Reproducibility principles

Every formal experiment must preserve:

- exact Git commit and dirty-state record;
- machine, OS, driver, CUDA, Python, ROS, and simulator versions;
- immutable data source identity and checksums where feasible;
- configuration and random seeds;
- raw per-run outputs before aggregation;
- evaluator version and metric definitions;
- failed runs and exclusions;
- a clear `GO`, `NO-GO`, or `BLOCKED` decision against predeclared criteria.

See `docs/PROJECT_CONTRACT.md`, `docs/STAGE0_PLAN.md`, and `docs/REPRODUCIBILITY_CONTRACT.md`.

## Remote

```text
git@github.com:kaiwen123-yang/CryoLocoManip.git
```

## License

No project-wide license has been selected yet. Third-party code and data must retain their original licenses and provenance.