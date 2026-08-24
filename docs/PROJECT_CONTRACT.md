# CryoLocoManip Project Contract

## Status

The thesis direction is **not frozen**. CryoLocoManip is currently a falsification program for deciding whether snow–ice support–manipulation coupling is a scientifically distinct and practically solvable problem for legged mobile manipulators.

## Candidate scientific hypothesis

For a legged mobile manipulator operating on uncertain snow–ice support, manipulation forces and payload-induced whole-body redistribution can change slip, sinkage, brittle-crust failure, support margins, and feasible task forces in ways that are not adequately handled by ordinary rigid-ground whole-body control or a single terrain-class label.

A useful method should therefore predict and control the consequences of a **candidate combined locomotion–manipulation action**, rather than only classify terrain or track an end-effector trajectory.

## Primary embodiment and task boundary

Stage 0 and the first formal papers use one primary embodiment:

- a quadruped base;
- one 5–7 DoF arm;
- a gripper or probing end effector.

The initial task family is deliberately bounded:

1. stationary probing or scraping with controlled normal/tangential force;
2. pushing or pulling under asymmetric support;
3. carrying an uncertain payload across weak or deformable support.

Humanoids, dexterous hands, heterogeneous teams, and full polar-science mission planning are not dependencies of the core thesis.

## Required comparison question

Every proposed snow–ice-aware component must answer:

> Does it improve the task-progress versus tail-risk Pareto frontier beyond strong robust WBC/MPC under matched sensing, compute, data, and action budgets?

Improvements caused only by slower motion, larger safety margins, additional sensors, or privileged simulator state do not establish the hypothesis.

## Conditional VLA branch

VLM/VLA methods are not assumed necessary. They may enter only as a high-level open-vocabulary task and skill interface after the physical coupling and control branches are validated.

A VLA branch proceeds only if it materially outperforms strong finite-state, symbolic, or skill-routing baselines on unseen objects, paraphrased instructions, task recombination, or long-horizon instruction following while using the same low-level skills and safety controller.

## Decision outcomes

### GO-A — coupling confirmed, VLA necessary

Proceed with support–manipulation prediction, trustworthy whole-body control, and an open-vocabulary high-level interface.

### GO-B — coupling confirmed, VLA unnecessary

Proceed with support–manipulation prediction and trustworthy whole-body control. VLA remains an optional interface, not an innovation claim.

### NO-GO-C — coupling is not distinct

If strong robust WBC/MPC matches the snow–ice-aware method under fair budgets, stop this thesis branch. Re-evaluate action-conditioned traversability, active terrain probing, seasonal interaction memory, or polar navigation integrity.

## Prohibited claims before evidence closure

Do not claim:

- first universal polar VLA;
- a general snow–ice world model;
- safe autonomous operation from IID calibration alone;
- real snow–ice performance from a custom low-fidelity simulator;
- cross-embodiment generalization from robot-ID tokens alone;
- that terrain-classification accuracy establishes action safety;
- that more sensors constitute meaningful fusion without decision-level benefit.

## Canonical storage

The canonical WSL checkout is `/mnt/g/CryoLocoManip`, corresponding to `G:\CryoLocoManip`. No maintained checkout may exist under the WSL Linux filesystem.