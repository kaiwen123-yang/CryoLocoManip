# CODEX_01R — BorealTC Exact-Version and Clean-Commit Corrective Rerun

## Purpose

This is a narrow external-review correction to Stage 0B.1. The existing audit logic and substantive findings are retained. Do **not** expand scope, retrain models, or add a new method.

The correction closes two reproducibility gaps:

1. the split/pipeline audit was executed with `scikit-learn 1.7.2`, while the locked upstream `requirements-lock.txt` specifies `scikit-learn==1.4.0` together with `numpy==1.26.4`, `pandas==2.2.0`, and `scipy==1.12.0`;
2. the formal run was created before the audit implementation was committed, so its run id and manifest point to base commit `767f1c5` rather than a clean commit containing the executed audit code.

A corrected pass requires an exact-version split reconstruction and a fresh formal run from a clean committed tree.

## Authoritative documents

Read in this order:

1. `docs/STAGE0B1_BOREALTC_SOURCE_CLOSURE.md`
2. `docs/REPRODUCIBILITY_CONTRACT.md`
3. `docs/audits/BOREALTC_SOURCE_CLOSURE.md`
4. `prompts/CODEX_01_BOREALTC_SOURCE_CLOSURE.md`
5. this corrective prompt

This prompt narrows and strengthens the execution evidence; it does not alter the scientific scope or the locked upstream identities.

## Terminal states

Use exactly one:

- `PASS_BOREALTC_SOURCE_CLOSURE`
- `BLOCKED_BOREALTC_SOURCE_CLOSURE_SKLEARN_PIN`
- `BLOCKED_BOREALTC_SOURCE_CLOSURE_CLEAN_RUN`
- `FAIL_BOREALTC_SOURCE_CLOSURE_VERSION_DIVERGENCE`
- `FAIL_BOREALTC_SOURCE_CLOSURE_REPRODUCIBILITY`

The report must include `review_revision=R1`.

---

## 0. Storage and Git rules

Canonical checkout:

```text
/mnt/g/CryoLocoManip
```

Remote:

```text
git@github.com:kaiwen123-yang/CryoLocoManip.git
```

All environments, caches, temporary files, and run outputs must remain below `/mnt/g/CryoLocoManip`. Do not create or retain any project asset under `/home`, `~`, `/root`, `/opt`, or another WSL Linux-filesystem path.

Start with:

```bash
cd /mnt/g/CryoLocoManip
git fetch origin --prune
git switch stage0/borealtc-source-closure
git pull --ff-only origin stage0/borealtc-source-closure
```

Record start HEAD. Before the formal corrected run, the worktree must be clean:

```bash
test -z "$(git status --porcelain)"
```

The pinned upstream checkout must remain detached, unmodified, and clean at:

```text
/mnt/g/CryoLocoManip/third_party/BorealTC
0146dcd9fa08c34a9075a80448ac04c0a947b568
```

Do not merge, tag, force-push, rewrite history, or open a PR.

---

## 1. Exact lightweight core environment

Create a separate minimal environment below G drive:

```text
/mnt/g/CryoLocoManip/cache/venvs/borealtc-source-closure-pin
```

Use Python 3.10 and `venv --copies`; apply the already documented drvfs `lib64` workaround if required. Redirect `TMPDIR`, `XDG_CACHE_HOME`, and `PIP_CACHE_DIR` below `/mnt/g/CryoLocoManip/cache/`.

Install only the upstream core versions needed for source/pipeline/split/evaluator closure:

```text
numpy==1.26.4
pandas==2.2.0
pyarrow==15.0.0
scipy==1.12.0
scikit-learn==1.4.0
tqdm==4.66.2
```

Do not install or change:

- `mamba-ssm`;
- `causal-conv1d`;
- Lightning;
- Optuna;
- CUDA extensions;
- simulators;
- ROS;
- VLA dependencies;
- any model-training dependency not required by the scripts above.

The existing Stage 0B.1 environment may be used only for the already-closed public `borealtc.py`/Torch API smoke. Do not recreate an 8 GB Torch environment merely for this correction.

Save exact freezes for both environments when used.

---

## 2. Exact-version split and pipeline closure

Using the pinned core environment, execute the committed audit code without modification:

```bash
python -m tools.borealtc.inventory
python -m tools.borealtc.reconstruct_paper_pipeline
python -m tools.borealtc.audit_splits
python -m tools.borealtc.evaluate_committed_results --pipeline-json <new_pipeline_json> --upstream-smoke
```

Use the current scripts' required arguments and formal run directory contract; the pseudocommands above state the required modules, not permission to omit mandatory flags.

Create:

```text
version_pin_comparison.json
```

It must compare the prior run and exact-pin rerun for at least:

- pre-split order SHA256;
- all train/test fold membership SHA256 values;
- per-fold partition counts;
- run-overlap percentages;
- temporal-adjacency counts;
- augmentation slides/strides;
- per-fold test-window counts;
- total test windows;
- CNN and Mamba independent metrics;
- confusion matrices;
- upstream evaluator byte-identity results.

Required result:

- every split membership hash and every derived split statistic must match the prior run exactly;
- every metric must remain within the existing declared tolerance;
- any divergence is `FAIL_BOREALTC_SOURCE_CLOSURE_VERSION_DIVERGENCE`, not a warning.

If exact upstream versions cannot be installed or imported without inventing policy, return `BLOCKED_BOREALTC_SOURCE_CLOSURE_SKLEARN_PIN`.

---

## 3. Fresh formal run from a clean committed tree

The corrected formal run must begin while the CryoLocoManip worktree is clean and must use the current committed HEAD containing all audit code.

The new run id must include that clean commit's short SHA, not `767f1c5`.

Required run root:

```text
runs/stage0b1/borealtc_source_closure/<new_run_id>/
```

In addition to all Stage 0B.1 outputs, the corrected run must contain:

```text
manifest.json
git_state.txt
command.sh
status.json
environment_core_pin.txt
environment_api_smoke.txt        # only if the old Torch environment is reused
requirements_core_pin.freeze.txt
requirements_api_smoke.freeze.txt # only if applicable
version_pin_comparison.json
stdout.log
stderr.log
sha256_manifest.txt
```

Mandatory Git state:

```text
manifest.git.commit = the clean execution commit
manifest.git.dirty = false
```

`git_state.txt` must include repository root, branch, HEAD, origin, `git status --porcelain`, and the pinned upstream HEAD/clean state.

`command.sh` must contain the exact executed commands in order and be non-secret.

`status.json` must state the terminal status, review revision `R1`, start/end timestamps, run id, exit code, mandatory checks, warnings, and evidence boundary.

The SHA256 manifest must cover all formal run outputs except itself and may additionally cover the committed audit scripts used by the run.

If any audit code must be changed to correct a bug:

1. make the code change;
2. add/adjust tests;
3. commit and push the code change first;
4. confirm a clean worktree;
5. start a completely new formal run from that clean commit;
6. only then make a later documentation-only commit referring to the clean run.

Do not claim a formal rerun from uncommitted code.

---

## 4. Public API smoke

The public `borealtc.py` API smoke does not need to be repeated in the exact core environment because it requires Torch and was already closed. It must, however, be included in the final evidence chain by either:

- re-running it in the existing G-drive Torch environment from the same clean project commit; or
- explicitly linking and hash-verifying the prior API-smoke output while labeling it as a parent diagnostic run.

Do not merge the public API semantics with the paper preprocessing semantics.

---

## 5. Required tracked updates after the clean rerun

After the formal clean run passes, update only the evidence references needed to point to it:

- `docs/audits/BOREALTC_SOURCE_CLOSURE.md`;
- `literature/evidence_matrix.csv`;
- `literature/notes/larocque2024_borealtc.md`.

Required additions:

- exact upstream core version closure;
- old-versus-new membership-hash comparison result;
- clean execution commit and new run id;
- `dirty=false` statement;
- retained limitation that model training itself has not been rerun;
- unchanged evidence boundary: no quadruped contact, action-conditioned outcomes, support–manipulation coupling, WBC improvement, VLA necessity, seasonal action memory, or thesis novelty.

Documentation must not silently overwrite the historical first run. Preserve it as the parent/preliminary audit and identify the corrected run as the authoritative Stage 0B.1 run.

Commit the documentation updates and any new/updated tests as one logically scoped commit, then push:

```text
origin/stage0/borealtc-source-closure
```

Do not open a PR, merge, or tag.

---

## 6. Validation

At minimum run:

```bash
bash scripts/preflight_wsl.sh
python3 scripts/verify_workspace.py
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m compileall -q src scripts tools tests
```

Run the scientific audit tests in the exact pinned environment as well. Record pass counts and all exclusions.

The final repository worktree must be clean after the final commit.

---

## 7. Final report

Return:

1. exact terminal state;
2. `review_revision=R1`;
3. start HEAD, clean-run execution HEAD, and final documentation commit;
4. exact pinned and API-smoke environment identities;
5. old and new run ids;
6. membership-hash comparison and any divergence;
7. Table I and Table III closure summary;
8. split/run/adjacency findings;
9. public API evidence link/re-execution status;
10. required run files and hashes;
11. tests and commands;
12. storage confirmation and disk usage;
13. warnings/exclusions;
14. exact evidence boundary;
15. branch push confirmation and clean final status.

A corrected pass is allowed only when exact-version split identity and clean-commit run identity are both closed.