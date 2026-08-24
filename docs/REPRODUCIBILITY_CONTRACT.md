# Reproducibility Contract

## 1. Canonical workspace

The only maintained checkout is:

```text
/mnt/g/CryoLocoManip
```

No experiment may silently read from or write to a second checkout under `/home`, `/root`, `/opt`, or another WSL Linux-filesystem path.

## 2. Local-only paths

Machine-specific paths belong in ignored files under:

```text
configs/local/
```

Use `configs/paths.local.example.yaml` as the public template. Never commit credentials, access tokens, private keys, personally identifying paths, or licensed dataset secrets.

## 3. Data and third-party sources

For each source, record:

- canonical paper and repository identity;
- commit, tag, release, or download date;
- license and redistribution constraints;
- raw file inventory;
- checksums where practical;
- any conversion or filtering step;
- whether the source is complete, partially released, or unavailable.

Do not silently replace missing data, code, labels, or simulator semantics. Mark the result `BLOCKED` or `PARTIAL_DIAGNOSTIC_ONLY` when a faithful replacement is not justified.

## 4. Run directory contract

Formal runs are stored below ignored `runs/` using:

```text
runs/<stage>/<method>/<run_id>/
```

Recommended `run_id`:

```text
YYYYMMDDTHHMMSSZ_<git-short-sha>_<config-id>_<seed>
```

Every run directory must contain, where applicable:

```text
manifest.json
config.resolved.yaml
environment.txt
git_state.txt
command.sh
stdout.log
stderr.log
metrics.raw.csv
status.json
```

Model checkpoints and large intermediate artifacts stay in the run directory or an explicitly referenced G-drive object store; they are not committed.

## 5. Manifest minimum fields

`manifest.json` must include:

- stage and method;
- Git commit and dirty status;
- command and working directory;
- hostname or anonymized machine ID;
- OS, kernel, Python, compiler, CUDA, driver, ROS, and simulator versions;
- dataset/config/evaluator identity;
- random seeds;
- start/end timestamps;
- exit code;
- declared exclusions or failed cases;
- parent or resumed run identity.

## 6. Evaluation contract

- Raw method outputs are immutable after evaluation begins.
- Evaluators are versioned and never use reference answers to select coordinate signs, offsets, thresholds, contacts, or model variants.
- Hyperparameter selection, validation, and final test sets remain separate.
- Report both central performance and tail/safety metrics.
- Report all failed, timed-out, and excluded runs with reasons.
- Aggregate tables must be reproducible from raw per-run outputs by a committed script.

## 7. Comparison fairness

A method comparison must state whether methods have matched:

- observations and privileged information;
- action and control rates;
- task progress/time budget;
- sensor and active-probing budget;
- compute budget;
- training data and augmentation;
- safety margin and termination policy.

A method that is slower, more conservative, or uses additional sensing is not automatically superior. The relevant comparison is the progress–risk–energy frontier.

## 8. Statistical contract

For formal claims:

- use multiple independent seeds or repeated trials;
- preserve paired scenario identities where possible;
- report confidence intervals or another declared uncertainty summary;
- include tail failure and unsafe-authorization metrics;
- avoid declaring significance from a single seed or a few successful demonstrations.

## 9. Claim ledger

Each report must distinguish:

- `REPRODUCED`: independently recomputed from released artifacts;
- `AUDITED`: code/data/metric contract checked but full result not retrained;
- `CLAIMED_BY_SOURCE`: reported only by the original authors;
- `INFERENCE`: reasoned from evidence but not directly measured;
- `BLOCKED`: evidence unavailable;
- `DIAGNOSTIC_ONLY`: mechanism test that does not support a real-world claim.

## 10. Git discipline

- Do not commit generated datasets, checkpoints, simulator caches, or raw runs.
- Use small, logically scoped commits.
- Do not rewrite shared history without explicit approval.
- Record the exact commit for every formal run.
- Preserve third-party repositories under ignored `third_party/` with their own Git history or as explicitly reviewed submodules; do not copy untracked source fragments into the core codebase without provenance.

## 11. Stage decision

Every stage closes with a machine-readable `status.json` and a human-readable report whose terminal state is exactly one of:

```text
PASS
NO_GO
BLOCKED
PARTIAL_DIAGNOSTIC_ONLY
```

No stage advances merely because a demo runs.