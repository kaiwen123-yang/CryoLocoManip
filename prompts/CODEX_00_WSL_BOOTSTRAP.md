# Codex Task 00 — Bootstrap the Canonical WSL/G-Drive Workspace

## Role

You are the execution agent for the CryoLocoManip research repository. This task is operational and reproducibility-focused. Do not reinterpret the scientific hypothesis, add a new algorithm, download public datasets, install a heavy simulator, or begin model training.

## Authoritative repository and workspace

```text
Remote: git@github.com:kaiwen123-yang/CryoLocoManip.git
Windows path: G:\CryoLocoManip
WSL path: /mnt/g/CryoLocoManip
Default branch: main
```

The canonical checkout must be `/mnt/g/CryoLocoManip`. Do not create a maintained checkout, worktree, environment, dataset, cache, or run directory under `/home`, `/root`, `/opt`, or the WSL Linux filesystem.

## Safety and non-destructive rules

1. Do not use `git reset --hard`, `git clean -fdx`, destructive removal, or force-push.
2. Do not overwrite a non-empty `/mnt/g/CryoLocoManip` directory that is not the expected repository.
3. If the path already contains uncommitted work, preserve it and stop with `BLOCKED_EXISTING_LOCAL_CHANGES` unless it is obviously the clean expected clone.
4. Do not change global Git, SSH, WSL, CUDA, Docker, or system package settings.
5. Do not install packages in this task.
6. Do not download datasets, weights, simulator assets, or third-party repositories.
7. Do not merge a pull request or tag a release.

## Step 1 — Verify WSL and G-drive mount

Run and record:

```bash
uname -a
cat /etc/os-release
printf 'WSL_DISTRO_NAME=%s\n' "${WSL_DISTRO_NAME:-unset}"
findmnt -T /mnt/g
stat -f -c '%T' /mnt/g
df -h /mnt/g
```

Required:

- `/mnt/g` exists and is writable;
- it resolves to the Windows G drive rather than the WSL root filesystem;
- enough free space exists for future datasets and simulation assets.

If not satisfied, stop with `BLOCKED_G_DRIVE_MOUNT` and make no repository changes.

## Step 2 — Verify SSH without changing configuration

Run:

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -T git@github.com 2>&1 | tee /tmp/cryolocomanip_ssh_check.txt
```

GitHub normally returns a non-zero shell exit even when authentication succeeds. Treat the check as successful only if the output explicitly identifies successful GitHub authentication for the expected user. Otherwise stop with `BLOCKED_GITHUB_SSH`.

## Step 3 — Establish the canonical checkout

Handle the path conservatively:

- If `/mnt/g/CryoLocoManip` does not exist, clone directly there.
- If it exists and is empty, clone into it.
- If it is already the expected Git repository, inspect it; do not reclone.
- If it is non-empty and not the expected repository, stop with `BLOCKED_PATH_COLLISION`.

Canonical command when cloning is required:

```bash
git clone git@github.com:kaiwen123-yang/CryoLocoManip.git /mnt/g/CryoLocoManip
```

Then:

```bash
cd /mnt/g/CryoLocoManip
git remote -v
git fetch --prune origin
git status --short --branch
git log -n 5 --oneline --decorate
```

The origin SSH URL must match exactly. Do not add a second remote unless the task is later amended.

## Step 4 — Run the committed workspace preflight

From the repository root:

```bash
bash scripts/preflight_wsl.sh
```

The script must return `PRECHECK_PASS`. Preserve its ignored local report under `artifacts/runtime/`.

Inspect and read before editing:

```text
README.md
docs/PROJECT_CONTRACT.md
docs/STAGE0_PLAN.md
docs/REPRODUCIBILITY_CONTRACT.md
docs/REPOSITORY_LAYOUT.md
configs/paths.local.example.yaml
```

## Step 5 — Create an isolated bootstrap branch

Only after the checkout is clean and preflight passes:

```bash
git switch -c stage0/bootstrap-wsl
```

If that branch already exists locally or remotely, stop and report it rather than deleting or overwriting it.

## Step 6 — Add only the minimum bootstrap implementation

Create the following committed files:

### `pyproject.toml`

Provide a minimal Python project definition for `cryolocomanip` with Python `>=3.10`. Add only lightweight development dependencies for later use, such as `pytest`, `ruff`, `pyyaml`, and `jsonschema`; do not install them now. Configure Ruff and pytest conservatively.

### `src/cryolocomanip/__init__.py`

Expose only a package version such as `0.0.0`. Do not add scientific implementation.

### `scripts/collect_environment.sh`

Create a non-destructive script that writes a timestamped environment inventory below ignored `artifacts/runtime/`. It should collect, when available:

- OS and kernel;
- WSL distro;
- mount and disk information for `/mnt/g`;
- Git version and repository state;
- Python and Conda availability;
- compiler/CMake;
- NVIDIA driver, GPU, CUDA and `nvcc`;
- Docker;
- ROS distribution and installed ROS packages at a summary level;
- system memory and CPU summary.

Missing optional tools must be reported as `unavailable`, not treated as fatal.

### `scripts/verify_workspace.py`

Create a standard-library-only validator that:

- confirms the repository root resolves exactly to `/mnt/g/CryoLocoManip`;
- confirms origin is the exact SSH remote;
- confirms required local directories exist on `/mnt/g`;
- confirms `configs/local/paths.local.yaml` resolves all project paths below `/mnt/g/CryoLocoManip` unless explicitly allow-listed;
- exits non-zero on violations;
- never modifies files.

### `tests/test_repository_contract.py`

Add lightweight standard-library or pytest-compatible tests for path-template parsing and repository layout assumptions. Tests must not depend on machine-specific local files or heavy packages.

### `literature/README.md`

Define the planned evidence matrix fields without populating unverified papers. Include at least: identity, year, venue, publication status, source tier, robot embodiment, environment, state, observation, action, uncertainty, active interaction, datasets, code, reproduced status, limitations, future work, direct overlap, and claim level.

### `schemas/run_manifest.schema.json`

Define a minimal JSON Schema for the run-manifest requirements in `docs/REPRODUCIBILITY_CONTRACT.md`.

Do not create empty experiment directories simply to imitate the target layout.

## Step 7 — Validate locally

Run:

```bash
python3 scripts/verify_workspace.py
bash scripts/collect_environment.sh
python3 -m compileall src scripts tests
python3 -m unittest discover -s tests -p 'test_*.py' -v
bash scripts/preflight_wsl.sh
git diff --check
git status --short
```

If `pytest` or Ruff is already available, they may also be run, but do not install them in this task.

## Step 8 — Commit and push the branch

Before committing, inspect the complete diff. Ensure no local path secrets, credentials, generated runtime inventories, caches, or large files are staged.

Commit subject:

```text
Bootstrap canonical WSL research workspace
```

Push only the branch:

```bash
git push -u origin stage0/bootstrap-wsl
```

Do not merge to `main` and do not create a tag.

## Required final report

Return a concise but complete terminal report containing:

1. terminal state: `PASS_BOOTSTRAP_WSL`, `BLOCKED_*`, or `FAIL_*`;
2. WSL distro, kernel, filesystem type, and G-drive free space;
3. canonical repository path and origin;
4. start and end HEADs;
5. branch and commit SHA;
6. exact commands executed;
7. validation results;
8. files added or modified;
9. warnings, missing optional tools, or possible duplicate checkouts;
10. remote branch URL or identity;
11. confirmation that no dataset, simulator, checkpoint, or heavy environment was installed;
12. confirmation that no checkout or project data was created under the WSL Linux filesystem.

Do not continue to BorealTC or any other baseline in this task.