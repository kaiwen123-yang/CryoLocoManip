# CryoLocoManip Environment Isolation and Host-Hygiene Contract

## Status

Mandatory for every local, WSL, native-Ubuntu, server, and Codex execution.

This contract protects the user's existing development systems from project-specific dependency drift. A stage may not be marked `PASS` if its runtime identity is unclear or if it modifies the host/base environment without an explicit, separately approved infrastructure task.

---

## 1. Core rule

Project dependencies must be installed into an isolated, stage-specific environment. The host operating system, system Python, Conda `base`, user site-packages, shell startup files, and globally installed CUDA/toolchain files are read-only unless a later prompt explicitly authorizes a host-infrastructure change.

For WSL, the canonical project root is:

```text
/mnt/g/CryoLocoManip
```

All project environments, package caches, source builds, model caches, temporary files, datasets, logs, and run artifacts must remain beneath that root.

---

## 2. Current Stage 0B.2A environment decision

The completed BorealTC runtime closure used project-local Python virtual environments:

```text
/mnt/g/CryoLocoManip/cache/venvs/borealtc-runtime-compat
/mnt/g/CryoLocoManip/cache/venvs/borealtc-runtime-declared-probe  # removed after audit
```

This is already isolated from:

- `/usr/lib/python*`;
- `/usr/local/lib/python*`;
- `~/.local/lib/python*`;
- Conda `base`;
- any unrelated ROS, navigation, MuJoCo, or robotics environment.

Do **not** migrate the accepted BorealTC compatibility environment to Conda merely for consistency. Environment migration would create a second unresolved runtime and invalidate the tested binary identity. Keep the accepted environment immutable for Stage 0B.2B reproduction.

---

## 3. Allowed environment types

### 3.1 Default: project-local `venv --copies`

Use for Python/pip projects when all required native dependencies can be built or installed faithfully. This is the preferred method for the accepted BorealTC stack because it is already tested and hash-recorded.

Required prefix:

```text
/mnt/g/CryoLocoManip/cache/venvs/<stage-or-method-name>
```

### 3.2 Conda or micromamba: conditional

Use only when a method genuinely requires Conda-distributed native libraries or when the upstream project declares Conda as the authoritative runtime.

Required locations:

```text
CONDA_ENVS_PATH=/mnt/g/CryoLocoManip/cache/conda/envs
CONDA_PKGS_DIRS=/mnt/g/CryoLocoManip/cache/conda/pkgs
```

Rules:

- never install project packages into Conda `base`;
- use an explicit prefix, not an implicit environment name resolving under `~/.conda`;
- record `conda list --explicit`, `conda env export --from-history`, and the exact solver identity;
- do not mix Conda and pip silently; pip-installed packages must be frozen separately;
- if DrvFS hardlink/symlink behavior makes the environment unreliable, return `BLOCKED_ENVIRONMENT_ON_DRVFS` rather than moving it into the WSL virtual disk without authorization.

### 3.3 Containers: not currently authorized in WSL

The current Docker data root is `/var/lib/docker`, inside the WSL virtual disk. Therefore Docker image pulls/builds are prohibited for CryoLocoManip in WSL until Docker's full data root is deliberately relocated or the task moves to native Ubuntu.

Native-Ubuntu Isaac Sim/Isaac Lab work will receive a separate multi-host and storage contract.

---

## 4. Forbidden host modifications

Unless a future infrastructure prompt explicitly authorizes them, all stages must forbid:

```text
sudo apt install / apt-get install
sudo pip / system pip
pip install --user
conda install into base
writing to /usr/local, /usr/lib, /opt, /etc
editing ~/.bashrc, ~/.profile, ~/.zshrc, ~/.condarc, /etc/environment
creating project source/data/envs under /home, ~, /root, /opt, /tmp
Docker pulls/builds while Docker root is inside the WSL VHD
changing the system CUDA symlink or compiler alternatives
```

The existing NVIDIA driver, `/usr/local/cuda`, compiler, CMake, Ninja, Git, SSH, and WSL GPU bridge may be used read-only.

---

## 5. Mandatory runtime redirections

Every stage must source a project-local environment script that sets at least:

```text
PIP_CACHE_DIR=/mnt/g/CryoLocoManip/cache/pip
XDG_CACHE_HOME=/mnt/g/CryoLocoManip/cache/xdg
TMPDIR=/mnt/g/CryoLocoManip/cache/tmp
TORCH_EXTENSIONS_DIR=/mnt/g/CryoLocoManip/cache/torch_extensions
CUDA_CACHE_PATH=/mnt/g/CryoLocoManip/cache/cuda
TRITON_CACHE_DIR=/mnt/g/CryoLocoManip/cache/triton
HF_HOME=/mnt/g/CryoLocoManip/cache/huggingface
TRANSFORMERS_CACHE=/mnt/g/CryoLocoManip/cache/huggingface/transformers
PYTHONNOUSERSITE=1
PIP_REQUIRE_VIRTUALENV=1
PIP_USER=0
```

For Conda/micromamba stages also set the two locations in §3.2.

These variables must be scoped to the project shell or run script. They must not be appended to global shell startup files.

---

## 6. Required pre/post host snapshots

Before and after an environment-building stage, record in the ignored formal run directory:

```text
host_python_before.txt / host_python_after.txt
user_site_before.txt / user_site_after.txt
conda_before.txt / conda_after.txt
shell_rc_hashes_before.txt / shell_rc_hashes_after.txt
system_package_snapshot_before.txt / system_package_snapshot_after.txt
docker_state_before.txt / docker_state_after.txt
storage_before.txt / storage_after.txt
```

Minimum checks:

- `python3 -m pip --version` and `python3 -m pip list --user`;
- `python3 -c 'import site; print(site.getusersitepackages())'`;
- `conda info --envs` when Conda exists;
- SHA256 of relevant user shell startup files;
- `dpkg-query -W` or an equivalent package snapshot;
- Docker root and image count;
- `/mnt/g` and WSL virtual-disk free space;
- project and upstream Git cleanliness.

Expected result: no unexplained host-level delta.

---

## 7. Environment lifecycle

Each stage-specific environment must be classified as one of:

- `RETAINED_AUTHORITATIVE`: required to reproduce an accepted result;
- `RETAINED_DIAGNOSTIC`: retained temporarily because it contains evidence not yet archived elsewhere;
- `REBUILDABLE_AND_REMOVED`: freeze, source hashes, build commands, and logs retained; environment deleted;
- `QUARANTINED_FAILED_BUILD`: isolated from future runs until explicitly removed.

A stage report must state the classification and disk usage of every environment it creates.

No environment may be reused for another paper or method merely because it happens to import successfully.

---

## 8. Pass/fail policy

Return a blocker instead of proceeding when:

- a dependency can be installed only globally;
- a build requires changing `/usr/local/cuda` or the system compiler configuration;
- Conda insists on writing package/env state into the WSL virtual disk and cannot be redirected reliably;
- Docker requires pulling into `/var/lib/docker`;
- the active Python prefix is outside the authorized project environment;
- the user site becomes importable during a formal run;
- host package or shell startup state changes unexpectedly.

Allowed blocker names include:

```text
BLOCKED_HOST_MODIFICATION_REQUIRED
BLOCKED_ENVIRONMENT_OUTSIDE_PROJECT_ROOT
BLOCKED_ENVIRONMENT_ON_DRVFS
BLOCKED_DOCKER_STORAGE_POLICY
FAIL_HOST_ENVIRONMENT_POLLUTION
```

---

## 9. Evidence boundary

Environment isolation protects reproducibility and the user's workstation. It is not scientific evidence for terrain classification, foot-snow interaction, support-manipulation coupling, whole-body control, VLA, or thesis novelty.
