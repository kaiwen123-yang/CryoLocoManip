#!/usr/bin/env python3
"""Verify the canonical CryoLocoManip workspace contract.

Standard-library-only, read-only checks:

1. the repository root resolves exactly to /mnt/g/CryoLocoManip;
2. origin is exactly git@github.com:kaiwen123-yang/CryoLocoManip.git;
3. the required ignored local directories exist below the project root;
4. configs/local/paths.local.yaml resolves every project path below the
   project root, unless the path lies below a root declared in the optional
   `allowed_external_roots` list, which must itself stay below /mnt/g.

Prints WORKSPACE_OK and exits 0 when every check passes; otherwise prints one
VIOLATION line per failed check and exits 1. The script never modifies files.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

CANONICAL_ROOT = "/mnt/g/CryoLocoManip"
EXPECTED_ORIGIN = "git@github.com:kaiwen123-yang/CryoLocoManip.git"
GDRIVE_MOUNT = "/mnt/g"

REQUIRED_LOCAL_DIRS = (
    "data",
    "third_party",
    "simulators",
    "checkpoints",
    "cache",
    "runs",
    "outputs",
    "configs/local",
    "artifacts/runtime",
    "logs",
)

LOCAL_PATHS_FILE = "configs/local/paths.local.yaml"
ALLOWED_EXTERNAL_KEY = "allowed_external_roots"


def clean_scalar(value: str) -> str:
    """Strip whitespace, trailing inline comments, and surrounding quotes."""
    value = value.strip()
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return value


def parse_simple_yaml(text: str) -> dict:
    """Parse the flat, two-level `key: value` subset of YAML used by
    configs/paths.local.example.yaml: comments, blank lines, top-level
    scalars, and one nesting level of mappings or `- item` lists.

    Raises ValueError on anything outside that subset.
    """
    root: dict = {}
    current = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            if ":" not in stripped:
                raise ValueError(f"line {lineno}: expected 'key: value' or 'key:'")
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = clean_scalar(value)
            if value:
                root[key] = value
                current = None
            else:
                root[key] = {}
                current = key
        else:
            if current is None:
                raise ValueError(f"line {lineno}: unexpected indented line")
            block = root[current]
            if stripped.startswith("- "):
                if block == {}:
                    block = []
                    root[current] = block
                if not isinstance(block, list):
                    raise ValueError(f"line {lineno}: mixed list and mapping entries")
                block.append(clean_scalar(stripped[2:]))
            else:
                if ":" not in stripped:
                    raise ValueError(f"line {lineno}: expected 'key: value'")
                if not isinstance(block, dict):
                    raise ValueError(f"line {lineno}: mixed list and mapping entries")
                key, _, value = stripped.partition(":")
                block[key.strip()] = clean_scalar(value)
    return root


def is_within(path: str, root: str) -> bool:
    """Purely lexical containment check for absolute POSIX paths."""
    candidate = PurePosixPath(os.path.normpath(str(path)))
    base = PurePosixPath(os.path.normpath(str(root)))
    return candidate == base or base in candidate.parents


def git_output(repo_root: Path, *args: str):
    """Return stripped stdout of a read-only git command, or None."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def check_local_paths_config(repo_root: Path) -> list:
    config_path = repo_root / LOCAL_PATHS_FILE
    if not config_path.is_file():
        return [f"{LOCAL_PATHS_FILE} is missing (run scripts/preflight_wsl.sh first)"]
    try:
        config = parse_simple_yaml(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{LOCAL_PATHS_FILE} could not be parsed: {exc}"]

    violations = []

    project_root = config.get("project_root")
    if project_root != CANONICAL_ROOT:
        violations.append(
            f"{LOCAL_PATHS_FILE}: project_root must be {CANONICAL_ROOT}, "
            f"found {project_root!r}"
        )

    allowed_external = config.get(ALLOWED_EXTERNAL_KEY, [])
    if isinstance(allowed_external, dict) and not allowed_external:
        allowed_external = []
    if not isinstance(allowed_external, list):
        violations.append(f"{LOCAL_PATHS_FILE}: {ALLOWED_EXTERNAL_KEY} must be a list")
        allowed_external = []
    for ext_root in allowed_external:
        if not is_within(ext_root, GDRIVE_MOUNT):
            violations.append(
                f"{LOCAL_PATHS_FILE}: {ALLOWED_EXTERNAL_KEY} entry {ext_root!r} "
                f"is not below {GDRIVE_MOUNT}"
            )

    paths = config.get("paths")
    if not isinstance(paths, dict) or not paths:
        violations.append(f"{LOCAL_PATHS_FILE}: a non-empty 'paths:' mapping is required")
        return violations

    for key, value in paths.items():
        if not isinstance(value, str) or not value.startswith("/"):
            violations.append(
                f"{LOCAL_PATHS_FILE}: paths.{key} must be an absolute POSIX path, "
                f"found {value!r}"
            )
            continue
        if is_within(value, CANONICAL_ROOT):
            continue
        if any(
            is_within(value, ext) and is_within(ext, GDRIVE_MOUNT)
            for ext in allowed_external
        ):
            continue
        violations.append(
            f"{LOCAL_PATHS_FILE}: paths.{key}={value} resolves outside "
            f"{CANONICAL_ROOT} and is not allow-listed"
        )
    return violations


def check_workspace(repo_root: Path) -> list:
    violations = []

    real_root = Path(os.path.realpath(repo_root))
    if str(real_root) != CANONICAL_ROOT:
        violations.append(
            f"repository root must resolve to {CANONICAL_ROOT}, found {real_root}"
        )

    toplevel = git_output(real_root, "rev-parse", "--show-toplevel")
    if toplevel is None:
        violations.append("git toplevel could not be resolved (is git available?)")
    elif os.path.realpath(toplevel) != str(real_root):
        violations.append(
            f"git toplevel {toplevel} does not match script location {real_root}"
        )

    origin = git_output(real_root, "remote", "get-url", "origin")
    if origin != EXPECTED_ORIGIN:
        violations.append(
            f"origin must be exactly {EXPECTED_ORIGIN}, found {origin or '<missing>'}"
        )

    for rel in REQUIRED_LOCAL_DIRS:
        if not (real_root / rel).is_dir():
            violations.append(
                f"required local directory missing: {rel} (run scripts/preflight_wsl.sh)"
            )

    violations.extend(check_local_paths_config(real_root))
    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    violations = check_workspace(repo_root)
    if violations:
        for violation in violations:
            print(f"VIOLATION: {violation}", file=sys.stderr)
        print(f"WORKSPACE_FAIL: {len(violations)} violation(s)", file=sys.stderr)
        return 1
    print(f"WORKSPACE_OK: canonical workspace verified at {CANONICAL_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
