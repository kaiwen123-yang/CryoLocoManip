"""Source-level runtime ambiguity audit for Stage 0B.2A (prompt §4).

Usage:

    python -m tools.borealtc.runtime.source_semantics --out <runtime_source_semantics.json>

Parses, from the pinned upstream checkout only: requirements files, the
Dockerfile, README installation statements, SLURM launch scripts, the training
scripts' literal configuration dicts, and the released checkpoint file
identities. Classifies every declared dependency as pinned/underdetermined and
states explicitly which parts of the runtime stack the sources do NOT
determine. Pure parsing helpers are unit-tested without the upstream tree.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

try:
    from tools.borealtc import _common
except ImportError:  # executed as a loose script
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tools.borealtc import _common

# Dependencies that materially determine training semantics or the CUDA
# runtime, as imported by the released training path (utils/models.py,
# utils/preprocessing.py, utils/datamodule.py, utils/dataset.py,
# utils/frequency.py, utils/augmentations.py, utils/transforms.py).
MATERIAL_DEPENDENCIES = (
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "scikit-learn",
    "torch",
    "torchvision",
    "lightning",
    "einops",
    "mamba-ssm",
    "causal_conv1d",
    "pipeline",
    "torchmetrics",
    "tqdm",
    "packaging",
    "tensorboard",
)


def parse_requirement_line(line: str) -> dict | None:
    """Parse one requirements line into {name, spec_type, version, raw}.

    spec_type: exact | range | unpinned | git_unpinned | git_pinned.
    Returns None for comments/blank lines.
    """
    raw = line.rstrip("\n")
    stripped = raw.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("git+"):
        name = stripped.rstrip("/").rsplit("/", 1)[-1]
        name = name.split("@", 1)[0].removesuffix(".git")
        pinned = "@" in stripped.split("/")[-1]
        return {
            "raw": stripped,
            "name": name,
            "spec_type": "git_pinned" if pinned else "git_unpinned",
            "version": None,
        }
    m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(==|>=|<=|~=|!=|>|<)?\s*(\S+)?$", stripped)
    if not m:
        return {"raw": stripped, "name": stripped, "spec_type": "unparsed", "version": None}
    name, op, version = m.group(1), m.group(2), m.group(3)
    if op == "==":
        spec = "exact"
    elif op:
        spec = "range"
    else:
        spec = "unpinned"
    return {"raw": stripped, "name": name, "spec_type": spec, "version": version if op else None}


def parse_requirements(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        parsed = parse_requirement_line(line)
        if parsed is not None:
            out.append(parsed)
    return out


def extract_dockerfile_facts(text: str) -> dict:
    facts: dict = {
        "base_image": None,
        "torch_cuda_arch_list": None,
        "arch_list_max_sm": None,
        "arch_list_has_ptx": None,
        "pip_torch_install_lines": [],
        "requirements_file_copied_as": None,
        "python_install_lines": [],
        "env_lines": [],
    }
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("FROM "):
            facts["base_image"] = line.split(None, 1)[1]
        if "TORCH_CUDA_ARCH_LIST" in line:
            m = re.search(r'TORCH_CUDA_ARCH_LIST="([^"]+)"', line)
            if m:
                arch = m.group(1)
                facts["torch_cuda_arch_list"] = arch
                caps = re.findall(r"(\d+\.\d+)(\+PTX)?", arch)
                if caps:
                    facts["arch_list_max_sm"] = max(float(c[0]) for c in caps)
                    facts["arch_list_has_ptx"] = any(c[1] for c in caps)
        if line.startswith("ENV "):
            facts["env_lines"].append(line)
        if re.search(r"pip3? install .*\btorch\b", line):
            facts["pip_torch_install_lines"].append(line)
        if re.search(r"apt-get install .*python3", line):
            facts["python_install_lines"].append(line)
        m = re.match(r"COPY\s+(\S+)\s+\./(\S+)", line)
        if m and "requirements" in m.group(1):
            facts["requirements_file_copied_as"] = f"{m.group(1)} -> {m.group(2)}"
    return facts


def extract_literal_configs(source: str, names: tuple[str, ...]) -> dict:
    """Extract top-level `NAME = {...}` dict assignments from a script.

    Each dict value is literal-evaluated per key; non-literal values (for
    example `len(terrains)`) are recorded as {"__expr__": "<source>"} so that
    nothing is silently invented.
    """
    tree = ast.parse(source)
    found: dict = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in names:
            continue
        value = node.value
        if isinstance(value, ast.Dict):
            entry = {}
            for key_node, val_node in zip(value.keys, value.values):
                key = ast.literal_eval(key_node)
                try:
                    entry[key] = ast.literal_eval(val_node)
                except ValueError:
                    entry[key] = {"__expr__": ast.unparse(val_node)}
            found[target.id] = entry
        else:
            try:
                found[target.id] = ast.literal_eval(value)
            except ValueError:
                found[target.id] = {"__expr__": ast.unparse(value)}
    return found


def classify_lock(entries: list[dict]) -> dict:
    """Split lock entries into pinned/underdetermined groups."""
    exact = [e for e in entries if e["spec_type"] == "exact"]
    under = [e for e in entries if e["spec_type"] in ("unpinned", "range", "git_unpinned")]
    material_under = [e for e in under if e["name"] in MATERIAL_DEPENDENCIES]
    return {
        "exact_count": len(exact),
        "exact": sorted(e["raw"] for e in exact),
        "underdetermined": sorted(e["raw"] for e in under),
        "material_underdetermined": sorted(e["name"] for e in material_under),
    }


def _git_blob_id(upstream: Path, rel: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(upstream), "ls-tree", "HEAD", rel],
        capture_output=True,
        text=True,
        timeout=60,
    )
    parts = proc.stdout.split()
    return parts[2] if len(parts) >= 3 else "unavailable"


def collect(upstream: Path) -> dict:
    lock_text = (upstream / "requirements-lock.txt").read_text(encoding="utf-8")
    req_text = (upstream / "requirements.txt").read_text(encoding="utf-8")
    docker_text = (upstream / "Dockerfile").read_text(encoding="utf-8")
    readme_text = (upstream / "README.md").read_text(encoding="utf-8")
    mamba_src = (upstream / "mamba_train.py").read_text(encoding="utf-8")
    cnn_src = (upstream / "cnn_train.py").read_text(encoding="utf-8")
    models_src = (upstream / "utils" / "models.py").read_text(encoding="utf-8")

    lock_entries = parse_requirements(lock_text)
    req_entries = parse_requirements(req_text)
    docker = extract_dockerfile_facts(docker_text)

    models_top_imports = [
        line
        for line in models_src.splitlines()
        if line.startswith(("import ", "from "))
    ]
    slurm_dir = upstream / "slurm-scripts"
    slurm = {}
    for script in sorted(slurm_dir.glob("slurm_train_borealtc_*.sh")):
        text = script.read_text(encoding="utf-8")
        slurm[script.name] = {
            "container_runtime": "podman" if "podman run" in text else "docker",
            "dataset_env": re.findall(r"-e DATASET='(\w+)'", text),
            "entry": re.findall(r"python3 (\S+\.py)", text),
        }

    checkpoints = {}
    for ckpt in sorted((upstream / "checkpoints").glob("*.ckpt")):
        rel = f"checkpoints/{ckpt.name}"
        checkpoints[ckpt.name] = {
            "size_bytes": ckpt.stat().st_size,
            "git_blob_id": _git_blob_id(upstream, rel),
            "sha256": _common.sha256_file(ckpt),
        }

    torch_lock = [e for e in lock_entries if e["name"] == "torch"]
    torchvision_lock = [e for e in lock_entries if e["name"] == "torchvision"]

    readme_install = []
    grab = False
    for line in readme_text.splitlines():
        if line.strip().startswith("### Installation"):
            grab = True
        elif grab and line.startswith("## "):
            break
        elif grab:
            readme_install.append(line)

    determinations = {
        "torch_pin": {
            "declared": torch_lock[0]["raw"] if torch_lock else None,
            "is_pinned": bool(torch_lock and torch_lock[0]["spec_type"] == "exact"),
            "constraint_source": (
                torchvision_lock[0]["raw"] if torchvision_lock else None
            ),
            "note": (
                "torch is declared without a version; torchvision==0.17.0 "
                "transitively constrains torch to 2.2.0, but the CUDA wheel "
                "variant (index/backend) is nowhere declared"
            ),
        },
        "cuda_wheel_source": {
            "declared_index": None,
            "note": (
                "no --index-url/--extra-index-url appears in requirements or "
                "Dockerfile; pip default PyPI resolution decides the CUDA "
                "backend of the torch wheel at install time"
            ),
        },
        "gpu_arch_list": {
            "dockerfile_torch_cuda_arch_list": docker["torch_cuda_arch_list"],
            "max_compute_capability": docker["arch_list_max_sm"],
            "includes_ptx_for_max": docker["arch_list_has_ptx"],
        },
        "torchmetrics_vs_torch_metrics": {
            "lock_declares": "torch-metrics==1.1.7",
            "code_imports": "torchmetrics",
            "note": (
                "requirements-lock pins the unrelated PyPI package "
                "'torch-metrics'; the 'torchmetrics' package actually imported "
                "by utils/models.py arrives only as an unpinned transitive "
                "dependency of lightning"
            ),
        },
        "pipeline_git_dependency": {
            "declared": "git+https://github.com/willGuimont/pipeline",
            "is_pinned": False,
            "imported_by": "utils/models.py, utils/datamodule.py, utils/dataset.py",
        },
        "mamba_import_required_for_cnn": {
            "value": any("mamba_ssm" in line for line in models_top_imports),
            "note": (
                "utils/models.py imports mamba_ssm at module top level, so the "
                "CNN training path cannot even be imported without an "
                "installed mamba-ssm/causal_conv1d pair"
            ),
        },
        "readme_dockerfile_gpu_drift": {
            "readme_mentions_dockerfile_gpu": "DockerfileGPU" in readme_text,
            "dockerfile_gpu_exists": (upstream / "DockerfileGPU").exists(),
        },
    }

    return {
        "timestamp_utc": _common.utc_iso(),
        "upstream": _common.verify_upstream_identity(require_clean=True),
        "requirements_lock": {
            "sha256": _common.sha256_text(lock_text),
            "entries": lock_entries,
            "classification": classify_lock(lock_entries),
        },
        "requirements_txt": {
            "sha256": _common.sha256_text(req_text),
            "entries": req_entries,
            "classification": classify_lock(req_entries),
        },
        "dockerfile": {"sha256": _common.sha256_text(docker_text), **docker},
        "readme_installation_excerpt": readme_install,
        "slurm_train_scripts": slurm,
        "training_configs": {
            "mamba_train.py": extract_literal_configs(
                mamba_src,
                ("ssm_cfg_imu", "ssm_cfg_pro", "mamba_train_opt", "RANDOM_STATE"),
            ),
            "cnn_train.py": extract_literal_configs(
                cnn_src, ("cnn_par", "cnn_train_opt", "RANDOM_STATE")
            ),
        },
        "models_py_top_level_imports": models_top_imports,
        "released_checkpoints": checkpoints,
        "determinations": determinations,
        "declared_stack_uniquely_determined": False,
        "declared_stack_determination_note": (
            "The source artifacts pin the scientific core exactly but leave "
            "torch's version to transitive resolution, its CUDA backend to the "
            "default index, torchmetrics and the pipeline git dependency "
            "unpinned, and the Python version implicit (Ubuntu 22.04 python3 = "
            "3.10). An installed environment therefore cannot be labeled "
            "EXACT_DECLARED_STACK on version identity alone."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = collect(_common.upstream_dir())
    _common.write_json(Path(args.out), data)
    lock = data["requirements_lock"]["classification"]
    print(
        f"SOURCE_SEMANTICS_WRITTEN: {args.out} exact={lock['exact_count']} "
        f"material_underdetermined={lock['material_underdetermined']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
