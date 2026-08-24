"""Stage 0B.1 R1: compare two formal closure runs field by field.

Produces `version_pin_comparison.json` for the external-review corrective
rerun: split/pipeline IDENTITY fields (membership hashes, orders, counts,
overlap statistics, slides/strides, confusion matrices, smoke byte-identity)
must match EXACTLY; independent metrics must agree within the declared
+/-0.01 pp tolerance. Any identity divergence or out-of-tolerance metric is
a failure (non-zero exit), never a warning.

Usage:
    python -m tools.borealtc.compare_runs \
        --old-run-dir <prior_run> --new-run-dir <corrected_run> \
        --out <version_pin_comparison.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.borealtc import _common
except ImportError:  # executed as a loose script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.borealtc import _common

CORE_PACKAGES = ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "tqdm", "torch")
FREEZE_CANDIDATES = ("requirements_core_pin.freeze.txt", "requirements.freeze.txt")


def load_json(run_dir: Path, name: str) -> dict:
    path = run_dir / name
    if not path.is_file():
        raise _common.AuditError(f"missing run artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def core_versions(run_dir: Path) -> dict[str, str]:
    for name in FREEZE_CANDIDATES:
        path = run_dir / name
        if not path.is_file():
            continue
        versions = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if "==" in line:
                pkg, _, ver = line.partition("==")
                if pkg.strip().lower() in CORE_PACKAGES:
                    versions[pkg.strip().lower()] = ver.strip()
        return {"freeze_file": name, **versions}
    return {"freeze_file": "absent"}


def identity_fields(run_dir: Path) -> dict:
    pipe = load_json(run_dir, "pipeline_reconstruction.json")
    split = load_json(run_dir, "split_audit.json")
    ev = load_json(run_dir, "evaluator_report.json")
    fields: dict = {
        "sampling_freq_hz": pipe["sampling_freq_hz"],
        "pre_split_partition_counts": pipe["pre_split_partition_counts"],
        "pre_split_order_sha256": pipe["pre_split_order_sha256"],
        "split_fold_hashes": pipe["split_fold_hashes"],
        "split_deterministic_on_rerun": pipe["split_deterministic_on_rerun"],
        "fold_test_partition_counts": pipe["fold_test_partition_counts"],
        "fold_test_window_counts": pipe["fold_test_window_counts"],
        "total_test_windows_all_folds": pipe["total_test_windows_all_folds"],
        "augmentation_n_slides": pipe["augmentation_plan_independent"]["n_slides"],
        "augmentation_strides": pipe["augmentation_plan_independent"]["aug_stride"],
        "run_overlap_by_fold": [
            {
                "fold": s["fold"],
                "runs_in_both": s["run_overlap"]["runs_in_both"],
                "pct_test_runs_also_in_train": s["run_overlap"][
                    "pct_test_runs_also_in_train"
                ],
                "pct_test_partitions_from_train_runs": s["run_overlap"][
                    "pct_test_partitions_from_train_runs"
                ],
                "adjacent_train_test_pairs": s["adjacent_train_test_pairs"],
                "duplicate_partitions": s["duplicate_partitions"],
            }
            for s in split["fold_summaries"]
        ],
    }
    for model in ("CNN", "Mamba"):
        d = ev["models"][model]
        fields[f"{model}_n_samples"] = d["n_samples"]
        fields[f"{model}_artifact_sha256"] = d["artifact"]["sha256"]
        fields[f"{model}_confusion_matrix"] = d["confusion_matrix"]
        fields[f"{model}_support"] = {
            c: d["per_class"][c]["support"] for c in d["per_class"]
        }
        fields[f"{model}_fold_window_counts_inferred"] = d[
            "fold_test_window_counts_inferred"
        ]
    smoke = ev.get("upstream_metric_smoke")
    fields["upstream_smoke_identity"] = (
        {
            m: {
                "identical_text": v["identical_text"],
                "identical_parsed": v["identical_parsed"],
            }
            for m, v in smoke["compared"].items()
        }
        if smoke and not smoke.get("exception")
        else f"unavailable: {smoke.get('exception') if smoke else 'not run'}"
    )
    return fields


def metric_fields(run_dir: Path) -> dict:
    ev = load_json(run_dir, "evaluator_report.json")
    out: dict = {}
    for model in ("CNN", "Mamba"):
        d = ev["models"][model]
        m: dict[str, float] = {
            "accuracy_pct": d["accuracy_pct"],
            "macro_f1_pct": d["macro_f1_pct"],
            "weighted_f1_pct": d["weighted_f1_pct"],
            "balanced_accuracy_pct": d["balanced_accuracy_pct"],
            "natural_weighted_accuracy_pct_diagnostic": d[
                "natural_weighted_accuracy_pct_diagnostic"
            ],
        }
        for cls, per in d["per_class"].items():
            m[f"precision_{cls}"] = per["precision_pct"]
            m[f"recall_{cls}"] = per["recall_pct"]
            m[f"f1_{cls}"] = per["f1_pct"]
        ap = d.get("upstream_ap_diagnostic", {}).get("reproduced_value_pct")
        if ap is not None:
            m["upstream_ap_diagnostic_pct"] = ap
        out[model] = m
    return out


def compare(old_dir: Path, new_dir: Path) -> dict:
    old_id = identity_fields(old_dir)
    new_id = identity_fields(new_dir)
    identity: dict = {}
    identity_match = True
    for key in old_id:
        match = old_id[key] == new_id.get(key)
        identity_match &= match
        entry: dict = {"match": match}
        if not match:
            entry["old"] = old_id[key]
            entry["new"] = new_id.get(key)
        identity[key] = entry

    old_metrics = metric_fields(old_dir)
    new_metrics = metric_fields(new_dir)
    metrics: dict = {}
    metrics_ok = True
    max_delta = 0.0
    for model in old_metrics:
        metrics[model] = {}
        for name, old_value in old_metrics[model].items():
            new_value = new_metrics[model][name]
            delta = abs(new_value - old_value)
            max_delta = max(max_delta, delta)
            within = _common.within_tolerance(old_value, new_value)
            metrics_ok &= within
            metrics[model][name] = {
                "old": round(old_value, 6),
                "new": round(new_value, 6),
                "abs_delta_pp": round(delta, 9),
                "within_tolerance": within,
            }

    verdict_ok = identity_match and metrics_ok
    return {
        "old_run": old_dir.name,
        "new_run": new_dir.name,
        "old_environment": core_versions(old_dir),
        "new_environment": core_versions(new_dir),
        "tolerance_pp": _common.METRIC_TOLERANCE_PP,
        "identity": identity,
        "identity_match": identity_match,
        "metrics": metrics,
        "metrics_within_tolerance": metrics_ok,
        "max_metric_abs_delta_pp": round(max_delta, 9),
        "verdict": (
            "IDENTITY_MATCH_AND_METRICS_WITHIN_TOLERANCE"
            if verdict_ok
            else "DIVERGENCE"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run-dir", required=True)
    parser.add_argument("--new-run-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = compare(Path(args.old_run_dir).resolve(), Path(args.new_run_dir).resolve())
    _common.write_json(Path(args.out).resolve(), result)

    mismatched = [k for k, v in result["identity"].items() if not v["match"]]
    for key in mismatched:
        print(f"IDENTITY_DIVERGENCE: {key}")
    for model, entries in result["metrics"].items():
        for name, entry in entries.items():
            if not entry["within_tolerance"]:
                print(
                    f"METRIC_DIVERGENCE: {model} {name} "
                    f"old={entry['old']} new={entry['new']}"
                )
    print(
        f"identity_match={result['identity_match']} "
        f"metrics_within_tolerance={result['metrics_within_tolerance']} "
        f"max_metric_abs_delta_pp={result['max_metric_abs_delta_pp']}"
    )
    print(f"COMPARE_{'OK' if result['verdict'] != 'DIVERGENCE' else 'FAIL'}")
    return 0 if result["verdict"] != "DIVERGENCE" else 1


if __name__ == "__main__":
    sys.exit(main())
