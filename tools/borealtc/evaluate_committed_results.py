"""Stage 0B.1 independent evaluation of the canonical committed results.

Reads exactly the two locked result arrays

    results/husky/results_CNN_hamming_mw_1.7.npy
    results/husky/results_mamba_optim2_mw_1.7.npy

from the pinned upstream checkout and, independently of upstream
`compile_metrics.py`:

- validates array structure, label domain, terrain ordering, and the fold
  concatenation layout (recovered from the label block structure);
- computes accuracy, per-class precision/recall/F1/support, macro-F1,
  weighted-F1, balanced accuracy, confusion matrices, and timing summaries;
- compares every two-decimal metric against the committed upstream metric
  files and against the Stage 0B.1 contract targets within +/-0.01 pp;
- audits the upstream `average_precision_score` call as DIAGNOSTIC ONLY
  (its scalar output is not a standard multiclass average precision);
- optionally re-runs the unchanged upstream `compile_metrics.process_results`
  in a disposable copy under the run directory (--upstream-smoke) and
  compares its output text against the committed metric files.

All expected values enter only as comparison references, never as outputs.

Usage:
    python -m tools.borealtc.evaluate_committed_results \
        --run-dir <run_dir> [--pipeline-json <pipeline_reconstruction.json>] \
        [--upstream-smoke]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

try:
    from tools.borealtc import _common
except ImportError:  # executed as a loose script
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.borealtc import _common


def load_canonical_array(model: str) -> tuple[dict, dict]:
    """Load one canonical result artifact; returns (values, provenance)."""
    import numpy as np

    rel = _common.CANONICAL_RESULTS[model]
    path = _common.upstream_dir() / rel
    if not path.is_file():
        raise _common.AuditError(f"missing canonical result artifact: {path}")
    provenance = {"path": rel, "sha256": _common.sha256_file(path)}
    # Pinned, hash-recorded upstream artifact; allow_pickle is required by its
    # dict payload and is accepted only for this verified file.
    values = np.load(path, allow_pickle=True).item()
    if not isinstance(values, dict):
        raise _common.AuditError(f"{rel}: payload is not a dict")
    missing = {"terrains", "pred", "true"} - set(values)
    if missing:
        raise _common.AuditError(f"{rel}: missing keys {sorted(missing)}")
    return values, provenance


def evaluate_model(model: str, values: dict, natural_counts: dict[str, int]) -> dict:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    terrains = [t for t in values["terrains"] if t != _common.MIXED_CLASS]
    if tuple(terrains) != _common.OFFICIAL_CLASSES:
        raise _common.AuditError(
            f"{model}: terrain order {terrains} != {_common.OFFICIAL_CLASSES}"
        )
    pred_raw = np.asarray(values["pred"])
    true_raw = np.asarray(values["true"])
    if pred_raw.shape != true_raw.shape or pred_raw.ndim != 1:
        raise _common.AuditError(f"{model}: pred/true shape mismatch")

    names = np.array(terrains)
    # Mirror upstream compile_metrics: int64 arrays are indices into the
    # terrain list; anything else is already terrain names.
    if pred_raw.dtype.type is np.int64:
        domain = set(np.unique(np.concatenate([pred_raw, true_raw])).tolist())
        if not domain <= set(range(len(terrains))):
            raise _common.AuditError(
                f"{model}: label domain {sorted(domain)} out of range"
            )
        y_pred = names[pred_raw]
        y_true = names[true_raw]
        true_idx = true_raw
        label_encoding = "int64 indices"
    else:
        unknown = set(np.unique(np.concatenate([pred_raw, true_raw]))) - set(terrains)
        if unknown:
            raise _common.AuditError(f"{model}: unknown labels {sorted(unknown)}")
        y_pred = pred_raw
        y_true = true_raw
        name_to_idx = {t: i for i, t in enumerate(terrains)}
        true_idx = np.array([name_to_idx[t] for t in true_raw])
        label_encoding = "terrain-name strings"

    prec = precision_score(y_true, y_pred, labels=names, average=None)
    reca = recall_score(y_true, y_pred, labels=names, average=None)
    f1 = f1_score(y_true, y_pred, labels=names, average=None)
    support = Counter(y_true.tolist())
    cm = confusion_matrix(y_true, y_pred, labels=names)

    fold_blocks = _common.infer_fold_blocks(true_idx.tolist())
    recalls_pct = {
        c: 100.0 * float(reca[i]) for i, c in enumerate(_common.OFFICIAL_CLASSES)
    }
    out = {
        "n_samples": int(len(y_true)),
        "label_encoding": label_encoding,
        "terrains": terrains,
        "accuracy_pct": 100.0 * float(accuracy_score(y_true, y_pred)),
        "per_class": {
            c: {
                "precision_pct": 100.0 * float(prec[i]),
                "recall_pct": 100.0 * float(reca[i]),
                "f1_pct": 100.0 * float(f1[i]),
                "support": int(support[c]),
            }
            for i, c in enumerate(_common.OFFICIAL_CLASSES)
        },
        "macro_f1_pct": 100.0 * float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1_pct": 100.0
        * float(f1_score(y_true, y_pred, average="weighted")),
        "balanced_accuracy_pct": 100.0
        * float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm.tolist(),
        "fold_test_window_counts_inferred": [
            dict(zip(_common.OFFICIAL_CLASSES, fb)) for fb in fold_blocks
        ],
        "natural_weighted_accuracy_pct_diagnostic": _common.natural_weighted_accuracy(
            recalls_pct, natural_counts
        ),
    }
    for key in ("ftime", "ptime"):
        if key not in values:
            out[f"{key}_summary_s"] = "absent"
            continue
        arr = np.asarray(values[key], dtype=float)
        if arr.size == 0:
            out[f"{key}_summary_s"] = "present_but_empty"
            continue
        out[f"{key}_summary_s"] = {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "sum": float(arr.sum()),
        }
    return out


def audit_upstream_ap(values: dict) -> dict:
    """Replicate the exact upstream average_precision_score call, as a
    diagnostic. The call feeds integer class indices of predictions as if
    they were detection scores for a single binary problem, so its scalar
    output is not a standard multiclass average precision; the paper
    Table III closure does not depend on it.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score

    terrains = np.array([t for t in values["terrains"] if t != _common.MIXED_CLASS])
    names = terrains[np.asarray(values["pred"])]
    tests = terrains[np.asarray(values["true"])]
    terr_idx = {t: i for i, t in enumerate(terrains)}
    try:
        ap = average_precision_score(
            np.array([terr_idx[y] for y in tests]).reshape(-1, 1),
            np.array([terr_idx[y] for y in names]).reshape(-1, 1),
            average=None,
        ).item()
        return {"reproduced_value_pct": 100.0 * ap, "exception": None}
    except Exception as exc:  # record, never mask: diagnostic only
        return {"reproduced_value_pct": None, "exception": f"{type(exc).__name__}: {exc}"}


def compare_metrics(model: str, result: dict) -> tuple[list[list[object]], list[str]]:
    """Compare independent metrics against committed .dat and contract."""
    dat_path = _common.upstream_dir() / _common.CANONICAL_METRIC_FILES[model]
    dat = _common.parse_metric_dat(dat_path.read_text(encoding="utf-8"))
    contract = _common.EXPECTED_TABLE3[model]

    rows: list[list[object]] = []
    mismatches: list[str] = []

    def add(metric: str, ours: float, dat_value, contract_value) -> None:
        ours2 = round(ours, 2)
        ok_dat = (
            _common.within_tolerance(ours2, dat_value) if dat_value is not None else ""
        )
        ok_contract = (
            _common.within_tolerance(ours2, contract_value)
            if contract_value is not None
            else ""
        )
        if ok_dat is False:
            mismatches.append(
                f"{model} {metric}: independent {ours2} vs committed {dat_value}"
            )
        if ok_contract is False:
            mismatches.append(
                f"{model} {metric}: independent {ours2} vs contract {contract_value}"
            )
        rows.append(
            [
                model,
                metric,
                ours2,
                dat_value if dat_value is not None else "",
                contract_value if contract_value is not None else "",
                round(ours2 - dat_value, 4) if dat_value is not None else "",
                ok_dat,
                ok_contract,
            ]
        )

    add("accuracy", result["accuracy_pct"], dat.get("acc"), contract["accuracy"])
    for cls in _common.OFFICIAL_CLASSES:
        short = cls[:3]
        per = result["per_class"][cls]
        expect = contract["per_class"][cls]
        add(f"precision-{cls}", per["precision_pct"], dat.get(f"p-{short}"), expect["precision"])
        add(f"recall-{cls}", per["recall_pct"], dat.get(f"r-{short}"), expect["recall"])
        add(f"f1-{cls}", per["f1_pct"], dat.get(f"f-{short}"), expect["f1"])
    return rows, mismatches


def upstream_metric_smoke(run_dir: Path) -> dict:
    """Run the UNCHANGED upstream compile_metrics.process_results in a
    disposable copy below the run directory (never inside third_party).
    """
    up = _common.upstream_dir()
    smoke_dir = run_dir / "upstream_metric_smoke"
    (smoke_dir / "results" / "husky").mkdir(parents=True, exist_ok=True)
    script_src = up / _common.UPSTREAM_METRIC_SCRIPT
    script_copy = smoke_dir / _common.UPSTREAM_METRIC_SCRIPT
    shutil.copyfile(script_src, script_copy)
    src_sha = _common.sha256_file(script_src)
    if _common.sha256_file(script_copy) != src_sha:
        raise _common.AuditError("compile_metrics.py copy hash mismatch")
    for rel in _common.CANONICAL_RESULTS.values():
        shutil.copyfile(up / rel, smoke_dir / rel)

    outcome: dict = {"script_sha256": src_sha, "compared": {}, "exception": None}
    cwd = os.getcwd()
    try:
        os.chdir(smoke_dir)
        spec = importlib.util.spec_from_file_location(
            "upstream_compile_metrics", script_copy
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # creates ./metrics, imports tqdm/scipy
        for model, rel in _common.CANONICAL_RESULTS.items():
            module.process_results(Path(rel))
            generated = (
                smoke_dir / _common.CANONICAL_METRIC_FILES[model]
            ).read_text(encoding="utf-8")
            committed = (up / _common.CANONICAL_METRIC_FILES[model]).read_text(
                encoding="utf-8"
            )
            outcome["compared"][model] = {
                "identical_text": generated == committed,
                "identical_parsed": _common.parse_metric_dat(generated)
                == _common.parse_metric_dat(committed),
            }
    except Exception as exc:
        outcome["exception"] = f"{type(exc).__name__}: {exc}"
    finally:
        os.chdir(cwd)
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--pipeline-json", default=None)
    parser.add_argument("--upstream-smoke", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()

    identity = _common.verify_upstream_identity()
    print(f"upstream OK: {identity['commit']}")

    pipeline = None
    natural_counts = dict(_common.EXPECTED_PARTITION_COUNTS)
    natural_source = "contract configuration (no pipeline json supplied)"
    if args.pipeline_json:
        pipeline = json.loads(Path(args.pipeline_json).read_text(encoding="utf-8"))
        natural_counts = {
            k: int(v) for k, v in pipeline["pre_split_partition_counts"].items()
        }
        natural_source = "reconstructed pipeline (pre_split_partition_counts)"

    report: dict = {
        "upstream": identity,
        "natural_distribution_source": natural_source,
        "models": {},
    }
    metric_rows: list[list[object]] = []
    comparison_rows: list[list[object]] = []
    all_mismatches: list[str] = []
    failures: list[str] = []

    for model in ("CNN", "Mamba"):
        values, provenance = load_canonical_array(model)
        result = evaluate_model(model, values, natural_counts)
        result["artifact"] = provenance
        result["upstream_ap_diagnostic"] = audit_upstream_ap(values)

        if pipeline is not None:
            expected_folds = [
                {k: int(v) for k, v in fold.items()}
                for fold in pipeline["fold_test_window_counts"]
            ]
            result["fold_counts_match_reconstruction"] = (
                result["fold_test_window_counts_inferred"] == expected_folds
            )
            if not result["fold_counts_match_reconstruction"]:
                failures.append(
                    f"{model}: inferred fold test-window counts do not match the "
                    f"reconstructed split"
                )
            if result["n_samples"] != int(pipeline["total_test_windows_all_folds"]):
                failures.append(
                    f"{model}: n_samples {result['n_samples']} != reconstructed "
                    f"{pipeline['total_test_windows_all_folds']}"
                )

        rows, mismatches = compare_metrics(model, result)
        comparison_rows.extend(rows)
        all_mismatches.extend(mismatches)
        report["models"][model] = result

        for cls in _common.OFFICIAL_CLASSES:
            per = result["per_class"][cls]
            metric_rows.append(
                [
                    model,
                    cls,
                    _common.PAPER_DISPLAY_NAMES[cls],
                    round(per["precision_pct"], 4),
                    round(per["recall_pct"], 4),
                    round(per["f1_pct"], 4),
                    per["support"],
                ]
            )
        metric_rows.append(
            [
                model,
                "AGGREGATE",
                "",
                round(result["accuracy_pct"], 4),
                round(result["balanced_accuracy_pct"], 4),
                round(result["macro_f1_pct"], 4),
                result["n_samples"],
            ]
        )

        cm_path = run_dir / f"confusion_{model.lower()}.csv"
        _common.write_csv(
            cm_path,
            ["true\\pred", *_common.OFFICIAL_CLASSES],
            [
                [cls, *result["confusion_matrix"][i]]
                for i, cls in enumerate(_common.OFFICIAL_CLASSES)
            ],
        )

    if args.upstream_smoke:
        smoke = upstream_metric_smoke(run_dir)
        report["upstream_metric_smoke"] = smoke
        if smoke["exception"]:
            print(f"SMOKE_WARNING: upstream smoke failed: {smoke['exception']}")
        else:
            agree = all(
                v["identical_parsed"] for v in smoke["compared"].values()
            )
            print("SMOKE_OK" if agree else "SMOKE_MISMATCH")
            if not agree:
                failures.append("upstream smoke output disagrees with committed .dat")

    _common.write_csv(
        run_dir / "independent_metrics.csv",
        [
            "model",
            "class",
            "paper_display_name",
            "precision_pct_or_accuracy",
            "recall_pct_or_balanced_accuracy",
            "f1_pct_or_macro_f1",
            "support_or_n_samples",
        ],
        metric_rows,
    )
    _common.write_csv(
        run_dir / "paper_result_comparison.csv",
        [
            "model",
            "metric",
            "independent_pct",
            "committed_dat_pct",
            "contract_expected_pct",
            "delta_vs_dat_pp",
            "within_tolerance_dat",
            "within_tolerance_contract",
        ],
        comparison_rows,
    )
    report["metric_mismatches"] = all_mismatches
    report["failures"] = failures
    _common.write_json(run_dir / "evaluator_report.json", report)

    for line in all_mismatches:
        print(f"MISMATCH: {line}")
    for line in failures:
        print(f"PROBLEM: {line}")
    for model in ("CNN", "Mamba"):
        r = report["models"][model]
        print(
            f"{model}: n={r['n_samples']} acc={r['accuracy_pct']:.2f}% "
            f"macroF1={r['macro_f1_pct']:.2f}% balacc={r['balanced_accuracy_pct']:.2f}% "
            f"natural-weighted acc (diagnostic)={r['natural_weighted_accuracy_pct_diagnostic']:.2f}%"
        )
    ok = not all_mismatches and not failures
    print("EVAL_OK" if ok else "EVAL_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
