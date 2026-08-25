"""Machine-readable storage budget for Stage 0B.2A (prompt §3).

Usage:

    python -m tools.borealtc.runtime.storage baseline --path <storage_budget.json>
    python -m tools.borealtc.runtime.storage update   --path <...> --label <event>
    python -m tools.borealtc.runtime.storage finalize --path <...>

Tracks recursive sizes of the roots that Stage 0B.2A may grow, evaluates the
25 GiB new-persistent-asset cap and the 75 GiB free reserve after every
environment attempt, and never deletes anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.borealtc import _common
    from tools.borealtc.runtime import _rt
except ImportError:  # executed as a loose script
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tools.borealtc import _common
    from tools.borealtc.runtime import _rt

# Roots whose growth counts against the Stage 0B.2A cap, relative to the
# repository root. third_party/BorealTC and the Stage 0B.1 evidence are
# tracked to prove they do NOT grow or shrink.
TRACKED_RELATIVE = (
    "cache/venvs/borealtc-runtime-declared-probe",
    "cache/venvs/borealtc-runtime-compat",
    "cache/pip",
    "cache/tmp",
    "cache/xdg",
    "cache/torch_extensions",
    "logs/stage0b2a",
    "runs/stage0b2a",
)
INVARIANT_RELATIVE = (
    "third_party/BorealTC",
    "cache/venvs/borealtc-source-closure",
    "cache/venvs/borealtc-source-closure-pin",
    "runs/stage0b1",
)


def snapshot(root: Path) -> dict:
    tracked = {rel: _rt.du_bytes(root / rel) for rel in TRACKED_RELATIVE}
    invariant = {rel: _rt.du_bytes(root / rel) for rel in INVARIANT_RELATIVE}
    return {
        "timestamp_utc": _common.utc_iso(),
        "tracked_bytes": tracked,
        "invariant_bytes": invariant,
        "gdrive_free_bytes": _rt.free_bytes("/mnt/g"),
    }


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _event(budget: dict, label: str, snap: dict) -> dict:
    ev = {
        "label": label,
        **snap,
        "evaluation": _rt.evaluate_storage(
            budget["baseline"]["tracked_bytes"],
            snap["tracked_bytes"],
            snap["gdrive_free_bytes"],
        ),
        "invariants_unchanged": {
            rel: snap["invariant_bytes"][rel]
            == budget["baseline"]["invariant_bytes"].get(rel)
            for rel in INVARIANT_RELATIVE
        },
    }
    return ev


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("baseline", "update", "finalize"):
        p = sub.add_parser(name)
        p.add_argument("--path", required=True)
        if name == "update":
            p.add_argument("--label", required=True)
    args = parser.parse_args()

    path = Path(args.path)
    root = _common.repo_root()
    if args.action == "baseline":
        budget = {
            "stage": "stage0b2a",
            "cap_new_persistent_gib": _rt.MAX_NEW_PERSISTENT_GIB,
            "min_free_reserve_gib": _rt.MIN_FREE_RESERVE_GIB,
            "tracked_roots": list(TRACKED_RELATIVE),
            "invariant_roots": list(INVARIANT_RELATIVE),
            "baseline": snapshot(root),
            "events": [],
            "final": None,
        }
        _common.write_json(path, budget)
        free = _rt.bytes_to_gib(budget["baseline"]["gdrive_free_bytes"])
        print(f"STORAGE_BASELINE_WRITTEN: {path} free={free} GiB")
        return 0

    budget = _load(path)
    snap = snapshot(root)
    if args.action == "update":
        ev = _event(budget, args.label, snap)
        budget["events"].append(ev)
        _common.write_json(path, budget)
        e = ev["evaluation"]
        print(
            f"STORAGE_EVENT[{args.label}]: added={e['added_since_baseline_gib']} GiB "
            f"free={e['free_gib']} GiB cap_ok={e['cap_ok']} reserve_ok={e['reserve_ok']}"
        )
        return 0 if (e["cap_ok"] and e["reserve_ok"]) else 1

    ev = _event(budget, "final", snap)
    e = ev["evaluation"]
    verdict = "WITHIN_BUDGET" if (e["cap_ok"] and e["reserve_ok"]) else "OVER_BUDGET"
    invariants_ok = all(ev["invariants_unchanged"].values())
    budget["final"] = {**ev, "verdict": verdict, "invariants_ok": invariants_ok}
    _common.write_json(path, budget)
    print(
        f"STORAGE_FINAL: {verdict} added={e['added_since_baseline_gib']} GiB "
        f"free={e['free_gib']} GiB invariants_ok={invariants_ok}"
    )
    return 0 if verdict == "WITHIN_BUDGET" and invariants_ok else 1


if __name__ == "__main__":
    sys.exit(main())
