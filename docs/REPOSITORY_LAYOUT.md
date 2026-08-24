# Repository Layout

CryoLocoManip separates committed research logic from large local assets and immutable run products.

```text
CryoLocoManip/
├── .github/                    # CI and issue/PR templates when introduced
├── artifacts/
│   ├── curated/                # compact, reviewed evidence suitable for Git
│   └── runtime/                # ignored machine/run artifacts
├── configs/
│   ├── local/                  # ignored machine paths and secrets
│   └── paths.local.example.yaml
├── docs/                       # scientific, mathematical, data, and experiment contracts
├── experiments/
│   ├── stage0_borealtc/
│   ├── stage0_interaction_prediction/
│   ├── stage0_mobile_manipulation/
│   └── stage0_coupling_falsification/
├── literature/                 # deduplicated evidence matrix and reading contracts
├── prompts/                    # versioned Codex execution prompts
├── reports/                    # reviewed stage reports, not raw logs
├── schemas/                    # run, evidence, and result schemas
├── scripts/                    # reproducible entrypoints and audits
├── src/cryolocomanip/          # reusable project-owned implementation
├── tests/                      # unit, contract, and regression tests
├── data/                       # ignored public/private datasets
├── third_party/                # ignored source checkouts or reviewed submodules
├── simulators/                 # ignored simulator installations/assets
├── checkpoints/                # ignored model weights
├── cache/                      # ignored caches
├── runs/                       # ignored immutable run directories
└── outputs/                    # ignored temporary generated outputs
```

## Ownership rules

- `src/cryolocomanip/` contains only reusable code owned by this project.
- Reproductions remain isolated under `experiments/<stage>/` and may call pinned third-party checkouts.
- Do not copy third-party implementations into `src/` without a provenance and license review.
- Raw data, checkpoints, and runtime outputs never enter Git.
- Compact final CSV/JSON tables may enter `artifacts/curated/` only after their generating run and script are identified.
- A report in `reports/` must link to the exact run manifests and evaluator commit that support it.

## Experiment directory minimum

Each formal experiment directory should eventually contain:

```text
README.md
METHOD_CONTRACT.md
DATA_CONTRACT.md
EVALUATION_CONTRACT.md
configs/
scripts/
tests/
```

Do not create empty complexity merely to match the tree. Add a directory when its first auditable artifact exists.