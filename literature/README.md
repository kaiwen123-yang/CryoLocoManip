# Literature Evidence Matrix

This directory holds the deduplicated literature evidence matrix for
CryoLocoManip. Stage 0.0 defines only the schema; no paper entry is committed
until its identity, artifacts, and claims have been verified against the
actual source.

## Storage plan

- `literature/evidence_matrix.csv` — one row per deduplicated source, using
  the fields below. Created when the first verified entry exists, not before.
- `literature/notes/<citekey>.md` — optional per-source reading notes linked
  from the matrix.

Deduplicate by DOI or arXiv identifier first, then by (title, year, authors).

## Required fields

| Field | Meaning | Format / allowed values |
| --- | --- | --- |
| `identity` | Stable citation identity | citekey + DOI or arXiv id + full title |
| `year` | Year of the cited version | four-digit year |
| `venue` | Venue or archive | e.g. `RA-L`, `ICRA`, `CoRL`, `T-RO`, `arXiv` |
| `publication_status` | Review status of the cited version | `peer_reviewed` \| `preprint` \| `workshop` \| `thesis` \| `tech_report` \| `dataset_release` |
| `source_tier` | Evidence quality tier | `T1` peer-reviewed with released code and data; `T2` peer-reviewed or preprint with partial artifacts; `T3` paper only, no artifacts; `T4` blog/talk/unverified |
| `robot_embodiment` | Platform studied | e.g. `quadruped`, `quadruped_arm`, `humanoid`, `tracked`, `wheeled`, `manipulator_only`, `none_dataset_only` |
| `environment` | Terrain or medium studied | e.g. `snow`, `ice`, `crust`, `sand`, `deformable_lab`, `rigid_indoor`, `simulation_only` |
| `state` | State the method models or estimates | free text, e.g. contact state, terrain parameters, support margin |
| `observation` | Sensing modalities used | e.g. proprioception, IMU, vision, depth, force/torque |
| `action` | Action representation; whether outcomes are action-conditioned | `none` \| `terrain_class_only` \| `velocity_commands` \| `joint_commands` \| `wrench` \| free text |
| `uncertainty` | Uncertainty treatment | `none` \| `aleatoric` \| `epistemic` \| `calibrated_eval` \| `set_based` |
| `active_interaction` | Whether the robot acts to gain information | `none` \| `probing` \| `active_sensing` \| `active_learning` |
| `datasets` | Datasets used or released, with availability | name(s) + `public` \| `on_request` \| `private` |
| `code` | Code availability and license | link + license + `released` \| `partial` \| `none` |
| `reproduced_status` | Our reproduction state | `NOT_ATTEMPTED` \| `IN_PROGRESS` \| `REPRODUCED` \| `PARTIALLY_REPRODUCED` \| `FAILED` \| `BLOCKED` |
| `limitations` | Limitations relevant to our thesis | free text |
| `future_work` | Future work stated by the authors | free text |
| `direct_overlap` | Overlap with the CryoLocoManip thesis spine | `none` \| `low` \| `medium` \| `high` + one-line justification |
| `claim_level` | Strength of the evidence we rely on | exactly one ledger state from `docs/REPRODUCIBILITY_CONTRACT.md` §9: `REPRODUCED` \| `AUDITED` \| `CLAIMED_BY_SOURCE` \| `INFERENCE` \| `BLOCKED` \| `DIAGNOSTIC_ONLY` |

## Rules

1. No entry may be added from memory alone; verify identity, venue, and
   artifact availability against the actual source before committing a row.
2. A `claim_level` stronger than `CLAIMED_BY_SOURCE` requires a linked run
   manifest or audit note.
3. Keep one row per source; revisions update the row rather than duplicating
   it.
4. Do not commit PDFs or third-party text; store identifiers and notes only.
