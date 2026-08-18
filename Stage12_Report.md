# DPC PartIQ — Stage 12 Report: Missing-Data Stress Test

## Part 1 — Random corruption of non-critical fields

Cell-level random nulling + silent imputation (train median for numeric, train mode
for categorical) across all non-critical columns, at 10/20/30%, evaluated on the
test set through the combined rules+ML pipeline (`src/inference.py`).

| Corruption | Micro-F1 (12 ML targets) | Degradation vs clean |
|---|---|---|
| 0% (clean) | 0.906 | — |
| 10% | 0.855 | −0.051 |
| 20% | 0.797 | −0.109 |
| 30% | 0.741 | −0.165 |

Degrades roughly linearly rather than falling off a cliff — no single corruption
level causes a sudden collapse, which is the behavior you want from a model relying
on many weakly-redundant features rather than one brittle field. Per-target
breakdown in `reports/stage12_missing_data_stress_test.csv`.

*(Note: the clean baseline here, 0.906, is lower than Stage 10's headline 0.977 —
that number covered all 15 targets including the 3 perfect deterministic rules and
used raw ML predictions with no engineering-rule overlay. This 0.906 is ML-only, 12
targets, WITH the safety/missing-data rule overlays applied on top — the overlays
trade a little raw accuracy for deliberately forcing conservative outcomes on
borderline rows, which is the intended behavior, not a regression.)*

## Part 2 — Critical-field integrity check

This is the part worth reading closely. Corrupted each of the four fields that
directly drive a deterministic gate — `Safety_Critical`, `IP_Restriction`,
`Standard_Commercial_Part`, `has_duplicate_group` — on 100 sampled rows and checked
whether the pipeline silently produces a confident (and possibly wrong)
recommendation, or explicitly flags the gap.

**First pass found a real bug, not just a design question.** Injecting an
unseen category (`"Unknown"`) into a critical categorical field crashed XGBoost's
`Repair` model outright with a hard `XGBoostError` — a real failure mode, and the
wrong one: a corrupted critical field should degrade to "send this to a human,"
not throw an unhandled exception in production. Separately, corrupting
`has_duplicate_group` to a sentinel value flowed straight through the
`Consolidate_Duplicate` rule as a bare int cast, producing an invalid `-1` label
silently rather than flagging anything.

**Both are now fixed in `src/inference.py`, not worked around in the test:**
- Any per-target model call that raises now abstains on that row (falls back
  row-by-row so one bad row doesn't blank the whole batch) and forces
  `Engineering_Review = 1` with an explicit `override_reason`, instead of crashing
  or guessing.
- `Consolidate_Duplicate` now validates `has_duplicate_group` is a clean 0/1 before
  trusting it; anything else forces review instead of silently resolving to 0.

Results after the fix:

| Critical field corrupted | Downstream target | Predictions changed | Forced to review | Abstained |
|---|---|---|---|---|
| Safety_Critical | Engineering_Review | 55/100 | **100/100** | 100/100 |
| IP_Restriction | IP_Legal_Review | 7/100 | **100/100** | 100/100 |
| Standard_Commercial_Part | Standard_Commercial_Action | 28/100 | **100/100** | 100/100 |
| has_duplicate_group | Consolidate_Duplicate | 9/100 | **100/100** | 100/100 |

All four now correctly force 100% of affected rows to review rather than producing
an unflagged (or crashed) output. This is a preview of Stage 17/18's full
abstention/OOD layer — what's here handles the "can't score this row at all" case;
genuine out-of-distribution detection for rows the model *can* score but shouldn't
trust is still to come.

## Files produced

- `reports/stage12_missing_data_stress_test.csv`
- `reports/stage12_critical_field_integrity.csv`
- `src/inference.py` — now the canonical rules+ML pipeline (also used going forward
  for the challenge cases, and later the FastAPI backend)
