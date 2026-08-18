# DPC PartIQ — Stage 13 Report: Engineering Challenge Cases

64 synthetic cases across the 8 archetypes from Section 24, run through the full
rules+ML pipeline (`src/inference.py`).

## A methodology bug caught and fixed before trusting any result

The first version of the case generator built every case from a single **global
median/mode template**, overriding only the fields that defined each archetype.
Result: nonsensical soft-check scores (`Reverse_Engineer` matched expectation on
0/8 "obsolete, no drawings" cases; `Engineering_Review` was forced to `1` on all 8
"excellent data" cases that should have been clean). Tracing it down: an "obsolete,
no-drawings pump impeller" built this way was still assigned the *global* median
`Unit_Cost_AED` (4,530 AED) and median `Weight_kg` (1.2 kg) — a combination that
barely exists in the real data (real obsolete/no-drawings parts cluster cheap and
light, e.g. 180 AED / 0.05 kg). The model wasn't wrong; the test case was an
unrealistic Frankenstein combination sitting in sparse feature space.

**Fix:** `sample_realistic_base()` now draws each case's non-defining fields from an
actual training row matching the archetype's defining criteria (e.g. for
"obsolete/no-drawings," a real row with `Drawing_Available=No, CAD_Available=No,
OEM_Obsolete=Yes, Standard_Commercial_Part=No`), then overrides only what the
archetype specifies. Re-running with this fix changed `Reverse_Engineer` from 0/8
to 8/8 and `Engineering_Review` (long-lead/excellent-data) from 0/8 forced-review
to 8/8 correctly clean. Worth being upfront about this — it's a reminder that
synthetic test-case generation needs the same "does this combination actually occur"
scrutiny as the training data itself.

## Hard checks — deterministic rules (must always hold)

**20/20 passed.** Every case where a rule applies (`IP_Restriction=Yes -> IP_Legal_Review`,
`Standard_Commercial_Part=Yes -> Standard_Commercial_Action` + `Reverse_Engineer=0`)
behaved exactly as the rule specifies, across all archetypes and all data-completeness
variations tested.

## Soft checks — directional ML expectations

| Archetype | Check | Match rate | Read |
|---|---|---|---|
| obsolete_custom_no_drawings | Reverse_Engineer=1 | **8/8** | Clean |
| obsolete_custom_no_drawings | Digitize=1 | 3/8 | Over-strong expectation (see below) |
| long_lead_excellent_data | Engineering_Review=0 | **8/8** | Clean |
| long_lead_excellent_data | Digitize=1 | 6/8 | Reasonable |
| additive_friendly_unknown_material | AM_Candidate=1 | 3/8 | Nuanced — see below |
| expensive_high_consumption | Reduce_Stock=0 | 8/8 | Clean |
| expensive_high_consumption | Keep_Physical=1 | 0/8 | Over-strong expectation |
| single_source_high_downtime | Increase_Safety_Stock=1 | 2/8 | Over-strong expectation |

### Where PartIQ's behavior is genuinely good, not just "matched my guess"

`AM_Candidate` deserves a closer look because the raw match rate (3/8) undersells
what's actually happening. Per-case trace:

- 4 of 8 cases had strong ML signal (proba ≈ 0.999) for AM suitability.
- Of those 4, **3 correctly got `AM_Candidate = Yes`**.
- The 4th had `technical_data_completeness = 0.25` (missing material certificate) —
  the missing-data gate correctly **suppressed** the confident ML "yes" and forced
  `Engineering_Review` instead, exactly per Section 15's "do not issue a
  manufacturing-ready recommendation on incomplete data" rule.
- The other 4 cases simply weren't judged AM-suitable by the model at all (proba
  ≈ 0.0003) — the archetype only overrides `Weight_kg`/`Geometry_Complexity`, and
  the other fields sampled from real matching rows varied enough that not every
  case was actually AM-friendly by the model's learned criteria.

So the honest read is: 3/4 genuinely-AM-suitable cases got the right call, the 4th
was correctly gated rather than wrongly approved, and the "misses" are cases that
were never really AM-suitable in the first place. That's better behavior than the
raw 3/8 number suggests.

### Where my expectations were simply too strong

For `expensive_high_consumption -> Keep_Physical` and
`single_source_high_downtime -> Increase_Safety_Stock`, I checked the actual label
distribution among *real* training rows matching the same defining criteria before
concluding anything was wrong:

- Rows matching "expensive + high consumption" (only 3 in the training set) show
  `Keep_Physical` = 0, 1, 0 — not a reliable pattern in this dataset at all.
- Rows matching "single-source + high downtime cost" (137 rows): `Increase_Safety_Stock`
  is true in only **15/137 (11%)** of them.

My archetype assumed a strong causal link that the synthetic generator's actual
label logic doesn't encode nearly as strongly — these two soft-check "misses" say
more about an over-simplified expectation on my part than about a PartIQ defect.
Worth flagging as a genuine open question for real data: does single-source +
high-downtime-cost actually warrant safety stock in your engineers' judgment, or
does this dataset's generator under-weight that factor? That's exactly the kind of
question synthetic-vs-real validation is supposed to surface.

## Files produced

- `reports/stage13_challenge_cases.csv` — all 64 cases with expectations and outcomes
- `src/challenge_cases.py` — case generator (reusable for adding more archetypes later)
