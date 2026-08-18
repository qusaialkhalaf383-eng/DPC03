# DPC PartIQ — Stage 5–6 Report: Baseline + Boosting Benchmark

**Synthetic demo data. Results below are not validated real-world industrial accuracy.**

## 1. Setup

Approach A (independent binary classifier per target, as you preferred). Untuned
defaults for all four model families — Optuna tuning is Stage 7 and only applies
*after* this finding is addressed (see Section 3). Threshold fixed at 0.50 for this
pass; per-target threshold optimization is Stage 12. Class imbalance handled via
class weighting (`class_weight='balanced'` / `auto_class_weights='Balanced'` /
`scale_pos_weight`), not resampling, per Section 8.

## 2. Validation F1 — model comparison

| Target | LogReg | CatBoost | XGBoost | LightGBM | Winner |
|---|---|---|---|---|---|
| AM_Candidate | 0.923 | 0.982 | 0.978 | **0.985** | LightGBM |
| CNC_Candidate | 0.979 | **0.994** | 0.992 | 0.994 | CatBoost |
| Cast_Forge_Candidate | 0.672 | **0.978** | 0.966 | 0.925 | CatBoost |
| Consolidate_Duplicate | 1.000 | 1.000 | 1.000 | 1.000 | *(see §3)* |
| Digitize | 0.760 | **0.859** | 0.756 | 0.824 | CatBoost |
| Engineering_Review | 0.883 | **0.980** | 0.939 | 0.964 | CatBoost |
| IP_Legal_Review | 1.000 | 1.000 | 1.000 | 1.000 | *(see §3)* |
| Increase_Safety_Stock | 0.886 | 0.899 | 0.844 | **0.906** | LightGBM |
| Keep_Physical | 0.941 | **0.978** | 0.966 | 0.968 | CatBoost |
| Local_Source | 0.902 | **0.953** | 0.890 | 0.930 | CatBoost |
| OEM_Purchase | 0.947 | 1.000 | 0.987 | 1.000 | *(see §3, watchlist)* |
| Reduce_Stock | 0.615 | 0.902 | **0.920** | 0.902 | XGBoost |
| Repair | 0.694 | **0.893** | 0.825 | 0.893 | CatBoost |
| Reverse_Engineer | 1.000 | 1.000 | 0.993 | 1.000 | *(see §3, watchlist)* |
| Standard_Commercial_Action | 1.000 | 1.000 | 1.000 | 1.000 | *(see §3)* |

CatBoost wins or ties on 10 of 15 targets even untuned, which is expected given the
categorical-heavy feature set — consistent with your instinct to give it special
attention. But five targets show suspiciously perfect scores across *every* model,
including plain Logistic Regression. That's the finding below.

## 3. Critical finding: 3 targets are deterministic copies of a single input field

Per your instruction to surface (not hide) methodology problems — this needed a stop
before Stage 7 tuning, because tuning a model that already perfectly memorizes a 1:1
field mapping wastes compute and produces a misleading "100% accurate AI" headline.

Crosstab verification against the training set (2,100 rows):

| Target | Field | Match |
|---|---|---|
| `Consolidate_Duplicate` | `has_duplicate_group` (derived from `Duplicate_Group_ID`) | 2,100 / 2,100 — zero counter-examples |
| `IP_Legal_Review` | `IP_Restriction` | 2,100 / 2,100 — zero counter-examples |
| `Standard_Commercial_Action` | `Standard_Commercial_Part` | 2,100 / 2,100 — zero counter-examples |

These aren't leakage in the raw-data sense (all three source fields are legitimate
pre-decision inputs) — they're evidence the synthetic label generator implemented
exactly the deterministic engineering rules your own Section 15 describes ("IP gate",
"Standard component gate") directly as the label formula. An ML model "learning" a 1:1
copy isn't learning anything; it's just an expensive way to restate the rule.

**Fix applied:** moved these three to `config/rules.yaml` as hard `deterministic_gates`
and dropped them from the ML benchmark/tuning set. This is exactly the "Rules+ML"
pattern your own Section 34 table already anticipated for `IP_Legal_Review` — turns out
it applies to two more targets as well.

**Two more flagged, not converted to rules:** `Reverse_Engineer` and `OEM_Purchase`
aren't 100% determined by any single field, but a shallow depth-3 decision tree already
reaches 95.4% and 98.3% *training* accuracy respectively using just 3–4 raw fields
(`Drawing_Available`, `CAD_Available`, `Original_Process`, `OEM_Status`,
`Safety_Critical`, `OEM_Obsolete`). They stay as ML targets, but they're added to a
`near_deterministic_watchlist` in `rules.yaml` — the Stage 22 generalization challenge
(held-out sector/component family) needs to confirm this is real multi-factor signal
and not just a slightly-more-complex synthetic rule before trusting the near-1.0
validation F1.

## 4. Targets where ML is doing genuine work

The remaining targets show real, varied performance gaps between model families —
this is where boosting actually earns its place over the baseline:

- `Digitize` (0.760 → 0.859), `Repair` (0.694 → 0.893), `Reduce_Stock` (0.615 → 0.920),
  `Cast_Forge_Candidate` (0.672 → 0.978) — Logistic Regression struggles noticeably,
  boosted trees add clear value.
- `Engineering_Review`, `Local_Source`, `Keep_Physical` — meaningful but smaller gaps,
  still favor CatBoost.

These 10 targets (+ the 2 watchlist targets = 12 total) are the actual candidates for
Stage 7 Optuna tuning.

## 5. Files produced

- `config/rules.yaml` — deterministic gates + near-deterministic watchlist
- `reports/stage5_6_benchmark_raw.csv` — full per-model, per-target metrics
- `reports/stage5_6_benchmark_summary.csv` — the table above
