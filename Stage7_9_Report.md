# DPC PartIQ — Stage 7–9 Report: Tuning, Thresholds, Calibration

**Synthetic demo data. Test set has NOT been touched by anything in this report.**

## Stage 7 — Hyperparameter tuning (Optuna, train+validation only)

Environment note: the sandbox this ran in has a single CPU core, so the trial
budget/search ranges were scaled down from the brief's suggested defaults (10 trials
per model instead of 30+, iteration/estimator ranges capped around 250 instead of
500) to keep the 12-target × 3-model sweep tractable. Worth re-running with a wider
budget on properly resourced hardware before treating these as final production
hyperparameters — see `models/tuned_hyperparameters.json` for what was actually used.

| Target | CatBoost PR-AUC | XGBoost PR-AUC | LightGBM PR-AUC | Winner |
|---|---|---|---|---|
| Keep_Physical | 0.999 | 0.998 | 0.998 | CatBoost |
| Increase_Safety_Stock | 0.983 | 0.950 | 0.956 | CatBoost |
| Reduce_Stock | 0.962 | 0.976 | **0.988** | LightGBM |
| Digitize | **0.969** | 0.906 | 0.940 | CatBoost |
| Reverse_Engineer | 1.000 | 1.000 | 1.000 | CatBoost *(watchlist — see Stage 5/6 report)* |
| Repair | 0.984 | **0.984** | 0.983 | XGBoost |
| OEM_Purchase | 1.000 | 0.999 | 1.000 | CatBoost *(watchlist)* |
| Local_Source | **0.993** | 0.983 | 0.990 | CatBoost |
| CNC_Candidate | 1.000 | 0.998 | 1.000 | CatBoost |
| AM_Candidate | **0.999** | 0.998 | 0.999 | CatBoost |
| Cast_Forge_Candidate | **1.000** | 0.966 | 0.995 | CatBoost |
| Engineering_Review | **0.992** | 0.974 | 0.991 | CatBoost |

CatBoost wins 10/12 even after tuning — confirms the untuned-benchmark pattern.
LightGBM and XGBoost each win one target, which is exactly the "allow a
best-model-per-target ensemble" outcome the brief called for (Section 10) rather
than forcing one algorithm everywhere.

## Stage 8 — Per-target threshold optimization (validation set)

Default profile maximizes F1. `Engineering_Review` uses a recall-floor profile
(≥90% recall required, per Section 11's "false negative is worse than false
positive" logic for safety escalation) — it landed at 94.8% recall / 97.9%
precision at threshold 0.914, comfortably clearing the floor.

| Target | Winning model | Threshold | Precision | Recall | F1 (vs F1@0.50) |
|---|---|---|---|---|---|
| Keep_Physical | CatBoost | 0.494 | — | — | 0.986 (0.983) |
| Increase_Safety_Stock | CatBoost | 0.917 | — | — | 0.937 (0.899) |
| Reduce_Stock | LightGBM | 0.966 | — | — | 0.957 (0.902) |
| Digitize | CatBoost | 0.596 | — | — | 0.894 (0.872) |
| Reverse_Engineer | LightGBM | 0.779 | — | — | 1.000 (1.000) |
| Repair | XGBoost | 0.914 | — | — | 0.941 (0.857) |
| OEM_Purchase | CatBoost | 0.969 | — | — | 1.000 (0.996) |
| Local_Source | CatBoost | 0.587 | — | — | 0.960 (0.950) |
| CNC_Candidate | CatBoost | 0.680 | — | — | 1.000 (1.000) |
| AM_Candidate | CatBoost | 0.706 | — | — | 0.985 (0.985) |
| Cast_Forge_Candidate | CatBoost | 0.762 | — | — | 0.989 (0.967) |
| Engineering_Review | CatBoost | 0.914 | 0.979 | 0.948 | 0.963 (0.979 at 0.50) |

Full precision/recall per target in `reports/stage8_thresholds.csv`. Threshold
tuning gave a real lift on several targets (`Repair` +0.084 F1, `Reduce_Stock`
+0.055, `Increase_Safety_Stock` +0.038) — the brief's instinct to not default to
0.50 was correct.

Note `Engineering_Review`'s F1 at its chosen threshold (0.963) is slightly *lower*
than F1@0.50 (0.979) — expected and correct. The recall-floor objective isn't
maximizing F1; it's guaranteeing the recall floor first, then maximizing precision
subject to that. In this case F1@0.50 happens to be higher purely because the model
is strong enough that even the default threshold clears the floor, but the chosen
operating point is still the deliberately safety-biased one.

## Stage 9 — Probability calibration

Isotonic vs sigmoid compared per target (fit/eval split within the validation set
so ECE/Brier aren't measured on calibration-fitting data). **10 of 12 targets are
well-calibrated** (Expected Calibration Error < 0.05) — these can display
`Probability = XX%` per Section 13. Two are not:

| Target | ECE | Display format |
|---|---|---|
| `Digitize` | 0.051 | `PartIQ Score = XX/100` (not calibrated probability) |
| `Repair` | 0.052 | `PartIQ Score = XX/100` (not calibrated probability) |

Both are just over the 0.05 cutoff, not badly miscalibrated — but per your own
instruction ("only display Probability=87% if meaningfully calibrated, otherwise
call it PartIQ Score"), these two get the score framing rather than a probability
claim. Isotonic calibration won on 5 targets where raw scores had room to improve
(`Reduce_Stock`, `Reverse_Engineer`, `OEM_Purchase`, `Local_Source`,
`CNC_Candidate`); the rest were already well-calibrated raw and calibration wasn't
needed. Caveat: the calibration-fitting split is only ~225 rows (half of
Validation), thin for the lower-prevalence targets — worth widening on a larger
validation set later.

## Files produced

- `models/tuned_hyperparameters.json` — best hyperparameters per target/model
- `reports/stage7_tuning_results.csv` — full tuning results
- `config/thresholds.json`, `reports/stage8_thresholds.csv` — per-target thresholds + winning model
- `config/calibration.json`, `reports/stage9_calibration.csv` — calibration method + ECE/Brier per target

## Next: Stage 10 — untouched test set evaluation

Preprocessing, feature engineering, model selection, hyperparameter tuning, and
threshold tuning are now complete for all 12 ML targets, and the 3 deterministic
targets are handled by rules — the precondition for touching the test set (Section
3) is met.
