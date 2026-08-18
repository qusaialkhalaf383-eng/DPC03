# DPC PartIQ — Stage 10–11 Report: Test Evaluation & Generalization Challenge

**Synthetic demo data. Not validated real-world industrial accuracy.**

## Stage 10 — Untouched test set (450 rows, first and only use)

| Target | Model | F1 | Precision | Recall | FPR | FNR | Support (pos/total) |
|---|---|---|---|---|---|---|---|
| Keep_Physical | CatBoost | 0.980 | 0.990 | 0.970 | 0.008 | 0.030 | 198/450 |
| Increase_Safety_Stock | CatBoost | 0.936 | 0.917 | 0.957 | 0.005 | 0.043 | 23/450 |
| Reduce_Stock | LightGBM | 0.923 | 0.909 | 0.938 | 0.007 | 0.062 | 32/450 |
| Consolidate_Duplicate | Rule | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 60/450 |
| Standard_Commercial_Action | Rule | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 63/450 |
| Digitize | CatBoost | 0.936 | 0.943 | 0.930 | 0.011 | 0.070 | 71/450 |
| Reverse_Engineer | LightGBM | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 85/450 |
| Repair | XGBoost | 0.973 | 1.000 | 0.947 | 0.000 | 0.053 | 38/450 |
| OEM_Purchase | CatBoost | 0.989 | 1.000 | 0.979 | 0.000 | 0.021 | 94/450 |
| Local_Source | CatBoost | 0.945 | 0.952 | 0.939 | 0.023 | 0.061 | 147/450 |
| CNC_Candidate | CatBoost | 0.998 | 1.000 | 0.996 | 0.000 | 0.004 | 238/450 |
| AM_Candidate | CatBoost | 0.988 | 0.976 | 1.000 | 0.009 | 0.000 | 122/450 |
| Cast_Forge_Candidate | CatBoost | 0.981 | 1.000 | 0.962 | 0.000 | 0.038 | 53/450 |
| Engineering_Review | CatBoost | 0.940 | 0.981 | 0.903 | 0.006 | 0.097 | 113/450 |
| IP_Legal_Review | Rule | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 37/450 |

**Overall multi-label metrics:**

| Metric | Value |
|---|---|
| Micro F1 | 0.9766 |
| Macro F1 | 0.9726 |
| Weighted F1 | 0.9764 |
| Hamming loss | 0.0095 |
| Subset accuracy (all 15 labels exactly right) | 0.8644 |
| Mean average precision | 0.9941 |

Test metrics track validation metrics closely across every target (no target shows
more than a few points of drop) — no evidence of overfitting to the validation set
during threshold/calibration selection. `Engineering_Review` recall holds at 90.3% on
test, still clearing its 90% floor out of sample.

## Stage 11 — Generalization challenge: held-out sector

Held out **Marine & Offshore** entirely (380/3,000 rows, 12.7% of the dataset) —
combined Train+Validation+Test, removed every Marine & Offshore row, retrained each
target's winning model on the remaining 2,620 rows, and evaluated only on the
held-out sector.

| Target | Held-out sector F1 | Original test F1 | Degradation |
|---|---|---|---|
| Keep_Physical | 0.991 | 0.980 | −0.011 (improved) |
| Increase_Safety_Stock | 0.930 | 0.936 | +0.006 |
| Reduce_Stock | 1.000 | 0.923 | −0.077 (improved) |
| Digitize | 0.935 | 0.936 | +0.001 |
| **Reverse_Engineer** | 0.984 | 1.000 | **+0.016** |
| Repair | 0.976 | 0.973 | −0.003 (improved) |
| **OEM_Purchase** | 1.000 | 0.989 | **−0.011 (improved)** |
| Local_Source | 0.936 | 0.945 | +0.009 |
| CNC_Candidate | 1.000 | 0.998 | −0.002 (improved) |
| AM_Candidate | 0.995 | 0.988 | −0.007 (improved) |
| Cast_Forge_Candidate | 0.990 | 0.981 | −0.009 (improved) |
| Engineering_Review | 0.981 | 0.940 | −0.041 (improved) |
| Consolidate_Duplicate / Standard_Commercial_Action / IP_Legal_Review (rules) | 1.000 | 1.000 | 0.000 |

**Mean F1 degradation across all 15 targets: −0.0086** (i.e., essentially flat — a
handful of targets actually scored *better* on the unseen sector). **Zero targets
degraded by more than 0.10.**

**This resolves the near-deterministic watchlist from Stage 5/6.** `Reverse_Engineer`
and `OEM_Purchase` were flagged because a shallow decision tree reached 95–98% train
accuracy on just 3–4 fields, raising the question of whether the model was learning
a real pattern or a synthetic-generator shortcut specific to the sectors it was
trained on. An entirely unseen sector shows no meaningful drop for either target
(`Reverse_Engineer` +0.016, `OEM_Purchase` actually improved) — the relationship
those fields encode (`Drawing_Available` / `CAD_Available` / `Original_Process` /
`OEM_Status` → reverse-engineering need; `Original_Process` / `Safety_Critical` /
`OEM_Obsolete` → OEM purchase decision) generalizes across sectors. It's still a
simple, near-deterministic rule *within this synthetic generator's logic* — that's a
property of how the demo data was built, not a flaw in the model — but it's not
sector-specific memorization.

**Caveat for the record:** the synthetic generator likely encodes similar rule logic
across all 8 sectors (it's demo data, not independently-sourced field observations),
so this challenge test is a genuine but limited check — it tells us the model isn't
overfitting to sector-specific noise, but it can't fully substitute for validation
against real, independently-collected engineering decisions once real data is
available (see Section 1's caution against presenting synthetic-data performance as
validated field performance).

## Files produced

- `reports/stage10_test_metrics_per_target.csv`, `stage10_test_metrics_overall.json`
- `reports/stage11_generalization_challenge.csv`

## Next

Stage 12 (missing-data stress test) and Stage 13 (programmatic engineering challenge
cases) are the remaining robustness checks before building the engineering rules
engine code, the portfolio optimizer, and the API/Streamlit layers.
