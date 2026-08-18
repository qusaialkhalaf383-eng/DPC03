# DPC PartIQ — Stage 1–2 Report: Dataset Inspection & Leakage Audit

**Source file:** `DPC_PartIQ_Demo_Training_Data_3000.xlsx`
**Status:** Synthetic demo data only. Not customer data. Not validated field performance.

---

## 1. Structural note

The workbook has two title/subtitle rows and one blank row before the real header.
Pandas' default `header=0` therefore misreads column 1 as a giant title string.
Correct read: `pd.read_excel(path, header=3)` → clean **3,000 rows × 95 columns**,
matching the spec exactly.

## 2. Shape & integrity

| Check | Result |
|---|---|
| Rows | 3,000 |
| Columns | 95 |
| `Part_ID` uniqueness | 3,000 / 3,000 unique — no duplicate IDs |
| Fully duplicated rows | 0 |
| `Model_Split` present | Yes — **Train 2,100 / Validation 450 / Test 450** (exact 70/15/15) |
| Negative stock / lead time / pressure / MOQ / weight | None found |
| `Approved_Local_Suppliers > Supplier_Count` | None found |
| `Supplier_Count == 0` | None found |
| Categorical spelling variants (Sector, Equipment_Class, Material_Group, OEM_Status, Geometry_Complexity, Tolerance_Class) | None found — all values are clean, canonical |
| `Standard_Commercial_Part=Yes` combined with `Reverse_Engineer=1` | 0 rows — no contradiction |

**Verdict:** this is a clean synthetic dataset — no impossible values or categorical
inconsistencies were found. That's expected for generated data; it does **not** imply
real operational data will be this clean, and the pipeline (Stage 4) still needs to be
built defensively.

## 3. Missing values

Only 7 columns have missing values, and all of them are expected / structural, not data-quality issues:

| Column | Missing | Explanation |
|---|---|---|
| `Duplicate_Group_ID` | 2,657 (88.6%) | Only populated for the 343 rows belonging to an intentional duplicate-part group |
| `Actual_Route`, `Actual_Cost_AED`, `Actual_Lead_Time_Days`, `Inspection_Result`, `Field_Operating_Hours`, `Field_Failure` | 2,463 each (82.1%) | Real-world outcome fields, only populated where `Outcome_Available = Yes` (537 rows) |

No missing values exist in any legitimate model input column or any of the 15 targets.

## 4. Target label audit (the 15 multi-label targets)

All 15 targets are clean int64 binary (0/1), no missing values. Positive rates:

| Target | Positive rate |
|---|---|
| Keep_Physical | 46.1% |
| Increase_Safety_Stock | 5.5% |
| Reduce_Stock | 7.1% |
| Consolidate_Duplicate | 11.4% |
| Standard_Commercial_Action | 13.5% |
| Digitize | 17.1% |
| Reverse_Engineer | 17.0% |
| Repair | 7.6% |
| OEM_Purchase | 21.4% |
| Local_Source | 33.1% |
| CNC_Candidate | 54.5% |
| AM_Candidate | 28.4% |
| Cast_Forge_Candidate | 12.3% |
| Engineering_Review | 23.0% |
| IP_Legal_Review | 6.8% |

Several targets (`Increase_Safety_Stock`, `Reduce_Stock`, `Repair`, `IP_Legal_Review`) are
meaningfully imbalanced (5–8% positive) and will need class-weighting / threshold tuning
per Stage 8 & 12, not naive 0.50 cutoffs. Train/Validation/Test positive rates were spot
checked for `Digitize`, `Engineering_Review`, `IP_Legal_Review`, `Keep_Physical` and stay
within a few percentage points across splits — the provided split looks reasonably
stratified already, no evidence of a broken split.

## 5. Leakage audit — critical finding

**5 of 323 duplicate-part groups (≈1.5%) span more than one `Model_Split` value**
(e.g. `DUP-00103`, `DUP-00261`, `DUP-00361`, `DUP-00478`, `DUP-01924` each have one member
in Train and one in Validation or Test). This means near-identical parts (same
Component_Category / Equipment_Class / Material_Group combination) can appear on both
sides of the split boundary — a mild grouped-leakage risk, primarily relevant to the
`Consolidate_Duplicate` target.

**Decision:** Section 3 of the brief is explicit — *"If the dataset contains a column
named `Model_Split`, use Train/Validation/Test exactly as provided. Do not randomly
mix them."* Given that instruction, we will **keep the provided split as-is** rather than
re-splitting, but:
- document this as a known limitation in the final validation report (Section 30),
- report `Consolidate_Duplicate` test metrics with a footnote flagging the 5 affected groups,
- make the grouped-split utility available in `src/preprocess.py` as an opt-in function,
  so it's a one-line change if you'd rather re-split by
  `Component_Category + Equipment_Class + Material_Group + Original_Process` instead.

No other leakage vectors were found at the row level (no duplicate rows, no ID overlap).

## 6. Column classification (full detail in `config/column_classification.yaml`)

| Category | Count | Examples |
|---|---|---|
| Legitimate model input | 44 | Sector, Equipment_Class, Weight_kg, Lead_Time_Days, Safety_Critical, CAD_Available, IP_Restriction |
| Model target | 15 | Keep_Physical … IP_Legal_Review |
| Derived/scoring output (excluded from inputs) | 16 | Supply_Risk_Score, Digitization_Priority_Score, CNC_Suitability_Score, Manufacturing_Readiness_Score |
| Post-decision information (excluded) | 14 | Primary_Recommendation, Decision_Rationale, Actual_Route, Actual_Cost_AED, Field_Failure |
| Metadata / identifier (excluded) | 6 | Data_Type, Part_ID, Equipment_ID, Duplicate_Group_ID (raw), Outcome_Available, Model_Split |

**Special case:** `Duplicate_Group_ID` — the raw ID is excluded, but a derived boolean
`has_duplicate_group` is permitted as an engineered input (it's a data-matching fact
available pre-decision, not an outcome of the decision). It is flagged for extra scrutiny
on the `Consolidate_Duplicate` target specifically since it will likely dominate SHAP
importance there — this is expected and will be called out in Stage 21, not treated as
an error, but its predictive strength should not be mistaken for a general finding.

The 16 derived-score columns are effectively the "answer key" — several (e.g.
`Digitization_Priority_Score`, `CNC_Suitability_Score`) are near-direct numeric restatements
of the corresponding binary target and must never enter the feature matrix.
`Primary_Recommendation` / `Secondary_Actions` / `Decision_Rationale` are even more direct
leakage — they encode (or in the rationale's case, narrate) the target combination itself.

## 7. Next step

Stage 3 (confirm inputs/targets) is effectively this document — pending your sign-off,
Stage 4 (preprocessing pipeline: `src/preprocess.py`, `src/features.py`, engineered
features per Section 5) is next.
