"""
DPC PartIQ - Standalone Single-File Demo App
================================================
Everything in one file on purpose: no src/ package, no sys.path surgery, no
"is the folder structure nested or flat" fragility -- the exact class of error
that broke earlier Streamlit Cloud deploys. Deploy by pointing Streamlit Cloud
at THIS file directly.

Still needs these DATA folders as siblings of this file (not code, so not
merged in): config/, data/, models/, reports/. See the deployment checklist in
the accompanying message for the exact file list.

    streamlit run streamlit_app.py
"""
from __future__ import annotations
import io
import json
import datetime
import warnings
from pathlib import Path
from datetime import datetime as dt

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml
import joblib
import shap
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 Image, HRFlowable, KeepTogether)
from reportlab.lib.enums import TA_CENTER

from catboost import CatBoostClassifier, Pool
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# Canonical paths -- defined ONCE here, used by every function below.
# Assumes config/, data/, models/, reports/ sit as siblings of this file.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "config"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_BUNDLE_DIR = MODELS_DIR / "DPC_PartIQ_Demo_v0.1"
RULES_PATH = CONFIG_DIR / "rules.yaml"
SEED = 42
CALIBRATION = json.loads((CONFIG_DIR / "calibration.json").read_text())
BRAND_GREEN = colors.HexColor("#0b3d2e")
WARN_BG = colors.HexColor("#fff3cd")
WARN_TEXT = colors.HexColor("#664d03")


# ============================================================================
# Source: src/features.py
# ============================================================================
"""
DPC PartIQ - Feature Engineering
=================================
Builds derived features from RAW LEGITIMATE INPUT columns only.
Never touches: model targets, derived/scoring outputs, or post-decision columns.
See config/column_classification.yaml for the source-of-truth column categories.

All ratios use safe division (no div-by-zero -> inf/nan propagating into models).
"""

EPS = 1e-6


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, returning 0.0 wherever the denominator is ~0 instead of inf/nan."""
    denom = denominator.astype(float).copy()
    result = numerator.astype(float) / denom.replace(0, np.nan)
    return result.fillna(0.0)


def _yes_no_to_bin(series: pd.Series) -> pd.Series:
    """Map Yes/No (and common variants) to 1/0. Leaves numeric 0/1 columns unchanged."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    mapping = {"yes": 1, "no": 0, "y": 1, "n": 0, "true": 1, "false": 0}
    mapped = series.astype(str).str.strip().str.lower().map(mapping)
    if mapped.isna().any():
        bad = sorted(series[mapped.isna()].dropna().unique().tolist())
        raise ValueError(f"_yes_no_to_bin: unrecognized values (not Yes/No): {bad}")
    return mapped.astype(float)


# Columns that represent Yes/No technical-data-availability flags used in several
# engineered features below. Kept as a named list so Stage 16 (completeness model)
# can reuse the exact same definition.
TECH_AVAILABILITY_COLS = [
    "Drawing_Available",
    "CAD_Available",
    "Material_Certificate_Available",
    "Inspection_History_Available",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered columns to a copy of df. Input df must already contain the raw
    legitimate-input columns (see column_classification.yaml). Returns a NEW dataframe
    (does not mutate the input) with engineered columns appended.
    """
    out = df.copy()

    # --- normalize a few Yes/No flags used repeatedly below -----------------
    for col in TECH_AVAILABILITY_COLS + ["Failure_Mode_Known", "Rotating_Part", "Safety_Critical",
                                          "IP_Restriction", "OEM_Obsolete", "Standard_Commercial_Part",
                                          "Repair_History"]:
        if col in out.columns:
            out[f"_{col}_bin"] = _yes_no_to_bin(out[col])

    # --- Inventory / stock ----------------------------------------------------
    monthly_consumption = out["Annual_Consumption"] / 12.0
    out["stock_coverage_months"] = _safe_div(out["Current_Stock_Qty"], monthly_consumption)
    out["years_of_stock"] = out["stock_coverage_months"] / 12.0

    out["estimated_annual_consumption_value_aed"] = out["Annual_Consumption"] * out["Unit_Cost_AED"]

    # units of demand expected to occur during the current lead time window
    out["lead_time_demand_units"] = (out["Lead_Time_Days"] / 365.0) * out["Annual_Consumption"]
    # does current stock cover demand through the next lead-time window?
    out["lead_time_to_consumption_ratio"] = _safe_div(out["lead_time_demand_units"], out["Current_Stock_Qty"] + EPS)
    out["stock_covers_lead_time"] = (out["Current_Stock_Qty"] >= out["lead_time_demand_units"]).astype(int)

    # excess stock: holding materially more than min stock policy calls for
    out["stock_to_min_ratio"] = _safe_div(out["Current_Stock_Qty"], out["Min_Stock_Qty"] + EPS)
    out["excess_stock_indicator"] = ((out["stock_to_min_ratio"] > 3) & (out["stock_coverage_months"] > 24)).astype(int)

    out["inventory_turnover"] = _safe_div(out["Annual_Consumption"], out["Current_Stock_Qty"] + EPS)

    # --- Procurement / supply base --------------------------------------------
    out["supplier_concentration_indicator"] = _safe_div(pd.Series(1.0, index=out.index), out["Supplier_Count"] + EPS)
    out["is_single_source"] = (out["Supplier_Count"] <= 1).astype(int)
    out["local_supplier_availability_ratio"] = _safe_div(out["Approved_Local_Suppliers"], out["Supplier_Count"] + EPS)
    out["days_since_purchase_to_lead_time_ratio"] = _safe_div(out["Last_Purchase_Days_Ago"], out["Lead_Time_Days"] + EPS)
    out["emergency_purchase_rate_3y"] = out["Emergency_Purchase_Count_3Y"] / 3.0

    # --- Technical data completeness (engineered input feature; independent of
    #     the excluded Data_Completeness_Score derived column) -----------------
    tech_bin_cols = [f"_{c}_bin" for c in TECH_AVAILABILITY_COLS if f"_{c}_bin" in out.columns]
    if tech_bin_cols:
        out["technical_data_completeness"] = out[tech_bin_cols].mean(axis=1)
    else:
        out["technical_data_completeness"] = np.nan

    # --- Criticality / risk-relevant combinations (from raw inputs only) -----
    out["downtime_cost_per_year_aed"] = out["Downtime_Cost_AED_per_day"] * out["Failure_Frequency_per_year"]
    out["mtbf_years"] = out["MTBF_months"] / 12.0

    # --- Duplicate-group flag (special case; see column_classification.yaml) -
    # ALWAYS present in the output, even if the caller never supplied
    # Duplicate_Group_ID (e.g. an external API caller who doesn't know internal
    # duplicate-tracking IDs) -- default to "no known duplicate" rather than letting
    # the column vanish and break every downstream consumer that expects it.
    if "Duplicate_Group_ID" in out.columns:
        out["has_duplicate_group"] = out["Duplicate_Group_ID"].notna().astype(int)
    else:
        out["has_duplicate_group"] = 0

    # clean up intermediate helper columns (keep only the final engineered ones + bins)
    return out


ENGINEERED_FEATURE_NAMES = [
    "stock_coverage_months",
    "years_of_stock",
    "estimated_annual_consumption_value_aed",
    "lead_time_demand_units",
    "lead_time_to_consumption_ratio",
    "stock_covers_lead_time",
    "stock_to_min_ratio",
    "excess_stock_indicator",
    "inventory_turnover",
    "supplier_concentration_indicator",
    "is_single_source",
    "local_supplier_availability_ratio",
    "days_since_purchase_to_lead_time_ratio",
    "emergency_purchase_rate_3y",
    "technical_data_completeness",
    "downtime_cost_per_year_aed",
    "mtbf_years",
    "has_duplicate_group",
]


# ============================================================================
# Source: src/rules_engine.py
# ============================================================================
"""
DPC PartIQ - Engineering Rules Engine
========================================
Standalone module (Section 15/31). Loads config/rules.yaml so rules are
configurable without touching code. Two responsibilities:

  1. deterministic_targets(): the 3 targets fully replaced by a rule (no ML at all),
     confirmed in Stage 5/6 to be 1:1 functions of a single raw field.
  2. apply_overlays(): post-ML deterministic gates that can OVERRIDE an ML
     prediction (safety gate, missing-data gate) -- these run after ML, per the
     Section 35 architecture: DATA -> ML -> ENGINEERING RULES -> ... -> HUMAN APPROVAL.
     ML never has final authority on safety.
"""



MANUFACTURING_READY_TARGETS = ["CNC_Candidate", "AM_Candidate", "Cast_Forge_Candidate"]


def load_rules(path: Path = RULES_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def deterministic_target_names(rules: dict | None = None) -> list[str]:
    rules = rules or load_rules()
    return [g["target"] for g in rules["deterministic_gates"]]


def apply_deterministic_gates(X: pd.DataFrame, rules: dict | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Returns (predictions for the 3 rule targets, review_flag for rows where the
    driving field couldn't be trusted -- e.g. has_duplicate_group wasn't a clean 0/1)."""
    rules = rules or load_rules()
    preds = pd.DataFrame(index=X.index)
    review_flag = pd.Series(False, index=X.index)

    preds["IP_Legal_Review"] = (X["IP_Restriction"].astype(str) == "Yes").astype(int)
    preds["Standard_Commercial_Action"] = (X["Standard_Commercial_Part"].astype(str) == "Yes").astype(int)

    if "has_duplicate_group" in X.columns:
        dup_raw = pd.to_numeric(X["has_duplicate_group"], errors="coerce")
        valid = dup_raw.isin([0, 1])
        preds["Consolidate_Duplicate"] = 0
        preds.loc[valid, "Consolidate_Duplicate"] = dup_raw[valid].astype(int)
        review_flag = ~valid
    else:
        preds["Consolidate_Duplicate"] = 0
        review_flag = pd.Series(True, index=X.index)

    return preds, review_flag


def apply_overlays(X: pd.DataFrame, preds: pd.DataFrame, rules: dict | None = None) -> pd.DataFrame:
    """Post-ML deterministic overlays: safety gate + missing-data gate. Mutates a COPY
    of preds and returns it, with an 'override_reason' column describing what fired."""
    rules = rules or load_rules()
    preds = preds.copy()
    reasons = pd.Series([""] * len(X), index=X.index)

    completeness = X["technical_data_completeness"] if "technical_data_completeness" in X.columns else pd.Series(1.0, index=X.index)
    safety_critical = X["Safety_Critical"].astype(str) == "Yes" if "Safety_Critical" in X.columns else pd.Series(False, index=X.index)

    safety_threshold = 0.75  # SAFETY_GATE_COMPLETENESS_THRESHOLD, mirrors rules.yaml's safety_gate
    safety_fire = safety_critical & (completeness < safety_threshold)
    preds.loc[safety_fire, "Engineering_Review"] = 1
    reasons.loc[safety_fire] += "safety_gate(safety_critical+low_completeness); "

    missing_threshold = 0.50  # MISSING_DATA_GATE_THRESHOLD, mirrors rules.yaml's missing_data_gate
    missing_fire = completeness < missing_threshold
    for t in MANUFACTURING_READY_TARGETS:
        if t in preds.columns:
            preds.loc[missing_fire, t] = 0
    preds.loc[missing_fire, "Engineering_Review"] = 1
    reasons.loc[missing_fire] += "missing_data_gate(completeness<0.50); "

    preds["override_reason"] = reasons
    return preds


# ============================================================================
# Source: src/completeness.py
# ============================================================================
"""
DPC PartIQ - Engineering Data Completeness Model
====================================================
Section 16. Produces a 0-100 score + LOW/MEDIUM/HIGH category from the raw
availability flags. This is the SAME definition as the `technical_data_completeness`
engineered feature in src/features.py (kept identical deliberately -- two different
completeness numbers for the same part would be confusing and untrustworthy). This
module exists to expose it as a clean, documented, independently-callable scoring
function for the API/Streamlit layers rather than requiring callers to know the
internal feature-engineering column name.
"""

AVAILABILITY_FIELDS = [
    "Drawing_Available",
    "CAD_Available",
    "Material_Certificate_Available",
    "Inspection_History_Available",
]


def completeness_score(row: dict | pd.Series) -> float:
    """0-100 score from the four availability flags, equal-weighted."""
    n_yes = sum(1 for f in AVAILABILITY_FIELDS if str(row.get(f, "No")).strip().lower() == "yes")
    return round(100.0 * n_yes / len(AVAILABILITY_FIELDS), 1)


def completeness_category(score_0_100: float) -> str:
    if score_0_100 < 34:
        return "LOW"
    if score_0_100 < 67:
        return "MEDIUM"
    return "HIGH"


def completeness_report(row: dict | pd.Series) -> dict:
    score = completeness_score(row)
    missing = [f for f in AVAILABILITY_FIELDS if str(row.get(f, "No")).strip().lower() != "yes"]
    return {
        "completeness_score": score,
        "completeness_category": completeness_category(score),
        "missing_fields": missing,
        "manufacturing_ready": score >= 50.0,  # mirrors the missing_data_gate threshold in rules.yaml
    }


def completeness_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized version for portfolio-scale scoring."""
    n_yes = pd.Series(0, index=df.index)
    for f in AVAILABILITY_FIELDS:
        if f in df.columns:
            n_yes = n_yes + (df[f].astype(str).str.strip().str.lower() == "yes").astype(int)
    score = (100.0 * n_yes / len(AVAILABILITY_FIELDS)).round(1)
    category = pd.cut(score, bins=[-0.1, 33.9, 66.9, 100.1], labels=["LOW", "MEDIUM", "HIGH"])
    return pd.DataFrame({
        "completeness_score": score,
        "completeness_category": category,
        "manufacturing_ready": score >= 50.0,
    }, index=df.index)


# ============================================================================
# Source: src/optimizer.py
# ============================================================================
"""
DPC PartIQ - Portfolio Optimizer
====================================
Section 19/20. Compares strategies with a documented, ILLUSTRATIVE lifecycle cost
model. Every number here is a demo-data estimate built on stated assumptions, not a
calibrated financial model -- Section 19 explicitly requires this to be labeled as
such, and Section 20 requires "Observed Input / Model Prediction / Illustrative
Financial Estimate" to stay visibly separate.

Cost components (per Section 19):
  inventory carrying cost + procurement cost + expected stockout cost
  + expected downtime cost + obsolescence cost + digitization cost
  + reverse engineering cost + qualification cost + manufacturing cost
  + logistics cost + risk penalty

ASSUMPTIONS (all illustrative, tune per real finance input before any real decision):
"""

ASSUMPTIONS = {
    "carrying_cost_rate_annual": 0.20,       # % of inventory value held per year (storage, capital, insurance)
    "stockout_cost_multiplier": 3.0,          # a stockout costs ~3x the part's unit cost in expedite/downtime friction
    "obsolescence_writeoff_rate": 0.15,       # annual probability-weighted write-off risk for obsolete-OEM parts
    "digitization_cost_aed": 3500,            # flat cost to CAD-model + document a part once
    "reverse_engineering_cost_aed": 12000,    # flat cost for metrology + drawing reconstruction
    "qualification_cost_aed": 8000,           # flat cost for first-article qualification of a new manufacturing route
    "cnc_unit_cost_multiplier": 0.6,          # CNC-machined replacement typically ~60% of OEM unit cost, once qualified
    "am_unit_cost_multiplier": 0.5,           # additive manufacturing replacement, ~50% of OEM unit cost
    "cast_forge_unit_cost_multiplier": 0.55,  # casting/forging replacement, ~55% of OEM unit cost
    "local_source_lead_time_reduction_pct": 0.6,  # local sourcing typically cuts lead time ~60%
    "logistics_cost_rate": 0.05,              # % of unit cost for import logistics/customs on OEM purchase route
    "risk_penalty_per_safety_critical_aed": 5000,  # flat risk penalty added for safety-critical parts on any non-Keep route
    "horizon_years": 5,
}


def _row_get(row, key, default=0.0):
    v = row.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def estimate_lifecycle_costs(row: dict | pd.Series, assumptions: dict = ASSUMPTIONS) -> dict:
    """Returns illustrative 5-year lifecycle cost estimate for each candidate strategy.
    All outputs are Illustrative Financial Estimates -- never present as validated cost."""
    a = assumptions
    unit_cost = _row_get(row, "Unit_Cost_AED", 0)
    inv_value = _row_get(row, "Inventory_Value_AED", unit_cost)
    annual_consumption = _row_get(row, "Annual_Consumption", 0)
    downtime_cost_per_day = _row_get(row, "Downtime_Cost_AED_per_day", 0)
    failure_freq = _row_get(row, "Failure_Frequency_per_year", 0)
    lead_time_days = _row_get(row, "Lead_Time_Days", 0)
    safety_critical = str(row.get("Safety_Critical", "No")).strip().lower() == "yes"
    oem_obsolete = str(row.get("OEM_Obsolete", "No")).strip().lower() == "yes"
    horizon = a["horizon_years"]

    carrying_cost = inv_value * a["carrying_cost_rate_annual"] * horizon
    procurement_cost = unit_cost * annual_consumption * horizon
    expected_stockout_cost = unit_cost * a["stockout_cost_multiplier"] * failure_freq * horizon * 0.3
    expected_downtime_cost = downtime_cost_per_day * failure_freq * horizon * (lead_time_days / 30.0) * 0.1
    obsolescence_cost = inv_value * a["obsolescence_writeoff_rate"] * horizon if oem_obsolete else 0.0
    logistics_cost = unit_cost * annual_consumption * a["logistics_cost_rate"] * horizon
    risk_penalty = a["risk_penalty_per_safety_critical_aed"] if safety_critical else 0.0

    strategies = {}

    strategies["Keep_Physical_Current_Strategy"] = round(
        carrying_cost + procurement_cost + expected_stockout_cost + expected_downtime_cost + obsolescence_cost, 0)

    strategies["Increase_Safety_Stock"] = round(
        carrying_cost * 1.4 + procurement_cost + expected_downtime_cost * 0.3 + obsolescence_cost, 0)

    strategies["Reduce_Stock"] = round(
        carrying_cost * 0.5 + procurement_cost + expected_stockout_cost * 1.5 + expected_downtime_cost * 1.2, 0)

    strategies["Digitize_Only"] = round(
        a["digitization_cost_aed"] + carrying_cost * 0.5 + procurement_cost, 0)

    strategies["Digitize_Reverse_Engineer"] = round(
        a["digitization_cost_aed"] + a["reverse_engineering_cost_aed"] + a["qualification_cost_aed"]
        + procurement_cost * a["cnc_unit_cost_multiplier"], 0)

    strategies["CNC_Manufacture"] = round(
        a["qualification_cost_aed"] + procurement_cost * a["cnc_unit_cost_multiplier"]
        + carrying_cost * 0.3 + risk_penalty, 0)

    strategies["Additive_Manufacture"] = round(
        a["qualification_cost_aed"] + procurement_cost * a["am_unit_cost_multiplier"]
        + carrying_cost * 0.2 + risk_penalty, 0)

    strategies["Casting_Forging"] = round(
        a["qualification_cost_aed"] * 1.3 + procurement_cost * a["cast_forge_unit_cost_multiplier"]
        + carrying_cost * 0.3 + risk_penalty, 0)

    strategies["Local_Source"] = round(
        procurement_cost * 1.05 + carrying_cost * (1 - a["local_source_lead_time_reduction_pct"] * 0.3)
        + expected_downtime_cost * (1 - a["local_source_lead_time_reduction_pct"]), 0)

    strategies["OEM_Purchase"] = round(
        procurement_cost + logistics_cost + carrying_cost + expected_downtime_cost, 0)

    strategies["Repair_Refurbish"] = round(
        procurement_cost * 0.4 + carrying_cost * 0.6 + expected_downtime_cost * 0.5, 0)

    cheapest = min(strategies, key=strategies.get)

    return {
        "strategies_5yr_aed": strategies,
        "recommended_by_cost_only": cheapest,
        "recommended_cost_aed": strategies[cheapest],
        "note": "ILLUSTRATIVE FINANCIAL ESTIMATE from stated assumptions on synthetic demo data. "
                "Not a validated cost model -- constraints (safety, IP, service level, manufacturing "
                "readiness, engineering approval) are NOT applied here; see optimize_with_constraints().",
    }


def optimize_with_constraints(row: dict | pd.Series, preds_row: pd.Series,
                               assumptions: dict = ASSUMPTIONS) -> dict:
    """Applies the Section 19 constraints on top of the raw cost estimate: the
    cheapest strategy is only actionable if it doesn't conflict with a safety, IP,
    service-level, manufacturing-readiness, or engineering-approval constraint."""
    cost_result = estimate_lifecycle_costs(row, assumptions)
    strategies = dict(cost_result["strategies_5yr_aed"])

    blocked = []
    safety_critical = str(row.get("Safety_Critical", "No")).strip().lower() == "yes"
    ip_restricted = str(row.get("IP_Restriction", "No")).strip().lower() == "yes"
    engineering_review_required = bool(preds_row.get("Engineering_Review", 0))
    manufacturing_ready = not bool(preds_row.get("is_novel_component", 0))

    if ip_restricted:
        # IP-restricted parts cannot be reverse-engineered or third-party manufactured
        # without legal clearance -- those routes are blocked pending IP_Legal_Review.
        for s in ["Digitize_Reverse_Engineer", "CNC_Manufacture", "Additive_Manufacture", "Casting_Forging"]:
            blocked.append((s, "IP_Restriction=Yes: requires IP_Legal_Review clearance before this route"))
            strategies.pop(s, None)

    if engineering_review_required:
        for s in ["CNC_Manufacture", "Additive_Manufacture", "Casting_Forging"]:
            if s in strategies:
                blocked.append((s, "Engineering_Review required before a manufacturing-ready route can be approved"))
                strategies.pop(s, None)

    if safety_critical and not manufacturing_ready:
        for s in ["CNC_Manufacture", "Additive_Manufacture", "Casting_Forging"]:
            if s in strategies:
                blocked.append((s, "Safety-critical + novel/incomplete data: manufacturing route not approved"))
                strategies.pop(s, None)

    if not strategies:
        recommended = "Engineering_Review_Required"
        recommended_cost = None
    else:
        recommended = min(strategies, key=strategies.get)
        recommended_cost = strategies[recommended]

    return {
        "constrained_strategies_5yr_aed": strategies,
        "blocked_strategies": blocked,
        "recommended_strategy": recommended,
        "recommended_cost_aed": recommended_cost,
        "human_approval_required": True,  # Section 35: never automatic final authority
        "note": cost_result["note"],
    }


def business_opportunity_summary(row: dict | pd.Series, preds_row: pd.Series) -> dict:
    """Section 20: separates Observed Input / Model Prediction / Illustrative Estimate."""
    unit_cost = _row_get(row, "Unit_Cost_AED", 0)
    stock_qty = _row_get(row, "Current_Stock_Qty", 0)
    inv_value = _row_get(row, "Inventory_Value_AED", unit_cost * stock_qty)

    cost_result = estimate_lifecycle_costs(row)
    current_cost = cost_result["strategies_5yr_aed"]["Keep_Physical_Current_Strategy"]
    best_alt = min(
        (v for k, v in cost_result["strategies_5yr_aed"].items() if k != "Keep_Physical_Current_Strategy"),
        default=current_cost,
    )
    opportunity = max(0.0, current_cost - best_alt)

    return {
        "observed_input": {
            "unit_cost_aed": unit_cost,
            "current_stock_qty": stock_qty,
            "inventory_value_aed": round(inv_value, 0),
        },
        "model_prediction": {
            "digitize": int(preds_row.get("Digitize", 0)),
            "reverse_engineer": int(preds_row.get("Reverse_Engineer", 0)),
            "local_source": int(preds_row.get("Local_Source", 0)),
            "manufacturing_candidate": int(any(preds_row.get(t, 0) for t in
                                                ["CNC_Candidate", "AM_Candidate", "Cast_Forge_Candidate"])),
        },
        "illustrative_financial_estimate": {
            "physical_inventory_value_aed": round(inv_value, 0),
            "carrying_cost_exposure_5yr_aed": round(inv_value * ASSUMPTIONS["carrying_cost_rate_annual"] * ASSUMPTIONS["horizon_years"], 0),
            "estimated_5yr_lifecycle_opportunity_aed": round(opportunity, 0),
        },
        "disclaimer": "Illustrative estimates from synthetic demo data and stated assumptions. Not validated financial guidance.",
    }


# ============================================================================
# Source: src/data_validation.py
# ============================================================================
"""
DPC PartIQ - Schema Validation
=================================
Section 29. Shared by the FastAPI backend and the Streamlit upload page. Validates
a part record (single dict or a DataFrame) against the 43 raw legitimate-input
fields, separates missing fields into REQUIRED (analysis can't proceed safely
without them) vs OPTIONAL (filled with a train-set default, degrading confidence
rather than blocking), and never crashes on a malformed row -- it returns a
structured validation result instead.
"""



# REQUIRED: the fields that drive a deterministic rule, or that most heavily shape the
# recommendation. Missing any of these means the recommendation is not reliable enough
# to act on -- the API returns can_proceed=False rather than guessing.
REQUIRED_FIELDS = [
    "Safety_Critical", "IP_Restriction", "Standard_Commercial_Part",
    "Unit_Cost_AED", "Current_Stock_Qty", "Annual_Consumption",
    "Lead_Time_Days", "Supplier_Count", "OEM_Status", "OEM_Obsolete",
]

_DEFAULTS_PATH = CONFIG_DIR / "raw_field_defaults.json"
_RAW_DEFAULTS = json.loads(_DEFAULTS_PATH.read_text()) if _DEFAULTS_PATH.exists() else {}
ALL_RAW_FIELDS = list(_RAW_DEFAULTS.keys())
OPTIONAL_FIELDS = [f for f in ALL_RAW_FIELDS if f not in REQUIRED_FIELDS]


def validate_part(record: dict) -> dict:
    """Validates a single part dict. Returns a report + a filled-in copy (optional
    fields backfilled with train-set defaults; required fields are NEVER backfilled)."""
    present = {k for k, v in record.items() if v is not None and str(v).strip() != ""}
    missing_required = [f for f in REQUIRED_FIELDS if f not in present]
    missing_optional = [f for f in OPTIONAL_FIELDS if f not in present]

    filled = dict(record)
    for f in missing_optional:
        filled[f] = _RAW_DEFAULTS[f]

    can_proceed = len(missing_required) == 0
    return {
        "can_proceed": can_proceed,
        "missing_required_fields": missing_required,
        "missing_optional_fields": missing_optional,
        "n_fields_provided": len(present),
        "n_fields_expected": len(ALL_RAW_FIELDS),
        "message": ("Ready for analysis." if can_proceed else
                    f"Cannot analyze: missing {len(missing_required)} required field(s): {missing_required}. "
                    f"{len(missing_optional)} optional field(s) will be backfilled with typical values if you proceed anyway."),
        "filled_record": filled,
    }


def validate_dataframe(df: pd.DataFrame) -> dict:
    """Validates an uploaded portfolio file. Column-level check (not row-level) --
    a column entirely absent is different from a column with some blank cells, and
    the response reports both."""
    columns_present = set(df.columns)
    missing_required_columns = [f for f in REQUIRED_FIELDS if f not in columns_present]
    missing_optional_columns = [f for f in OPTIONAL_FIELDS if f not in columns_present]

    partial_missing = {}
    for f in REQUIRED_FIELDS:
        if f in df.columns:
            n_blank = df[f].isna().sum()
            if n_blank > 0:
                partial_missing[f] = int(n_blank)

    can_proceed = len(missing_required_columns) == 0
    return {
        "can_proceed": can_proceed,
        "n_rows": len(df),
        "missing_required_columns": missing_required_columns,
        "missing_optional_columns": missing_optional_columns,
        "required_fields_with_blank_cells": partial_missing,
        "message": ("Ready for portfolio analysis." if can_proceed else
                    f"Cannot analyze: missing required column(s): {missing_required_columns}."),
    }


# ============================================================================
# Source: src/narrative.py
# ============================================================================
"""
DPC PartIQ - Narrative Analysis Layer
=========================================
Turns raw predictions/scores/SHAP values into a coherent, readable per-part
analysis: an executive summary paragraph, risk/urgency flags, and peer
benchmarking against similar parts in the portfolio. This is presentation logic
built entirely on top of the existing pipeline (inference.py, explain.py,
optimizer.py) -- no new ML, just synthesis of what's already computed into
something a person (engineer or client) can actually read in one pass instead of
scanning six separate numbers.
"""

BENCHMARK_METRICS = {
    "Unit_Cost_AED": "Unit cost",
    "Lead_Time_Days": "Lead time",
    "stock_coverage_months": "Stock coverage",
    "Downtime_Cost_AED_per_day": "Downtime cost/day",
    "Supplier_Count": "Supplier count",
}

RECOMMENDATION_LABELS = {
    "Keep_Physical": "keep physical stock",
    "Increase_Safety_Stock": "increase safety stock",
    "Reduce_Stock": "reduce excess stock",
    "Consolidate_Duplicate": "consolidate with a duplicate part",
    "Standard_Commercial_Action": "treat as standard commercial procurement",
    "Digitize": "digitize (CAD/drawing capture)",
    "Reverse_Engineer": "reverse-engineer",
    "Repair": "repair/refurbish",
    "OEM_Purchase": "source through the OEM",
    "Local_Source": "source locally",
    "CNC_Candidate": "manufacture via CNC",
    "AM_Candidate": "manufacture via additive manufacturing",
    "Cast_Forge_Candidate": "manufacture via casting/forging",
    "Engineering_Review": "route to engineering review",
    "IP_Legal_Review": "route to IP/legal review",
}


def risk_flags(row: dict | pd.Series) -> list[dict]:
    """Badges: (label, severity) for the header of a part report."""
    flags = []
    if str(row.get("Safety_Critical", "No")).strip() == "Yes":
        flags.append({"label": "Safety critical", "severity": "danger"})
    crit = str(row.get("Asset_Criticality", ""))
    if crit == "Critical":
        flags.append({"label": "Critical asset", "severity": "danger"})
    elif crit == "High":
        flags.append({"label": "High criticality", "severity": "warning"})
    if str(row.get("OEM_Obsolete", "No")).strip() == "Yes":
        flags.append({"label": "OEM obsolete", "severity": "warning"})
    if str(row.get("IP_Restriction", "No")).strip() == "Yes":
        flags.append({"label": "IP restricted", "severity": "warning"})
    try:
        if float(row.get("Supplier_Count", 99)) <= 1:
            flags.append({"label": "Single-source", "severity": "warning"})
    except (TypeError, ValueError):
        pass
    try:
        if float(row.get("stock_coverage_months", 99)) < 1:
            flags.append({"label": "Low stock coverage", "severity": "danger"})
    except (TypeError, ValueError):
        pass
    return flags


def peer_benchmark(df_engineered: pd.DataFrame, row: pd.Series, group_col: str = "Equipment_Class",
                    metrics: dict = BENCHMARK_METRICS, min_group_size: int = 20) -> dict:
    """Percentile rank of this part vs peers sharing the same group_col value.
    Falls back to whole-portfolio comparison if the peer group is too small."""
    group_value = row.get(group_col)
    peers = df_engineered[df_engineered[group_col] == group_value]
    used_whole_portfolio = len(peers) < min_group_size
    if used_whole_portfolio:
        peers = df_engineered

    results = {}
    for col, label in metrics.items():
        if col not in df_engineered.columns:
            continue
        try:
            value = float(row[col])
        except (TypeError, ValueError, KeyError):
            continue
        peer_values = pd.to_numeric(peers[col], errors="coerce").dropna()
        if len(peer_values) < 5:
            continue
        percentile = float((peer_values < value).mean() * 100)
        results[col] = {
            "label": label,
            "value": round(value, 2),
            "peer_median": round(float(peer_values.median()), 2),
            "percentile": round(percentile, 0),
            "n_peers": len(peer_values),
        }
    return {
        "group_col": group_col,
        "group_value": group_value,
        "used_whole_portfolio": used_whole_portfolio,
        "metrics": results,
    }


def _active_recommendations(preds_row: pd.Series, all_targets: list[str]) -> list[str]:
    return [t for t in all_targets if int(preds_row.get(t, 0)) == 1]


def executive_summary(row: dict | pd.Series, preds_row: pd.Series, probas_row: pd.Series,
                       completeness: dict, cost_result: dict, all_targets: list[str],
                       part_id: str = "This part") -> str:
    """One-paragraph plain-English synthesis: what to do, why, how confident, what it costs."""
    active = _active_recommendations(preds_row, all_targets)
    review_required = bool(preds_row.get("Engineering_Review", 0)) or not completeness.get("manufacturing_ready", True)

    if not active:
        action_phrase = "no specific action is currently recommended"
    else:
        labels = [RECOMMENDATION_LABELS.get(t, t) for t in active if t not in ("Engineering_Review",)]
        if labels:
            if len(labels) == 1:
                action_phrase = f"the recommended action is to {labels[0]}"
            else:
                action_phrase = "the recommended actions are: " + ", ".join(labels[:-1]) + f", and {labels[-1]}"
        else:
            action_phrase = "no specific action is currently recommended beyond review"

    completeness_phrase = {
        "HIGH": "based on complete engineering data",
        "MEDIUM": "based on partially complete engineering data",
        "LOW": "based on limited engineering data",
    }.get(completeness.get("completeness_category", "MEDIUM"), "based on available data")

    urgency_bits = []
    try:
        coverage = float(row.get("stock_coverage_months", 99))
        if coverage < 1:
            urgency_bits.append(f"only {coverage:.1f} months of stock coverage remain")
    except (TypeError, ValueError):
        pass
    if str(row.get("OEM_Obsolete", "No")).strip() == "Yes":
        urgency_bits.append("the OEM has discontinued support")
    urgency_phrase = f" This is time-sensitive: {', and '.join(urgency_bits)}." if urgency_bits else ""

    review_phrase = (
        " This recommendation requires human engineering sign-off before action is taken."
        if review_required else
        " No mandatory review gate is currently triggered, though engineering judgment always applies."
    )

    cost_phrase = ""
    if cost_result.get("recommended_cost_aed") is not None:
        cheapest = cost_result.get("recommended_strategy", "")
        current = cost_result.get("constrained_strategies_5yr_aed", {}).get("Keep_Physical_Current_Strategy")
        rec_cost = cost_result.get("recommended_cost_aed")
        if current is not None and rec_cost is not None and current > rec_cost:
            savings = current - rec_cost
            cost_phrase = (f" The illustrative 5-year cost model estimates AED {savings:,.0f} in potential "
                            f"savings versus the current keep-physical baseline, if this route is approved.")

    return (
        f"{part_id}: {action_phrase[0].upper() + action_phrase[1:]}, {completeness_phrase}."
        f"{urgency_phrase}{review_phrase}{cost_phrase}"
    )


RISK_WEIGHTS = {
    "Safety critical": 3.0,
    "Critical asset": 2.0,
    "High criticality": 1.0,
    "OEM obsolete": 1.5,
    "IP restricted": 0.5,
    "Single-source": 1.0,
    "Low stock coverage": 2.5,
}


def priority_score(flags: list[dict], opportunity_aed: float, opportunity_scale_aed: float,
                    review_required: bool) -> float:
    """Composite 0-100 priority score: urgency (risk flags) + normalized cost opportunity
    + a bump for anything already requiring engineering review. Weights are a starting
    point for triage ordering, not a scientifically calibrated formula -- the point is to
    turn 15 separate signals into ONE ranking a person can actually work down a list by."""
    urgency = sum(RISK_WEIGHTS.get(f["label"], 0.5) for f in flags)
    urgency_norm = min(urgency / 10.0, 1.0)  # cap contribution once flags stack up

    opp_norm = 0.0
    if opportunity_scale_aed > 0:
        opp_norm = min(max(opportunity_aed, 0) / opportunity_scale_aed, 1.0)

    review_bump = 0.15 if review_required else 0.0

    score = (0.5 * urgency_norm + 0.35 * opp_norm + review_bump) * 100
    return round(min(score, 100.0), 1)


def compute_priority_queue(df_raw: pd.DataFrame, df_engineered: pd.DataFrame, preds: pd.DataFrame,
                            top_n: int = 20) -> pd.DataFrame:
    """Ranks every part in a portfolio by priority_score(). Reuses the same cost model as
    optimizer.py (no new financial logic -- one estimate_lifecycle_costs() call per row,
    which is pure arithmetic, not model inference, so this is fast even at 3,000 rows)."""
                                                       # for callers who only need risk_flags/
                                                       # peer_benchmark and don't have optimizer.py

    opportunities = []
    for i in range(len(df_raw)):
        row = df_raw.iloc[i].to_dict()
        cost_result = estimate_lifecycle_costs(row)
        strategies = cost_result["strategies_5yr_aed"]
        current = strategies.get("Keep_Physical_Current_Strategy", 0)
        best_alt = min((v for k, v in strategies.items() if k != "Keep_Physical_Current_Strategy"), default=current)
        opportunities.append(max(0.0, current - best_alt))

    opportunity_scale = max(np.percentile(opportunities, 90), 1.0) if opportunities else 1.0

    rows = []
    for i in range(len(df_raw)):
        raw_row = df_raw.iloc[i]
        flags = risk_flags(raw_row)
        review_required = bool(preds.iloc[i].get("Engineering_Review", 0))
        score = priority_score(flags, opportunities[i], opportunity_scale, review_required)
        rows.append({
            "Part_ID": raw_row.get("Part_ID"),
            "Priority score": score,
            "Top risk factors": ", ".join(f["label"] for f in flags[:3]) or "None flagged",
            "Est. 5yr opportunity (AED)": round(opportunities[i], 0),
            "Engineering review": "Yes" if review_required else "No",
            "Sector": raw_row.get("Sector"),
            "Equipment class": raw_row.get("Equipment_Class"),
        })

    queue = pd.DataFrame(rows).sort_values("Priority score", ascending=False).reset_index(drop=True)
    return queue.head(top_n)


# ============================================================================
# Source: src/report_pdf.py
# ============================================================================
"""
DPC PartIQ - PDF Part Report
================================
Generates a client-shareable one-part PDF report: executive summary, risk flags,
peer benchmark, active recommendations with driver charts, and the lifecycle cost
comparison. Built on reportlab (layout) + matplotlib (chart images) -- matplotlib
chosen over exporting Plotly figures because it has no extra system dependency
(Plotly's static image export needs kaleido, which is one more thing that can fail
on a constrained deploy target like Streamlit Cloud).
"""

def _ordinal(n: float) -> str:
    n = int(round(n))
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _driver_chart_image(drivers: list[dict], title: str) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(5.5, 2.2), dpi=150)
    features = [d["feature"] for d in drivers][::-1]
    values = [d["shap_value"] for d in drivers][::-1]
    bar_colors = ["#2ca02c" if v > 0 else "#d62728" for v in values]
    ax.barh(features, values, color=bar_colors)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.axvline(0, color="#888888", linewidth=0.6)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def _cost_chart_image(cost_df: pd.DataFrame) -> io.BytesIO:
    # Costs here can span orders of magnitude (a few thousand AED for a digitize-only
    # route vs. over a million for keeping physical stock) -- on a plain linear scale
    # the cheap, often-recommended options render as a bar too thin to show its own
    # fill color. Coloring the value LABEL text (not just the bar) carries the
    # Recommended/Blocked/Available signal even when the bar itself is a sliver.
    cost_df = cost_df.sort_values("5yr cost (AED)", ascending=False)  # cheapest ends up at top of barh
    color_map = {"Recommended": "#1a7d3a", "Available": "#4a5fc1", "Blocked": "#888888"}
    fig, ax = plt.subplots(figsize=(5.5, 2.8), dpi=150)
    colors_list = [color_map.get(s, "#4a5fc1") for s in cost_df["Status"]]
    bars = ax.barh(cost_df["Strategy"], cost_df["5yr cost (AED)"], color=colors_list,
                    edgecolor="#333333", linewidth=0.4)
    max_val = cost_df["5yr cost (AED)"].max()
    for bar, val, status in zip(bars, cost_df["5yr cost (AED)"], cost_df["Status"]):
        label = f"{val:,.0f}" + {"Recommended": "  (recommended)", "Blocked": "  (blocked)"}.get(status, "")
        ax.text(bar.get_width() + max_val * 0.02, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=6.5, color=color_map.get(status, "#333333"),
                fontweight="bold" if status == "Recommended" else "normal")
    for tick_label, status in zip(ax.get_yticklabels(), cost_df["Status"]):
        tick_label.set_color(color_map.get(status, "#333333"))
    ax.set_xlabel("5-yr illustrative cost (AED)", fontsize=8)
    ax.set_xlim(0, max_val * 1.35)  # headroom for the value + status labels
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_part_report_pdf(
    part_id: str,
    summary_text: str,
    flags: list[dict],
    preds_row: pd.Series,
    completeness: dict,
    bench: dict,
    explanation: dict,
    cost_df: pd.DataFrame,
    active_ml_targets: list[str],
    calibration: dict,
    probas_row: pd.Series,
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                             leftMargin=0.65 * inch, rightMargin=0.65 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleGreen", parent=styles["Title"], textColor=BRAND_GREEN, fontSize=18)
    h2 = ParagraphStyle("H2Green", parent=styles["Heading2"], textColor=BRAND_GREEN, fontSize=12, spaceBefore=10)
    body = styles["Normal"]
    small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=8, textColor=colors.grey)

    story = []
    story.append(Paragraph("DPC PartIQ &mdash; Part Intelligence Report", title_style))
    story.append(Paragraph(f"Part ID: {part_id} &nbsp;&nbsp;|&nbsp;&nbsp; Generated {datetime.datetime.now().strftime('%Y-%m-%d')}", body))
    story.append(Spacer(1, 6))

    warn_table = Table([[Paragraph(
        "Demonstration model trained on synthetic industrial data. Results do not represent "
        "validated field performance.", ParagraphStyle("Warn", parent=body, textColor=WARN_TEXT, fontSize=8.5))]],
        colWidths=[6.7 * inch])
    warn_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), WARN_BG),
                                     ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#ffe69c")),
                                     ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                                     ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
    story.append(warn_table)
    story.append(Spacer(1, 10))

    if flags:
        flag_text = "  ".join(f"[{f['label']}]" for f in flags)
        story.append(Paragraph(flag_text, ParagraphStyle("Flags", parent=body, textColor=colors.HexColor("#b3261e"), fontSize=9)))
        story.append(Spacer(1, 6))

    story.append(Paragraph("Executive summary", h2))
    story.append(Paragraph(summary_text, body))

    if preds_row.get("Engineering_Review") or preds_row.get("is_novel_component"):
        story.append(Spacer(1, 4))
        story.append(Paragraph("<b>Engineering approval required</b> before acting on this recommendation.",
                                ParagraphStyle("Alert", parent=body, textColor=colors.HexColor("#b3261e"))))

    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#e2e5e9")))

    story.append(Paragraph("How this part compares", h2))
    if bench.get("metrics"):
        bench_rows = [["Metric", "This part", "Peer median", "Percentile"]]
        for d in bench["metrics"].values():
            bench_rows.append([d["label"], str(d["value"]), str(d["peer_median"]), _ordinal(d["percentile"])])
        t = Table(bench_rows, colWidths=[1.8 * inch, 1.5 * inch, 1.5 * inch, 1.3 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_GREEN), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafbfc")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        peer_note = f"Benchmarked against {bench['group_value']}" + (
            " (portfolio-wide, peer group too small)" if bench.get("used_whole_portfolio") else "")
        story.append(Paragraph(peer_note, small))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Data completeness", h2))
    story.append(Paragraph(f"{completeness['completeness_score']}/100 ({completeness['completeness_category']})"
                            + (f" &mdash; missing: {', '.join(completeness['missing_fields'])}" if completeness["missing_fields"] else ""), body))

    if active_ml_targets:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Why: top drivers per active recommendation", h2))
        for t in active_ml_targets[:4]:  # cap at 4 to keep the report to a reasonable length
            entry = explanation["targets"].get(t, {})
            if entry.get("top_drivers"):
                p = probas_row.get(t)
                cal = calibration.get(t, {})
                label = cal.get("display_format", "PartIQ Score = XX/100").replace(
                    "XX", f"{p * 100:.0f}" if pd.notna(p) else "n/a")
                driver_section = [
                    Paragraph(f"<b>{t.replace('_', ' ')}</b> &mdash; {label}",
                              ParagraphStyle("SubHead", parent=body, fontSize=9.5, spaceBefore=6)),
                    Image(_driver_chart_image(entry["top_drivers"], ""), width=5.2 * inch, height=2.0 * inch),
                ]
                story.append(KeepTogether(driver_section))

    story.append(Spacer(1, 8))
    cost_section = [
        Paragraph("Lifecycle strategy cost comparison", h2),
        Paragraph("Illustrative estimates from stated assumptions on synthetic demo data "
                  "&mdash; not validated financial guidance.", small),
        Image(_cost_chart_image(cost_df), width=5.5 * inch, height=2.8 * inch),
    ]
    story.append(KeepTogether(cost_section))

    story.append(Spacer(1, 14))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#e2e5e9")))
    story.append(Paragraph("DPC PartIQ Demo v0.1 &mdash; synthetic training data &mdash; not validated field performance.", small))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


# ============================================================================
# Source: src/inference.py
# ============================================================================
"""
DPC PartIQ - Inference Pipeline (rules + ML + OOD combined)
================================================================
Single entry point used by the robustness/challenge tests, the FastAPI backend, and
the Streamlit app. Implements the Section 35 architecture:

  DATA -> ML -> ENGINEERING RULES -> UNCERTAINTY CHECK -> RECOMMENDATION -> HUMAN APPROVAL

Supports save()/load() for versioned model bundles (Section 25) so the ~26s cost of
fitting all 12 ML models only has to be paid once, not on every process start.
"""





ALL_15_TARGETS_ORDER = [
    "Keep_Physical", "Increase_Safety_Stock", "Reduce_Stock", "Consolidate_Duplicate",
    "Standard_Commercial_Action", "Digitize", "Reverse_Engineer", "Repair", "OEM_Purchase",
    "Local_Source", "CNC_Candidate", "AM_Candidate", "Cast_Forge_Candidate",
    "Engineering_Review", "IP_Legal_Review",
]


def _fit_model(model_name, params, X_train, y_train_t, categorical_features):
    if model_name == "CatBoost":
        m = CatBoostClassifier(**params, auto_class_weights="Balanced", random_seed=SEED,
                                verbose=False, allow_writing_files=False, thread_count=-1)
        m.fit(Pool(X_train, y_train_t, cat_features=categorical_features))
    elif model_name == "XGBoost":
        pos = y_train_t.sum(); neg = len(y_train_t) - pos
        spw = (neg / pos) if pos > 0 else 1.0
        m = xgb.XGBClassifier(**params, tree_method="hist", enable_categorical=True,
                               scale_pos_weight=spw, random_state=SEED, eval_metric="logloss", n_jobs=-1)
        m.fit(X_train, y_train_t)
    else:
        m = lgb.LGBMClassifier(**params, class_weight="balanced", random_state=SEED, verbosity=-1, n_jobs=-1)
        m.fit(X_train, y_train_t, categorical_feature=categorical_features)
    return m


def _raw_proba(model_name, model, X, categorical_features):
    if model_name == "CatBoost":
        return model.predict_proba(Pool(X, cat_features=categorical_features))[:, 1]
    return model.predict_proba(X)[:, 1]


def _fit_ood_detector(X_train: pd.DataFrame, numeric_features: list[str]) -> tuple[IsolationForest, dict]:
    """Section 18: simple OOD detection via Isolation Forest on the numeric feature
    space. Not a full categorical-aware novelty model -- a pragmatic first pass that
    flags rows whose numeric profile (stock levels, costs, lead times, ratios...)
    looks unlike anything in training, regardless of which target is being scored."""
    Xn = X_train[numeric_features].fillna(X_train[numeric_features].median())
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=SEED, n_jobs=-1)
    iso.fit(Xn)
    train_scores = iso.decision_function(Xn)
    cutoff = float(np.percentile(train_scores, 2))
    return iso, {"score_cutoff": cutoff}


class PartIQPipeline:
    """Fits (or loads) all 12 ML targets' winning models + OOD detector; call .predict(X) many times."""

    VERSION = "DPC_PartIQ_Demo_v0.1"

    def __init__(self, _skip_fit=False):
        if _skip_fit:
            return
        X_train = pd.read_pickle(PROCESSED_DIR / "X_train.pkl")
        y_train = pd.read_pickle(PROCESSED_DIR / "y_train.pkl")
        manifest = json.loads((PROCESSED_DIR / "feature_manifest.json").read_text())
        tuned_hp = json.loads((MODELS_DIR / "tuned_hyperparameters.json").read_text())
        thresholds = json.loads((CONFIG_DIR / "thresholds.json").read_text())
        rules = load_rules()
        rule_targets = deterministic_target_names(rules)

        self.categorical_features = manifest["categorical_features"]
        self.numeric_features = manifest["numeric_features"]
        self.feature_columns = list(X_train.columns)  # EXACT training order -- XGBoost enforces
                                                         # column-name order at predict time even
                                                         # when the same set of names is present;
                                                         # CatBoost/LightGBM are order-tolerant, which
                                                         # is why a numeric+categorical reconstruction
                                                         # silently broke only the XGBoost-backed target.
        self.thresholds = thresholds
        self.rules = rules
        self.rule_targets = rule_targets
        self.ml_targets = [t for t in manifest["target_columns"] if t not in rule_targets]
        self.trained_at = datetime.datetime.utcnow().isoformat()

        self.models = {}
        for target in self.ml_targets:
            info = thresholds[target]
            winner = info["winning_model"]
            params = tuned_hp[target][winner]
            self.models[target] = {
                "model": _fit_model(winner, params, X_train, y_train[target], self.categorical_features),
                "model_name": winner,
                "threshold": info["threshold"],
            }

        self.ood_detector, self.ood_meta = _fit_ood_detector(X_train, self.numeric_features)

    # -- persistence ---------------------------------------------------------
    def save(self, bundle_dir: Path = DEFAULT_BUNDLE_DIR):
        bundle_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "models": self.models,
            "ood_detector": self.ood_detector,
            "ood_meta": self.ood_meta,
            "categorical_features": self.categorical_features,
            "numeric_features": self.numeric_features,
            "feature_columns": self.feature_columns,
            "thresholds": self.thresholds,
            "rules": self.rules,
            "rule_targets": self.rule_targets,
            "ml_targets": self.ml_targets,
            "trained_at": self.trained_at,
            "version": self.VERSION,
        }, bundle_dir / "pipeline_bundle.joblib")
        metadata = {
            "version": self.VERSION,
            "trained_at": self.trained_at,
            "ml_targets": self.ml_targets,
            "rule_targets": self.rule_targets,
            "winning_models": {t: info["model_name"] for t, info in self.models.items()},
            "thresholds": {t: self.thresholds[t]["threshold"] for t in self.ml_targets},
        }
        with open(bundle_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        return bundle_dir

    @classmethod
    def load(cls, bundle_dir: Path = DEFAULT_BUNDLE_DIR) -> "PartIQPipeline":
        data = joblib.load(bundle_dir / "pipeline_bundle.joblib")
        obj = cls(_skip_fit=True)
        for k, v in data.items():
            setattr(obj, k, v)
        return obj

    # -- inference -------------------------------------------------------------
    def _align_dtypes(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.categorical_features:
            if c in X.columns and X[c].dtype.name != "category":
                X[c] = X[c].astype("category")
        for c in self.numeric_features:
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce")
        return X

    def ood_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        Xn = X[self.numeric_features].fillna(X[self.numeric_features].median())
        scores = self.ood_detector.decision_function(Xn)
        is_novel = scores < self.ood_meta["score_cutoff"]
        return pd.DataFrame({"ood_score": scores, "is_novel": is_novel}, index=X.index)

    def predict_raw(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """ML + rules, NO engineering-rule overrides applied yet. Returns (predictions, probabilities).
        Accepts a DataFrame with AT LEAST self.feature_columns present -- extra columns
        (Part_ID, raw pre-engineering fields, etc.) are fine and are dropped here rather
        than leaking into the model call, where an unexpected column count/order would
        break CatBoost/XGBoost's Pool/DMatrix construction."""
        X = self._align_dtypes(X)
        missing_features = [c for c in self.feature_columns if c not in X.columns]
        if missing_features:
            raise ValueError(f"Input is missing required engineered/feature columns: {missing_features}")
        X_features = X[self.feature_columns]  # exact columns, exact order the models were trained on

        preds = pd.DataFrame(index=X.index, columns=ALL_15_TARGETS_ORDER, dtype=float)
        probas = pd.DataFrame(index=X.index, columns=ALL_15_TARGETS_ORDER, dtype=float)

        rule_preds, dup_review_flag = apply_deterministic_gates(X, self.rules)
        for t in self.rule_targets:
            preds[t] = rule_preds[t]
        probas[self.rule_targets] = np.nan

        abstained = pd.Series(False, index=X.index)
        for target, info in self.models.items():
            try:
                proba = _raw_proba(info["model_name"], info["model"], X_features, self.categorical_features)
                probas[target] = proba
                preds[target] = (proba >= info["threshold"]).astype(int)
            except Exception:
                proba = np.full(len(X), np.nan)
                pred = np.zeros(len(X), dtype=int)
                for i, (idx, row) in enumerate(X_features.iterrows()):
                    try:
                        p = _raw_proba(info["model_name"], info["model"], X_features.loc[[idx]], self.categorical_features)[0]
                        proba[i] = p
                        pred[i] = int(p >= info["threshold"])
                    except Exception:
                        abstained.loc[idx] = True
                probas[target] = proba
                preds[target] = pred

        self._last_abstained = abstained | dup_review_flag
        return preds.astype(int, errors="ignore"), probas

    def predict(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Full pipeline: ML + rules -> engineering-rule overlays -> abstention -> OOD flag.
        Returns (final predictions incl. override_reason/ood columns, raw probabilities)."""
        preds, probas = self.predict_raw(X)
        final = apply_overlays(X, preds, self.rules)

        if getattr(self, "_last_abstained", None) is not None and self._last_abstained.any():
            final.loc[self._last_abstained, "Engineering_Review"] = 1
            final.loc[self._last_abstained, "override_reason"] += "abstained_unscoreable_input; "

        ood = self.ood_scores(self._align_dtypes(X))
        final["ood_score"] = ood["ood_score"]
        final["is_novel_component"] = ood["is_novel"].astype(int)
        novel = ood["is_novel"]
        final.loc[novel, "Engineering_Review"] = 1
        final.loc[novel, "override_reason"] = final.loc[novel, "override_reason"] + "novel_component_ood; "

        return final, probas


_PIPELINE_SINGLETON = None


def get_pipeline(prefer_bundle: bool = True) -> PartIQPipeline:
    """Loads the saved bundle if present (fast path); otherwise fits fresh (slow path,
    used the first time / in dev). Call PartIQPipeline().save() once to create the bundle."""
    global _PIPELINE_SINGLETON
    if _PIPELINE_SINGLETON is None:
        if prefer_bundle and (DEFAULT_BUNDLE_DIR / "pipeline_bundle.joblib").exists():
            _PIPELINE_SINGLETON = PartIQPipeline.load(DEFAULT_BUNDLE_DIR)
        else:
            _PIPELINE_SINGLETON = PartIQPipeline()
    return _PIPELINE_SINGLETON


# ============================================================================
# Source: src/explain.py
# ============================================================================
"""
DPC PartIQ - Explainability
==============================
Section 14/21. For a single part + target, returns the top contributing features
(SHAP values) in the direction that matters for a business user: "+" pushes toward
the positive recommendation, "-" pushes away from it. Rule-based targets (IP_Legal_Review
etc.) don't get SHAP explanations -- they get the rule condition instead, which IS the
full explanation.
"""


warnings.filterwarnings("ignore", module="shap")


TOP_N_DRIVERS = 5

# cache SHAP explainers per target -- building one is not free
_EXPLAINER_CACHE: dict = {}


def _get_explainer(pipeline, target: str):
    if target in _EXPLAINER_CACHE:
        return _EXPLAINER_CACHE[target]
    info = pipeline.models[target]
    explainer = shap.TreeExplainer(info["model"])
    _EXPLAINER_CACHE[target] = explainer
    return explainer


def explain_prediction(pipeline, X_row: pd.DataFrame, target: str) -> dict:
    """X_row: a single-row DataFrame with the pipeline's feature columns already aligned."""
    if target in pipeline.rule_targets:
        rule_map = {g["target"]: g for g in pipeline.rules["deterministic_gates"]}
        gate = rule_map.get(target, {})
        return {
            "target": target,
            "type": "rule",
            "condition": gate.get("condition"),
            "explanation": f"Deterministic rule: {gate.get('condition')} -> {gate.get('action')}",
        }

    info = pipeline.models[target]
    X_aligned = pipeline._align_dtypes(X_row)[pipeline.feature_columns]
    explainer = _get_explainer(pipeline, target)

    try:
        if info["model_name"] == "CatBoost":
            from catboost import Pool
            shap_values = explainer.shap_values(Pool(X_aligned, cat_features=pipeline.categorical_features))
        else:
            shap_values = explainer.shap_values(X_aligned)
    except Exception as e:
        return {"target": target, "type": "ml", "error": f"SHAP unavailable for this row: {e}"}

    sv = np.asarray(shap_values).reshape(-1)
    order = np.argsort(-np.abs(sv))[:TOP_N_DRIVERS]
    drivers = []
    for i in order:
        feature = pipeline.feature_columns[i]
        value = X_aligned.iloc[0][feature]
        drivers.append({
            "feature": feature,
            "value": value if not isinstance(value, (np.generic,)) else value.item(),
            "shap_value": round(float(sv[i]), 4),
            "direction": "+" if sv[i] > 0 else "-",
        })

    return {"target": target, "type": "ml", "model": info["model_name"], "top_drivers": drivers}


def explain_part(pipeline, X_row: pd.DataFrame, preds_row: pd.Series, probas_row: pd.Series,
                  calibration: dict | None = None) -> dict:
    """Full Section 14-style explanation bundle for one part across all 15 targets."""
    completeness = completeness_report(X_row.iloc[0].to_dict())
    result = {
        "completeness": completeness,
        "targets": {},
    }
    for target in pipeline.rule_targets + pipeline.ml_targets:
        entry = explain_prediction(pipeline, X_row, target)
        entry["prediction"] = int(preds_row[target])
        if target in pipeline.ml_targets:
            proba = probas_row[target]
            entry["score"] = None if pd.isna(proba) else round(float(proba) * 100, 1)
            cal = (calibration or {}).get(target, {})
            entry["display_format"] = cal.get("display_format", "PartIQ Score = XX/100")
        result["targets"][target] = entry

    result["engineering_review_required"] = bool(preds_row.get("Engineering_Review", 0)) or not completeness["manufacturing_ready"]
    result["override_reason"] = preds_row.get("override_reason", "")
    result["is_novel_component"] = bool(preds_row.get("is_novel_component", 0))
    return result



# ============================================================================
# Streamlit UI
# ============================================================================

st.set_page_config(page_title="DPC PartIQ", page_icon="\U0001F527", layout="wide")


@st.cache_resource(show_spinner="Loading PartIQ model...")
def load_pipeline():
    return get_pipeline()


@st.cache_data(show_spinner=False)
def load_demo_portfolio():
    df = pd.read_pickle(PROJECT_ROOT / "data" / "raw" / "dpc_partiq_raw.pkl")
    return df


@st.cache_data(show_spinner=False)
def load_demo_portfolio_engineered():
    return engineer_features(load_demo_portfolio())


def disclaimer_banner():
    st.warning(
        "**Demonstration model trained on synthetic industrial data. "
        "Results do not represent validated field performance.**",
        icon="\u26A0\uFE0F",
    )


def run_predictions(df: pd.DataFrame, pipeline):
    row_df = engineer_features(df)
    preds, probas = pipeline.predict(row_df)
    return preds, probas


# ---------------------------------------------------------------------------
st.title("DPC PartIQ")
st.caption("AI-Powered Spare Parts Intelligence & Optimization Platform")
disclaimer_banner()

pipeline = load_pipeline()

page = st.sidebar.radio(
    "Navigate",
    ["Portfolio Dashboard", "Part Explorer", "Manufacturing Intelligence", "Model Validation"],
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Model version: `{pipeline.VERSION}`")
st.sidebar.caption(f"Trained: {pipeline.trained_at[:19]} UTC")

demo_df = load_demo_portfolio()
demo_engineered = load_demo_portfolio_engineered()

# ===========================================================================
# PAGE 1 - Portfolio Dashboard
# ===========================================================================
if page == "Portfolio Dashboard":
    st.header("Portfolio Dashboard")

    st.subheader("1. Load a portfolio")
    upload = st.file_uploader("Upload CSV or XLSX (or use the built-in demo dataset below)", type=["csv", "xlsx"])
    use_demo = st.checkbox("Use built-in demo dataset (3,000 synthetic parts)", value=upload is None)

    if upload is not None:
        try:
            df = pd.read_csv(upload) if upload.name.endswith(".csv") else pd.read_excel(upload)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()
        validation = validate_dataframe(df)
        if not validation["can_proceed"]:
            st.error(validation["message"])
            st.json(validation)
            st.stop()
        if validation["missing_optional_columns"]:
            st.info(f"{len(validation['missing_optional_columns'])} optional column(s) missing -- "
                    f"backfilled with typical values: {validation['missing_optional_columns'][:8]}"
                    f"{'...' if len(validation['missing_optional_columns']) > 8 else ''}")
            for f in validation["missing_optional_columns"]:
                df[f] = _RAW_DEFAULTS[f]
        for col in ALL_RAW_FIELDS:
            if col not in df.columns:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode.iloc[0] if len(mode) else "Unknown")
    elif use_demo:
        df = demo_df
    else:
        st.info("Upload a file or check the demo-dataset box to continue.")
        st.stop()

    with st.spinner("Running PartIQ analysis..."):
        preds, probas = run_predictions(df, pipeline)
        completeness = completeness_dataframe(engineer_features(df))

    st.subheader("2. Portfolio summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Parts analyzed", len(df))
    c2.metric("Engineering review flagged", int(preds["Engineering_Review"].sum()))
    c3.metric("Digitization candidates", int(preds["Digitize"].sum()) if "Digitize" in preds else "n/a")
    c4.metric("IP legal review flagged", int(preds["IP_Legal_Review"].sum()))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Reverse engineering candidates", int(preds["Reverse_Engineer"].sum()) if "Reverse_Engineer" in preds else "n/a")
    mfg_candidates = preds[["CNC_Candidate", "AM_Candidate", "Cast_Forge_Candidate"]].max(axis=1).sum()
    c6.metric("Local manufacturing candidates", int(mfg_candidates))
    c7.metric("Novel components (OOD)", int(preds["is_novel_component"].sum()))
    high_risk_data = (completeness["completeness_category"] == "LOW").sum()
    c8.metric("Low data-completeness parts", int(high_risk_data))

    st.subheader("3. Recommendation breakdown")
    target_cols = [c for c in preds.columns if c not in ("override_reason", "ood_score", "is_novel_component")]
    rec_counts = preds[target_cols].sum().sort_values(ascending=True)
    fig = px.bar(rec_counts, orientation="h", labels={"value": "Parts flagged", "index": "Recommendation"},
                 title="Parts flagged per recommendation category")
    fig.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig, width='stretch')

    st.subheader("4. Priority action queue")
    st.caption("The whole portfolio ranked by urgency (safety, criticality, obsolescence, stock risk) "
               "plus estimated cost opportunity -- one ordered list instead of 15 separate counts.")
    row_engineered = engineer_features(df).reset_index(drop=True)
    with st.spinner("Ranking portfolio by priority..."):
        priority_queue = compute_priority_queue(df, row_engineered, preds, top_n=20)
    st.dataframe(priority_queue, width='stretch', hide_index=True)

    st.subheader("5. Illustrative business opportunity")
    st.caption("Illustrative estimates from stated assumptions on synthetic demo data -- not validated financial guidance.")
    sample_n = min(200, len(df))
    opp_rows = []
    for i in range(sample_n):
        opp = business_opportunity_summary(row_engineered.iloc[i].to_dict(), preds.iloc[i])
        opp_rows.append(opp["illustrative_financial_estimate"]["estimated_5yr_lifecycle_opportunity_aed"])
    total_opportunity_estimate = sum(opp_rows) * (len(df) / sample_n)
    st.metric(f"Estimated 5-yr portfolio opportunity (extrapolated from {sample_n}-part sample)",
              f"AED {total_opportunity_estimate:,.0f}")

# ===========================================================================
# PAGE 2 - Part Explorer
# ===========================================================================
elif page == "Part Explorer":
    st.header("Part Explorer")

    search_mode = st.radio("Search by", ["Part ID", "Component / Equipment / Sector filter", "Manual entry"], horizontal=True)

    selected_row = None
    if search_mode == "Part ID":
        part_id = st.selectbox("Select Part ID", demo_df["Part_ID"].tolist())
        selected_row = demo_df[demo_df["Part_ID"] == part_id].iloc[[0]]
    elif search_mode == "Component / Equipment / Sector filter":
        c1, c2, c3 = st.columns(3)
        sector = c1.selectbox("Sector", ["Any"] + sorted(demo_df["Sector"].unique().tolist()))
        equip = c2.selectbox("Equipment Class", ["Any"] + sorted(demo_df["Equipment_Class"].unique().tolist()))
        comp = c3.selectbox("Component Category", ["Any"] + sorted(demo_df["Component_Category"].unique().tolist()))
        filtered = demo_df.copy()
        if sector != "Any":
            filtered = filtered[filtered["Sector"] == sector]
        if equip != "Any":
            filtered = filtered[filtered["Equipment_Class"] == equip]
        if comp != "Any":
            filtered = filtered[filtered["Component_Category"] == comp]
        st.caption(f"{len(filtered)} matching parts")
        if len(filtered):
            part_id = st.selectbox("Select Part ID from filtered results", filtered["Part_ID"].tolist())
            selected_row = demo_df[demo_df["Part_ID"] == part_id].iloc[[0]]
    else:
        st.caption("Enter values for a hypothetical part. Unfilled fields use typical (median/mode) values.")
        manual = {}
        cols = st.columns(3)
        manual["Safety_Critical"] = cols[0].selectbox("Safety Critical", ["No", "Yes"])
        manual["IP_Restriction"] = cols[1].selectbox("IP Restriction", ["No", "Yes"])
        manual["Standard_Commercial_Part"] = cols[2].selectbox("Standard Commercial Part", ["No", "Yes"])
        manual["Unit_Cost_AED"] = cols[0].number_input("Unit Cost (AED)", min_value=0.0, value=1000.0)
        manual["Current_Stock_Qty"] = cols[1].number_input("Current Stock Qty", min_value=0.0, value=2.0)
        manual["Annual_Consumption"] = cols[2].number_input("Annual Consumption", min_value=0.0, value=1.0)
        manual["Lead_Time_Days"] = cols[0].number_input("Lead Time (days)", min_value=0.0, value=90.0)
        manual["Supplier_Count"] = cols[1].number_input("Supplier Count", min_value=0.0, value=2.0)
        manual["OEM_Status"] = cols[2].selectbox("OEM Status", ["Active", "Limited Support", "End-of-Life Announced", "Obsolete"])
        manual["OEM_Obsolete"] = cols[0].selectbox("OEM Obsolete", ["No", "Yes"])
        if st.button("Analyze this part"):
            validation = validate_part(manual)
            selected_row = pd.DataFrame([validation["filled_record"]])

    if selected_row is not None:
        with st.spinner("Running PartIQ analysis..."):
            row_engineered = engineer_features(selected_row)
            preds, probas = pipeline.predict(row_engineered)
            preds_row, probas_row = preds.iloc[0], probas.iloc[0]
            raw_row = selected_row.iloc[0]
            comp = completeness_report(raw_row.to_dict())
            cost = optimize_with_constraints(raw_row.to_dict(), preds_row)
            all_targets = pipeline.rule_targets + pipeline.ml_targets
            part_label = str(raw_row.get("Part_ID", "This part"))
            summary_text = executive_summary(raw_row, preds_row, probas_row, comp, cost, all_targets, part_id=part_label)
            flags = risk_flags(raw_row)
            bench = peer_benchmark(demo_engineered, row_engineered.iloc[0])

        # --- Executive summary ------------------------------------------------
        st.subheader(f"Part intelligence report: {part_label}")
        if flags:
            colors = {"danger": ("#fdecea", "#b3261e"), "warning": ("#fff4e5", "#8a5300")}
            badge_html = " ".join(
                f'<span style="background:{colors[f["severity"]][0]}; color:{colors[f["severity"]][1]}; '
                f'padding:3px 10px; border-radius:12px; font-size:12px; margin-right:6px;">{f["label"]}</span>'
                for f in flags
            )
            st.markdown(badge_html, unsafe_allow_html=True)
        st.info(summary_text)

        if preds_row["Engineering_Review"] or preds_row["is_novel_component"]:
            st.error("**Engineering approval required** before acting on this recommendation.")
        if preds_row["override_reason"]:
            st.caption(f"Rule overrides fired: {preds_row['override_reason']}")

        # --- Peer benchmark -----------------------------------------------------
        st.subheader("How this part compares")
        n_peers = next(iter(bench["metrics"].values()))["n_peers"] if bench["metrics"] else 0
        group_note = " (peer group too small, compared portfolio-wide)" if bench["used_whole_portfolio"] else ""
        st.caption(f"Benchmarked against {bench['group_value']} ({n_peers} peers){group_note}")
        if bench["metrics"]:
            bench_rows = [{"Metric": d["label"], "This part": d["value"], "Peer median": d["peer_median"],
                            "Percentile": d["percentile"]} for d in bench["metrics"].values()]
            bench_df = pd.DataFrame(bench_rows)
            fig = px.bar(bench_df, x="Percentile", y="Metric", orientation="h", range_x=[0, 100])
            fig.add_vline(x=50, line_dash="dot", line_color="gray")
            fig.update_layout(height=220, margin=dict(t=10, b=10), showlegend=False,
                               xaxis_title="Percentile vs peers (50 = typical)")
            st.plotly_chart(fig, width='stretch', key="peer_benchmark_chart")
            st.dataframe(bench_df.set_index("Metric"), width='stretch')

        st.metric("Data completeness", f"{comp['completeness_score']}/100", comp["completeness_category"])
        if comp["missing_fields"]:
            st.caption(f"Missing: {', '.join(comp['missing_fields'])}")

        # --- Why: driver breakdown for every ACTIVE recommendation --------------
        st.subheader("Why: driver breakdown per active recommendation")
        explanation = explain_part(pipeline, row_engineered, preds_row, probas_row, CALIBRATION)
        active_ml_targets = [t for t in pipeline.ml_targets if preds_row[t] == 1]
        if not active_ml_targets:
            st.caption("No ML-driven recommendations are currently active for this part.")
        for t in active_ml_targets:
            entry = explanation["targets"][t]
            p = probas_row[t]
            cal = CALIBRATION.get(t, {})
            score_label = cal.get("display_format", "PartIQ Score = XX/100").replace(
                "XX", f"{p * 100:.0f}" if pd.notna(p) else "n/a")
            with st.expander(f"{t.replace('_', ' ')} — {score_label}", expanded=(len(active_ml_targets) == 1)):
                if entry.get("top_drivers"):
                    drivers_df = pd.DataFrame(entry["top_drivers"])
                    fig = px.bar(drivers_df, x="shap_value", y="feature", orientation="h", color="direction",
                                 color_discrete_map={"+": "#2ca02c", "-": "#d62728"})
                    fig.update_layout(height=220, margin=dict(t=10), showlegend=False)
                    st.plotly_chart(fig, width='stretch', key=f"driver_chart_{t}")
                else:
                    st.caption(entry.get("error", "No SHAP drivers available."))

        with st.expander("Explore any target's score and drivers (including inactive ones)"):
            target_pick = st.selectbox("Target", pipeline.ml_targets, key="target_pick_explore")
            entry = explanation["targets"][target_pick]
            p = probas_row[target_pick]
            st.progress(float(p) if pd.notna(p) else 0.0,
                        text=f"score: {p * 100:.1f}" if pd.notna(p) else "n/a")
            if entry.get("top_drivers"):
                drivers_df = pd.DataFrame(entry["top_drivers"])
                fig = px.bar(drivers_df, x="shap_value", y="feature", orientation="h", color="direction",
                             color_discrete_map={"+": "#2ca02c", "-": "#d62728"})
                fig.update_layout(height=220, margin=dict(t=10), showlegend=False)
                st.plotly_chart(fig, width='stretch', key="explore_target_chart")

        # --- Lifecycle cost comparison --------------------------------------------
        st.subheader("Lifecycle strategy cost comparison")
        st.caption("Illustrative estimates from stated assumptions on synthetic demo data -- not validated financial guidance.")
        blocked = {b[0]: b[1] for b in cost["blocked_strategies"]}
        raw_all = estimate_lifecycle_costs(raw_row.to_dict())["strategies_5yr_aed"]
        cost_rows = []
        for strat, val in raw_all.items():
            status = "Blocked" if strat in blocked else ("Recommended" if strat == cost["recommended_strategy"] else "Available")
            cost_rows.append({"Strategy": strat.replace("_", " "), "5yr cost (AED)": val,
                               "Status": status, "Reason": blocked.get(strat, "")})
        cost_df = pd.DataFrame(cost_rows).sort_values("5yr cost (AED)")
        fig = px.bar(cost_df, x="5yr cost (AED)", y="Strategy", orientation="h", color="Status",
                     color_discrete_map={"Recommended": "#2ca02c", "Available": "#7f9cf5", "Blocked": "#cccccc"},
                     hover_data=["Reason"])
        fig.update_layout(height=380, margin=dict(t=10))
        st.plotly_chart(fig, width='stretch', key="cost_comparison_chart")
        if blocked:
            st.caption("Blocked: " + "; ".join(f"{k.replace('_', ' ')} — {v}" for k, v in blocked.items()))

        # --- Downloadable PDF report -----------------------------------------------
        st.subheader("Download report")
        st.caption("A shareable one-part summary -- executive summary, benchmark, driver charts, and cost comparison in one PDF.")
        try:
            pdf_bytes = generate_part_report_pdf(
                part_label, summary_text, flags, preds_row, comp, bench, explanation,
                cost_df, active_ml_targets, CALIBRATION, probas_row,
            )
            st.download_button("Download PDF report", data=pdf_bytes,
                                file_name=f"PartIQ_Report_{part_label}.pdf", mime="application/pdf")
        except Exception as e:
            st.caption(f"PDF generation unavailable: {e}")


# ===========================================================================
# PAGE 3 - Manufacturing Intelligence
# ===========================================================================
elif page == "Manufacturing Intelligence":
    st.header("Manufacturing Intelligence")
    st.caption("Compares CNC / AM / Casting-Forging / Repair / OEM routes across the demo portfolio.")

    with st.spinner("Scoring portfolio..."):
        preds, probas = run_predictions(demo_df, pipeline)

    route_cols = ["CNC_Candidate", "AM_Candidate", "Cast_Forge_Candidate", "Repair", "OEM_Purchase"]
    route_counts = preds[route_cols].sum()
    fig = px.pie(values=route_counts.values, names=route_counts.index,
                 title="Manufacturing / procurement route distribution")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Route suitability by material group")
    combo = pd.concat([demo_engineered[["Material_Group"]].reset_index(drop=True),
                        preds[route_cols].reset_index(drop=True)], axis=1)
    by_material = combo.groupby("Material_Group")[route_cols].mean().round(3) * 100
    st.dataframe(by_material.style.background_gradient(cmap="Blues", axis=None), width='stretch')

    st.subheader("Multi-candidate parts")
    st.caption("Parts flagged for more than one manufacturing route -- worth a closer engineering look.")
    multi = preds[route_cols].sum(axis=1)
    multi_candidates = demo_df.loc[multi[multi > 1].index, ["Part_ID", "Component_Category", "Equipment_Class"]]
    st.dataframe(multi_candidates.head(30), width='stretch')

# ===========================================================================
# PAGE 4 - Model Validation
# ===========================================================================
elif page == "Model Validation":
    st.header("Model Validation")
    disclaimer_banner()

    reports_dir = PROJECT_ROOT / "reports"

    st.subheader("Test-set performance (final, untouched test set)")
    per_target_path = reports_dir / "stage10_test_metrics_per_target.csv"
    if per_target_path.exists():
        metrics_df = pd.read_csv(per_target_path)
        st.dataframe(metrics_df[["target", "model", "f1", "precision", "recall",
                                   "false_positive_rate", "false_negative_rate", "support_positive"]],
                     width='stretch')
        fig = px.bar(metrics_df, x="target", y="f1", color="model", title="F1 by target (test set)")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, width='stretch')

    overall_path = reports_dir / "stage10_test_metrics_overall.json"
    if overall_path.exists():
        overall = json.loads(overall_path.read_text())
        cols = st.columns(len(overall) - 1)
        for i, (k, v) in enumerate([kv for kv in overall.items() if kv[0] != "n_test_rows"]):
            cols[i % len(cols)].metric(k.replace("_", " ").title(), f"{v:.4f}" if isinstance(v, float) else v)

    st.subheader("Calibration")
    calib_path = reports_dir / "stage9_calibration.csv"
    if calib_path.exists():
        st.dataframe(pd.read_csv(calib_path), width='stretch')

    st.subheader("Generalization challenge (held-out sector)")
    gen_path = reports_dir / "stage11_generalization_challenge.csv"
    if gen_path.exists():
        gen_df = pd.read_csv(gen_path)
        fig = px.bar(gen_df, x="target", y="f1_degradation", title="F1 degradation on held-out Marine & Offshore sector")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, width='stretch')

    st.subheader("Missing-data stress test")
    stress_path = reports_dir / "stage12_missing_data_stress_test.csv"
    if stress_path.exists():
        stress_df = pd.read_csv(stress_path)
        overall_by_level = stress_df.groupby("corruption_level")[["f1_corrupt", "f1_clean"]].mean()
        st.line_chart(overall_by_level)

    st.subheader("Model info")
    st.json({
        "version": pipeline.VERSION,
        "trained_at": pipeline.trained_at,
        "ml_targets": pipeline.ml_targets,
        "rule_based_targets": pipeline.rule_targets,
    })
