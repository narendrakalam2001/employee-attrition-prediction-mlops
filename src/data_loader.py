# ============================================================
# DATA LOADER + FEATURE ENGINEERING — Employee Attrition ML System
# ============================================================

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# ── Required columns for IBM HR Analytics Attrition dataset ──
REQUIRED_COLS = [
    "age", "distancefromhome", "monthlyincome", "totalworkingyears",
    "yearsatcompany", "overtime", "jobsatisfaction", "attrition"
]

# ── Columns that carry zero signal (constant / pure identifiers) ──
DROP_COLS = ["employeecount", "employeenumber", "over18", "standardhours"]


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_input_data(df: pd.DataFrame) -> pd.DataFrame:

    # ── Normalize column names ────────────────────────────────
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
        .str.lower()
    )

    # ── Drop irrelevant ID / constant columns ─────────────────
    for col in DROP_COLS:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            logger.info("Dropped column: %s", col)

    # ── Target encoding: Yes/No → 1/0 ─────────────────────────
    if "attrition" in df.columns and df["attrition"].dtype == object:
        df["attrition"] = df["attrition"].map({"Yes": 1, "No": 0, "yes": 1, "no": 0})

    # ── Check required columns ───────────────────────────────
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Target validation ────────────────────────────────────
    if not set(df["attrition"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Target 'attrition' must contain only 0 and 1")

    # ── Null check ───────────────────────────────────────────
    nulls = df.isnull().sum().sum()
    if nulls > 0:
        logger.warning("Dataset contains %d missing values — will be handled in preprocessing", nulls)

    # ── Minimum size check ───────────────────────────────────
    if df.shape[0] < 100:
        raise ValueError("Dataset too small for training (< 100 rows)")

    # ── Deduplication ────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(ignore_index=True, inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate rows", dropped)

    logger.info("Data validation passed  |  shape=%s  |  attrition_rate=%.3f",
                df.shape, df["attrition"].mean())

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    HR-analytics-grade feature engineering for employee attrition risk.
    Every feature maps to a real retention signal used by People Analytics
    teams (TCS / Accenture / Infosys-style attrition models).
    """
    df = df.copy()

    # ── Overtime flag (binary encode Yes/No → 1/0) ────────────
    if "overtime" in df.columns and df["overtime"].dtype == object:
        df["overtime"] = df["overtime"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # ── Tenure & mobility ratios ──────────────────────────────
    df["tenure_ratio"] = df["yearsatcompany"] / (df["totalworkingyears"].replace(0, 1) + 1)

    if "yearssincelastpromotion" in df.columns:
        df["promotion_gap_ratio"] = df["yearssincelastpromotion"] / (df["yearsatcompany"] + 1)

    if "yearswithcurrmanager" in df.columns:
        df["manager_stability"] = df["yearswithcurrmanager"] / (df["yearsatcompany"] + 1)

    if "yearsincurrentrole" in df.columns:
        df["role_stagnation"] = df["yearsincurrentrole"] / (df["yearsatcompany"] + 1)

    if "numcompaniesworked" in df.columns:
        df["job_hopping_score"] = df["numcompaniesworked"] / (df["totalworkingyears"] + 1)
        df["avg_years_per_company"] = df["totalworkingyears"] / (df["numcompaniesworked"] + 1)

    # ── Income signals ─────────────────────────────────────────
    if "joblevel" in df.columns:
        df["income_per_level"] = df["monthlyincome"] / df["joblevel"].replace(0, 1)

    if "percentsalaryhike" in df.columns:
        df["hike_ratio"] = df["percentsalaryhike"] / 100.0

    # ── Composite satisfaction score ──────────────────────────
    satisfaction_cols = [c for c in [
        "environmentsatisfaction", "jobsatisfaction",
        "relationshipsatisfaction", "worklifebalance"
    ] if c in df.columns]

    if satisfaction_cols:
        df["satisfaction_composite"] = df[satisfaction_cols].mean(axis=1)
        df["low_satisfaction_flag"] = (df["satisfaction_composite"] <= 2.0).astype(int)

    # ── Risk flags (map to hard business rules used downstream) ─
    df["high_distance_flag"] = (df["distancefromhome"] >= 20).astype(int)
    df["low_tenure_flag"]    = (df["yearsatcompany"] <= 1).astype(int)

    if "yearssincelastpromotion" in df.columns:
        df["long_promotion_gap_flag"] = (df["yearssincelastpromotion"] >= 5).astype(int)

    # ── Interaction features ───────────────────────────────────
    df["overtime_x_low_satisfaction"] = df["overtime"] * df.get("low_satisfaction_flag", 0)
    df["income_x_joblevel"] = df["monthlyincome"] * df.get("joblevel", 1)

    logger.info("Feature engineering done  |  total columns=%d", df.shape[1])

    return df


# ============================================================
# FEATURE TYPE DETECTION
# ============================================================

def detect_feature_types(df: pd.DataFrame, threshold: int = 10):
    """
    Auto-detect: ordinal / continuous / binary columns.
    Excludes target column 'attrition'.
    """
    ordinal_cols    = []
    continuous_cols = []
    binary_cols     = []

    for col in df.columns:
        if col == "attrition":
            continue

        dtype_name  = df[col].dtype.name
        n_unique    = df[col].nunique(dropna=False)

        if dtype_name in ("object", "category", "bool"):
            ordinal_cols.append(col)

        elif np.issubdtype(df[col].dtype, np.number):
            if n_unique == 2:
                binary_cols.append(col)
            elif 3 <= n_unique <= threshold:
                ordinal_cols.append(col)
            else:
                continuous_cols.append(col)
        else:
            ordinal_cols.append(col)

    logger.info("Feature types  |  ordinal=%d  continuous=%d  binary=%d",
                len(ordinal_cols), len(continuous_cols), len(binary_cols))

    return ordinal_cols, continuous_cols, binary_cols
