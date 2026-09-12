# ============================================================
# ATTRITION ENGINE — Employee Attrition ML System
# ============================================================
# HR-grade 3-tier retention decision system:
#   RETAIN    → low risk, no action needed
#   MONITOR   → elevated risk, add to manager watch-list
#   INTERVENE → high risk, trigger retention action (1:1, comp review, etc.)
#
# Decision hierarchy:
#   1. Hard business rules (override ML) — known strong attrition drivers
#   2. ML model probability + risk bands
# ============================================================

import pandas as pd
import logging

from src.config import (
    RISK_BANDS, LOW_SATISFACTION_THRESHOLD, HIGH_DISTANCE_FROM_HOME_KM,
    LOW_TENURE_YEARS, LONG_PROMOTION_GAP_YEARS
)

logger = logging.getLogger(__name__)


# ============================================================
# RISK BAND — probability → LOW / MEDIUM / HIGH
# ============================================================

def get_risk_band(prob: float) -> str:
    for band, (low, high) in RISK_BANDS.items():
        if low <= prob < high:
            return band
    return "HIGH"


# ============================================================
# ATTRITION ENGINE — row-level decisions
# ============================================================

def attrition_engine(employee_df: pd.DataFrame, probs, threshold: float) -> list:
    """
    For each employee row → returns decision string.

    Rule priority:
      1. overtime + low satisfaction        → INTERVENE_RULE (hard rule)
      2. tenure <= 1yr (onboarding risk)     → MONITOR_TENURE
      3. promotion gap >= 5yr                → MONITOR_PROMOTION
      4. prob >= threshold                   → INTERVENE_MODEL
      5. prob >= threshold * 0.6             → MONITOR_MODEL
      6. else                                → RETAIN
    """
    decisions = []

    for idx, (_, row) in enumerate(employee_df.iterrows()):

        p = probs[idx]

        # ── Hard rule: overtime + chronically dissatisfied ────
        low_sat = row.get("satisfaction_composite", 4) <= LOW_SATISFACTION_THRESHOLD
        if row.get("overtime", 0) == 1 and low_sat:
            decisions.append("INTERVENE_RULE")
            continue

        # ── Soft rule: very new hire (onboarding flight risk) ──
        if "yearsatcompany" in row and row["yearsatcompany"] <= LOW_TENURE_YEARS:
            decisions.append("MONITOR_TENURE")
            continue

        # ── Soft rule: long promotion drought ──────────────────
        if "yearssincelastpromotion" in row and row["yearssincelastpromotion"] >= LONG_PROMOTION_GAP_YEARS:
            decisions.append("MONITOR_PROMOTION")
            continue

        # ── ML model decisions ────────────────────────────────
        if p >= threshold:
            decisions.append("INTERVENE_MODEL")

        elif p >= threshold * 0.6:
            decisions.append("MONITOR_MODEL")

        else:
            decisions.append("RETAIN")

    return decisions


# ============================================================
# ATTRITION SCORING — single employee (for API)
# ============================================================

def score_employee(row: dict, prob: float, threshold: float) -> dict:
    """
    Returns structured retention-risk output for a single employee.
    Used by FastAPI prediction endpoint.
    """
    risk_band = get_risk_band(prob)

    rule_triggered = None

    low_sat = row.get("satisfaction_composite", 4) <= LOW_SATISFACTION_THRESHOLD

    if row.get("overtime", 0) == 1 and low_sat:
        decision       = "INTERVENE"
        rule_triggered = "OVERTIME_PLUS_LOW_SATISFACTION"

    elif row.get("yearsatcompany", 99) <= LOW_TENURE_YEARS:
        decision       = "MONITOR"
        rule_triggered = "NEW_HIRE_FLIGHT_RISK"

    elif row.get("yearssincelastpromotion", 0) >= LONG_PROMOTION_GAP_YEARS:
        decision       = "MONITOR"
        rule_triggered = "LONG_PROMOTION_GAP"

    # ── ML model decision ─────────────────────────────────────
    elif prob >= threshold:
        decision = "INTERVENE"

    elif prob >= threshold * 0.6:
        decision = "MONITOR"

    else:
        decision = "RETAIN"

    return {
        "attrition_probability": round(float(prob), 4),
        "risk_band":             risk_band,
        "decision":              decision,
        "rule_triggered":        rule_triggered,
    }
