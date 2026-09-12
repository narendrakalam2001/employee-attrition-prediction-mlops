# ============================================================
# METRICS — Employee Attrition ML System
# ============================================================

import numpy as np
import pandas as pd
import logging

from sklearn.metrics import precision_recall_curve, roc_curve

logger = logging.getLogger(__name__)


# ============================================================
# THRESHOLD TUNING
# ============================================================

def tune_threshold(
    y_true:           np.ndarray,
    y_prob:           np.ndarray,
    target_recall:    float = None,
    target_precision: float = None
) -> float:
    """
    Default: maximize F1.
    Optional: constrain on recall or precision first.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
    best_idx  = np.nanargmax(f1_scores)
    best_thr  = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    if target_recall is not None:
        idxs = np.where(recall >= target_recall)[0]
        if idxs.size > 0:
            chosen = idxs[np.argmax(precision[idxs])]
            best_thr = thresholds[chosen] if chosen < len(thresholds) else 0.5

    if target_precision is not None:
        idxs = np.where(precision >= target_precision)[0]
        if idxs.size > 0:
            chosen = idxs[np.argmax(recall[idxs])]
            best_thr = thresholds[chosen] if chosen < len(thresholds) else 0.5

    logger.info("Best threshold: %.4f  |  precision=%.4f  recall=%.4f  f1=%.4f",
                best_thr,
                precision[best_idx],
                recall[best_idx],
                f1_scores[best_idx])

    return float(best_thr)


# ============================================================
# PSI — Population Stability Index
# ============================================================

def psi(expected, actual, buckets: int = 10) -> float:
    """
    Population Stability Index — measures distribution shift.

    PSI < 0.1   → no significant shift (stable)
    PSI 0.1–0.2 → moderate shift (monitor closely)
    PSI > 0.2   → major shift (retrain recommended)

    Correct approach:
      1. Compute quantile bin EDGES from `expected` (reference)
      2. Bin BOTH `expected` and `actual` using those SAME edges
      3. Compare bin proportions

    Common bug to avoid:
      Ranking both distributions independently → both become
      uniform → PSI always ~0. This is wrong.
    """
    try:
        expected = np.asarray(expected, dtype=float)
        actual   = np.asarray(actual,   dtype=float)

        # ── Step 1: compute bin edges from reference distribution ──
        quantiles  = np.linspace(0, 100, buckets + 1)
        bin_edges  = np.percentile(expected, quantiles)

        # Make edges unique to avoid empty bins — add small epsilon
        bin_edges  = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0

        # Extend edges to cover full range of actual distribution
        bin_edges[0]  = min(bin_edges[0],  actual.min()) - 1e-9
        bin_edges[-1] = max(bin_edges[-1], actual.max()) + 1e-9

        # ── Step 2: bin BOTH using the SAME edges ─────────────────
        exp_hist, _ = np.histogram(expected, bins=bin_edges)
        act_hist, _ = np.histogram(actual,   bins=bin_edges)

        # ── Step 3: convert to proportions, avoid div-by-zero ─────
        exp_pct = exp_hist / (exp_hist.sum() + 1e-9)
        act_pct = act_hist / (act_hist.sum() + 1e-9)

        exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
        act_pct = np.where(act_pct == 0, 1e-6, act_pct)

        # ── Step 4: PSI formula ────────────────────────────────────
        psi_value = float(np.sum((exp_pct - act_pct) * np.log(exp_pct / act_pct)))

        return psi_value

    except Exception as e:
        logger.warning("PSI computation failed: %s", e)
        return float("nan")


# ============================================================
# KS STATISTIC
# ============================================================

def ks_statistic(y_true, y_prob) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


# ============================================================
# RECALL @ K
# ============================================================

def recall_at_k(y_true, y_prob, k: float = 0.05) -> float:
    df = (
        pd.DataFrame({"y": y_true, "p": y_prob})
        .sort_values("p", ascending=False)
    )
    top_n = int(len(df) * k)
    return float(df.iloc[:top_n]["y"].sum() / (df["y"].sum() + 1e-9))


# ============================================================
# LIFT @ K
# ============================================================

def lift_at_k(y_true, y_prob, k: float = 0.05) -> float:
    base = np.mean(y_true)
    return float(recall_at_k(y_true, y_prob, k) / (base + 1e-9))


# ============================================================
# COST-SENSITIVE EVALUATION  (HR / People Analytics grade)
# ============================================================

def cost_sensitive_evaluation(
    X_test,
    y_true,
    y_pred,
    income_col:             str   = "monthlyincome",
    replacement_multiplier: float = 9.0,     # ≈ 9x MonthlyIncome ≈ ₹5–15L replacement cost
    retention_action_cost:  float = 15000.0  # ₹ cost of a proactive retention intervention
) -> dict:
    """
    Business-impact framing used by People Analytics teams:

        Missed attriter (False Negative) → full replacement cost
            (recruiting + onboarding + 3-6 month ramp-up productivity loss)
        Flagged stayer  (False Positive) → wasted retention intervention cost
            (manager 1:1, comp review, etc. spent on someone who wasn't leaving)

    Returns estimated ₹ business impact of the model's decision boundary.
    """

    df = X_test.copy()
    df["y_true"] = y_true.values if hasattr(y_true, "values") else y_true
    df["y_pred"] = y_pred

    # ── Missed attriters (FN) → full replacement cost ─────────
    fn_mask = (df["y_true"] == 1) & (df["y_pred"] == 0)
    if income_col in df.columns:
        fn_cost = float((df.loc[fn_mask, income_col] * replacement_multiplier).sum())
    else:
        fn_cost = float(fn_mask.sum() * 500000)  # fallback ₹5L flat

    # ── Unnecessary retention spend (FP) ───────────────────────
    fp_mask = (df["y_true"] == 0) & (df["y_pred"] == 1)
    fp_cost = float(fp_mask.sum() * retention_action_cost)

    total_cost = fn_cost + fp_cost

    result = {
        "missed_attriters_count":     int(fn_mask.sum()),
        "false_alarm_count":          int(fp_mask.sum()),
        "estimated_replacement_loss": round(fn_cost, 2),
        "retention_intervention_cost": round(fp_cost, 2),
        "total_estimated_cost_inr":   round(total_cost, 2),
    }

    logger.info(
        "Cost eval  |  missed=%d  false_alarms=%d  replacement_loss=%.2f  "
        "retention_cost=%.2f  total=%.2f",
        result["missed_attriters_count"],
        result["false_alarm_count"],
        result["estimated_replacement_loss"],
        result["retention_intervention_cost"],
        result["total_estimated_cost_inr"],
    )

    return result


# ============================================================
# DRIFT REPORT — feature mean shift
# ============================================================

def simple_drift_report(X_ref: "pd.DataFrame", X_new: "pd.DataFrame", top_n: int = 10):
    diffs = (X_ref.mean() - X_new.mean()).abs()
    rel   = (diffs / (X_ref.std().replace(0, 1))).sort_values(ascending=False)
    return rel.head(top_n)
