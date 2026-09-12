# ============================================================
# CONFIGURATION — Employee Attrition ML System
# ============================================================

import os

# ── Reproducibility ──────────────────────────────────────────
RANDOM_STATE   = 42
N_JOBS         = -1

# ── Cross-validation ─────────────────────────────────────────
CV_FOLDS             = 5
RANDOM_SEARCH_ITERS  = 20

# ── Feature selection ────────────────────────────────────────
SELECT_K = 15

# ── Outlier clipping ─────────────────────────────────────────
CLIP_FOLD = 1.5

# ── Feature engineering thresholds ───────────────────────────
ORDINAL_UNIQUE_THRESHOLD = 10

# ── Attrition risk bands (probability → tier) ────────────────
RISK_BANDS = {
    "LOW":    (0.00, 0.30),
    "MEDIUM": (0.30, 0.60),
    "HIGH":   (0.60, 1.01),
}

# ── Business rule thresholds (HR domain) ──────────────────────
# Hard/soft rules used by the retention engine, in addition to the ML model.
LOW_SATISFACTION_THRESHOLD   = 2      # EnvironmentSatisfaction / JobSatisfaction <= 2 (1-4 scale)
HIGH_DISTANCE_FROM_HOME_KM   = 20     # DistanceFromHome >= 20 km
LOW_TENURE_YEARS             = 1      # YearsAtCompany <= 1 → onboarding flight risk
LONG_PROMOTION_GAP_YEARS     = 5      # YearsSinceLastPromotion >= 5

# ── PSI drift thresholds ──────────────────────────────────────
PSI_MODERATE         = 0.10    # PSI >= 0.10 → moderate drift, monitor
PSI_HIGH             = 0.20    # PSI >= 0.20 → critical drift, retrain

# ── Score monitoring alert ───────────────────────────────────
SCORE_MEAN_ALERT     = 0.35

# ── Challenger promotion gates ────────────────────────────────
MIN_F1_IMPROVEMENT      = 0.005
MIN_ROCAUC_THRESHOLD    = 0.75
MAX_GENERALIZATION_GAP  = 0.12

# ── Cost-sensitive evaluation (business impact, INR) ──────────
# TCS / Accenture / Infosys-style replacement-cost economics:
#   Replacing a trained employee typically costs ₹5–15L (recruiting,
#   onboarding, ramp-up productivity loss). We use 9x MonthlyIncome as
#   a proxy for full replacement cost (industry rule of thumb: 0.5x-2x
#   annual salary depending on role seniority).
REPLACEMENT_COST_MULTIPLIER = 9.0     # x MonthlyIncome ≈ ₹5-15L for mid-level ICs
RETENTION_ACTION_COST       = 15000.0 # ₹ cost of a retention intervention (manager 1:1, raise review, etc.)

# ── Paths ────────────────────────────────────────────────────
MODEL_DIR   = "attrition_models"
METRICS_LOG = "attrition_models/metrics_log.csv"
TESTS_DIR   = "tests"
SERVING_DIR = "serving"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("logs",    exist_ok=True)
