# ============================================================
# PREDICTION SERVICE — Employee Attrition ML System
# ============================================================

import pandas as pd
import numpy as np
import logging

from src.attrition_engine import score_employee
from src.data_loader import add_engineered_features

logger = logging.getLogger(__name__)


# ============================================================
# PREPARE FEATURES — for API inference
# ============================================================

def prepare_features(input_data: dict) -> pd.DataFrame:
    """
    Takes raw API input dict → returns engineered feature DataFrame.
    Mirrors the training pipeline feature engineering.
    """
    df = pd.DataFrame([input_data])
    df = add_engineered_features(df)
    return df


# ============================================================
# PREDICT — single employee
# ============================================================

def predict_employee(model, input_data: dict, threshold: float) -> dict:
    """
    Full prediction flow for one employee:
      1. Feature engineering
      2. Model probability
      3. Attrition engine scoring (rules + ML)

    Returns structured retention-risk output dict.
    """
    df = prepare_features(input_data)

    try:
        prob = float(model.predict_proba(df)[0][1])
    except Exception as e:
        logger.error("predict_proba failed: %s", e)
        prob = 0.5

    # ── Compute derived fields needed by attrition engine rules ─
    row = df.iloc[0].to_dict()

    result = score_employee(row, prob, threshold)

    logger.info(
        "Prediction  |  prob=%.4f  band=%s  decision=%s",
        result["attrition_probability"],
        result["risk_band"],
        result["decision"]
    )

    return result
