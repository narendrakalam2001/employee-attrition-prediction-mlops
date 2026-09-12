# ============================================================
# EMPLOYEE ATTRITION API — FastAPI Serving
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import logging
import time
import os
import json

from src.model_loader             import load_latest_model
from services.prediction_service  import predict_employee

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Employee Attrition Prediction API")

# ── Load model on startup ─────────────────────────────────────
try:
    model, threshold = load_latest_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error("Model loading failed: %s", e)
    model     = None
    threshold = 0.5


# ============================================================
# INPUT SCHEMA  — mirrors IBM HR Analytics Attrition dataset
# ============================================================

class EmployeeInput(BaseModel):
    age:                       int
    businesstravel:            str = "Travel_Rarely"    # Non-Travel / Travel_Rarely / Travel_Frequently
    dailyrate:                 float = 800.0
    department:                str = "Research & Development"
    distancefromhome:          int
    education:                 int = 3                  # 1=Below College .. 5=Doctor
    educationfield:             str = "Life Sciences"
    environmentsatisfaction:   int = 3                  # 1=Low .. 4=Very High
    gender:                    str = "Male"
    hourlyrate:                float = 65.0
    jobinvolvement:            int = 3                  # 1=Low .. 4=Very High
    joblevel:                  int = 2
    jobrole:                   str = "Sales Executive"
    jobsatisfaction:           int
    maritalstatus:             str = "Married"
    monthlyincome:             float
    monthlyrate:               float = 15000.0
    numcompaniesworked:        int = 1
    overtime:                  str = "No"                # Yes / No
    percentsalaryhike:         int = 13
    performancerating:         int = 3
    relationshipsatisfaction:  int = 3
    stockoptionlevel:          int = 0
    totalworkingyears:         int
    trainingtimeslastyear:     int = 2
    worklifebalance:           int = 3                   # 1=Bad .. 4=Best
    yearsatcompany:            int
    yearsincurrentrole:        int = 2
    yearssincelastpromotion:   int = 1
    yearswithcurrmanager:      int = 2


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Employee Attrition Prediction API is live 🚀",
        "docs":    "/docs",
        "health":  "/health"
    }

@app.get("/health")
def health():
    return {"status": "running", "model_loaded": model is not None}

@app.get("/model_info")
def model_info():
    registry_path = "attrition_models/latest_model.json"
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            return json.load(f)
    return {"error": "Model registry not found"}


# ── Prediction endpoint ───────────────────────────────────────

@app.post("/predict")
def predict(employee: EmployeeInput):

    start      = time.time()
    input_data = employee.dict()

    result     = predict_employee(model, input_data, threshold)

    # ── Log prediction ────────────────────────────────────────
    log_record = {
        "timestamp":            time.time(),
        "age":                  input_data["age"],
        "monthlyincome":        input_data["monthlyincome"],
        "attrition_probability":result["attrition_probability"],
        "risk_band":            result["risk_band"],
        "decision":             result["decision"],
        "rule_triggered":       result.get("rule_triggered"),
    }

    log_path = "logs/prediction_logs.csv"
    os.makedirs("logs", exist_ok=True)

    log_df = pd.DataFrame([log_record])
    if os.path.exists(log_path):
        log_df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_path, index=False)

    result["latency_seconds"] = round(time.time() - start, 4)

    return result
