# ============================================================
# EVALUATION — Employee Attrition ML System
# ============================================================

import os
import json
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.calibration       import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection   import cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.inspection        import permutation_importance
from sklearn.pipeline          import Pipeline

from src.config  import RANDOM_STATE, N_JOBS, CV_FOLDS, MODEL_DIR
from src.metrics import tune_threshold, psi, ks_statistic, recall_at_k, lift_at_k, cost_sensitive_evaluation
from src.attrition_engine import attrition_engine

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================
# SAFE PREDICT PROBA
# ============================================================

def safe_predict_proba(pipe, X):
    try:
        return pipe.predict_proba(X)[:, 1]
    except Exception:
        logger.warning("predict_proba failed — falling back to manual extraction")
        try:
            pre = pipe.named_steps["preprocessor"].transform(X)
            sel = pipe.named_steps.get("selector", None)
            X_in = sel.transform(pre) if sel is not None else pre
            clf  = pipe.named_steps["classifier"]
            if hasattr(clf, "predict_proba"):
                return clf.predict_proba(X_in)[:, 1]
        except Exception as e:
            logger.error("safe_predict_proba fully failed: %s", e)
    return None


# ============================================================
# EVALUATE ALL MODELS — summary table
# ============================================================

def evaluate_models(
    pipelines:    Dict,
    X_train:      pd.DataFrame,
    y_train:      pd.Series,
    X_test:       pd.DataFrame,
    y_test:       pd.Series,
) -> pd.DataFrame:

    rows = []

    for name, pipe in pipelines.items():

        logger.info("Evaluating: %s", name)

        # ── Accuracy ──────────────────────────────────────────
        try:
            train_acc = accuracy_score(y_train, pipe.predict(X_train))
        except Exception:
            train_acc = None

        test_acc  = accuracy_score(y_test, pipe.predict(X_test))

        # ── Probabilities + threshold ─────────────────────────
        y_prob = safe_predict_proba(pipe, X_test)

        if y_prob is not None:
            thr    = tune_threshold(y_test.values, y_prob)
            y_pred = (y_prob >= thr).astype(int)
        else:
            thr    = None
            y_pred = pipe.predict(X_test)

        # ── Core metrics ──────────────────────────────────────
        prec     = precision_score(y_test, y_pred,       zero_division=0)
        rec      = recall_score(y_test, y_pred,           zero_division=0)
        f1       = f1_score(y_test, y_pred,               zero_division=0)
        roc_auc  = roc_auc_score(y_test, y_prob)          if y_prob is not None else None
        pr_auc   = average_precision_score(y_test, y_prob) if y_prob is not None else None
        ks       = ks_statistic(y_test, y_prob)            if y_prob is not None else None
        brier    = brier_score_loss(y_test, y_prob)        if y_prob is not None else None

        # ── CV stability ──────────────────────────────────────
        try:
            cv_scores = cross_val_score(
                pipe, X_train, y_train,
                scoring = "f1",
                cv      = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                n_jobs  = N_JOBS
            )
            cv_mean = float(cv_scores.mean())
            cv_std  = float(cv_scores.std())
        except Exception:
            cv_mean = None
            cv_std  = None

        rows.append({
            "model":           name,
            "train_acc":       round(train_acc, 4) if train_acc else None,
            "test_acc":        round(test_acc,  4),
            "train_test_gap":  round(abs((train_acc or 0) - test_acc), 4),
            "cv_mean_f1":      round(cv_mean, 4) if cv_mean else None,
            "cv_std_f1":       round(cv_std,  4) if cv_std  else None,
            "cv_test_gap":     round(abs((cv_mean or 0) - f1), 4),
            "precision":       round(prec,    4),
            "recall":          round(rec,     4),
            "f1":              round(f1,      4),
            "roc_auc":         round(roc_auc, 4) if roc_auc else None,
            "pr_auc":          round(pr_auc,  4) if pr_auc  else None,
            "ks":              round(ks,      4) if ks      else None,
            "brier":           round(brier,   4) if brier   else None,
            "threshold":       round(thr,     4) if thr     else None,
        })

    summary = (
        pd.DataFrame(rows)
        .sort_values(["f1", "roc_auc"], ascending=False)
        .reset_index(drop=True)
    )

    summary.to_csv(os.path.join(MODEL_DIR, "model_experiment_results.csv"), index=False)

    return summary


# ============================================================
# SELECT BEST MODEL — with generalization filter
# ============================================================

def select_best_model(
    summary:      pd.DataFrame,
    pipelines:    Dict,
    scaled_pipes: Dict,
    unscaled_pipes:Dict
) -> Tuple[str, object]:
    # `pipelines` is the full combined dict (scaled + unscaled + NeuralNet etc.)
    # and is used as the final fallback lookup below.

    def _filter(df, gen_gap, cv_std, min_f1, min_prec, min_rec):
        mask = (
            df["train_test_gap"].notna() &
            df["cv_std_f1"].notna() &
            (df["train_test_gap"] <= gen_gap) &
            (df["cv_std_f1"]      <= cv_std)  &
            (df["f1"]             >= min_f1)   &
            (df["precision"]      >= min_prec) &
            (df["recall"]         >= min_rec)
        )
        return df[mask].copy()

    # ── Progressive relaxation ────────────────────────────────
    thresholds = [
        (0.05, 0.05, 0.70, 0.60, 0.60),
        (0.08, 0.07, 0.65, 0.55, 0.55),
        (0.10, 0.10, 0.60, 0.50, 0.50),
        (0.20, 0.20, 0.00, 0.00, 0.00),
    ]

    candidates = pd.DataFrame()
    for gg, cvs, mf1, mp, mr in thresholds:
        candidates = _filter(summary, gg, cvs, mf1, mp, mr)
        if not candidates.empty:
            logger.info("Candidates found with gen_gap<=%.2f cv_std<=%.2f min_f1>=%.2f", gg, cvs, mf1)
            break

    if candidates.empty:
        selected_name = summary.iloc[0]["model"]
    else:
        candidates    = candidates.sort_values(
            ["f1", "roc_auc", "cv_std_f1"], ascending=[False, False, True]
        ).reset_index(drop=True)
        selected_name = candidates.iloc[0]["model"]

    logger.info("Selected model: %s", selected_name)

    selected_pipe = (
        scaled_pipes.get(selected_name)
        or unscaled_pipes.get(selected_name)
        or pipelines.get(selected_name)
    )
    if selected_pipe is None:
        raise RuntimeError(f"Selected model '{selected_name}' not found in any pipeline dict")

    return selected_name, selected_pipe


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_with_holdout(selected_pipe, X_cal, y_cal, method: str = "isotonic"):
    """
    Holdout calibration — no SMOTE on calibration set.
    """
    clf = selected_pipe.named_steps["classifier"]
    pre = selected_pipe.named_steps["preprocessor"]
    sel = selected_pipe.named_steps.get("selector", None)

    X_pre = pre.transform(X_cal)
    X_sel = sel.transform(X_pre) if sel is not None else X_pre

    method_use = method
    if method == "isotonic" and len(y_cal) < 200:
        method_use = "sigmoid"
        logger.info("Calibration: switching to sigmoid (cal set < 200 rows)")

    try:
        try:
            cal = CalibratedClassifierCV(estimator=clf,      cv="prefit", method=method_use)
        except TypeError:
            cal = CalibratedClassifierCV(base_estimator=clf, cv="prefit", method=method_use)

        cal.fit(X_sel, y_cal)

        steps = [("preprocessor", pre)]
        if sel is not None:
            steps.append(("selector", sel))
        steps.append(("classifier", cal))

        cal_pipe = Pipeline(steps)
        logger.info("Calibration done  |  method=%s", method_use)
        return cal_pipe

    except Exception as e:
        logger.exception("Calibration failed: %s", e)
        return None


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def compute_feature_importance(selected_pipe, X_train, y_train, top_k: int = 20):

    clf   = selected_pipe.named_steps["classifier"]
    names = _get_feature_names(selected_pipe, X_train)

    # ── Tree-based importances ────────────────────────────────
    if hasattr(clf, "feature_importances_"):
        imp   = np.asarray(clf.feature_importances_)
        names = names if names and len(names) == len(imp) else [f"f{i}" for i in range(len(imp))]
        fi    = pd.Series(imp, index=names).sort_values(ascending=False).head(top_k)
        return fi

    # ── Linear model coefficients ─────────────────────────────
    if hasattr(clf, "coef_"):
        coef  = np.abs(clf.coef_).ravel()
        names = names if names and len(names) == len(coef) else [f"f{i}" for i in range(len(coef))]
        fi    = pd.Series(coef, index=names).sort_values(ascending=False).head(top_k)
        return fi

    # ── Permutation importance (fallback) ─────────────────────
    logger.info("Using permutation importance (slow fallback)...")
    try:
        r   = permutation_importance(
            selected_pipe, X_train, y_train,
            n_repeats=10, scoring="f1", n_jobs=N_JOBS, random_state=RANDOM_STATE
        )
        idx = np.argsort(r.importances_mean)[::-1][:top_k]
        fi  = pd.Series(r.importances_mean[idx],
                        index=[f"f{i}" for i in idx])
        return fi
    except Exception as e:
        logger.exception("Permutation importance failed: %s", e)
        return None


def _get_feature_names(pipe, X_sample):
    """
    Extracts clean feature names after preprocessor + selector.
    Strips sklearn prefixes like 'ord__', 'skewed__', 'bin__' etc.
    Returns actual column names like 'income', 'ccavg', 'mortgage'.
    """
    try:
        pre   = pipe.named_steps["preprocessor"]
        raw_names = list(pre.get_feature_names_out())

        # ── Strip sklearn ColumnTransformer prefixes ──────────
        # e.g. 'ord__family' → 'family'
        #      'skewed__income' → 'income'
        #      'non_skew__age' → 'age'
        #      'bin__online' → 'online'
        clean_names = []
        for n in raw_names:
            if "__" in n:
                clean_names.append(n.split("__", 1)[1])
            else:
                clean_names.append(n)

        # ── Apply selector mask if present ────────────────────
        sel = pipe.named_steps.get("selector", None)
        if sel is not None:
            mask        = sel.get_support()
            clean_names = [n for n, m in zip(clean_names, mask) if m]

        return clean_names

    except Exception as e:
        logger.warning("Could not extract feature names: %s", e)
        return None


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

def compute_shap(selected_pipe, X_train, X_explain, max_explain: int = 200):

    if not SHAP_AVAILABLE:
        logger.info("SHAP not installed — skipping")
        return None

    clf  = selected_pipe.named_steps["classifier"]
    pre  = selected_pipe.named_steps["preprocessor"]
    sel  = selected_pipe.named_steps.get("selector", None)

    try:
        X_pre = pre.transform(X_train)
        X_tr  = sel.transform(X_pre) if sel is not None else X_pre
        X_ex  = sel.transform(pre.transform(X_explain)) if sel is not None else pre.transform(X_explain)
        X_ex  = np.asarray(X_ex)[:max_explain]

        names = _get_feature_names(selected_pipe, X_train)

        # ── TreeExplainer for tree models ─────────────────────
        tree_types = ("RandomForestClassifier", "ExtraTreesClassifier",
                      "GradientBoostingClassifier", "XGBClassifier",
                      "LGBMClassifier", "CatBoostClassifier")

        if type(clf).__name__ in tree_types:
            explainer = shap.TreeExplainer(clf)
            try:
                vals = np.array(explainer(X_ex).values)
            except Exception:
                vals = np.array(explainer.shap_values(X_ex))
                if isinstance(vals, list):
                    vals = vals[1]

        else:
            bg_sample = shap.sample(X_tr, min(200, len(X_tr)))

            def model_fn(x):
                return clf.predict_proba(x)[:, 1]

            explainer = shap.KernelExplainer(model_fn, bg_sample)
            vals      = np.array(explainer.shap_values(X_ex, nsamples=100))

        if vals.ndim == 1:
            vals = vals.reshape(1, -1)

        mean_abs = np.abs(vals).mean(axis=0)
        feat_names = names[:mean_abs.shape[0]] if names else [f"f{i}" for i in range(mean_abs.shape[0])]

        fi_series = pd.Series(mean_abs, index=feat_names).sort_values(ascending=False)

        try:
            shap.summary_plot(vals, features=X_ex, feature_names=feat_names, show=False)
        except Exception:
            pass

        logger.info("SHAP done  |  explainer=%s", type(explainer).__name__)

        return {"shap_top": fi_series.head(20).to_dict(), "explainer": type(explainer).__name__}

    except Exception as e:
        logger.exception("SHAP failed: %s", e)
        return None


# detect_leakage() moved to src/leakage_check.py


# ============================================================
# SAVE MODEL  (model card saving moved to src/model_card.py)
# ============================================================

def save_challenger_artifact(
    selected_name:   str,
    pipe,
    model_card:      dict,
    version:         str = "v1",
    model_card_path: str = ""
):
    """
    Saves the trained CHALLENGER pipeline as .joblib — does NOT touch the
    registry (latest_model.json). The registry must only be updated by
    run_challenger_comparison() / _update_registry() in model_loader.py,
    and ONLY when the challenger actually beats the champion on all gates.

    BUG FIX: the previous save_model_and_card() always overwrote
    latest_model.json immediately, BEFORE run_challenger_comparison() ran.
    That meant _load_champion_metrics() inside run_challenger_comparison()
    loaded the CHALLENGER's own just-written metrics as the "champion" —
    so every retrain was silently comparing itself to itself (proof: every
    REJECTED entry in challenger_log.json had champion_f1 == challenger_f1
    exactly). This function fixes that by deferring the registry write.

    Returns challenger_model_path (NOT yet necessarily the champion).
    """
    model_path = os.path.join(MODEL_DIR, f"attrition_model_{selected_name}_{version}.joblib")
    joblib.dump(pipe, model_path)

    logger.info("Challenger model saved → %s (registry untouched until gates pass)", model_path)

    return model_path


def save_model_and_card(
    selected_name:   str,
    pipe,
    model_card:      dict,
    version:         str = "v1",
    model_card_path: str = ""
):
    """
    DEPRECATED for use inside the training pipeline's main flow — kept only
    for direct/manual model promotion (e.g. first-time setup scripts).

    Saves trained model as .joblib AND immediately overwrites latest_model.json.
    Training pipeline must use save_challenger_artifact() + run_challenger_comparison()
    instead, so the registry is gated by the promotion logic (see bug-fix note above).
    """
    model_path = os.path.join(MODEL_DIR, f"attrition_model_{selected_name}_{version}.joblib")
    joblib.dump(pipe, model_path)

    # ── Registry pointer ─────────────────────────────────────
    active_thr = (
        model_card.get("thresholds", {}).get("active")
        or model_card.get("threshold_calibrated")
        or model_card.get("threshold_uncalibrated")
        or 0.5
    )
    registry = {
        "model_name":      os.path.basename(model_path),
        "threshold":       active_thr,
        "model_card_path": model_card_path or ""
    }
    with open(os.path.join(MODEL_DIR, "latest_model.json"), "w") as f:
        json.dump(registry, f, indent=2)

    logger.info("Model saved   → %s", model_path)
    logger.info("Registry      → risk_models/latest_model.json")

    return model_path


# ============================================================
# MLFLOW LOGGING
# ============================================================

def mlflow_log_run(
    run_name:      str,
    selected_name: str,
    pipe,
    model_card:    dict,
    X_train_sample: "pd.DataFrame" = None
):
    """
    Logs model run to MLflow.
    Fixes applied:
      1. artifact_path → name  (deprecated warning fix)
      2. pip_requirements      (pip inference warning fix)
      3. input_example         (signature warning fix)
    """
    if not MLFLOW_AVAILABLE:
        logger.info("MLflow not installed — skipping")
        return

    try:
        # Force a safe local tracking URI instead of trusting whatever is in
        # MLFLOW_TRACKING_URI / MLFLOW_REGISTRY_URI. On Windows, a stray env
        # var pointing at a %-encoded path (e.g. spaces in the project path
        # turned into "%20") makes mlflow's sqlite/file resolver blow up with
        # PermissionError/FileNotFoundError. Pinning it here to a local sqlite
        # DB avoids that AND avoids the newer MLflow versions' deprecation of
        # the plain file-based ("./mlruns") tracking store, which now raises
        # "filesystem tracking backend is in maintenance mode" on write.
        _mlflow_db = os.path.abspath("mlflow.db")
        mlflow.set_tracking_uri(f"sqlite:///{Path(_mlflow_db).as_posix()}")

        mlflow.set_experiment("employee_attrition_experiments")

        # ── Read metrics from nested model_card structure ─────
        metrics_dict = model_card.get("metrics", model_card)

        with mlflow.start_run(run_name=run_name):

            # ── Params ────────────────────────────────────────
            mlflow.log_param("model_name",  selected_name)
            mlflow.log_param("selector_k",  model_card.get("pipeline_config", {}).get("selector_k", "?"))
            mlflow.log_param("threshold",   model_card.get("thresholds", {}).get("active", "?"))

            # ── Metrics ───────────────────────────────────────
            mlflow.log_metric("test_f1",       float(metrics_dict.get("test_f1",       0)))
            mlflow.log_metric("test_precision", float(metrics_dict.get("test_precision", 0)))
            mlflow.log_metric("test_recall",    float(metrics_dict.get("test_recall",   0)))
            mlflow.log_metric("roc_auc",        float(metrics_dict.get("roc_auc",       0)))
            mlflow.log_metric("ks_statistic",   float(metrics_dict.get("ks_statistic",  0)))
            mlflow.log_metric("brier_score",    float(metrics_dict.get("brier_score",   0)))

            # ── Log model — all 4 warnings fixed ─────────────
            try:
                log_kwargs = {
                    "sk_model":        pipe,
                    "name":            "model",           # Fix 1: artifact_path → name
                    "pip_requirements": [                  # Fix 2: explicit pip list
                        "scikit-learn==1.3.2",
                        "lightgbm",
                        "xgboost",
                        "imbalanced-learn",
                        "pandas",
                        "numpy",
                    ],
                    # Fix 4: newer MLflow defaults to skops serialization, which
                    # rejects our custom Clipper class + certain sklearn/MLP
                    # internals as "untrusted types". cloudpickle (MLflow's
                    # traditional format) has no such allow-list and handles
                    # arbitrary custom classes fine — this is the officially
                    # recommended workaround for pipelines with custom transformers.
                    "serialization_format": "cloudpickle",
                }
                if X_train_sample is not None:            # Fix 3: input_example for signature
                    # Only cast numeric columns to float64 (avoids MLflow's
                    # integer-column schema warning). Casting the WHOLE frame
                    # blindly breaks here because this dataset has string
                    # categorical columns (e.g. "businesstravel" = "Travel_Rarely")
                    # that cannot become floats — that caused the crash below.
                    sample = X_train_sample.iloc[:5].copy()
                    numeric_cols = sample.select_dtypes(include="number").columns
                    sample[numeric_cols] = sample[numeric_cols].astype(float)
                    log_kwargs["input_example"] = sample

                mlflow.sklearn.log_model(**log_kwargs)

            except Exception as e:
                logger.warning("mlflow log_model failed: %s", e)

        logger.info("MLflow run logged: %s", run_name)

    except Exception as e:
        logger.exception("MLflow logging failed: %s", e)