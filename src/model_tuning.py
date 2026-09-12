# ============================================================
# MODEL TUNING — Employee Attrition ML System
# ============================================================

import numpy as np
import logging

from typing import Dict, List, Tuple

from sklearn.base              import BaseEstimator
from sklearn.compose           import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection   import RandomizedSearchCV, StratifiedKFold

from sklearn.linear_model  import LogisticRegression, SGDClassifier
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import (RandomForestClassifier, GradientBoostingClassifier,
                                   AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.naive_bayes   import GaussianNB, BernoulliNB
from xgboost               import XGBClassifier

from imblearn.pipeline     import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

from src.config        import RANDOM_STATE, N_JOBS, CV_FOLDS, RANDOM_SEARCH_ITERS, SELECT_K
from src.preprocessing import safe_k

logger = logging.getLogger(__name__)


# ============================================================
# HELPER — compute safe n_iter for RandomizedSearchCV
# ============================================================

def _compute_n_iter(param_dist: dict, budget: int) -> int:
    if not param_dist:
        return 1
    prod = 1
    for v in param_dist.values():
        try:
            prod *= len(v)
        except TypeError:
            prod *= budget
    return min(budget, max(1, prod))


# ============================================================
# MODEL GRIDS
# ============================================================

# Distance / linear models  →  need scaled input
scaled_models: Dict[str, Tuple[BaseEstimator, dict]] = {

    "LogisticRegression": (
        LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
        {
            "classifier__penalty":      ["l1", "l2"],
            "classifier__C":            [0.01, 0.1, 1, 10],
            "classifier__solver":       ["liblinear", "saga"],
            "classifier__class_weight": [None, "balanced"],
        }
    ),

    "KNN": (
        KNeighborsClassifier(),
        {
            "classifier__n_neighbors": [3, 5, 9],
            "classifier__weights":     ["uniform", "distance"],
        }
    ),

    "SGD": (
        SGDClassifier(random_state=RANDOM_STATE, max_iter=2000, tol=1e-3),
        {
            "classifier__loss":         ["log_loss"],
            "classifier__alpha":        [1e-4, 1e-3, 1e-2],
            "classifier__penalty":      ["l2", "elasticnet"],
            "classifier__class_weight": [None, "balanced"],
        }
    ),

    "GaussianNB": (
        GaussianNB(), {}
    ),
}

# Tree-based models  →  work on raw / clipped input
# class_weight="balanced" + scale_pos_weight handle imbalance INSIDE model
# SMOTENC handles it at data level — both together = robust imbalance handling
unscaled_models: Dict[str, Tuple[BaseEstimator, dict]] = {

    "DecisionTree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        {
            "classifier__max_depth":        [5, 10, 20, None],
            "classifier__min_samples_leaf": [1, 2, 4],
        }
    ),

    "RandomForest": (
        RandomForestClassifier(n_jobs=N_JOBS, random_state=RANDOM_STATE, class_weight="balanced"),
        {
            "classifier__n_estimators":     [100, 200],
            "classifier__max_depth":        [None, 10, 20],
            "classifier__min_samples_leaf": [1, 2],
        }
    ),

    "ExtraTrees": (
        ExtraTreesClassifier(n_jobs=N_JOBS, random_state=RANDOM_STATE, class_weight="balanced"),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth":    [None, 10, 20],
        }
    ),

    "GradientBoosting": (
        # GradientBoosting has no class_weight → tuned via sample_weight / subsample
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "classifier__n_estimators":  [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth":     [3, 5],
            "classifier__subsample":     [0.8, 1.0],
        }
    ),

    "AdaBoost": (
        AdaBoostClassifier(random_state=RANDOM_STATE),
        {
            "classifier__n_estimators":  [50, 100, 200],
            "classifier__learning_rate": [0.01, 0.1, 1.0],
        }
    ),

    "XGBoost": (
        # scale_pos_weight = approx neg/pos ratio for imbalanced data
        XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
        {
            "classifier__n_estimators":    [100, 200],
            "classifier__learning_rate":   [0.05, 0.1],
            "classifier__max_depth":       [3, 5],
            "classifier__scale_pos_weight":[5, 9, 15],   # ~neg/pos ratio for 9.6% positive rate
        }
    ),

    "BernoulliNB": (
        BernoulliNB(), {}
    ),
}

if LGBMClassifier is not None:
    unscaled_models["LightGBM"] = (
        LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, is_unbalance=True),
        {
            "classifier__n_estimators":  [100, 200],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth":     [-1, 10],
            "classifier__num_leaves":    [31, 63],
        }
    )

if CatBoostClassifier is not None:
    unscaled_models["CatBoost"] = (
        CatBoostClassifier(
            verbose=0, random_state=RANDOM_STATE, auto_class_weights="Balanced",
            allow_writing_files=False   # prevents catboost_info/ dir creation —
                                         # fixes race condition when n_jobs>1 spawns
                                         # parallel folds that all try to mkdir it (Windows)
        ),
        {
            "classifier__iterations":    [100, 200],
            "classifier__learning_rate": [0.03, 0.1],
            "classifier__depth":         [4, 6],
        }
    )


# ============================================================
# TUNE MODELS
# ============================================================

def tune_models(
    models:         Dict[str, Tuple[BaseEstimator, dict]],
    preprocessor:   ColumnTransformer,
    cat_indices:    List[int],
    X_train:        "pd.DataFrame",
    y_train:        "pd.Series",
    use_smote:      bool = True,
    selector_k:     int  = SELECT_K
) -> Dict[str, ImbPipeline]:
    """
    For each model:
      preprocessor → [SMOTENC] → SelectKBest → classifier
    Tuned with RandomizedSearchCV (scoring = F1).
    Returns dict of {model_name: best_pipeline}.
    """

    final_pipelines: Dict[str, ImbPipeline] = {}

    k_safe = safe_k(selector_k, preprocessor, X_train)
    logger.info("Selector k set to %d (requested %d)", k_safe, selector_k)

    for name, (clf, param_dist) in models.items():

        logger.info("Tuning: %s", name)

        # ── Build pipeline steps ──────────────────────────────
        steps = [("preprocessor", preprocessor)]

        if use_smote and len(cat_indices) > 0:
            steps.append((
                "smote",
                SMOTENC(categorical_features=cat_indices, random_state=RANDOM_STATE)
            ))

        steps.append(("selector", SelectKBest(mutual_info_classif, k=k_safe)))
        steps.append(("classifier", clf))

        pipe = ImbPipeline(steps)

        # ── Randomized search ─────────────────────────────────
        n_iter = _compute_n_iter(param_dist, RANDOM_SEARCH_ITERS)

        search = RandomizedSearchCV(
            pipe,
            param_distributions = param_dist,
            n_iter              = n_iter,
            scoring             = "f1",
            cv                  = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            n_jobs              = N_JOBS,
            random_state        = RANDOM_STATE,
            verbose             = 0
        )

        search.fit(X_train, y_train)

        logger.info("%s best params: %s", name, search.best_params_)

        final_pipelines[name] = search.best_estimator_

    return final_pipelines


# ============================================================
# NEURAL NETWORK — trained separately (no CV search)
# ============================================================

def train_mlp_pipeline(X_train, y_train, preprocessor, cat_indices: List[int]):
    """
    MLP trained separately outside RandomizedSearchCV.
    Reason: MLP training time makes CV search impractical.
    """
    from sklearn.neural_network import MLPClassifier

    logger.info("Training Neural Network (MLP) ...")

    pipe = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote",         SMOTENC(categorical_features=cat_indices, random_state=RANDOM_STATE)),
        ("classifier",    MLPClassifier(
            hidden_layer_sizes   = (128, 64),
            activation           = "relu",
            solver               = "adam",
            alpha                = 0.0001,
            batch_size           = 512,
            learning_rate        = "adaptive",
            max_iter             = 50,
            early_stopping       = True,
            validation_fraction  = 0.1,
            n_iter_no_change     = 5,
            random_state         = RANDOM_STATE
        ))
    ])

    pipe.fit(X_train, y_train)

    logger.info("MLP training done")

    return pipe