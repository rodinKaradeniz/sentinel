"""Model pipeline factories.

Each factory returns an sklearn ``Pipeline`` so that all data-dependent steps
(imputation, scaling, one-hot categories) are **fit on the training split only** —
no leakage from val/test. Linear models need imputation + scaling + one-hot; tree
models need far less (Step 6-7).
"""
from __future__ import annotations

import json
from datetime import date

import joblib
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config, metrics
from . import data as data_mod
from . import preprocessing as pp
from .preprocessing import CATEGORICAL

MODEL_PATH = config.MODELS / "lightgbm_baseline.joblib"
META_PATH = config.MODELS / "lightgbm_baseline.meta.json"


def _numeric_columns(X) -> list[str]:
    return [c for c in X.columns if c not in CATEGORICAL]


def linear_preprocessor(X) -> ColumnTransformer:
    """Impute (median) + scale numerics; one-hot categoricals.

    Median imputation pairs with the ``*_missing`` flags from preprocessing.clean():
    the flag preserves "was it missing", the median fills a usable value. Scaling
    matters for Logistic Regression (coefficients/regularization are scale-sensitive).
    """
    numeric = _numeric_columns(X)
    num_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    return ColumnTransformer(
        [
            ("num", num_pipe, numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop=None), CATEGORICAL),
        ]
    )


def tree_preprocessor(X) -> ColumnTransformer:
    """Median-impute numerics + one-hot categoricals. No scaling (trees don't need it).

    sklearn's RandomForest can't consume NaN or strings, hence impute + one-hot.
    The ``*_missing`` flags still carry the "was it missing" signal.
    """
    numeric = _numeric_columns(X)
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def logistic_regression(X) -> Pipeline:
    """Logistic Regression baseline with class_weight='balanced' for the imbalance.

    'balanced' reweights the loss by inverse class frequency so the ~89:1 imbalance
    doesn't collapse the model to "always predict legit" (which would give recall 0).
    """
    return Pipeline(
        [
            ("prep", linear_preprocessor(X)),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=config.SEED,
                ),
            ),
        ]
    )


def _to_category(X):
    """Cast the categorical columns to pandas 'category' dtype.

    LightGBM auto-detects category dtype and splits on it natively (no one-hot),
    and handles NaN natively (no imputation) — the whole reason we picked it.
    """
    X = X.copy()
    for col in CATEGORICAL:
        X[col] = X[col].astype("category")
    return X


def lightgbm(X) -> Pipeline:
    """LightGBM baseline. Native categorical + NaN handling; balanced for imbalance.

    Sensible, untuned defaults (no early stopping on the validation set, so the
    LR/RF/LGBM comparison stays apples-to-apples — none of them saw val during fit).
    """
    return Pipeline(
        [
            ("cast", FunctionTransformer(_to_category)),
            (
                "clf",
                LGBMClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    class_weight="balanced",
                    random_state=config.SEED,
                    n_jobs=-1,
                    verbose=-1,
                ),
            ),
        ]
    )


def random_forest(X) -> Pipeline:
    """Random Forest with balanced class weights.

    min_samples_leaf caps tree depth implicitly: it keeps leaves from chasing
    individual rare frauds (overfit) and keeps training fast on ~795k rows.
    'balanced_subsample' rebalances within each bootstrap sample.
    """
    return Pipeline(
        [
            ("prep", tree_preprocessor(X)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=50,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=config.SEED,
                ),
            ),
        ]
    )


# --- baseline artifact lifecycle (Step 9) -----------------------------------


def _versions() -> dict:
    import lightgbm, numpy, pandas, sklearn

    return {
        "python": __import__("platform").python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
    }


def train_baseline():
    """Train the winning LightGBM on TRAIN, pick the cost-optimal threshold on VAL.

    Threshold is selected on validation (never on test) so the test report is an
    honest out-of-sample estimate. Returns (pipeline, threshold, val_result).
    """
    train, val, _ = pp.load_processed()
    Xtr, ytr = pp.split_xy(train)
    Xva, yva = pp.split_xy(val)

    pipe = lightgbm(Xtr)
    pipe.fit(Xtr, ytr)

    val_scores = pipe.predict_proba(Xva)[:, 1]
    co = metrics.cost_optimal_threshold(yva, val_scores, config.COST_FN, config.COST_FP)
    threshold = co["threshold"]
    val_result = metrics.evaluate(yva, val_scores, threshold, model="LightGBM (val)")
    return pipe, threshold, val_result


def save_baseline(pipe, threshold: float, extra_meta: dict | None = None):
    """Persist the pipeline (joblib) + a human-readable JSON metadata sidecar."""
    config.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "threshold": threshold}, MODEL_PATH)

    meta = {
        "model": "LightGBM",
        "created": date.today().isoformat(),
        "trained_on": "BAF Base.csv, months 0-5 (train)",
        "threshold_selected_on": "month 6 (val), cost-optimal",
        "threshold": threshold,
        "cost_matrix": {"false_negative": config.COST_FN, "false_positive": config.COST_FP},
        "seed": config.SEED,
        "dataset_sha256": data_mod.sha256(config.DATA_RAW / "Base.csv"),
        "versions": _versions(),
    }
    if extra_meta:
        meta.update(extra_meta)
    META_PATH.write_text(json.dumps(meta, indent=2))
    return MODEL_PATH, META_PATH


def load_baseline():
    """Load (pipeline, threshold) from the saved artifact."""
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle["threshold"]
