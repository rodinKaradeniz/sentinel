"""Preprocessing + temporal split for BAF.

``clean()`` is deterministic (no fitting), so it is safe to apply to the whole frame.
Anything that *learns* from data (one-hot categories, scaling stats) is deferred to the
per-model pipeline and fit on the training split only — see ml/src/models.py.
"""
from __future__ import annotations

import pandas as pd

from . import config

# --- column groups (from EDA, see reports/eda_findings.md) ------------------

CATEGORICAL = ["payment_type", "employment_status", "housing_status", "source", "device_os"]

# Constant column (all zeros) + the split key, neither belongs in the feature matrix.
DROP = ["device_fraud_count", "month"]

# Count fields where -1 means "unknown": flag + map -1 -> NaN.
SENTINEL_MINUS1 = [
    "prev_address_months_count",
    "bank_months_count",
    "current_address_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]

# Continuous field where negatives mean "not applicable": flag + negatives -> NaN.
SENTINEL_NEGATIVE = ["intended_balcon_amount"]

# Temporal split boundaries (months 0-7). Train early, validate/test on later months.
TRAIN_MONTHS = [0, 1, 2, 3, 4, 5]
VAL_MONTHS = [6]
TEST_MONTHS = [7]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(config.DATA_RAW / "Base.csv")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Handle negative sentinels and drop dead columns. No fitting → no leakage.

    Adds ``<col>_missing`` flags and replaces the sentinel values with NaN so the
    downstream model (LightGBM natively, or an imputer for LR/RF) can handle them.
    Keeps ``month`` on the frame here; it is dropped only when building X (it is
    needed first for the split).
    """
    df = df.copy()

    for col in SENTINEL_MINUS1:
        df[f"{col}_missing"] = (df[col] == -1).astype("int8")
        df.loc[df[col] == -1, col] = pd.NA

    for col in SENTINEL_NEGATIVE:
        df[f"{col}_missing"] = (df[col] < 0).astype("int8")
        df.loc[df[col] < 0, col] = pd.NA

    # device_fraud_count is constant; month is dropped later (needed for split).
    df = df.drop(columns=["device_fraud_count"])
    return df


def temporal_split(df: pd.DataFrame):
    """Split a cleaned frame into (train, val, test) by month. Earlier → later."""
    train = df[df["month"].isin(TRAIN_MONTHS)].copy()
    val = df[df["month"].isin(VAL_MONTHS)].copy()
    test = df[df["month"].isin(TEST_MONTHS)].copy()
    return train, val, test


def split_xy(df: pd.DataFrame):
    """Drop non-feature columns and separate the target. Call AFTER the split."""
    y = df[config.TARGET]
    X = df.drop(columns=[config.TARGET] + ["month"])
    return X, y


def build_processed(save: bool = True):
    """Full Step-4 pipeline: load → clean → temporal split → (optionally) persist.

    Returns (train, val, test) cleaned frames (target + features, incl. ``month``).
    """
    df = clean(load_raw())
    train, val, test = temporal_split(df)
    if save:
        config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        train.to_parquet(config.DATA_PROCESSED / "train.parquet", index=False)
        val.to_parquet(config.DATA_PROCESSED / "val.parquet", index=False)
        test.to_parquet(config.DATA_PROCESSED / "test.parquet", index=False)
    return train, val, test


def load_processed():
    """Load the persisted splits. Run build_processed() first if missing."""
    p = config.DATA_PROCESSED
    return (
        pd.read_parquet(p / "train.parquet"),
        pd.read_parquet(p / "val.parquet"),
        pd.read_parquet(p / "test.parquet"),
    )
