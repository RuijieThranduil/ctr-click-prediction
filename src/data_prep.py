"""
data_prep.py
------------
Reusable data loading + feature engineering for the Avazu CTR project.

Both the EDA notebook and the modeling notebook import from here, so the
exact same preprocessing is applied everywhere (no copy-paste drift).

Avazu schema (24 columns):
    id, click(label), hour(YYMMDDHH), C1, banner_pos,
    site_id, site_domain, site_category,
    app_id, app_domain, app_category,
    device_id, device_ip, device_model, device_type, device_conn_type,
    C14..C21 (anonymized categorical features)
"""

from __future__ import annotations
import pandas as pd

# --- column groups -----------------------------------------------------------
TARGET = "click"

# Low-cardinality categoricals -> safe to one-hot / index directly.
LOW_CARD_CATEGORICAL = [
    "C1", "banner_pos", "device_type", "device_conn_type",
    "C15", "C16", "C18", "site_category", "app_category",
]

# High-cardinality categoricals -> need hashing / target encoding (not one-hot).
HIGH_CARD_CATEGORICAL = [
    "site_id", "site_domain", "app_id", "app_domain",
    "device_id", "device_ip", "device_model",
    "C14", "C17", "C19", "C20", "C21",
]

# Columns engineered from `hour`.
TIME_FEATURES = ["hour_of_day", "day_of_week", "is_weekend"]

ID_COLS = ["id"]


def load_raw(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Load the Avazu csv. `hour` is read as int then parsed in add_time_features."""
    df = pd.read_csv(path, nrows=nrows)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the YYMMDDHH `hour` field into usable calendar features.

    Note: modern pandas removed `pd.datetime`; we use pd.to_datetime with an
    explicit format string instead (the old notebook's pd.datetime.strptime
    no longer runs).
    """
    df = df.copy()
    ts = pd.to_datetime(df["hour"].astype(str), format="%y%m%d%H")
    df["timestamp"] = ts
    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek          # 0 = Monday
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Avazu has no nulls, but keep a single defensive hook for reuse on raw data."""
    df = df.copy()
    # ensure label is int
    df[TARGET] = df[TARGET].astype(int)
    return df


def prepare(path: str, nrows: int | None = None) -> pd.DataFrame:
    """End-to-end: load -> clean -> time features. Returns a model-ready frame."""
    df = load_raw(path, nrows=nrows)
    df = basic_clean(df)
    df = add_time_features(df)
    return df


if __name__ == "__main__":
    # quick smoke test
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "data/filtered_train.csv"
    d = prepare(p, nrows=50000)
    print("rows:", len(d), "| CTR:", round(d[TARGET].mean(), 4))
    print("time feats:\n", d[["timestamp"] + TIME_FEATURES].head())
