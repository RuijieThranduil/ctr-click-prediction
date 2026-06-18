"""
modeling.py
-----------
CTR modeling pipeline: baseline -> XGBoost, with proper encoding of
high-cardinality features and honest, imbalance-aware evaluation.

Key design choices (and how they fix the original notebooks):
  * Evaluate with LogLoss (the Avazu competition metric) + ROC-AUC + PR-AUC,
    never accuracy (the label is 83/17 imbalanced).
  * High-cardinality IDs are kept via *smoothed target encoding* fit on the
    training split only (no leakage). The original Spark notebook dropped every
    column with >70 distinct values; we show that costs real performance.
  * Single train/test split done once, reused by every model for a fair compare.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

from data_prep import prepare, TARGET, LOW_CARD_CATEGORICAL, HIGH_CARD_CATEGORICAL, TIME_FEATURES

RANDOM_STATE = 42


# --- smoothed target encoder (leakage-safe, fit on train only) ---------------
class TargetEncoder:
    """Mean-of-target encoding with additive smoothing toward the global mean."""
    def __init__(self, cols, smoothing=20.0):
        self.cols = cols
        self.smoothing = smoothing
        self.maps_ = {}
        self.global_ = None

    def fit(self, X, y):
        self.global_ = y.mean()
        df = X[self.cols].copy()
        df["_y"] = y.values
        for c in self.cols:
            stats = df.groupby(c)["_y"].agg(["mean", "count"])
            # smoothed estimate
            enc = (stats["mean"] * stats["count"] + self.global_ * self.smoothing) / (stats["count"] + self.smoothing)
            self.maps_[c] = enc
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        for c in self.cols:
            out[c + "_te"] = X[c].map(self.maps_[c]).fillna(self.global_).astype(float)
        return out


def load_split(path, test_size=0.30):
    df = prepare(path)
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)


def evaluate(name, y_true, p):
    return {
        "model": name,
        "LogLoss": log_loss(y_true, p),
        "ROC_AUC": roc_auc_score(y_true, p),
        "PR_AUC": average_precision_score(y_true, p),
    }


def build_features(X_train, y_train, X_test, use_high_card=True):
    """Returns dense numpy matrices: low-card one-hot + time + (optional) target-encoded high-card."""
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=50)
    Xtr_ohe = ohe.fit_transform(X_train[LOW_CARD_CATEGORICAL])
    Xte_ohe = ohe.transform(X_test[LOW_CARD_CATEGORICAL])

    tr_time = X_train[TIME_FEATURES].to_numpy(float)
    te_time = X_test[TIME_FEATURES].to_numpy(float)

    blocks_tr, blocks_te = [Xtr_ohe, tr_time], [Xte_ohe, te_time]
    if use_high_card:
        te = TargetEncoder(HIGH_CARD_CATEGORICAL, smoothing=20.0).fit(X_train, y_train)
        blocks_tr.append(te.transform(X_train).to_numpy(float))
        blocks_te.append(te.transform(X_test).to_numpy(float))

    return np.hstack(blocks_tr), np.hstack(blocks_te)
