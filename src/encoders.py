"""Leakage-safe categorical encoders (pure pandas, no sklearn)."""
import pandas as pd


class TargetEncoder:
    """Mean-of-target encoding with additive smoothing toward the global mean.

    Fit on the training split only; unseen categories at transform time fall back
    to the global mean. Smoothing shrinks low-count categories toward the prior so
    rare values don't get overconfident estimates.
    """
    def __init__(self, cols, smoothing=20.0):
        self.cols = cols
        self.smoothing = smoothing
        self.maps_ = {}
        self.global_ = None

    def fit(self, X, y):
        y = pd.Series(y).reset_index(drop=True)
        self.global_ = float(y.mean())
        df = X[self.cols].reset_index(drop=True).copy()
        df["_y"] = y.values
        for c in self.cols:
            stats = df.groupby(c)["_y"].agg(["mean", "count"])
            enc = (stats["mean"] * stats["count"] + self.global_ * self.smoothing) / (stats["count"] + self.smoothing)
            self.maps_[c] = enc
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        for c in self.cols:
            out[c + "_te"] = X[c].map(self.maps_[c]).fillna(self.global_).astype(float)
        return out
