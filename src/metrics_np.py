"""Pure-numpy evaluation metrics (no scipy/sklearn needed).

Implemented from scratch so the pipeline runs anywhere and to make the math
explicit: LogLoss is the Avazu competition metric; ROC-AUC via the rank
(Mann-Whitney) identity; PR-AUC via the trapezoid under the precision-recall curve.
"""
import numpy as np


def log_loss(y, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def roc_auc(y, p):
    """AUC = P(score(pos) > score(neg)) using average ranks (handles ties)."""
    y = np.asarray(y); p = np.asarray(p)
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float)
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0  # 1-based average rank
        i = j + 1
    n_pos = y.sum(); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[y == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pr_auc(y, p):
    """Average precision = area under precision-recall curve (trapezoid)."""
    y = np.asarray(y); p = np.asarray(p)
    order = np.argsort(-p, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapz(precision, recall))


def calibration_bins(y, p, n_bins=10):
    """Return (mean_predicted, observed_rate) per equal-width probability bin."""
    y = np.asarray(y); p = np.asarray(p)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    mp, obs = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 0:
            mp.append(p[m].mean()); obs.append(y[m].mean())
    return np.array(mp), np.array(obs)
