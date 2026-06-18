"""Logistic regression from scratch (numpy): L2 + optional class weighting, full-batch GD."""
import numpy as np


def _sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


class LogRegNP:
    def __init__(self, lr=0.3, n_iter=1000, l2=1e-4, class_weight_balanced=False,
                 seed=42, verbose=False):
        self.lr = lr; self.n_iter = n_iter; self.l2 = l2
        self.balanced = class_weight_balanced; self.seed = seed; self.verbose = verbose

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.mu_ = X.mean(0); self.sd_ = X.std(0) + 1e-8
        Xs = (X - self.mu_) / self.sd_
        n, d = Xs.shape
        if self.balanced:
            pos = y.mean()
            w = np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))
        else:
            w = np.ones(n)
        W = w / w.mean()
        p0 = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        self.w_ = np.zeros(d); self.b_ = np.log(p0 / (1 - p0))
        for it in range(self.n_iter):
            p = _sigmoid(Xs @ self.w_ + self.b_)
            g = (p - y) * W
            self.w_ -= self.lr * (Xs.T @ g / n + self.l2 * self.w_)
            self.b_ -= self.lr * g.mean()
            if self.verbose and (it % 200 == 0 or it == self.n_iter - 1):
                pc = np.clip(p, 1e-15, 1 - 1e-15)
                ll = -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))
                print(f"    iter {it:4d}  train_logloss={ll:.4f}")
        return self

    def predict_proba(self, X):
        Xs = (np.asarray(X, float) - self.mu_) / self.sd_
        p = _sigmoid(Xs @ self.w_ + self.b_)
        return np.column_stack([1 - p, p])
