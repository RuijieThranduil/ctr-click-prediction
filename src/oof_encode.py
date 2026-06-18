"""Out-of-fold (cross-fitted) target encoding to prevent target leakage.

Why: plain target encoding computes a category's encoding from rows that include
that row's own label, so a model overfits the encoded column and the gain vanishes
(or reverses) on unseen data. OOF encoding computes each training row's encoding
from *other* folds only, matching the noisy distribution seen at test time.
"""
import numpy as np
import pandas as pd
from encoders import TargetEncoder


def oof_target_encode(train, y, test, cols, n_splits=5, smoothing=20.0, seed=42):
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    y = np.asarray(y, float)
    idx = np.arange(len(train))
    rng = np.random.RandomState(seed); rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)

    oof = np.zeros((len(train), len(cols)))
    for k in range(n_splits):
        val = folds[k]
        tr = np.concatenate([folds[j] for j in range(n_splits) if j != k])
        enc = TargetEncoder(cols, smoothing).fit(train.iloc[tr], y[tr])
        oof[val] = enc.transform(train.iloc[val]).values
    enc_full = TargetEncoder(cols, smoothing).fit(train, y)
    test_arr = enc_full.transform(test).values
    return oof, test_arr
