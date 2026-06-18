"""Run CTR experiments using the from-scratch numpy stack (no sklearn/scipy/xgboost).

Produces REAL numbers on the real Avazu sample for the logistic-regression family
and the high-card-encoding ablation, plus all evaluation figures.
"""
import sys, os, time, json
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_prep import prepare, TARGET, LOW_CARD_CATEGORICAL, HIGH_CARD_CATEGORICAL, TIME_FEATURES
from encoders import TargetEncoder
from oof_encode import oof_target_encode
from logreg_np import LogRegNP
import metrics_np as M

OUT = "outputs_figs"; os.makedirs(OUT, exist_ok=True)
RS = 42; t0 = time.time()

df = prepare("data/filtered_train.csv")
rng = np.random.RandomState(RS)
mask = rng.rand(len(df)) < 0.70
tr, te = df[mask].copy(), df[~mask].copy()
y_tr, y_te = tr[TARGET].values, te[TARGET].values
base = y_tr.mean()
print(f"train={len(tr):,} test={len(te):,} CTR={base:.4f}")


def onehot_lowcard(train, test, cols, min_count=50):
    """One-hot low-card cols; categories fixed on train; rare collapsed."""
    frames_tr, frames_te = [], []
    for c in cols:
        vc = train[c].value_counts()
        keep = vc[vc >= min_count].index
        a = pd.get_dummies(train[c].where(train[c].isin(keep), other="__rare__"), prefix=c)
        b = pd.get_dummies(test[c].where(test[c].isin(keep), other="__rare__"), prefix=c)
        b = b.reindex(columns=a.columns, fill_value=0)
        frames_tr.append(a); frames_te.append(b)
    return pd.concat(frames_tr, axis=1).values.astype(float), pd.concat(frames_te, axis=1).values.astype(float)


# --- feature blocks ---
oh_tr, oh_te = onehot_lowcard(tr, te, LOW_CARD_CATEGORICAL)
tm_tr, tm_te = tr[TIME_FEATURES].values.astype(float), te[TIME_FEATURES].values.astype(float)
hc_tr, hc_te = oof_target_encode(tr, y_tr, te, HIGH_CARD_CATEGORICAL, n_splits=5, smoothing=20.0)

X_low_tr = np.hstack([oh_tr, tm_tr]);            X_low_te = np.hstack([oh_te, tm_te])
X_full_tr = np.hstack([oh_tr, tm_tr, hc_tr]);    X_full_te = np.hstack([oh_te, tm_te, hc_te])
print("dims: low=%d full=%d" % (X_low_tr.shape[1], X_full_tr.shape[1]))


def run(name, Xtr, Xte, **kw):
    m = LogRegNP(**kw).fit(Xtr, y_tr)
    p = m.predict_proba(Xte)[:, 1]
    row = {"model": name, "LogLoss": M.log_loss(y_te, p),
           "ROC_AUC": M.roc_auc(y_te, p), "PR_AUC": M.pr_auc(y_te, p)}
    print(f"  {name:34s} LogLoss={row['LogLoss']:.4f} AUC={row['ROC_AUC']:.4f} PR={row['PR_AUC']:.4f}")
    return row, p


# baselines
results, preds = [], {}
# naive constant baseline (predict base rate) for context
p0 = np.full(len(y_te), base)
results.append({"model": "Baseline (predict mean CTR)", "LogLoss": M.log_loss(y_te, p0),
                "ROC_AUC": 0.5, "PR_AUC": M.pr_auc(y_te, p0)})
print(f"  {'Baseline (predict mean CTR)':34s} LogLoss={results[-1]['LogLoss']:.4f}")

r, p = run("LogReg (low-card only)", X_low_tr, X_low_te); results.append(r); preds[r["model"]] = p
r, p = run("LogReg (+ high-card target-enc)", X_full_tr, X_full_te); results.append(r); preds[r["model"]] = p

res = pd.DataFrame(results).set_index("model").round(4)
res.to_csv(f"{OUT}/model_results.csv")
print("\n=== RESULTS (held-out test) ===\n" + res.to_string())

# improvement summary
ll_low = res.loc["LogReg (low-card only)", "LogLoss"]
ll_full = res.loc["LogReg (+ high-card target-enc)", "LogLoss"]
auc_low = res.loc["LogReg (low-card only)", "ROC_AUC"]
auc_full = res.loc["LogReg (+ high-card target-enc)", "ROC_AUC"]
print(f"\nUsing high-card features: LogLoss {ll_low:.4f} -> {ll_full:.4f} "
      f"({100*(ll_low-ll_full)/ll_low:+.1f}%), AUC {auc_low:.4f} -> {auc_full:.4f} ({auc_full-auc_low:+.4f})")

# --- evaluation figures (ROC / PR / calibration) ---
flag = ["LogReg (low-card only)", "LogReg (+ high-card target-enc)"]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for name in flag:
    p = preds[name]
    # ROC
    order = np.argsort(-p); yt = y_te[order]
    tp = np.cumsum(yt) / yt.sum(); fp = np.cumsum(1 - yt) / (len(yt) - yt.sum())
    ax[0].plot(np.concatenate([[0], fp]), np.concatenate([[0], tp]), label=name)
    # PR
    tpc = np.cumsum(yt); fpc = np.cumsum(1 - yt)
    prec = tpc / (tpc + fpc); rec = tpc / yt.sum()
    ax[1].plot(rec, prec, label=name)
    # calibration
    mp, obs = M.calibration_bins(y_te, p, 10)
    ax[2].plot(mp, obs, marker="o", label=name)
ax[0].plot([0, 1], [0, 1], "k--", alpha=.4); ax[0].set_title("ROC"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend(fontsize=8)
ax[1].axhline(base, ls="--", color="gray"); ax[1].set_title("Precision-Recall"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].legend(fontsize=8)
ax[2].plot([0, 1], [0, 1], "k--", alpha=.4); ax[2].set_title("Calibration"); ax[2].set_xlabel("Mean predicted"); ax[2].set_ylabel("Observed CTR"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/06_model_curves.png"); plt.close(fig)

# metric comparison bars
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
plot_res = res.drop(index=["Baseline (predict mean CTR)"])
for a, metric, better in zip(ax, ["LogLoss", "ROC_AUC", "PR_AUC"], ["lower", "higher", "higher"]):
    plot_res[metric].plot(kind="barh", ax=a, color="#4C72B0"); a.set_title(f"{metric} ({better}=better)"); a.invert_yaxis()
plt.tight_layout(); plt.savefig(f"{OUT}/07_metric_comparison.png"); plt.close(fig)

json.dump(res.reset_index().to_dict("records"), open(f"{OUT}/model_results.json", "w"), indent=2)
print(f"\nElapsed {time.time()-t0:.0f}s")
