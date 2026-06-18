"""Run the model comparison and save results + figures."""
import sys, os, json, time
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

from modeling import load_split, build_features, evaluate, TARGET

OUT = "outputs_figs"; os.makedirs(OUT, exist_ok=True)
t0 = time.time()

X_tr, X_te, y_tr, y_te = load_split("data/filtered_train.csv")
base = y_tr.mean()
print(f"train={len(X_tr):,} test={len(X_te):,} CTR={base:.4f}")

# Feature matrices: with and without high-cardinality features
Xtr_full, Xte_full = build_features(X_tr, y_tr, X_te, use_high_card=True)
Xtr_low,  Xte_low  = build_features(X_tr, y_tr, X_te, use_high_card=False)
print("feature dims: full=%d  low_only=%d" % (Xtr_full.shape[1], Xtr_low.shape[1]))

results = []
preds = {}

# 1) Logistic Regression baseline (low-card only) — the "naive" reference
lr0 = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
lr0.fit(Xtr_low, y_tr)
p = lr0.predict_proba(Xte_low)[:, 1]; preds["LR (low-card only)"] = p
results.append(evaluate("LR (low-card only)", y_te, p))

# 2) Logistic Regression + target-encoded high-card
lr1 = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
lr1.fit(Xtr_full, y_tr)
p = lr1.predict_proba(Xte_full)[:, 1]; preds["LR (+ high-card TE)"] = p
results.append(evaluate("LR (+ high-card TE)", y_te, p))

# scale_pos_weight for imbalance
spw = (1 - base) / base

# 3) XGBoost WITHOUT high-card (mirrors the original 'drop high-card' choice)
xgb0 = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                     subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                     eval_metric="logloss", scale_pos_weight=spw, n_jobs=4,
                     random_state=42)
xgb0.fit(Xtr_low, y_tr)
p = xgb0.predict_proba(Xte_low)[:, 1]; preds["XGB (low-card only)"] = p
results.append(evaluate("XGB (low-card only)", y_te, p))

# 4) XGBoost WITH high-card target encoding — the full model
xgb1 = XGBClassifier(n_estimators=400, max_depth=7, learning_rate=0.1,
                     subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                     eval_metric="logloss", scale_pos_weight=spw, n_jobs=4,
                     random_state=42)
xgb1.fit(Xtr_full, y_tr)
p = xgb1.predict_proba(Xte_full)[:, 1]; preds["XGB (+ high-card TE)"] = p
results.append(evaluate("XGB (+ high-card TE)", y_te, p))

res = pd.DataFrame(results).set_index("model").round(4)
print("\n=== RESULTS (test set) ===")
print(res.to_string())
res.to_csv("outputs_figs/model_results.csv")

# --- ROC + PR + calibration for the two flagship models ---
flagship = ["LR (low-card only)", "XGB (+ high-card TE)"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for name in flagship:
    fpr, tpr, _ = roc_curve(y_te, preds[name]); axes[0].plot(fpr, tpr, label=name)
    pr, rc, _ = precision_recall_curve(y_te, preds[name]); axes[1].plot(rc, pr, label=name)
    frac_pos, mean_pred = calibration_curve(y_te, preds[name], n_bins=10)
    axes[2].plot(mean_pred, frac_pos, marker="o", label=name)
axes[0].plot([0, 1], [0, 1], "k--", alpha=.4); axes[0].set_title("ROC curve"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].legend()
axes[1].axhline(base, ls="--", color="gray"); axes[1].set_title("Precision-Recall"); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].legend()
axes[2].plot([0, 1], [0, 1], "k--", alpha=.4); axes[2].set_title("Calibration"); axes[2].set_xlabel("Mean predicted"); axes[2].set_ylabel("Observed CTR"); axes[2].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/06_model_curves.png"); plt.close(fig)

# --- results bar chart ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, metric, better in zip(axes, ["LogLoss", "ROC_AUC", "PR_AUC"], ["lower", "higher", "higher"]):
    res[metric].plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title(f"{metric} ({better} = better)"); ax.invert_yaxis()
plt.tight_layout(); plt.savefig(f"{OUT}/07_metric_comparison.png"); plt.close(fig)

print(f"\nElapsed {time.time()-t0:.0f}s. Saved results + figures.")
