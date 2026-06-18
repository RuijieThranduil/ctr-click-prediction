"""Builds notebooks/01_eda.ipynb programmatically."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
c = []

c.append(new_markdown_cell(
"""# Avazu CTR — Exploratory Data Analysis

**Goal:** understand what drives a click before modeling. The dataset is 10 days of
mobile ad impressions (Avazu Kaggle competition). We work on a reproducible ~1% random
sample (`data/filtered_train.csv`, seed=42) of the full 40.4M-row training set; the
overall click-through rate is preserved (~17%).

All preprocessing lives in `src/data_prep.py` so the EDA and the model see identical features."""))

c.append(new_code_cell(
"""import sys; sys.path.insert(0, "../src")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_prep import prepare, TARGET, LOW_CARD_CATEGORICAL, HIGH_CARD_CATEGORICAL

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3})

df = prepare("../data/filtered_train.csv")
base = df[TARGET].mean()
print(f"rows={len(df):,}  columns={df.shape[1]}  overall CTR={base:.4f}")
df.head()"""))

c.append(new_markdown_cell(
"""## 1. Data quality & class balance

The data is clean (no missing values). The label is imbalanced: only **~17%** of
impressions are clicks. This matters for modeling — accuracy is a useless metric here
(predicting "no click" for everyone already scores 83%), so we evaluate with
**LogLoss / ROC-AUC / PR-AUC** and use class weighting."""))

c.append(new_code_cell(
"""print("missing values:", int(df.isnull().sum().sum()))
df[TARGET].value_counts(normalize=True).rename({0:"no click",1:"click"})"""))

c.append(new_code_cell(
"""ax = df[TARGET].value_counts().sort_index().plot(
    kind="bar", color=["#4C72B0","#DD8452"], rot=0, figsize=(5,4))
ax.set_xticklabels(["No click","Click"]); ax.set_title(f"Class imbalance (CTR={base:.1%})")
ax.set_ylabel("Impressions"); plt.tight_layout(); plt.show()"""))

c.append(new_markdown_cell(
"""## 2. A helper: CTR and *lift* by category

For a categorical value, **lift = (CTR of that group) / (overall CTR)**. Lift > 1 means
that group clicks more than average. Lift is the single most useful lens in CTR work —
it tells you which segments a model should care about."""))

c.append(new_code_cell(
"""def ctr_lift(col, topn=10):
    g = df.groupby(col)[TARGET].agg(CTR="mean", impressions="count")
    g["lift"] = g["CTR"] / base
    return g.sort_values("impressions", ascending=False).head(topn).round(4)

ctr_lift("banner_pos")"""))

c.append(new_markdown_cell(
"""## 3. Temporal patterns

`hour` is encoded as `YYMMDDHH`. We parse it into hour-of-day and day-of-week.
CTR is fairly stable across the day (small lift range) — time-of-day is a *weak* signal
on its own, which is a useful finding: it tells us not to over-invest in temporal features."""))

c.append(new_code_cell(
"""fig, axes = plt.subplots(1, 2, figsize=(13,4))
df.groupby("hour_of_day")[TARGET].mean().plot(marker="o", ax=axes[0], color="#C44E52")
axes[0].axhline(base, ls="--", color="gray"); axes[0].set_title("CTR by hour of day"); axes[0].set_ylabel("CTR")
df.groupby("day_of_week")[TARGET].mean().plot(kind="bar", ax=axes[1], color="#4C72B0", rot=0)
axes[1].axhline(base, ls="--", color="gray"); axes[1].set_title("CTR by day of week (0=Mon)")
plt.tight_layout(); plt.show()"""))

c.append(new_markdown_cell(
"""## 4. The strong categorical signals

These are where the real predictive power is. Note especially:

- **device_conn_type**: CTR collapses from ~18% (type 0) to ~1.5% (type 5) — a >10x spread.
- **C18 / C15 / C16** (anonymized): some values carry 1.7x–2.5x lift. We don't know what
  they encode, but the model definitely should.
- **banner_pos**: positions 7 and 4 click ~1.7x more than the dominant position 0."""))

c.append(new_code_cell(
"""panels = ["banner_pos","device_type","device_conn_type","C18","C15","C16"]
fig, axes = plt.subplots(2, 3, figsize=(15,8))
for ax, col in zip(axes.ravel(), panels):
    g = df.groupby(col)[TARGET].agg(["mean","count"]).sort_values("count", ascending=False).head(6).sort_index()
    bars = ax.bar(g.index.astype(str), g["mean"], color="#4C72B0")
    ax.axhline(base, ls="--", color="gray")
    for b, m in zip(bars, g["mean"]):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{m/base:.1f}x", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"CTR by {col}")
plt.suptitle("CTR by category (dashed = overall; labels = lift)", fontweight="bold")
plt.tight_layout(); plt.show()"""))

c.append(new_markdown_cell(
"""## 5. Content categories: site vs app"""))

c.append(new_code_cell(
"""fig, axes = plt.subplots(1, 2, figsize=(14,5))
for ax, col in zip(axes, ["site_category","app_category"]):
    g = df.groupby(col)[TARGET].agg(["mean","count"]).sort_values("count", ascending=False).head(8)
    ax.barh(g.index.astype(str), g["mean"], color="#55A868"); ax.axvline(base, ls="--", color="gray")
    ax.invert_yaxis(); ax.set_title(f"CTR by {col} (top 8 by volume)"); ax.set_xlabel("CTR")
plt.tight_layout(); plt.show()"""))

c.append(new_markdown_cell(
"""## 6. Cardinality — why we can't one-hot everything

Several ID-like columns have tens of thousands of distinct values. One-hot encoding them
would explode the feature space and overfit. We therefore split features into:

- **low-cardinality** (≤ ~25 values): one-hot encode.
- **high-cardinality** (IDs, `device_ip`, `C14`, …): need hashing or target/frequency
  encoding. The original Spark notebook simply *dropped* these (kept only ≤70-cardinality
  columns) — that throws away signal, and improving on it is a concrete win for the model."""))

c.append(new_code_cell(
"""card = df[LOW_CARD_CATEGORICAL + HIGH_CARD_CATEGORICAL].nunique().sort_values(ascending=False)
card.to_frame("distinct_values")"""))

c.append(new_markdown_cell(
"""## Key takeaways (carried into modeling)

1. **Imbalanced label (17% CTR)** → optimize/evaluate with LogLoss & AUC, use class weights; never accuracy.
2. **Strongest signals are categorical**: `device_conn_type`, `C18`, `C15/C16`, `banner_pos`, content categories. Temporal features are weak.
3. **High-cardinality IDs hold signal** but need smart encoding (hashing/target encoding), not one-hot or dropping.
4. **Lift framing** gives an interpretable, business-friendly view of which segments convert — useful for both modeling and stakeholder communication."""))

nb["cells"] = c
nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
with open("notebooks/01_eda.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote notebooks/01_eda.ipynb with", len(c), "cells")
