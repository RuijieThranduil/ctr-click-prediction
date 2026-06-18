"""Generate EDA figures for the Avazu CTR project. Saves PNGs to outputs_figs/."""
import sys; sys.path.insert(0, "src")
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_prep import prepare, TARGET

OUT = "outputs_figs"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3,
                     "font.size": 10, "axes.titleweight": "bold"})

df = prepare("data/filtered_train.csv")
base = df[TARGET].mean()
N = len(df)


def ctr_by(col):
    g = df.groupby(col)[TARGET].agg(["mean", "count"])
    g["lift"] = g["mean"] / base
    return g


# 1) Class imbalance
fig, ax = plt.subplots(figsize=(5, 4))
vc = df[TARGET].value_counts().sort_index()
bars = ax.bar(["No click (0)", "Click (1)"], vc.values, color=["#4C72B0", "#DD8452"])
for b, v in zip(bars, vc.values):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n{v/N:.1%}", ha="center", va="bottom")
ax.set_title(f"Class imbalance — overall CTR = {base:.1%}")
ax.set_ylabel("Impressions")
fig.tight_layout(); fig.savefig(f"{OUT}/01_class_imbalance.png"); plt.close(fig)

# 2) CTR by hour of day
fig, ax = plt.subplots(figsize=(8, 4))
g = df.groupby("hour_of_day")[TARGET].mean()
ax.plot(g.index, g.values, marker="o", color="#C44E52")
ax.axhline(base, ls="--", color="gray", label=f"overall {base:.1%}")
ax.set_xlabel("Hour of day"); ax.set_ylabel("CTR"); ax.set_xticks(range(0, 24, 2))
ax.set_title("CTR by hour of day"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/02_ctr_by_hour.png"); plt.close(fig)

# 3) CTR by day across the campaign (temporal trend)
fig, ax = plt.subplots(figsize=(8, 4))
gd = df.groupby(df["timestamp"].dt.date)[TARGET].agg(["mean", "count"])
ax.bar(range(len(gd)), gd["count"], color="#CCCCCC", alpha=0.6)
ax.set_ylabel("Impressions", color="gray")
ax2 = ax.twinx()
ax2.plot(range(len(gd)), gd["mean"].values, marker="o", color="#C44E52")
ax2.axhline(base, ls="--", color="gray")
ax2.set_ylabel("CTR", color="#C44E52")
ax.set_xticks(range(len(gd))); ax.set_xticklabels([str(d)[5:] for d in gd.index], rotation=45)
ax.set_title("Volume (bars) and CTR (line) over 10 days")
fig.tight_layout(); fig.savefig(f"{OUT}/03_ctr_over_time.png"); plt.close(fig)

# 4) Grid of CTR-vs-category for the strongest low-card features
panels = ["banner_pos", "device_type", "device_conn_type", "C18", "C15", "C16"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.ravel(), panels):
    g = ctr_by(col).sort_values("count", ascending=False).head(6).sort_index()
    bars = ax.bar(g.index.astype(str), g["mean"], color="#4C72B0")
    ax.axhline(base, ls="--", color="gray")
    for b, lift in zip(bars, g["lift"]):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{lift:.1f}x",
                ha="center", va="bottom", fontsize=8)
    ax.set_title(f"CTR by {col}"); ax.set_ylabel("CTR")
fig.suptitle("CTR by category (dashed = overall CTR; labels = lift)", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/04_ctr_by_category.png"); plt.close(fig)

# 5) Site vs App category CTR (top by volume)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col in zip(axes, ["site_category", "app_category"]):
    g = ctr_by(col).sort_values("count", ascending=False).head(8)
    ax.barh(g.index.astype(str), g["mean"], color="#55A868")
    ax.axvline(base, ls="--", color="gray")
    ax.invert_yaxis(); ax.set_title(f"CTR by {col} (top 8 by volume)"); ax.set_xlabel("CTR")
fig.tight_layout(); fig.savefig(f"{OUT}/05_ctr_site_app_category.png"); plt.close(fig)

print("Saved figures to", OUT)
for f in sorted(os.listdir(OUT)):
    print(" -", f)
