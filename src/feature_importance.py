"""Feature importance for the LR+encoding model, and a demo of the original sort bug."""
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from data_prep import prepare, TARGET, LOW_CARD_CATEGORICAL, HIGH_CARD_CATEGORICAL, TIME_FEATURES
from oof_encode import oof_target_encode
from logreg_np import LogRegNP

OUT="outputs_figs"
df = prepare("data/filtered_train.csv")
rng=np.random.RandomState(42); m=rng.rand(len(df))<0.70
tr,te=df[m].copy(),df[~m].copy(); y_tr=tr[TARGET].values

# Build full feature matrix (one-hot low-card + time + OOF high-card), tracking names
def onehot(train,test,cols,min_count=50):
    names=[]; A=[]; B=[]
    for c in cols:
        vc=train[c].value_counts(); keep=vc[vc>=min_count].index
        a=pd.get_dummies(train[c].where(train[c].isin(keep),"__rare__"),prefix=c)
        b=pd.get_dummies(test[c].where(test[c].isin(keep),"__rare__"),prefix=c).reindex(columns=a.columns,fill_value=0)
        names+=list(a.columns); A.append(a.values.astype(float)); B.append(b.values.astype(float))
    return np.hstack(A),np.hstack(B),names

oh_tr,oh_te,oh_names=onehot(tr,te,LOW_CARD_CATEGORICAL)
tm_names=TIME_FEATURES
hc_tr,hc_te=oof_target_encode(tr,y_tr,te,HIGH_CARD_CATEGORICAL,n_splits=5,smoothing=20.0)
hc_names=[c+"_te" for c in HIGH_CARD_CATEGORICAL]

X=np.hstack([oh_tr, tr[TIME_FEATURES].values.astype(float), hc_tr])
names=oh_names+tm_names+hc_names
model=LogRegNP().fit(X,y_tr)

# importance = |standardized coefficient| (features already standardized inside model)
imp=pd.DataFrame({"feature":names,"coef":model.w_,"abs":np.abs(model.w_)})

# ---- BUG DEMO: original code sorted by feature NAME, not weight ----
weights=["%.4f"%w for w in model.w_]
buggy=sorted(zip(weights,names), key=lambda x:x[1], reverse=True)   # original: key=x[1] (name!)
fixed=imp.sort_values("abs",ascending=False)
print("ORIGINAL (buggy) top-5 'by importance' — actually alphabetical by name:")
for w,n in buggy[:5]: print(f"   {n:24s} {w}")
print("\nFIXED top-10 by |coefficient|:")
print(fixed.head(10).to_string(index=False))

# group importance back to original columns
def base_col(f):
    if f.endswith("_te"): return f[:-3]
    for c in LOW_CARD_CATEGORICAL:
        if f.startswith(c+"_"): return c
    return f
imp["column"]=imp["feature"].map(base_col)
grp=imp.groupby("column")["abs"].sum().sort_values(ascending=False)
fig,ax=plt.subplots(figsize=(8,6))
grp.head(15).iloc[::-1].plot(kind="barh",ax=ax,color="#55A868")
ax.set_title("Feature importance (sum |std. LR coef| per column)"); ax.set_xlabel("importance")
plt.tight_layout(); plt.savefig(f"{OUT}/08_feature_importance.png"); plt.close(fig)
grp.to_csv(f"{OUT}/feature_importance.csv")
print("\nTop columns:\n", grp.head(10).round(3).to_string())
