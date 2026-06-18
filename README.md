# Advertising Click-Through-Rate (CTR) Prediction

Predicting whether a mobile ad impression results in a click, using the
[Avazu CTR](https://www.kaggle.com/c/avazu-ctr-prediction) dataset (40.4M impressions, 24 features).
The project covers the full workflow: EDA -> leakage-safe feature engineering -> baseline and
gradient-boosted models -> honest, imbalance-aware evaluation - implemented both **locally
(pandas/NumPy)** and **at scale (PySpark/Databricks)**.

## Headline result

Correctly engineering the **high-cardinality features** that a naive pipeline discards lifts
**ROC-AUC from 0.654 -> 0.730** (+0.076) with logistic regression alone, while also improving
LogLoss and calibration. Out-of-fold target encoding was essential to avoid target leakage.

| Model | LogLoss (lower) | ROC-AUC (higher) | PR-AUC (higher) |
|---|---|---|---|
| Baseline (predict mean CTR) | 0.455 | 0.500 | 0.170 |
| Logistic Regression, low-cardinality features only | 0.431 | 0.654 | 0.285 |
| **Logistic Regression + high-cardinality OOF target encoding** | **0.405** | **0.730** | **0.356** |
| XGBoost (same features, see notebook) | - | ~0.74-0.76 | - |

*Results on a reproducible 1% stratified sample (404,892 rows, seed=42) of the full training set;
the 16.99% click-through rate matches the full dataset. Held-out 30% test split.*

## Key decisions (and why they matter)

- **Imbalance-aware metrics.** CTR is ~17%, so accuracy is meaningless (predicting "no click"
  for everyone scores 83%). Everything is evaluated with **LogLoss** (the Avazu competition
  metric), **ROC-AUC**, and **PR-AUC**, plus a calibration check.
- **High-cardinality features kept, not dropped.** IDs like `app_id`, `site_id`, `device_model`
  and `C14` have thousands of distinct values. One-hot encoding explodes; dropping them throws
  away signal. They are encoded with **smoothed target encoding computed out-of-fold** on the
  training set - this turned out to be the single biggest driver of performance.
- **Leakage control.** Naive target encoding (using a row's own label) actually *hurt* LogLoss
  (0.43 -> 0.54). Out-of-fold encoding fixed it and is documented in the notebook as a worked example.
- **Baseline-first.** A constant-rate baseline and a logistic-regression baseline are reported
  before any boosted model, so each step's added complexity is justified by a real gain.

## What's in here

```
.
├── notebooks/
│   ├── 01_eda.ipynb            # exploratory analysis, CTR-by-segment, lift
│   ├── 02_modeling.ipynb       # encoding, baseline->LR->XGBoost, evaluation, feature importance
│   └── 03_spark_pipeline.ipynb # scalable PySpark/Databricks version (runs on the full 40M rows)
├── src/
│   ├── data_prep.py            # loading + time features + feature groups
│   ├── encoders.py             # smoothed target encoder
│   ├── oof_encode.py           # out-of-fold (leakage-safe) target encoding
│   ├── logreg_np.py            # logistic regression from scratch (NumPy)
│   ├── metrics_np.py           # LogLoss / ROC-AUC / PR-AUC / calibration from scratch
│   ├── run_experiments_np.py   # reproduces the results table + curves
│   └── feature_importance.py   # importance ranking + original-bug demonstration
├── data/
│   └── filtered_train.csv      # 1% reproducible sample (created from train.csv)
└── outputs_figs/               # generated EDA + model figures
```

## Reproduce

```bash
# create the sample from the full Kaggle train.csv (or use your own filtered_train.csv)
awk 'BEGIN{srand(42)} NR==1 || rand()<0.01' train/train.csv > data/filtered_train.csv

# run the modeling experiments (NumPy-only path: no scikit-learn/xgboost required)
python src/run_experiments_np.py
python src/feature_importance.py
```

The logistic-regression pipeline and all metrics are implemented from scratch in NumPy, so the
core results reproduce with only `numpy`, `pandas`, and `matplotlib`. The XGBoost and tuning
sections in `02_modeling.ipynb` use `scikit-learn` + `xgboost` (run in Colab or a full local env).

## EDA highlights

- The strongest signals are **categorical**, not temporal: `device_conn_type` CTR ranges from
  ~18% to ~1.5% (>10x); `banner_pos` 7 clicks ~1.7x the dominant position; several anonymized
  features (`C18`, `C15`, `C16`) carry 1.7-2.5x lift.
- **Time of day is a weak signal** - useful to know, so effort goes to the features that matter.

## Tech

Python (pandas, NumPy, matplotlib), scikit-learn & XGBoost (production model), PySpark / Databricks
(scalable pipeline). Methodology: target/out-of-fold encoding, class-imbalance handling,
cross-validated hyperparameter tuning, ROC/PR/calibration evaluation.
