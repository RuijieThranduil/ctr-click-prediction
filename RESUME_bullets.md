# Resume bullet points — CTR Prediction project

Pick 2–4 depending on the role (DA roles: lean on the EDA/insight + communication bullets;
DS roles: lean on the modeling/leakage/scale bullets). Numbers are from the held-out test set on
a 405K-row stratified sample; **swap in your XGBoost numbers after running the notebook in Colab.**

## Concise (3-bullet version)

- Built an end-to-end click-through-rate prediction pipeline on the **40M-row Avazu** ad dataset
  (Python, pandas, scikit-learn, XGBoost; PySpark for scale), framing it as an imbalanced
  classification problem (~17% CTR) evaluated with **LogLoss, ROC-AUC, and PR-AUC**.
- Improved **ROC-AUC from 0.65 to 0.73** by engineering high-cardinality categorical features
  (ad/site/app IDs) with **out-of-fold target encoding**, after diagnosing and fixing a target-leakage
  bug that had degraded LogLoss.
- Delivered both a local (pandas/NumPy) and a **distributed PySpark/Databricks** implementation, and
  surfaced actionable EDA insights (e.g., connection type and banner position drive a >10× CTR spread).

## Detailed (pick individual lines)

- Engineered a leakage-safe feature pipeline for **40.4M ad impressions**, encoding high-cardinality
  ID features (1K–260K distinct values) via smoothed **out-of-fold target encoding**, lifting
  logistic-regression **ROC-AUC from 0.654 to 0.730** and improving model calibration.
- Diagnosed a **target-leakage** failure where naive target encoding worsened LogLoss (0.43→0.54);
  implemented K-fold out-of-fold encoding to resolve it — demonstrating rigorous validation discipline.
- Established baseline-first evaluation (constant-rate → logistic regression → gradient-boosted trees)
  using **imbalance-appropriate metrics** (LogLoss, ROC-AUC, PR-AUC) instead of accuracy on a 17%-positive label.
- Implemented logistic regression and all evaluation metrics **from scratch in NumPy** (validated
  against closed-form recovery tests), making the core results reproducible with no heavy dependencies.
- Conducted EDA on click behavior across device, connection, banner-position, and content-category
  segments, quantifying drivers with **CTR lift** and finding categorical features far more predictive
  than temporal ones.
- Authored a **scalable PySpark/Databricks** version of the model (StringIndexer + VectorAssembler +
  GBT, FeatureHasher for high-cardinality features) to run on the full dataset on a cluster.
- Refactored an inherited notebook: fixed a feature-importance ranking bug (sorted by name, not weight),
  modernized deprecated pandas date parsing, and corrected a cross-validation variable-shadowing bug.

## One-liner (for a skills/projects line)

> **CTR Prediction (Avazu, 40M rows):** imbalanced ad-click classification; out-of-fold target
> encoding lifted ROC-AUC 0.65→0.73; pandas + PySpark. [github link]

## Honesty notes (read before using)

- The 0.65→0.73 numbers are **real**, from logistic regression on the sample. The XGBoost line is an
  expectation — run `02_modeling.ipynb` in Colab and replace it with your measured value before claiming it.
- If asked in an interview, be ready to explain **why** out-of-fold encoding matters (leakage) and **why**
  LogLoss over accuracy (imbalance). These are the two strongest talking points.
- Put the code on GitHub with the README; "show, don't tell" beats any bullet.
