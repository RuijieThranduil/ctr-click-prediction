window.ctrProjectData = {
  meta: {
    title: "Advertising Click-Through-Rate Prediction",
    subtitle: "A compact case study in leakage-safe feature engineering, honest model evaluation, and scalable CTR prediction.",
    audience: "Recruiters, data science interviewers, and project reviewers",
    updated: "2026-06-18",
    source: "Kaggle Avazu CTR Prediction",
    sample: "1% reproducible random sample, 404,892 rows, seed=42",
    scope: "EDA, feature engineering, NumPy logistic regression, XGBoost notebook path, and PySpark/Databricks scale-out path",
    repository: "https://github.com/RuijieThranduil/ctr-click-prediction"
  },
  purpose: {
    question: "Can high-cardinality ad, site, app, and device identifiers improve click prediction without leaking the target?",
    judgment: "Yes. Out-of-fold target encoding keeps the signal from high-cardinality categorical features while preserving honest validation.",
    takeaway: "The strongest project signal is not a more complicated model; it is a careful feature pipeline and evaluation setup."
  },
  metrics: [
    {
      model: "Mean CTR baseline",
      logLoss: 0.455,
      rocAuc: 0.5,
      prAuc: 0.17,
      note: "Predicts the training click rate for every impression."
    },
    {
      model: "Logistic regression, low-cardinality only",
      logLoss: 0.431,
      rocAuc: 0.654,
      prAuc: 0.285,
      note: "Uses low-cardinality categorical features plus time features."
    },
    {
      model: "Logistic regression + OOF target encoding",
      logLoss: 0.405,
      rocAuc: 0.73,
      prAuc: 0.356,
      note: "Adds leakage-safe encodings for high-cardinality categorical features."
    }
  ],
  metricExplorer: {
    defaultMetric: "rocAuc",
    metrics: [
      {
        key: "logLoss",
        label: "LogLoss",
        direction: "lower is better",
        domain: [0.38, 0.47],
        summary: "The OOF-encoded model reduces held-out LogLoss versus the low-cardinality baseline."
      },
      {
        key: "rocAuc",
        label: "ROC-AUC",
        direction: "higher is better",
        domain: [0.48, 0.76],
        summary: "ROC-AUC shows the ranking lift from preserving high-cardinality categorical signal."
      },
      {
        key: "prAuc",
        label: "PR-AUC",
        direction: "higher is better",
        domain: [0.14, 0.38],
        summary: "PR-AUC is useful because clicks are the minority class."
      }
    ]
  },
  featureImportance: [
    {
      feature: "app_id",
      group: "High-cardinality app identity",
      importance: 0.552,
      note: "App identity is the strongest feature after smoothed target encoding."
    },
    {
      feature: "site_category",
      group: "Publisher context",
      importance: 0.385,
      note: "Site category is low-cardinality but carries strong segment-level CTR signal."
    },
    {
      feature: "app_category",
      group: "App context",
      importance: 0.343,
      note: "App category helps explain broad audience and inventory differences."
    },
    {
      feature: "site_id",
      group: "High-cardinality site identity",
      importance: 0.326,
      note: "Keeping site identifiers avoids throwing away meaningful publisher-level signal."
    },
    {
      feature: "site_domain",
      group: "Publisher context",
      importance: 0.232,
      note: "Domain-level context contributes beyond the broader category field."
    },
    {
      feature: "C14",
      group: "Anonymized ad feature",
      importance: 0.197,
      note: "Anonymized competition fields still encode useful ad or campaign context."
    },
    {
      feature: "C16",
      group: "Anonymized ad feature",
      importance: 0.187,
      note: "A medium-strength categorical signal retained in the model."
    },
    {
      feature: "C18",
      group: "Anonymized ad feature",
      importance: 0.146,
      note: "Useful in combination with other categorical and encoded features."
    }
  ],
  highlights: [
    {
      label: "ROC-AUC lift",
      value: "+0.076",
      detail: "0.654 to 0.730 after keeping high-cardinality features with OOF target encoding."
    },
    {
      label: "LogLoss reduction",
      value: "-5.9%",
      detail: "0.431 to 0.405 versus the low-cardinality logistic regression."
    },
    {
      label: "Click rate",
      value: "16.99%",
      detail: "The sample click-through rate closely matches the full Avazu training set."
    },
    {
      label: "Scale path",
      value: "40.4M rows",
      detail: "A PySpark/Databricks notebook sketches the same workflow for the full dataset."
    }
  ],
  workflow: [
    {
      step: "1",
      title: "Define the evaluation problem",
      body: "Treat CTR as an imbalanced binary classification task and use LogLoss, ROC-AUC, PR-AUC, and calibration instead of accuracy.",
      artifact: "README.md, metrics_np.py"
    },
    {
      step: "2",
      title: "Build a reproducible sample",
      body: "Generate a seeded 1% sample from the Kaggle training CSV so local experiments are fast and repeatable.",
      artifact: "scripts/make_sample.py"
    },
    {
      step: "3",
      title: "Separate feature groups",
      body: "Use one-hot encoding for low-cardinality categories, keep time features, and route high-cardinality IDs to target encoding.",
      artifact: "src/data_prep.py"
    },
    {
      step: "4",
      title: "Prevent target leakage",
      body: "Compute target encodings out-of-fold for training rows and fit the final encoder only on the training split for test rows.",
      artifact: "src/oof_encode.py"
    },
    {
      step: "5",
      title: "Compare baselines honestly",
      body: "Start with mean CTR and low-cardinality logistic regression, then add high-cardinality features to isolate their lift.",
      artifact: "src/run_experiments_np.py"
    },
    {
      step: "6",
      title: "Prepare the scale-out route",
      body: "Document a PySpark/Databricks pipeline for running the same idea on the full 40M-row dataset.",
      artifact: "notebooks/03_spark_pipeline.ipynb"
    }
  ],
  charts: [
    {
      id: "curves",
      title: "Model Evaluation Curves",
      type: "Evaluation",
      src: "../outputs_figs/06_model_curves.png",
      alt: "ROC, precision-recall, and calibration curves comparing low-cardinality and target-encoded logistic regression.",
      insight: "Adding OOF target-encoded high-cardinality features improves ranking quality and precision-recall behavior."
    },
    {
      id: "comparison",
      title: "Metric Comparison",
      type: "Evaluation",
      src: "../outputs_figs/07_metric_comparison.png",
      alt: "Bar chart comparing LogLoss, ROC-AUC, and PR-AUC across model variants.",
      insight: "The target-encoded logistic regression improves all reported metrics over the low-cardinality baseline."
    },
    {
      id: "importance",
      title: "Feature Importance",
      type: "Model",
      src: "../outputs_figs/08_feature_importance.png",
      alt: "Feature importance chart showing app, site, and categorical features as top contributors.",
      insight: "High-cardinality app and site identifiers are among the strongest predictive signals."
    },
    {
      id: "category",
      title: "CTR by Category",
      type: "EDA",
      src: "../outputs_figs/04_ctr_by_category.png",
      alt: "Exploratory chart showing click-through-rate differences across categorical fields.",
      insight: "Categorical segments show much larger CTR spread than simple time-of-day features."
    },
    {
      id: "hour",
      title: "CTR by Hour",
      type: "EDA",
      src: "../outputs_figs/02_ctr_by_hour.png",
      alt: "Chart showing CTR by hour of day.",
      insight: "Time of day exists as a signal, but it is weaker than categorical identity and context fields."
    }
  ],
  methodNotes: [
    "Accuracy is intentionally not a headline metric because a no-click classifier would look strong on an imbalanced CTR dataset.",
    "Plain target encoding can leak a row's own label into its feature value. The project uses cross-fitted encodings to avoid that.",
    "XGBoost is included as an optional notebook path, but the headline lift is already demonstrated with logistic regression.",
    "The public repo excludes Kaggle raw data and generated samples; users regenerate the sample locally."
  ],
  dataContract: {
    goal: "Keep page content separate from page layout.",
    payload: "This showcase reads window.ctrProjectData from showcase/project-data.js.",
    reuse: "Future project pages can replace this payload and reuse the same static rendering pattern."
  }
};
