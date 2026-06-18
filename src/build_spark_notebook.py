"""Builds notebooks/03_spark_pipeline.ipynb — a cleaned, methodology-aligned
PySpark/Databricks version of the original notebook."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook(); c = []

c.append(new_markdown_cell(
"""# Avazu CTR — Scalable Pipeline (PySpark / Databricks)

This is the **big-data** version of the same CTR model. It runs on Spark so it scales to the
full 40M-row Avazu dataset on a cluster, whereas the pandas notebook works on an in-memory
sample. The modeling decisions mirror `02_modeling.ipynb` so the two tell one consistent story.

**Improvements over the original Spark notebook**
1. Adds a **Logistic Regression baseline** before the gradient-boosted trees (you need a baseline
   to claim the GBT is worth its complexity).
2. Evaluates with **LogLoss** (the Avazu metric) *and* ROC-AUC — the original reported AUC only.
3. **Fixes the feature-importance bug**: the original sorted by feature *name*, not weight.
4. Documents the **high-cardinality trade-off** (the ≤70-distinct-values filter drops columns that
   the pandas notebook shows are the most predictive) and points to FeatureHasher as the scalable fix.

> Run this in Databricks (or any Spark environment). Set `data_path` to your uploaded CSV."""))

c.append(new_code_cell(
"""# Databricks provides `spark`. Load the data and derive hour-of-day from YYMMDDHH.
data_path = "dbfs:/FileStore/tables/filtered_train.csv"
impression = (spark.read.format("csv").option("header", "true").load(data_path)
              .selectExpr("*", "substr(hour, 7) as hr")
              .repartition(64))
if "_c0" in impression.columns:
    impression = impression.drop("_c0")

# cast numeric/categorical-int columns
INT_COLS = ['click','hour','C1','banner_pos','device_type','device_conn_type',
            'C14','C15','C16','C17','C18','C19','C20','C21','hr']
for col_name in INT_COLS:
    if col_name in impression.columns:
        impression = impression.withColumn(col_name, impression[col_name].cast('int'))

print("rows:", impression.count())
impression.groupBy("click").count().show()  # class balance (~17% CTR)"""))

c.append(new_markdown_cell(
"""## Feature engineering

As in the original, we treat all columns except `click` as categorical and `StringIndex` the
ones with manageable cardinality. We keep the `maxBins=70` filter so GBT training is tractable,
**but note the trade-off**: this drops high-cardinality columns (`app_id`, `site_id`, `device_model`,
`C14`, …) that `02_modeling.ipynb` identifies as the *most* predictive. On a cluster the scalable
way to keep them is `FeatureHasher` (hashing trick) or out-of-fold target encoding — see the note
at the bottom."""))

c.append(new_code_cell(
"""from pyspark.sql.functions import countDistinct
from pyspark.ml.feature import StringIndexer, VectorAssembler

str_cols = [f for f, t in impression.dtypes if t == 'string']
int_cols = [f for f, t in impression.dtypes if t == 'int']

counts = {f: impression.select(countDistinct(f)).first()[0] for f in str_cols + int_cols}
maxBins = 70
categorical = [f for f in (str_cols + int_cols) if counts[f] <= maxBins and f != 'click']
print("kept categorical (<=%d distinct):" % maxBins, categorical)
dropped = [f for f in (str_cols + int_cols) if counts[f] > maxBins]
print("dropped high-cardinality:", dropped, "<- these are predictive; see note below")

indexers = [StringIndexer(inputCol=f, outputCol=f + "_idx", handleInvalid="keep") for f in categorical]
assembler = VectorAssembler(inputCols=[f + "_idx" for f in categorical], outputCol="features")
label_indexer = StringIndexer(inputCol="click", outputCol="label")"""))

c.append(new_code_cell(
"""from pyspark.ml import Pipeline

featurizer = Pipeline(stages=indexers + [assembler, label_indexer]).fit(impression)
data = featurizer.transform(impression).select("label", "features", "hr").cache()
train, test = data.randomSplit([0.7, 0.3], seed=42)
print("train:", train.count(), "test:", test.count())"""))

c.append(new_markdown_cell("""## 1. Baseline: Logistic Regression"""))

c.append(new_code_cell(
"""from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

auc_eval = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",
                                         labelCol="label", metricName="areaUnderROC")
logloss_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                 probabilityCol="probability", metricName="logLoss")

lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)
lr_model = lr.fit(train)
lr_pred = lr_model.transform(test)
print("LR  ROC-AUC = %.4f" % auc_eval.evaluate(lr_pred))
print("LR  LogLoss = %.4f" % logloss_eval.evaluate(lr_pred))"""))

c.append(new_markdown_cell("""## 2. Gradient-Boosted Trees"""))

c.append(new_code_cell(
"""from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(featuresCol="features", labelCol="label",
                    maxBins=maxBins, maxDepth=10, maxIter=10, seed=42)
gbt_model = gbt.fit(train)
gbt_pred = gbt_model.transform(test)
print("GBT  ROC-AUC = %.4f" % auc_eval.evaluate(gbt_pred))
print("GBT  LogLoss = %.4f" % logloss_eval.evaluate(gbt_pred))"""))

c.append(new_markdown_cell(
"""## 3. Feature importance — corrected

**The bug:** the original did
`sorted(zip(weights, features), key=lambda x: x[1], reverse=True)`, where `x[1]` is the feature
*name* — so it sorted alphabetically, not by importance. Also, `model.featureImportances` aligns
with the **assembler input order** (`categorical`), so we zip against that, not against vector
metadata. The fix sorts by the numeric weight."""))

c.append(new_code_cell(
"""importances = gbt_model.featureImportances.toArray()
# align importances with assembler input columns
fi = sorted(zip(categorical, importances), key=lambda x: x[1], reverse=True)  # FIX: sort by weight x[1]
print("Top features by importance:")
for name, w in fi[:15]:
    print(f"  {name:24s} {w:.4f}")

# (optional) display as a Spark DataFrame in Databricks
fi_df = spark.createDataFrame(fi, ["feature", "importance"])
fi_df.orderBy("importance", ascending=False).show(truncate=False)"""))

c.append(new_markdown_cell(
"""## 4. Results & the scalability note

| Model | ROC-AUC | LogLoss |
|---|---|---|
| Logistic Regression (baseline) | _fill from run_ | _fill_ |
| Gradient-Boosted Trees | ~0.70 | _fill_ |

**Scaling up the high-cardinality features.** The `maxBins=70` filter keeps GBT tractable but
discards `app_id`, `site_id`, `device_model`, `C14`, … The pandas notebook shows these lift AUC
from ~0.65 to ~0.73. To keep them at Spark scale without exploding the feature space, use the
hashing trick:

```python
from pyspark.ml.feature import FeatureHasher
hasher = FeatureHasher(inputCols=high_card_cols, outputCol="hashed", numFeatures=2**18)
```

then assemble `hashed` alongside the indexed low-cardinality features. This is the scalable
analogue of the out-of-fold target encoding used in the pandas pipeline."""))

nb["cells"] = c
nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
with open("notebooks/03_spark_pipeline.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote notebooks/03_spark_pipeline.ipynb with", len(c), "cells")
