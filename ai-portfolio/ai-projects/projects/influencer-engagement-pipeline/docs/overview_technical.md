# Influencer Engagement Pipeline — Technical Overview

## Architecture

```
prepare_data.py          (data prep — offline)
       |
  Airflow DAG (dag.py)
       |
  ingest.py              PySpark CSV → Delta Lake
       |
  feature_engineer.py    Delta → engineered features → parquet
       |
  train.py               XGBoost 3-class classifier → MLflow
      / \
explain.py   drift.py    SHAP TreeExplainer | Evidently PSI
                  |
             alert.py    Mock retraining alert
       |
  app.py                 Streamlit dashboard (reads artifacts)
```

## Data Pipeline

**Source:** Social Media Engagement 2025 dataset (Kaggle, 20K rows, MIT license).

**Ingestion:** PySpark reads cleaned CSVs with schema validation and writes
Delta Lake tables. Delta provides ACID transactions and schema enforcement
over raw parquet — important for production data pipelines even at this scale.

**Feature Engineering:**
- `log_followers` — log1p transform to normalize heavy-tailed distribution
- `follow_ratio` — following/followers, capped at 10 (engagement bot signal)
- `content_engagement_density` — (likes + comments) / content_length
- `is_micro_influencer` — binary flag for followers < 50K
- `account_maturity` — log-scaled account age
- `share_to_like_ratio` — virality signal
- `comment_to_like_ratio` — conversation depth signal
- One-hot encoding for categoricals (gender, topic, device, language)

**Target:** `engagement_tier` (high/medium/low) derived from `engagement_rate`
using 33rd/66th percentile thresholds.

## Model

**XGBoost multi-class classifier** — the right tool for tabular data at this
scale. Trains in seconds locally, integrates cleanly with SHAP's TreeExplainer,
and is what a data team would actually deploy in production.

**Hyperparameters:** 300 estimators, max_depth=6, learning_rate=0.1,
subsample=0.8, colsample_bytree=0.8. Logged to MLflow for reproducibility.

**Metrics logged:** accuracy, macro/weighted F1, per-class precision/recall/F1,
confusion matrix, feature importance.

## Experiment Tracking (MLflow)

Local file-based MLflow tracking (`file:./data/mlflow`). Each training run
logs hyperparameters, metrics, the XGBoost model artifact, confusion matrix
plot, and classification report. Run `mlflow ui --backend-store-uri file:./data/mlflow`
to browse experiments.

## Explainability (SHAP)

SHAP TreeExplainer provides per-feature, per-prediction importance scores.
Generated artifacts:
- Beeswarm plot — feature value impact on predictions
- Bar plot — mean |SHAP value| global importance
- Raw SHAP values saved for interactive exploration in Streamlit

## Drift Monitoring (Evidently)

Compares training data distributions against a simulated future data slice
with injected drift (inflated follower counts, shifted topic mix, younger
user demographics). Uses PSI (Population Stability Index) thresholds:
- PSI < 0.1 — stable
- 0.1 < PSI < 0.2 — moderate drift, monitor
- PSI > 0.2 — significant drift, investigate
- PSI > 0.25 — critical, retrain recommended

Generates an HTML report and per-feature drift summary JSON.

## Orchestration (Airflow)

TaskFlow API (`@task` decorators) with explicit data dependencies.
Tasks pass lightweight dicts (file paths, metrics) via XCom — never DataFrames.
SequentialExecutor for local development. DAG is manually triggered
(`schedule=None`).

## Tradeoffs

- **Delta Lake at small scale:** Overkill for 20K rows, but demonstrates
  awareness of lakehouse patterns (ACID, schema evolution, time travel).
- **Airflow for a local pipeline:** Heavy dependency for local use, but shows
  production orchestration skills. Each task module works standalone too.
- **Synthetic drift injection:** The future data slice has artificial drift.
  In production, drift monitoring runs on each new batch of real data.
- **No GPU required:** XGBoost on tabular data trains on CPU in seconds.
  This was a deliberate choice to keep the project fully runnable locally.

## Production Deployment Path: Databricks

This project runs each component locally, but the stack maps directly to
Databricks — the managed platform that bundles all of these together:

| Local (this project) | Databricks equivalent |
|---|---|
| PySpark `local[*]` | Managed Spark cluster (auto-scaling) |
| `delta-spark` library | Delta Lake (native, optimized) |
| MLflow server (SQLite) | Managed MLflow (hosted tracking + registry) |
| Airflow TaskFlow DAG | Databricks Workflows / Jobs |
| Standalone `.py` scripts | Databricks Notebooks or Repos |

**Migration path:** Each `.py` module is already structured as a standalone
function (`run_*`) that takes paths and returns dicts. Moving to Databricks
would mean: (1) replace local file paths with DBFS/Unity Catalog paths,
(2) swap Airflow for Databricks Workflows, (3) point MLflow at the managed
tracking server. The core logic — feature engineering, XGBoost training,
SHAP analysis, Evidently drift monitoring — stays identical.

**Why build locally first:** Demonstrates understanding of the underlying
technologies without relying on a managed platform. In an interview: "I built
this with the same stack Databricks wraps, so I understand what's happening
under the hood."

## Upgrade Opportunities

- **PySpark feature engineering:** Currently loads Delta into pandas for
  feature engineering. At production scale (millions of rows), keep the
  entire feature pipeline in PySpark DataFrames to leverage distributed
  compute. The current approach works at 20K rows but wouldn't scale.
- **Streaming ingestion:** Replace batch CSV ingestion with Spark Structured
  Streaming for near-real-time engagement data processing.
- **Hyperparameter tuning:** Add Optuna or MLflow's hyperopt integration
  for automated XGBoost hyperparameter search.
- **Model registry:** Promote the best model to MLflow Model Registry
  with staging/production aliases for controlled rollouts.

## Relevance to Influencer Marketing

Every feature and design choice maps to Traackr's domain:
- Platform-aware engagement thresholds (TikTok vs LinkedIn baselines differ)
- Micro-influencer segmentation (different engagement dynamics below 50K)
- Content strategy features (posting cadence, content length, media usage)
- Drift monitoring on platform mix (e.g., TikTok's growing share)
- SHAP explanations for brand clients ("why this influencer?")
