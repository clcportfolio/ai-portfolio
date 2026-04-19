# Influencer Engagement Pipeline — Build Walkthrough

## Construction Narrative

This document walks through the project build order so you can explain
every design decision in an interview.

### Step 1: Data Preparation (`prepare_data.py`)

Started with the Social Media Engagement 2025 dataset from Kaggle (20K rows).
The raw data has text columns (post_content, hashtags) that aren't useful for
tabular ML, so we drop them early. We derive `account_age_days` from the
date columns, then create `engagement_tier` as the classification target
using 33rd/66th percentile thresholds on `engagement_rate`.

The drift slice is created by sampling 20% of the data and injecting
distribution shifts: inflated follower counts (1.5x), younger user ages,
oversampled trending topics, and more mobile devices. This gives Evidently
real drift to detect.

### Step 2: PySpark Ingestion (`ingest.py`)

PySpark reads the cleaned CSVs and writes Delta Lake format. At 20K rows
this is overkill — but it demonstrates the pattern. In production, this
would handle millions of rows with schema evolution and ACID guarantees.
The Delta tables live locally in `data/delta/`.

### Step 3: Feature Engineering (`feature_engineer.py`)

Reads Delta tables back into pandas for feature engineering. Key features:
- `log_followers` normalizes the heavy-tailed follower distribution
- `follow_ratio` catches engagement bots (high following/follower ratio)
- `content_engagement_density` normalizes engagement by content length
- `is_micro_influencer` flags accounts under 50K (different dynamics)
- `share_to_like_ratio` and `comment_to_like_ratio` capture engagement quality

Categoricals (gender, topic, device, language) are one-hot encoded.
Train/test split is 80/20 with stratification on the target.

### Step 4: Model Training (`train.py`)

XGBoost multi-class classifier. Chose XGBoost over deep learning because
it's the right tool for tabular data — trains in seconds, integrates with
SHAP TreeExplainer, and is what you'd actually deploy. Everything logs
to local MLflow: hyperparameters, per-class metrics, confusion matrix,
and the model artifact.

### Step 5: SHAP Explainability (`explain.py`)

TreeExplainer is fast for tree models. Generates beeswarm and bar plots
showing which features drive predictions. This is critical for the Traackr
use case — brands need to understand WHY an influencer is predicted to
perform well, not just the tier.

### Step 6: Drift Monitoring (`drift.py`)

Evidently compares training vs future data distributions using PSI.
The injected drift in the future slice should trigger detection on
follower_count, user_age, and topic distributions. The HTML report
is embedded in the Streamlit dashboard.

### Step 7: Alert (`alert.py`)

Simple threshold check on drift scores. PSI > 0.2 = warning,
PSI > 0.25 = critical. Writes an alert JSON that the dashboard reads.
In production, this would fire a Slack or PagerDuty notification.

### Step 8: Airflow DAG (`dag.py`)

Wires everything together with TaskFlow API. Tasks pass file paths via
XCom — never DataFrames. SHAP and drift monitoring run in parallel after
training since they're independent. The DAG file also works standalone
(`python dag.py`) for validation without Airflow installed.

### Step 9: Dashboard (`app.py`)

Four-tab Streamlit app that reads saved artifacts. No live computation —
everything is pre-computed by the pipeline. The Evidently HTML report
is embedded directly via `st.components.v1.html`.
