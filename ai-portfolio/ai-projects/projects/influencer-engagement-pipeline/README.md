# Influencer Engagement Pipeline

Predicts social media engagement tiers (high/medium/low) using XGBoost, orchestrated by an Airflow TaskFlow DAG with PySpark ingestion, Delta Lake storage, MLflow tracking, SHAP explainability, and Evidently drift monitoring. Built as a portfolio piece for influencer marketing data platforms.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Download CSV from https://www.kaggle.com/datasets/dagaca/social-media-engagement-2025
# Place at: data/raw/social_media_engagement.csv
python prepare_data.py
python ingest.py
python feature_engineer.py
python train.py
python explain.py
python drift.py
python alert.py
streamlit run app.py
```

## What you'll see

A four-tab Streamlit dashboard: model performance metrics with confusion matrix, SHAP feature importance plots, Evidently drift monitoring with per-feature PSI scores and alert status, and a pipeline overview with setup instructions.

## How it works

```
prepare_data.py     Clean Kaggle CSV, derive target, create drift slice
       |
  Airflow DAG
       |
  ingest.py         PySpark CSV → Delta Lake
       |
  feature_engineer   Delta → engineered features → parquet
       |
  train.py          XGBoost classifier → MLflow
      / \
  explain  drift    SHAP plots | Evidently PSI report
              |
          alert     Mock retraining alert
       |
  app.py            Streamlit dashboard
```

## Tech stack

- Apache Airflow (TaskFlow API orchestration)
- PySpark + Delta Lake (ingestion and storage)
- XGBoost + scikit-learn (classification)
- MLflow (experiment tracking)
- SHAP (feature importance)
- Evidently (drift monitoring)
- Streamlit (dashboard)
- Dataset: [Social Media Engagement 2025](https://www.kaggle.com/datasets/dagaca/social-media-engagement-2025) (Kaggle, MIT)
