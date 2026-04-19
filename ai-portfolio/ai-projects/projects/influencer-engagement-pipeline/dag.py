"""
dag.py — Influencer Engagement Pipeline
Airflow TaskFlow DAG orchestrating the full ML pipeline:

    ingest_csv_to_delta
         |
    engineer_features
         |
    train_model
        / \\
  explain   monitor_drift
  (SHAP)        |
           check_alert

Uses @task decorators (TaskFlow API) for explicit data dependencies.
Tasks pass lightweight dicts (file paths, metrics) via XCom.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project directory to path for imports
PROJECT_DIR = Path(__file__).parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from airflow.decorators import dag, task

    @dag(
        dag_id="influencer_engagement_pipeline",
        schedule=None,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["ml", "influencer", "engagement", "traackr"],
        doc_md="""
        ## Influencer Engagement ML Pipeline

        End-to-end ML pipeline for predicting social media engagement tiers.

        **Tasks:**
        1. **Ingest** — PySpark CSV → Delta Lake
        2. **Feature Engineering** — Derive ML features from raw data
        3. **Train** — XGBoost classifier logged to MLflow
        4. **Explain** — SHAP feature importance analysis
        5. **Drift Monitor** — Evidently PSI drift detection
        6. **Alert** — Mock retraining alert based on drift severity
        """,
    )
    def influencer_engagement_pipeline():

        @task()
        def ingest_csv_to_delta() -> dict:
            """PySpark reads cleaned CSVs and writes Delta Lake format."""
            from ingest import run_ingestion
            return run_ingestion()

        @task()
        def engineer_features(ingest_result: dict) -> dict:
            """Reads Delta tables, engineers features, outputs train/test parquet."""
            from feature_engineer import run_feature_engineering
            return run_feature_engineering(
                delta_train_path=ingest_result["delta_train_path"],
            )

        @task()
        def train_model(feature_result: dict) -> dict:
            """Trains XGBoost classifier and logs to MLflow."""
            from train import run_training
            return run_training(
                feature_dir=feature_result["feature_path"],
            )

        @task()
        def explain_model(train_result: dict) -> dict:
            """SHAP TreeExplainer analysis on the trained model."""
            from explain import run_shap_analysis
            return run_shap_analysis(
                model_path=train_result["model_path"],
                test_data_path=train_result["test_data_path"],
            )

        @task()
        def monitor_drift(train_result: dict) -> dict:
            """Evidently PSI drift: training vs simulated future data."""
            from drift import run_drift_monitoring
            return run_drift_monitoring(
                train_data_path=train_result["train_data_path"],
                future_data_path=train_result["future_data_path"],
            )

        @task()
        def check_alert(drift_result: dict) -> dict:
            """Check drift severity and log retraining alert."""
            from alert import run_alert_check
            return run_alert_check(drift_result)

        # Wire the DAG
        ingested = ingest_csv_to_delta()
        features = engineer_features(ingested)
        trained = train_model(features)

        # SHAP and drift run in parallel after training
        shap_result = explain_model(trained)
        drift_result = monitor_drift(trained)

        # Alert depends on drift results
        alert_result = check_alert(drift_result)

    # Instantiate the DAG
    influencer_engagement_pipeline()

except ImportError:
    # Airflow not installed — allow standalone validation
    pass


if __name__ == "__main__":
    try:
        from airflow.decorators import dag as _dag_check
        # Re-import to get the DAG object
        dag_obj = influencer_engagement_pipeline()
        print(f"DAG ID: {dag_obj.dag_id}")
        print(f"Tasks:  {sorted([t.task_id for t in dag_obj.tasks])}")
        print(f"Tags:   {dag_obj.tags}")
        print("\nDAG parsed successfully.")
    except ImportError:
        print("Airflow not installed — DAG validation skipped.")
        print("Install with: pip install apache-airflow")
        print("\nDAG structure (static):")
        print("  ingest_csv_to_delta")
        print("       |")
        print("  engineer_features")
        print("       |")
        print("  train_model")
        print("      / \\")
        print("  explain   monitor_drift")
        print("  (SHAP)        |")
        print("           check_alert")
