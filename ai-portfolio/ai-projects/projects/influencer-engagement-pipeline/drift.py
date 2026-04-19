"""
drift.py — Influencer Engagement Pipeline
Evidently PSI (Population Stability Index) drift monitoring.

Compares training data distributions against a simulated future
data slice to detect feature drift that would degrade model
performance. In production, this would run on each new data batch.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
FEATURE_DIR = PROJECT_DIR / "data" / "features"
DELTA_DIR = PROJECT_DIR / "data" / "delta"
DRIFT_DIR = PROJECT_DIR / "data" / "drift"

# PSI thresholds (industry standard)
PSI_WARN = 0.1     # Slight drift — monitor
PSI_MODERATE = 0.2  # Moderate drift — investigate
PSI_SEVERE = 0.25   # Significant drift — retrain


def load_future_as_features(future_delta_path: str) -> pd.DataFrame:
    """
    Load future Delta table and apply the same feature engineering
    as the training data for apples-to-apples drift comparison.
    """
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("influencer-drift-monitoring")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )

    try:
        df = spark.read.format("delta").load(future_delta_path).toPandas()
        logger.info("Loaded %d future rows from %s", len(df), future_delta_path)
        return df
    finally:
        spark.stop()


def run_drift_monitoring(
    train_data_path: str | None = None,
    future_data_path: str | None = None,
) -> dict:
    """
    Run Evidently drift analysis comparing training vs future data.
    Returns dict with per-feature drift results and overall status.
    """
    train_data_path = train_data_path or str(FEATURE_DIR / "train.parquet")
    future_data_path = future_data_path or str(DELTA_DIR / "future")
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)

    # Load training data (features + target)
    train_df = pd.read_parquet(train_data_path)
    train_df = train_df.drop(columns=["target"], errors="ignore")

    # Load and prepare future data
    future_df = load_future_as_features(future_data_path)

    # Align columns: only compare features present in both datasets
    # The future data may have raw columns; we need numeric-compatible columns
    numeric_train = train_df.select_dtypes(include=["number"])
    common_cols = [c for c in numeric_train.columns if c in future_df.columns]

    if not common_cols:
        logger.warning("No common numeric columns between train and future data")
        # Fall back to comparing raw numeric columns
        common_cols = [
            c for c in future_df.select_dtypes(include=["number"]).columns
            if c in train_df.columns
        ]

    reference = train_df[common_cols].copy() if common_cols else numeric_train
    current = future_df[[c for c in common_cols if c in future_df.columns]].copy()

    # Ensure same columns
    shared = sorted(set(reference.columns) & set(current.columns))
    reference = reference[shared]
    current = current[shared]

    logger.info(
        "Comparing %d reference cols vs %d current cols (%d shared)",
        len(reference.columns), len(current.columns), len(shared),
    )

    # Run Evidently drift report
    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)

    # Save HTML report
    html_path = DRIFT_DIR / "drift_report.html"
    snapshot.save_html(str(html_path))
    logger.info("Saved drift report to %s", html_path)

    # Extract per-feature drift results from the report
    # Evidently 0.7+ uses a flat metrics list with ValueDrift per column
    report_dict = snapshot.dict()

    drift_results = {}
    dataset_drift = False
    drift_share = 0.5  # default threshold

    for metric in report_dict.get("metrics", []):
        metric_name = metric.get("metric_name", "")
        config = metric.get("config", {})
        value = metric.get("value")

        # Dataset-level drift count
        if "DriftedColumnsCount" in metric_name:
            share = value.get("share", 0) if isinstance(value, dict) else 0
            dataset_drift = share >= drift_share
            continue

        # Per-column drift (ValueDrift)
        if "ValueDrift" in metric_name:
            col_name = config.get("column", "unknown")
            method = config.get("method", "unknown")
            threshold = config.get("threshold", 0.05)
            # value is the p-value; drift detected if p-value < threshold
            p_value = float(value) if value is not None else 1.0
            is_drifted = p_value < threshold

            severity = "ok"
            if p_value < 0.001:
                severity = "critical"
            elif p_value < 0.01:
                severity = "warning"
            elif p_value < threshold:
                severity = "slight"

            drift_results[col_name] = {
                "drift_score": round(p_value, 6),
                "is_drifted": is_drifted,
                "stat_test": method,
                "severity": severity,
            }

    # Identify drifted features
    drifted_features = [
        col for col, data in drift_results.items() if data["is_drifted"]
    ]
    critical_features = [
        col for col, data in drift_results.items() if data["severity"] == "critical"
    ]

    # Save drift summary JSON
    summary = {
        "dataset_drift": dataset_drift,
        "total_features_checked": len(drift_results),
        "drifted_features_count": len(drifted_features),
        "drifted_features": drifted_features,
        "critical_features": critical_features,
        "per_feature": drift_results,
    }
    summary_path = DRIFT_DIR / "drift_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.info(
        "Drift analysis: %d/%d features drifted (%d critical)",
        len(drifted_features), len(drift_results), len(critical_features),
    )

    return {
        "html_report_path": str(html_path),
        "summary_path": str(summary_path),
        "dataset_drift": dataset_drift,
        "drifted_features": drifted_features,
        "critical_features": critical_features,
        "drift_results": drift_results,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Evidently PSI drift monitoring")
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--future-data", type=str, default=None)
    args = parser.parse_args()

    result = run_drift_monitoring(
        train_data_path=args.train_data,
        future_data_path=args.future_data,
    )

    print("\n=== Drift Monitoring Summary ===")
    print(f"  Dataset drift detected: {result['dataset_drift']}")
    print(f"  Drifted features: {result['drifted_features']}")
    print(f"  Critical features: {result['critical_features']}")
    print(f"  HTML report: {result['html_report_path']}")
