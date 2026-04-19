"""
predict.py — Influencer Engagement Pipeline
Loads the production model from MLflow Model Registry and runs
inference on new data.

This is the "consuming" side of the pipeline — it demonstrates how
a downstream service (API, batch job, dashboard) would use the
registered model without knowing which algorithm won training.

Usage:
    # Single prediction from CLI
    python predict.py --followers 50000 --likes 1200 --comments 80 --shares 30

    # Batch prediction from CSV
    python predict.py --csv data/raw/engagement_future.csv

    # Load model programmatically
    from predict import load_production_model, predict_single
    model, class_names = load_production_model()
    result = predict_single(model, class_names, {...})
"""

import argparse
import json
import logging
import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from guardrails import validate_input, sanitize_output

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
FEATURE_DIR = PROJECT_DIR / "data" / "features"
MODEL_DIR = PROJECT_DIR / "data" / "model"

REGISTRY_MODEL_NAME = "influencer-engagement-classifier"
PRODUCTION_ALIAS = "production"


def _setup_mlflow() -> None:
    """Configure MLflow tracking URI from .env."""
    load_dotenv(PROJECT_DIR / ".env")
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
    mlflow.set_tracking_uri(uri)


def load_production_model() -> tuple:
    """
    Load the production model from MLflow Model Registry.

    Returns:
        (model, class_names) — the model as a pyfunc wrapper,
        and the list of class label names.

    This is the key pattern: the consuming code doesn't know or care
    whether the production model is XGBoost, LightGBM, or Random Forest.
    It just calls model.predict(). The registry alias decouples training
    from serving.
    """
    _setup_mlflow()

    model_uri = f"models:/{REGISTRY_MODEL_NAME}@{PRODUCTION_ALIAS}"
    logger.info("Loading model from: %s", model_uri)

    model = mlflow.pyfunc.load_model(model_uri)

    # Load class names
    classes_path = FEATURE_DIR / "class_names.txt"
    class_names = (
        classes_path.read_text().strip().split("\n")
        if classes_path.exists()
        else ["high", "low", "medium"]
    )

    # Load registry info for logging
    registry_path = MODEL_DIR / "registry_info.json"
    if registry_path.exists():
        info = json.loads(registry_path.read_text())
        logger.info(
            "Loaded %s v%s (%s) with alias '%s'",
            info.get("registered_model_name"),
            info.get("version"),
            info.get("model_type"),
            PRODUCTION_ALIAS,
        )

    return model, class_names


def predict_single(model, class_names: list[str], features: dict) -> dict:
    """
    Run prediction on a single sample.
    Applies guardrails on input and output.
    """
    # Validate input
    validate_input(features)

    # The model expects a DataFrame with the same columns as training.
    # For a single prediction, we need to match the feature schema.
    df = pd.DataFrame([features])

    # Predict
    prediction = model.predict(df)

    # Map numeric prediction back to class name
    pred_idx = int(prediction[0])
    tier = class_names[pred_idx] if pred_idx < len(class_names) else "unknown"

    result = {
        "engagement_tier": tier,
        "prediction_index": pred_idx,
        "model": REGISTRY_MODEL_NAME,
        "alias": PRODUCTION_ALIAS,
    }

    # Sanitize output
    result = sanitize_output(result)

    return result


def predict_batch(model, class_names: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """
    Run prediction on a batch DataFrame.
    Returns the DataFrame with an added 'predicted_tier' column.
    """
    predictions = model.predict(df)
    df = df.copy()
    df["predicted_tier"] = [
        class_names[int(p)] if int(p) < len(class_names) else "unknown"
        for p in predictions
    ]
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Run predictions using the production model from MLflow Registry",
    )
    parser.add_argument("--csv", type=str, help="Path to CSV for batch prediction")
    parser.add_argument("--followers", type=int, help="Follower count (single prediction)")
    parser.add_argument("--likes", type=int, help="Average likes")
    parser.add_argument("--comments", type=int, help="Average comments")
    parser.add_argument("--shares", type=int, help="Average shares")
    args = parser.parse_args()

    # Load production model
    model, class_names = load_production_model()

    if args.csv:
        # ── Batch prediction ────────────────────────────────────────────────
        logger.info("Running batch prediction on %s", args.csv)
        df = pd.read_parquet(args.csv) if args.csv.endswith(".parquet") else pd.read_csv(args.csv)

        # Drop non-feature columns if present
        drop_cols = ["engagement_tier", "predicted_tier", "target"]
        df_features = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        # Drop non-numeric columns (model expects numeric features)
        df_numeric = df_features.select_dtypes(include=[np.number])

        results = predict_batch(model, class_names, df_numeric)

        print(f"\n=== Batch Prediction Results ===")
        print(f"  Rows: {len(results)}")
        print(f"  Predicted distribution:")
        print(results["predicted_tier"].value_counts().to_string(header=False))

        # Save results
        out_path = PROJECT_DIR / "data" / "predictions.csv"
        results.to_csv(out_path, index=False)
        print(f"  Saved to: {out_path}")

    elif args.followers is not None:
        # ── Single prediction ───────────────────────────────────────────────
        # Build a minimal feature dict — in production you'd have all features
        features = {
            "followers_count": args.followers,
            "likes": args.likes or 0,
            "comments": args.comments or 0,
            "shares": args.shares or 0,
        }

        # This will likely fail because the model expects all 41 features.
        # That's intentional — it shows that single predictions need a
        # full feature vector, which is what feature_engineer.py produces.
        try:
            result = predict_single(model, class_names, features)
            print(f"\n=== Prediction ===")
            print(f"  Engagement tier: {result['engagement_tier']}")
            print(f"  Model: {result['model']}@{result['alias']}")
        except Exception as e:
            print(f"\n  Single prediction requires all {len(model.metadata.get_input_schema().inputs) if hasattr(model, 'metadata') else '41'} features.")
            print(f"  For single predictions, run the full feature engineering pipeline first.")
            print(f"  Error: {e}")
            print(f"\n  Use batch mode instead:")
            print(f"    python predict.py --csv data/features/test.parquet")

    else:
        # ── Demo: predict on test set ───────────────────────────────────────
        test_path = FEATURE_DIR / "test.parquet"
        if test_path.exists():
            logger.info("No args provided — running demo on test set")
            df = pd.read_parquet(test_path)
            y_true = df["target"]
            df_features = df.drop(columns=["target"])

            results = predict_batch(model, class_names, df_features)

            from sklearn.metrics import accuracy_score, f1_score
            y_pred_idx = [class_names.index(t) for t in results["predicted_tier"]]
            acc = accuracy_score(y_true, y_pred_idx)
            f1 = f1_score(y_true, y_pred_idx, average="macro")

            print(f"\n=== Demo: Test Set Predictions ===")
            print(f"  Rows:     {len(results)}")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  F1 Macro: {f1:.4f}")
            print(f"  Distribution:")
            print(results["predicted_tier"].value_counts().to_string(header=False))
            print(f"\n  Model loaded from: models:/{REGISTRY_MODEL_NAME}@{PRODUCTION_ALIAS}")
        else:
            print("No test data found. Run the pipeline first, or provide --csv.")
