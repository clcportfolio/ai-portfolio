"""
train.py — Influencer Engagement Pipeline
Trains multiple classifiers for engagement tier prediction and logs
each as a separate MLflow run for comparison.

Models: XGBoost, LightGBM, Random Forest
Predicts: high / medium / low engagement tier
Features: engineered from Social Media Engagement 2025 dataset

The best model (by F1 macro) is saved locally and registered to
MLflow Model Registry. Promotion to production is a manual step
(see commented-out code and docs).
"""

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
FEATURE_DIR = PROJECT_DIR / "data" / "features"
MODEL_DIR = PROJECT_DIR / "data" / "model"
MLFLOW_DIR = PROJECT_DIR / "data" / "mlflow"

# ── Model Configurations ────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "xgboost": {
        "class": xgb.XGBClassifier,
        "params": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "random_state": 42,
        },
        "fit_kwargs": lambda X_test, y_test: {
            "eval_set": [(X_test, y_test)],
            "verbose": False,
        },
        "log_fn": mlflow.xgboost.log_model,
    },
    "lightgbm": {
        "class": lgb.LGBMClassifier,
        "params": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "objective": "multiclass",
            "num_class": 3,
            "random_state": 42,
            "verbose": -1,
        },
        "fit_kwargs": lambda X_test, y_test: {
            "eval_set": [(X_test, y_test)],
        },
        "log_fn": mlflow.sklearn.log_model,
    },
    "random_forest": {
        "class": RandomForestClassifier,
        "params": {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        },
        "fit_kwargs": lambda X_test, y_test: {},
        "log_fn": mlflow.sklearn.log_model,
    },
}


# ── Training Logic ──────────────────────────────────────────────────────────


@mlflow.trace(name="train_single_model")
def _train_single_model(
    model_name: str,
    config: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: list[str],
) -> dict:
    """Train one model and log to MLflow. Returns metrics dict."""

    with mlflow.start_run(run_name=f"{model_name}-engagement-classifier") as run:
        # Log metadata
        mlflow.log_param("model_type", model_name)
        mlflow.log_params(config["params"])
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("feature_count", X_train.shape[1])

        # Train
        model = config["class"](**config["params"])
        fit_kwargs = config["fit_kwargs"](X_test, y_test)
        model.fit(X_train, y_train, **fit_kwargs)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)

        # Per-class metrics
        report = classification_report(
            y_test, y_pred,
            target_names=class_names,
            output_dict=True,
        )
        for cls_name in class_names:
            if cls_name in report:
                mlflow.log_metric(f"precision_{cls_name}", report[cls_name]["precision"])
                mlflow.log_metric(f"recall_{cls_name}", report[cls_name]["recall"])
                mlflow.log_metric(f"f1_{cls_name}", report[cls_name]["f1-score"])

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues")
        ax.set_title(f"{model_name} — Engagement Tier Confusion Matrix")
        cm_path = MODEL_DIR / f"confusion_matrix_{model_name}.png"
        fig.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(str(cm_path))

        # Log model to MLflow
        config["log_fn"](model, artifact_path="model")

        # Feature importance (tree-based models have this)
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            feature_names = list(X_train.columns)
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importance,
            }).sort_values("importance", ascending=False)
            imp_path = MODEL_DIR / f"feature_importance_{model_name}.csv"
            importance_df.to_csv(imp_path, index=False)
            mlflow.log_artifact(str(imp_path))

        run_id = run.info.run_id

    logger.info(
        "[%s] Accuracy: %.4f, F1 macro: %.4f, run: %s",
        model_name, accuracy, f1_macro, run_id,
    )

    return {
        "model_name": model_name,
        "model": model,
        "run_id": run_id,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "classification_report": report,
    }


@mlflow.trace(name="run_training_pipeline")
def run_training(
    feature_dir: str | None = None,
    mlflow_uri: str | None = None,
) -> dict:
    """
    Train all models, compare, and save the best one.
    Returns dict with best model info, all results, and data paths.
    """
    feature_dir = Path(feature_dir) if feature_dir else FEATURE_DIR
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Set up MLflow
    load_dotenv(PROJECT_DIR / ".env")
    mlflow_uri = mlflow_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("influencer-engagement-pipeline")

    # Load data
    train_df = pd.read_parquet(feature_dir / "train.parquet")
    test_df = pd.read_parquet(feature_dir / "test.parquet")

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # Load class names
    classes_path = feature_dir / "class_names.txt"
    class_names = (
        classes_path.read_text().strip().split("\n")
        if classes_path.exists()
        else ["high", "low", "medium"]
    )

    # ── Train all models ────────────────────────────────────────────────────

    all_results = []
    for model_name, config in MODEL_CONFIGS.items():
        logger.info("Training %s...", model_name)
        result = _train_single_model(
            model_name, config,
            X_train, y_train, X_test, y_test,
            class_names,
        )
        all_results.append(result)

    # ── Select best model by F1 macro ───────────────────────────────────────

    best = max(all_results, key=lambda r: r["f1_macro"])
    logger.info(
        "Best model: %s (F1 macro: %.4f)", best["model_name"], best["f1_macro"],
    )

    # Save best model locally
    best_model = best["model"]
    if isinstance(best_model, xgb.XGBClassifier):
        model_path = MODEL_DIR / "best_model.json"
        best_model.save_model(str(model_path))
    else:
        import joblib
        model_path = MODEL_DIR / "best_model.joblib"
        joblib.dump(best_model, model_path)

    # Save best model's classification report (for Streamlit)
    report_path = MODEL_DIR / "classification_report.json"
    report_path.write_text(json.dumps(best["classification_report"], indent=2))

    # Save best model's confusion matrix as the default
    best_cm_src = MODEL_DIR / f"confusion_matrix_{best['model_name']}.png"
    best_cm_dst = MODEL_DIR / "confusion_matrix.png"
    if best_cm_src.exists():
        import shutil
        shutil.copy2(best_cm_src, best_cm_dst)

    # Save best model's feature importance as default
    best_imp_src = MODEL_DIR / f"feature_importance_{best['model_name']}.csv"
    best_imp_dst = MODEL_DIR / "feature_importance.csv"
    if best_imp_src.exists():
        import shutil
        shutil.copy2(best_imp_src, best_imp_dst)

    # Save comparison summary (for Streamlit)
    comparison = []
    for r in all_results:
        comparison.append({
            "model": r["model_name"],
            "accuracy": round(r["accuracy"], 4),
            "f1_macro": round(r["f1_macro"], 4),
            "f1_weighted": round(r["f1_weighted"], 4),
            "run_id": r["run_id"],
            "is_best": r["model_name"] == best["model_name"],
        })
    comparison_path = MODEL_DIR / "model_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2))

    # ── Model Registry ──────────────────────────────────────────────────────
    #
    # Register the best model to MLflow Model Registry.
    # This creates a registered model entry that can be promoted through
    # stages: None → Staging → Production → Archived.
    #
    # In production, you would:
    #   1. Run this script — it auto-registers the best model
    #   2. Review the model in MLflow UI (localhost:5050)
    #   3. Compare metrics, SHAP plots, drift status
    #   4. Manually promote to Production via UI or the commands below
    #
    # To promote via CLI:
    #   from mlflow import MlflowClient
    #   client = MlflowClient()
    #   client.set_registered_model_alias(
    #       name="influencer-engagement-classifier",
    #       alias="production",
    #       version=<version_number>,
    #   )
    #
    # To load the production model:
    #   model = mlflow.pyfunc.load_model("models:/influencer-engagement-classifier@production")
    #

    model_uri = f"runs:/{best['run_id']}/model"
    registered = mlflow.register_model(
        model_uri=model_uri,
        name="influencer-engagement-classifier",
    )
    logger.info(
        "Registered model: %s, version: %s",
        registered.name, registered.version,
    )

    # Save registry info
    registry_info = {
        "registered_model_name": registered.name,
        "version": registered.version,
        "run_id": best["run_id"],
        "model_type": best["model_name"],
        "promotion_status": "None (promote manually via MLflow UI or CLI)",
    }
    registry_path = MODEL_DIR / "registry_info.json"
    registry_path.write_text(json.dumps(registry_info, indent=2))

    return {
        "model_path": str(model_path),
        "best_model": best["model_name"],
        "best_run_id": best["run_id"],
        "best_accuracy": best["accuracy"],
        "best_f1_macro": best["f1_macro"],
        "best_f1_weighted": best["f1_weighted"],
        "all_results": comparison,
        "registered_model_version": registered.version,
        "train_data_path": str(feature_dir / "train.parquet"),
        "test_data_path": str(feature_dir / "test.parquet"),
        "future_data_path": str(PROJECT_DIR / "data" / "delta" / "future"),
        "class_names": class_names,
        "classification_report": best["classification_report"],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train engagement classifiers")
    parser.add_argument("--feature-dir", type=str, default=None)
    parser.add_argument("--mlflow-uri", type=str, default=None)
    args = parser.parse_args()

    result = run_training(
        feature_dir=args.feature_dir,
        mlflow_uri=args.mlflow_uri,
    )

    print("\n=== Training Summary ===")
    print(f"\n  Model Comparison:")
    print(f"  {'Model':<16} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weight':>10} {'':>6}")
    print(f"  {'─'*16} {'─'*10} {'─'*10} {'─'*10} {'─'*6}")
    for r in result["all_results"]:
        marker = " ← best" if r["is_best"] else ""
        print(f"  {r['model']:<16} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f} {r['f1_weighted']:>10.4f}{marker}")

    print(f"\n  Best model:     {result['best_model']}")
    print(f"  MLflow run:     {result['best_run_id']}")
    print(f"  Registry ver:   {result['registered_model_version']}")
    print(f"  Model path:     {result['model_path']}")
    print(f"\n  To promote to production:")
    print(f"    mlflow models set-alias --name influencer-engagement-classifier --alias production --version {result['registered_model_version']}")
