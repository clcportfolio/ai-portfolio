"""
explain.py — Influencer Engagement Pipeline
SHAP feature importance analysis for the trained XGBoost model.

Generates beeswarm and bar plots showing which features drive
engagement tier predictions — critical for explaining to brand
clients WHY an influencer is predicted to perform well.
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
MODEL_DIR = PROJECT_DIR / "data" / "model"
FEATURE_DIR = PROJECT_DIR / "data" / "features"
SHAP_DIR = PROJECT_DIR / "data" / "shap"


def run_shap_analysis(
    model_path: str | None = None,
    test_data_path: str | None = None,
) -> dict:
    """
    Run SHAP TreeExplainer on the trained XGBoost model.
    Returns dict with plot paths and top feature rankings.
    """
    # Try best_model first, fall back to xgb_model
    if model_path is None:
        if (MODEL_DIR / "best_model.joblib").exists():
            model_path = str(MODEL_DIR / "best_model.joblib")
        elif (MODEL_DIR / "best_model.json").exists():
            model_path = str(MODEL_DIR / "best_model.json")
        else:
            model_path = str(MODEL_DIR / "xgb_model.json")
    test_data_path = test_data_path or str(FEATURE_DIR / "test.parquet")
    SHAP_DIR.mkdir(parents=True, exist_ok=True)

    # Load best model (could be XGBoost, LightGBM, or RF)
    model_path_p = Path(model_path)
    if model_path_p.suffix == ".json":
        model = xgb.XGBClassifier()
        model.load_model(model_path)
    elif model_path_p.suffix == ".joblib":
        import joblib
        model = joblib.load(model_path)
    else:
        raise ValueError(f"Unknown model format: {model_path_p.suffix}")

    test_df = pd.read_parquet(test_data_path)
    X_test = test_df.drop(columns=["target"])

    # Load class names
    classes_path = FEATURE_DIR / "class_names.txt"
    class_names = classes_path.read_text().strip().split("\n") if classes_path.exists() else ["high", "low", "medium"]

    # SHAP TreeExplainer (fast for tree models)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For multi-class, shap_values is a list of arrays (one per class)
    # or a 3D array. Normalize to 3D.
    if isinstance(shap_values, list):
        shap_values_3d = np.array(shap_values)  # (n_classes, n_samples, n_features)
        shap_values_3d = shap_values_3d.transpose(1, 2, 0)  # (n_samples, n_features, n_classes)
    else:
        shap_values_3d = shap_values

    # Mean absolute SHAP values across all classes for global importance
    mean_abs_shap = np.mean(np.abs(shap_values_3d), axis=(0, 2))
    feature_names = list(X_test.columns)

    # Top features ranked by mean |SHAP|
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    top_5 = importance_df.head(5)["feature"].tolist()

    # Save importance CSV
    importance_path = SHAP_DIR / "shap_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    # Bar plot — mean |SHAP| values
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = importance_df.head(15)
    ax.barh(
        range(len(top_n)),
        top_n["mean_abs_shap"].values,
        color="#2196F3",
    )
    ax.set_yticks(range(len(top_n)))
    ax.set_yticklabels(top_n["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance — Mean |SHAP| (All Classes)")
    fig.tight_layout()

    bar_path = SHAP_DIR / "shap_bar.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved SHAP bar plot to %s", bar_path)

    # Beeswarm plot — per-class SHAP values
    # Use class index 0 (typically 'high') for a single-class beeswarm
    if isinstance(shap_values, list):
        single_class_shap = shap_values[0]
    else:
        single_class_shap = shap_values_3d[:, :, 0] if shap_values_3d.ndim == 3 else shap_values

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        single_class_shap,
        X_test,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.title(f"SHAP Beeswarm — '{class_names[0]}' Engagement Class")
    beeswarm_path = SHAP_DIR / "shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved SHAP beeswarm plot to %s", beeswarm_path)

    # Save raw SHAP values for Streamlit
    np.save(SHAP_DIR / "shap_values.npy", shap_values_3d)
    X_test.to_parquet(SHAP_DIR / "shap_test_data.parquet", index=False)

    result = {
        "bar_plot_path": str(bar_path),
        "beeswarm_plot_path": str(beeswarm_path),
        "importance_csv_path": str(importance_path),
        "top_5_features": top_5,
        "feature_count": len(feature_names),
    }

    logger.info("SHAP analysis complete. Top 5 features: %s", top_5)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="SHAP feature importance analysis")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--test-data", type=str, default=None)
    args = parser.parse_args()

    result = run_shap_analysis(
        model_path=args.model_path,
        test_data_path=args.test_data,
    )

    print("\n=== SHAP Analysis Summary ===")
    print(f"  Top 5 features: {result['top_5_features']}")
    print(f"  Bar plot:       {result['bar_plot_path']}")
    print(f"  Beeswarm plot:  {result['beeswarm_plot_path']}")
