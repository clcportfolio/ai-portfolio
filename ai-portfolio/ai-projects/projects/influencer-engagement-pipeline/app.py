"""
app.py — Influencer Engagement Pipeline
Streamlit dashboard showing model performance, SHAP explanations,
data drift monitoring, and pipeline overview.

Run with: streamlit run app.py
"""

import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = DATA_DIR / "model"
SHAP_DIR = DATA_DIR / "shap"
DRIFT_DIR = DATA_DIR / "drift"
FEATURE_DIR = DATA_DIR / "features"

st.set_page_config(
    page_title="Influencer Engagement Pipeline",
    page_icon="📊",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("Influencer Engagement Pipeline")
st.sidebar.markdown(
    """
    **ML pipeline for predicting social media engagement tiers**
    using XGBoost, orchestrated by Apache Airflow.

    Built as a portfolio piece demonstrating data engineering
    and MLOps skills relevant to influencer marketing platforms.

    ---
    **Tech Stack:**
    - Apache Airflow (TaskFlow API)
    - PySpark + Delta Lake
    - XGBoost + scikit-learn
    - MLflow experiment tracking
    - SHAP explainability
    - Evidently drift monitoring
    - Streamlit dashboard

    ---
    **Dataset:**
    [Social Media Engagement 2025](https://www.kaggle.com/datasets/dagaca/social-media-engagement-2025)
    (Kaggle, MIT License)
    """
)

# ── Helper functions ─────────────────────────────────────────────────────────


def load_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def load_image(path: Path):
    if path.exists():
        return str(path)
    return None


# ── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "Model Performance",
    "Feature Importance (SHAP)",
    "Data Drift",
    "Pipeline Overview",
])

# ── Tab 1: Model Performance ────────────────────────────────────────────────

with tab1:
    st.header("Model Performance")

    report = load_json(MODEL_DIR / "classification_report.json")
    comparison = load_json(MODEL_DIR / "model_comparison.json")
    registry_info = load_json(MODEL_DIR / "registry_info.json")

    if report is None:
        st.warning(
            "No model artifacts found. Run the pipeline first:\n"
            "```\npython prepare_data.py\npython ingest.py\n"
            "python feature_engineer.py\npython train.py\n```"
        )
    else:
        # Model comparison table
        if comparison:
            st.subheader("Model Comparison")
            comp_df = pd.DataFrame(comparison)
            comp_df["best"] = comp_df["is_best"].map({True: "***", False: ""})
            comp_df = comp_df[["model", "accuracy", "f1_macro", "f1_weighted", "best", "run_id"]]
            comp_df.columns = ["Model", "Accuracy", "F1 (Macro)", "F1 (Weighted)", "", "MLflow Run ID"]
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Bar chart comparison
            fig = px.bar(
                pd.DataFrame(comparison),
                x="model", y=["accuracy", "f1_macro", "f1_weighted"],
                barmode="group",
                title="Model Comparison",
                labels={"value": "Score", "model": "Model", "variable": "Metric"},
                color_discrete_sequence=["#2196F3", "#4CAF50", "#FF9800"],
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Registry info
        if registry_info:
            st.subheader("Model Registry")
            col1, col2, col3 = st.columns(3)
            col1.metric("Registered Model", registry_info.get("model_type", ""))
            col2.metric("Version", registry_info.get("version", ""))
            col3.metric("Status", registry_info.get("promotion_status", "").split(" (")[0])

        st.divider()

        # Best model details
        st.subheader("Best Model Details")

        # Overall metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{report.get('accuracy', 0):.4f}")
        col2.metric("F1 (Macro)", f"{report.get('macro avg', {}).get('f1-score', 0):.4f}")
        col3.metric("F1 (Weighted)", f"{report.get('weighted avg', {}).get('f1-score', 0):.4f}")

        # Per-class metrics table
        st.subheader("Per-Class Metrics")
        class_names_path = FEATURE_DIR / "class_names.txt"
        class_names = (
            class_names_path.read_text().strip().split("\n")
            if class_names_path.exists()
            else ["high", "low", "medium"]
        )

        rows = []
        for cls in class_names:
            if cls in report:
                rows.append({
                    "Class": cls.title(),
                    "Precision": f"{report[cls]['precision']:.3f}",
                    "Recall": f"{report[cls]['recall']:.3f}",
                    "F1-Score": f"{report[cls]['f1-score']:.3f}",
                    "Support": int(report[cls]["support"]),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Confusion matrix
        cm_path = MODEL_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.subheader("Confusion Matrix")
            st.image(str(cm_path), width=600)

        # Feature importance (native XGBoost)
        importance_path = MODEL_DIR / "feature_importance.csv"
        if importance_path.exists():
            st.subheader("XGBoost Feature Importance (Gain)")
            imp_df = pd.read_csv(importance_path).head(15)
            fig = px.bar(
                imp_df, x="importance", y="feature",
                orientation="h", title="Top 15 Features by XGBoost Importance",
                color="importance", color_continuous_scale="Blues",
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
            st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: SHAP Feature Importance ──────────────────────────────────────────

with tab2:
    st.header("SHAP Feature Importance")

    importance_csv = SHAP_DIR / "shap_importance.csv"

    if not importance_csv.exists():
        st.warning(
            "No SHAP artifacts found. Run:\n```\npython explain.py\n```"
        )
    else:
        # SHAP importance bar chart (interactive via plotly)
        shap_df = pd.read_csv(importance_csv).head(20)
        fig = px.bar(
            shap_df, x="mean_abs_shap", y="feature",
            orientation="h",
            title="Mean |SHAP Value| — Global Feature Importance",
            color="mean_abs_shap",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=600)
        st.plotly_chart(fig, use_container_width=True)

        # SHAP plots (saved PNGs)
        col1, col2 = st.columns(2)
        bar_path = SHAP_DIR / "shap_bar.png"
        beeswarm_path = SHAP_DIR / "shap_beeswarm.png"

        if bar_path.exists():
            with col1:
                st.subheader("SHAP Bar Plot")
                st.image(str(bar_path))

        if beeswarm_path.exists():
            with col2:
                st.subheader("SHAP Beeswarm Plot")
                st.image(str(beeswarm_path))

        # Top features narrative
        top_5 = shap_df.head(5)["feature"].tolist()
        st.subheader("Key Takeaways")
        st.markdown(
            f"""
            The top features driving engagement tier predictions are:

            1. **{top_5[0]}** — The strongest predictor of engagement tier
            2. **{top_5[1]}** — Second most influential feature
            3. **{top_5[2]}** — Third most influential feature
            4. **{top_5[3]}** — Contributes meaningful signal
            5. **{top_5[4]}** — Rounds out the top 5

            These features align with influencer marketing intuition:
            follower-engagement dynamics, content strategy metrics,
            and audience composition all play roles in determining
            whether an influencer achieves high engagement.
            """
        )


# ── Tab 3: Data Drift ───────────────────────────────────────────────────────

with tab3:
    st.header("Data Drift Monitoring")

    drift_summary = load_json(DRIFT_DIR / "drift_summary.json")
    alert_data = load_json(DRIFT_DIR / "alert.json")

    if drift_summary is None:
        st.warning(
            "No drift artifacts found. Run:\n```\npython drift.py\npython alert.py\n```"
        )
    else:
        # Alert status banner
        if alert_data:
            severity = alert_data.get("severity", "OK")
            color_map = {
                "CRITICAL": "🔴", "WARNING": "🟡",
                "INFO": "🔵", "OK": "🟢",
            }
            icon = color_map.get(severity, "⚪")
            st.markdown(f"### {icon} Alert Status: **{severity}**")
            st.info(alert_data.get("recommended_action", ""))

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Dataset Drift", "Yes" if drift_summary["dataset_drift"] else "No")
        col2.metric("Features Checked", drift_summary["total_features_checked"])
        col3.metric("Drifted Features", drift_summary["drifted_features_count"])

        # Per-feature drift table
        st.subheader("Per-Feature Drift Scores")
        per_feature = drift_summary.get("per_feature", {})
        if per_feature:
            drift_rows = []
            for feat, details in per_feature.items():
                drift_rows.append({
                    "Feature": feat,
                    "Drift Score": details["drift_score"],
                    "Drifted": "Yes" if details["is_drifted"] else "No",
                    "Severity": details["severity"].title(),
                    "Test": details["stat_test"],
                })
            drift_df = pd.DataFrame(drift_rows).sort_values("Drift Score", ascending=False)
            st.dataframe(drift_df, use_container_width=True, hide_index=True)

            # Drift score chart
            fig = px.bar(
                drift_df.head(15), x="Drift Score", y="Feature",
                orientation="h", title="Per-Feature Drift Scores",
                color="Severity",
                color_discrete_map={
                    "Ok": "#4CAF50", "Slight": "#FFC107",
                    "Warning": "#FF9800", "Critical": "#F44336",
                },
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
            fig.add_vline(x=0.2, line_dash="dash", line_color="red",
                         annotation_text="PSI=0.2 (moderate drift)")
            st.plotly_chart(fig, use_container_width=True)

        # Embedded Evidently HTML report
        html_path = DRIFT_DIR / "drift_report.html"
        if html_path.exists():
            with st.expander("Full Evidently Drift Report (HTML)", expanded=False):
                st.components.v1.html(
                    html_path.read_text(), height=800, scrolling=True,
                )


# ── Tab 4: Pipeline Overview ────────────────────────────────────────────────

with tab4:
    st.header("Pipeline Overview")

    st.markdown(
        """
        ### Airflow TaskFlow DAG

        ```
        prepare_data.py (run manually — data prep)
                |
        ┌───────────────── Airflow DAG ─────────────────┐
        │                                                │
        │   ingest_csv_to_delta     (PySpark → Delta)    │
        │           |                                    │
        │   engineer_features       (Feature eng.)       │
        │           |                                    │
        │   train_model             (XGBoost + MLflow)   │
        │          / \\                                   │
        │   explain   monitor_drift                      │
        │   (SHAP)    (Evidently PSI)                    │
        │                  |                             │
        │             check_alert                        │
        │             (Retrain?)                         │
        │                                                │
        └────────────────────────────────────────────────┘
                |
        app.py (this dashboard — reads saved artifacts)
        ```

        ### Task Descriptions

        | Task | Module | Description |
        |---|---|---|
        | `ingest_csv_to_delta` | `ingest.py` | PySpark reads cleaned CSVs, validates schema, writes Delta Lake tables |
        | `engineer_features` | `feature_engineer.py` | Derives ML features (log_followers, follow_ratio, engagement density, etc.) |
        | `train_model` | `train.py` | XGBoost multi-class classifier, logs everything to MLflow |
        | `explain_model` | `explain.py` | SHAP TreeExplainer generates beeswarm and bar plots |
        | `monitor_drift` | `drift.py` | Evidently compares training vs future data distributions |
        | `check_alert` | `alert.py` | Evaluates drift severity, generates retraining recommendation |

        ### Setup Instructions

        ```bash
        # 1. Create and activate virtual environment
        python -m venv .venv && source .venv/bin/activate

        # 2. Install dependencies
        pip install -r requirements.txt

        # 3. Download dataset from Kaggle
        # Place CSV at: data/raw/social_media_engagement.csv

        # 4. Run data preparation
        python prepare_data.py

        # 5. Run pipeline steps individually
        python ingest.py
        python feature_engineer.py
        python train.py
        python explain.py
        python drift.py
        python alert.py

        # 6. Launch dashboard
        streamlit run app.py

        # 7. (Optional) Run via Airflow
        export AIRFLOW_HOME=./airflow_home
        airflow standalone
        # Symlink DAG: ln -s $(pwd)/dag.py $AIRFLOW_HOME/dags/dag.py
        # Trigger from Airflow UI at http://localhost:8080
        ```
        """
    )

    # Show data summary if available
    train_path = DATA_DIR / "raw" / "engagement_train.csv"
    if train_path.exists():
        with st.expander("Training Data Summary"):
            df = pd.read_csv(train_path, nrows=1000)
            st.write(f"**Shape:** {df.shape[0]:,} rows x {df.shape[1]} columns (showing first 1000)")
            st.dataframe(df.head(20), use_container_width=True, hide_index=True)

            if "engagement_tier" in df.columns:
                fig = px.histogram(
                    df, x="engagement_tier",
                    title="Engagement Tier Distribution",
                    color="engagement_tier",
                    color_discrete_map={
                        "high": "#4CAF50", "medium": "#FFC107", "low": "#F44336",
                    },
                )
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    print("Run this file with: streamlit run app.py")
