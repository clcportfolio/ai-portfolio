"""
feature_engineer.py — Influencer Engagement Pipeline
Reads Delta tables, engineers features relevant to influencer
marketing analytics, and outputs train/test parquet splits.

Feature design rationale:
- log_followers: Normalizes the heavy-tailed follower distribution
- follow_ratio: High ratio signals engagement bots or follow-for-follow
- content_engagement_density: Engagement normalized by content length
- is_micro_influencer: Micro-influencers (< 50K followers) have different
  engagement dynamics — a key Traackr segmentation
- account_maturity: Log-scaled account age captures diminishing returns
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
DELTA_DIR = PROJECT_DIR / "data" / "delta"
FEATURE_DIR = PROJECT_DIR / "data" / "features"

# Columns to drop before training (used for derivation only)
_DROP_COLS = ["engagement_rate"]

# Categorical columns to one-hot encode
_CAT_COLS = ["user_gender", "topic", "device", "language"]


def load_from_delta(delta_path: str) -> pd.DataFrame:
    """Load a Delta table into pandas via PySpark."""
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("influencer-feature-engineering")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )

    try:
        df = spark.read.format("delta").load(delta_path).toPandas()
        logger.info("Loaded %d rows from Delta: %s", len(df), delta_path)
        return df
    finally:
        spark.stop()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering transformations."""
    out = df.copy()

    # Log-transform follower count (heavy-tailed distribution)
    if "followers_count" in out.columns:
        out["log_followers"] = np.log1p(out["followers_count"])

    # Follow ratio: following / followers (capped at 10)
    if "followers_count" in out.columns and "following_count" in out.columns:
        out["follow_ratio"] = (
            out["following_count"] / out["followers_count"].clip(lower=1)
        ).clip(upper=10.0)

    # Content engagement density: (likes + comments) / content_length
    if all(c in out.columns for c in ["likes", "comments", "content_length"]):
        out["content_engagement_density"] = (
            (out["likes"] + out["comments"])
            / out["content_length"].clip(lower=1)
        )

    # Micro-influencer flag (< 50K followers)
    if "followers_count" in out.columns:
        out["is_micro_influencer"] = (out["followers_count"] < 50_000).astype(int)

    # Account maturity (log-scaled account age)
    if "account_age_days" in out.columns:
        out["account_maturity"] = np.log1p(out["account_age_days"])

    # Shares-to-likes ratio (virality signal)
    if "shares" in out.columns and "likes" in out.columns:
        out["share_to_like_ratio"] = (
            out["shares"] / out["likes"].clip(lower=1)
        ).clip(upper=5.0)

    # Comments-to-likes ratio (conversation depth)
    if "comments" in out.columns and "likes" in out.columns:
        out["comment_to_like_ratio"] = (
            out["comments"] / out["likes"].clip(lower=1)
        ).clip(upper=5.0)

    return out


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    cat_cols_present = [c for c in _CAT_COLS if c in df.columns]

    if cat_cols_present:
        df = pd.get_dummies(df, columns=cat_cols_present, drop_first=True, dtype=int)
        logger.info("One-hot encoded: %s", cat_cols_present)

    return df


def run_feature_engineering(
    delta_train_path: str | None = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Full feature engineering pipeline.
    Returns dict with output paths and feature metadata.
    """
    delta_train_path = delta_train_path or str(DELTA_DIR / "train")
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    # Load from Delta
    df = load_from_delta(delta_train_path)

    # Separate target before engineering
    target = df["engagement_tier"].copy()
    df = df.drop(columns=["engagement_tier"], errors="ignore")

    # Engineer features
    df = engineer_features(df)

    # Drop derivation-only columns
    df = df.drop(columns=[c for c in _DROP_COLS if c in df.columns], errors="ignore")

    # Encode categoricals
    df = encode_categoricals(df)

    # Encode target
    le = LabelEncoder()
    target_encoded = le.fit_transform(target)
    class_names = list(le.classes_)
    logger.info("Target classes: %s", class_names)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df, target_encoded, test_size=test_size, random_state=seed, stratify=target_encoded,
    )

    # Save
    train_path = FEATURE_DIR / "train.parquet"
    test_path = FEATURE_DIR / "test.parquet"

    X_train_out = X_train.copy()
    X_train_out["target"] = y_train
    X_train_out.to_parquet(train_path, index=False)

    X_test_out = X_test.copy()
    X_test_out["target"] = y_test
    X_test_out.to_parquet(test_path, index=False)

    # Save feature names for downstream use
    feature_names = [c for c in X_train.columns]
    feature_path = FEATURE_DIR / "feature_names.txt"
    feature_path.write_text("\n".join(feature_names))

    # Save label encoder classes
    classes_path = FEATURE_DIR / "class_names.txt"
    classes_path.write_text("\n".join(class_names))

    result = {
        "feature_path": str(FEATURE_DIR),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "feature_count": len(feature_names),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "class_names": class_names,
        "feature_names": feature_names,
    }

    logger.info(
        "Feature engineering complete: %d features, %d train / %d test rows",
        len(feature_names), len(X_train), len(X_test),
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Engineer features from Delta tables")
    parser.add_argument("--delta-path", type=str, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    result = run_feature_engineering(
        delta_train_path=args.delta_path,
        test_size=args.test_size,
    )

    print("\n=== Feature Engineering Summary ===")
    print(f"  Features:     {result['feature_count']}")
    print(f"  Train rows:   {result['train_rows']}")
    print(f"  Test rows:    {result['test_rows']}")
    print(f"  Classes:      {result['class_names']}")
    print(f"  Output dir:   {result['feature_path']}")
