"""
prepare_data.py — Influencer Engagement Pipeline
Downloads the Social Media Engagement 2025 dataset from Kaggle,
cleans it, derives the engagement_tier classification target,
and creates a drift-shifted future slice for Evidently monitoring.

Data source: https://www.kaggle.com/datasets/dagaca/social-media-engagement-2025
License: MIT
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
KAGGLE_DATASET = "dagaca/social-media-engagement-2025"


def download_dataset(output_dir: Path) -> Path:
    """Download dataset from Kaggle using opendatasets (prompts for creds if needed)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    expected_csv = output_dir / "synthetic_social_media_engagement.csv"
    if expected_csv.exists():
        logger.info("Dataset already exists at %s", expected_csv)
        return expected_csv

    # Also check the opendatasets subdirectory pattern
    od_subdir = output_dir / "social-media-engagement-2025"
    od_csv_candidates = list(od_subdir.glob("*.csv")) if od_subdir.exists() else []

    if od_csv_candidates:
        src = od_csv_candidates[0]
        src.rename(expected_csv)
        logger.info("Moved %s -> %s", src, expected_csv)
        return expected_csv

    try:
        import opendatasets as od

        od.download(
            f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}",
            data_dir=str(output_dir),
        )
        # opendatasets creates a subdirectory with the dataset name
        od_csv_candidates = list(od_subdir.glob("*.csv")) if od_subdir.exists() else []
        if od_csv_candidates:
            src = od_csv_candidates[0]
            src.rename(expected_csv)
            return expected_csv
        else:
            raise FileNotFoundError(
                f"Download succeeded but no CSV found in {od_subdir}"
            )
    except ImportError:
        raise RuntimeError(
            "opendatasets not installed. Install with: pip install opendatasets\n"
            "Or manually download from:\n"
            f"  https://www.kaggle.com/datasets/{KAGGLE_DATASET}\n"
            f"and place the CSV at: {expected_csv}"
        )


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """Load raw CSV and clean for ML use."""
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), csv_path)

    # Standardize column names to snake_case
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Drop text columns not useful for tabular ML
    text_cols = [c for c in ["post_content", "hashtags", "user_name", "location"] if c in df.columns]
    df = df.drop(columns=text_cols, errors="ignore")

    # Drop ID columns
    id_cols = [c for c in ["post_id", "user_id"] if c in df.columns]
    df = df.drop(columns=id_cols, errors="ignore")

    # Parse dates
    for col in ["account_creation_date", "post_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Derive account_age_days if both date columns exist
    if "account_creation_date" in df.columns and "post_date" in df.columns:
        df["account_age_days"] = (df["post_date"] - df["account_creation_date"]).dt.days
        df["account_age_days"] = df["account_age_days"].clip(lower=0)
        df = df.drop(columns=["account_creation_date", "post_date"], errors="ignore")

    # Encode boolean columns
    if "is_verified" in df.columns:
        df["is_verified"] = df["is_verified"].astype(int)
    if "has_media" in df.columns:
        df["has_media"] = df["has_media"].astype(int)

    # Encode categoricals as strings (for later one-hot or label encoding)
    cat_cols = ["user_gender", "topic", "device", "language"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # Drop rows with null engagement_rate
    if "engagement_rate" in df.columns:
        before = len(df)
        df = df.dropna(subset=["engagement_rate"])
        dropped = before - len(df)
        if dropped > 0:
            logger.info("Dropped %d rows with null engagement_rate", dropped)

    # Fill remaining numeric nulls with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df


def derive_engagement_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive engagement_tier target from engagement_rate using quantile thresholds.
    Bottom 33% = low, middle 33% = medium, top 33% = high.
    """
    q_low = df["engagement_rate"].quantile(0.33)
    q_high = df["engagement_rate"].quantile(0.66)

    conditions = [
        df["engagement_rate"] <= q_low,
        df["engagement_rate"] <= q_high,
        df["engagement_rate"] > q_high,
    ]
    labels = ["low", "medium", "high"]
    df["engagement_tier"] = np.select(conditions, labels, default="medium")

    logger.info(
        "Engagement tier thresholds: low <= %.4f, medium <= %.4f, high > %.4f",
        q_low, q_high, q_high,
    )
    logger.info("Class distribution:\n%s", df["engagement_tier"].value_counts().to_string())
    return df


def create_drift_slice(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Create a future data slice with injected distribution shifts
    to simulate real-world drift for Evidently monitoring.

    Drift injections:
    - Oversample 'technology' and 'fashion' topics
    - Inflate followers_count by 1.5x
    - Shift user_age distribution younger (subtract 3 years)
    - Increase engagement_rate slightly (growth scenario)
    """
    rng = np.random.default_rng(seed)

    # Sample ~20% of the data as the base for the future slice
    n_future = int(len(df) * 0.2)
    future = df.sample(n=n_future, random_state=seed).copy()

    # Drift 1: Inflate follower_count by 1.5x with noise
    if "followers_count" in future.columns:
        noise = rng.normal(1.5, 0.3, size=len(future))
        noise = np.clip(noise, 0.8, 3.0)
        future["followers_count"] = (future["followers_count"] * noise).astype(int)

    # Drift 2: Shift user_age younger
    if "user_age" in future.columns:
        future["user_age"] = (future["user_age"] - rng.integers(2, 6, size=len(future))).clip(lower=16)

    # Drift 3: Oversample certain topics (simulate trend shift)
    if "topic" in future.columns:
        trending_topics = ["technology", "fashion"]
        trending_mask = future["topic"].isin(trending_topics)
        trending_rows = future[trending_mask]
        if len(trending_rows) > 0:
            extra = trending_rows.sample(n=min(len(trending_rows), n_future // 4), replace=True, random_state=seed)
            future = pd.concat([future, extra], ignore_index=True)

    # Drift 4: Bump engagement_rate slightly
    if "engagement_rate" in future.columns:
        future["engagement_rate"] = future["engagement_rate"] * rng.normal(1.15, 0.1, size=len(future))
        future["engagement_rate"] = future["engagement_rate"].clip(lower=0)

    # Drift 5: Shift device distribution (more mobile)
    if "device" in future.columns:
        mobile_devices = ["iphone", "android"]
        swap_mask = rng.random(size=len(future)) < 0.3
        future.loc[swap_mask, "device"] = rng.choice(mobile_devices, size=swap_mask.sum())

    # Re-derive engagement tier for future slice
    future = derive_engagement_tier(future)

    logger.info("Created drift slice: %d rows (from %d original)", len(future), len(df))
    return future


def run_prepare(csv_path: Path | None = None, output_dir: Path | None = None) -> dict:
    """
    Full data preparation pipeline.
    Returns dict with paths and summary stats for downstream tasks.
    """
    output_dir = output_dir or RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get the raw CSV
    if csv_path is None:
        csv_path = download_dataset(output_dir)

    # Step 2: Clean
    df = load_and_clean(csv_path)

    # Step 3: Derive target
    df = derive_engagement_tier(df)

    # Step 4: Create drift slice from the clean data
    future_df = create_drift_slice(df)

    # Step 5: Save
    train_path = output_dir / "engagement_train.csv"
    future_path = output_dir / "engagement_future.csv"
    df.to_csv(train_path, index=False)
    future_df.to_csv(future_path, index=False)

    summary = {
        "train_path": str(train_path),
        "future_path": str(future_path),
        "train_rows": len(df),
        "future_rows": len(future_df),
        "columns": list(df.columns),
        "target_distribution": df["engagement_tier"].value_counts().to_dict(),
    }

    logger.info("Saved train data (%d rows) to %s", len(df), train_path)
    logger.info("Saved future data (%d rows) to %s", len(future_df), future_path)

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Prepare Social Media Engagement dataset")
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to raw CSV (skips Kaggle download if provided)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for cleaned CSVs (default: data/raw/)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    summary = run_prepare(csv_path=csv_path, output_dir=output_dir)

    print("\n=== Data Preparation Summary ===")
    print(f"  Train rows:  {summary['train_rows']}")
    print(f"  Future rows: {summary['future_rows']}")
    print(f"  Columns:     {len(summary['columns'])}")
    print(f"  Target distribution: {summary['target_distribution']}")
    print(f"  Train path:  {summary['train_path']}")
    print(f"  Future path: {summary['future_path']}")
