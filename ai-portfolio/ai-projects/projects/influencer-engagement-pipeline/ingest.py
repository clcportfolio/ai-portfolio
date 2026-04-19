"""
ingest.py — Influencer Engagement Pipeline
PySpark CSV ingestion to Delta Lake format.

Reads cleaned CSVs from data/raw/, validates schema,
and writes Delta tables to data/delta/ for downstream consumption.
"""

import argparse
import logging
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
DELTA_DIR = PROJECT_DIR / "data" / "delta"

EXPECTED_SCHEMA = StructType([
    StructField("user_gender", StringType(), True),
    StructField("user_age", IntegerType(), True),
    StructField("followers_count", IntegerType(), True),
    StructField("following_count", IntegerType(), True),
    StructField("is_verified", IntegerType(), True),
    StructField("topic", StringType(), True),
    StructField("content_length", IntegerType(), True),
    StructField("has_media", IntegerType(), True),
    StructField("device", StringType(), True),
    StructField("language", StringType(), True),
    StructField("likes", IntegerType(), True),
    StructField("comments", IntegerType(), True),
    StructField("shares", IntegerType(), True),
    StructField("engagement_rate", DoubleType(), True),
    StructField("account_age_days", IntegerType(), True),
    StructField("engagement_tier", StringType(), True),
])


def get_spark() -> SparkSession:
    """Create a local SparkSession with Delta Lake support."""
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("influencer-engagement-ingest")
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config(
            "spark.jars.packages",
            "io.delta:delta-spark_2.12:3.1.0",
        )
        .config("spark.driver.memory", "2g")
        .config("spark.sql.warehouse.dir", str(PROJECT_DIR / "spark-warehouse"))
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def validate_dataframe(df, name: str) -> None:
    """Run basic validation checks on ingested DataFrame."""
    row_count = df.count()
    if row_count == 0:
        raise ValueError(f"{name}: DataFrame is empty")

    null_counts = {
        col: df.filter(F.col(col).isNull()).count()
        for col in df.columns
    }
    high_null_cols = {k: v for k, v in null_counts.items() if v > row_count * 0.5}
    if high_null_cols:
        logger.warning(
            "%s: Columns with >50%% nulls: %s",
            name, high_null_cols,
        )

    logger.info("%s: Validated %d rows, %d columns", name, row_count, len(df.columns))


def run_ingestion(
    train_csv: str | None = None,
    future_csv: str | None = None,
) -> dict:
    """
    Ingest CSVs into Delta format.
    Returns dict with Delta paths and row counts for downstream tasks.
    """
    train_csv = train_csv or str(RAW_DIR / "engagement_train.csv")
    future_csv = future_csv or str(RAW_DIR / "engagement_future.csv")

    DELTA_DIR.mkdir(parents=True, exist_ok=True)

    spark = get_spark()

    try:
        result = {}

        for name, csv_path, delta_subdir in [
            ("train", train_csv, "train"),
            ("future", future_csv, "future"),
        ]:
            logger.info("Ingesting %s from %s", name, csv_path)

            df = (
                spark.read
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(csv_path)
            )

            validate_dataframe(df, name)

            delta_path = str(DELTA_DIR / delta_subdir)
            (
                df.write
                .format("delta")
                .mode("overwrite")
                .save(delta_path)
            )

            row_count = df.count()
            result[f"delta_{name}_path"] = delta_path
            result[f"{name}_row_count"] = row_count
            logger.info(
                "Wrote %d rows to Delta: %s", row_count, delta_path,
            )

        return result

    finally:
        spark.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Ingest CSVs to Delta Lake")
    parser.add_argument("--train-csv", type=str, default=None)
    parser.add_argument("--future-csv", type=str, default=None)
    args = parser.parse_args()

    result = run_ingestion(
        train_csv=args.train_csv,
        future_csv=args.future_csv,
    )

    print("\n=== Ingestion Summary ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
