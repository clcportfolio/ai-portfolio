"""
fetch_mtsamples.py — Medical Triage Classifier
Downloads the MTSamples dataset from HuggingFace Hub (harishnair04/mtsamples)
and saves it as a local CSV for data_prep.py to consume.

MTSamples is a collection of ~5000 anonymized medical transcription samples
across 40+ specialties. The HuggingFace copy avoids scraping the original site.

Run:  python fetch_mtsamples.py                   # save to data/mtsamples.csv
      python fetch_mtsamples.py --output custom.csv
      python fetch_mtsamples.py --preview          # print shape + sample rows only
"""

import argparse
import logging
import os
import sys

from datasets import load_dataset

logger = logging.getLogger(__name__)

# HuggingFace dataset identifier — public, no auth required
HF_DATASET = "harishnair04/mtsamples"

# Default output path (git-ignored via .gitignore)
DEFAULT_OUTPUT = os.path.join("data", "mtsamples.csv")


def fetch(output_path: str = DEFAULT_OUTPUT, preview_only: bool = False) -> dict:
    """
    Download MTSamples from HuggingFace and save as CSV.

    The raw HuggingFace dataset has an unnamed index column at position 0 —
    we drop it so the CSV starts with the actual data columns
    (description, medical_specialty, sample_name, transcription, keywords).

    Args:
        output_path: Where to write the CSV.
        preview_only: If True, print stats and samples but don't save.

    Returns:
        dict with row count, columns, and output path.
    """
    logger.info("Fetching %s from HuggingFace Hub...", HF_DATASET)
    ds = load_dataset(HF_DATASET)

    # Dataset has a single "train" split
    df = ds["train"].to_pandas()

    # Drop the unnamed index column (position 0) that the original CSV carried over
    if df.columns[0] == "" or df.columns[0].startswith("Unnamed"):
        df = df.iloc[:, 1:]
    logger.info("Loaded %d rows, %d columns: %s", len(df), len(df.columns), list(df.columns))

    # Basic sanity checks — data_prep.py expects these columns
    expected_cols = {"medical_specialty", "transcription"}
    missing = expected_cols - set(df.columns)
    if missing:
        logger.warning(
            "Expected columns %s not found. Available: %s",
            missing, list(df.columns),
        )

    # Preview mode — print stats and a few rows, don't save
    if preview_only:
        print(f"\nDataset: {HF_DATASET}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {list(df.columns)}\n")

        # Specialty distribution (top 15)
        if "medical_specialty" in df.columns:
            spec_counts = df["medical_specialty"].value_counts()
            print(f"Specialties ({len(spec_counts)} unique):")
            for spec, count in spec_counts.head(15).items():
                print(f"  {count:>4d}  {spec}")
            if len(spec_counts) > 15:
                print(f"  ... and {len(spec_counts) - 15} more\n")

        # Sample row
        print("Sample row:")
        row = df.iloc[0]
        for col in df.columns:
            val = str(row[col])[:100]
            print(f"  {col}: {val}{'...' if len(str(row[col])) > 100 else ''}")

        return {"rows": len(df), "columns": list(df.columns), "output_path": None}

    # Save to CSV
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)

    return {"rows": len(df), "columns": list(df.columns), "output_path": output_path}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Download MTSamples dataset from HuggingFace Hub",
    )
    parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Print dataset stats and sample rows without saving",
    )
    args = parser.parse_args()

    result = fetch(output_path=args.output, preview_only=args.preview)

    if result["output_path"]:
        size_mb = os.path.getsize(result["output_path"]) / (1024 * 1024)
        print(f"\nSaved {result['rows']} rows to {result['output_path']} ({size_mb:.1f} MB)")
        print(f"Columns: {result['columns']}")
        print(f"\nNext step: python data_prep.py {result['output_path']}")
