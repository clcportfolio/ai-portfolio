"""
data_prep.py — Medical Triage Classifier
Loads MTSamples CSV, maps medical specialties to urgency labels
(Routine/Urgent/Emergency), generates synthetic examples via Claude
to balance classes, splits into train/val/test, uploads to S3,
and logs dataset stats to MLflow.

Run:  python data_prep.py data/mtsamples.csv
      python data_prep.py data/mtsamples.csv --skip-synthetic  # skip Claude augmentation
      python data_prep.py --dry-run                            # validate logic only
"""

import argparse
import io
import json
import logging
import os
import sys
from collections import Counter

import mlflow
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)

# ── Urgency label mapping ────────────────────────────────────────────────────
# Maps MTSamples medical_specialty values to triage urgency levels.
# Mapping rationale:
#   Emergency — specialties that commonly handle acute, life-threatening presentations
#   Urgent    — specialties where timely intervention prevents deterioration
#   Routine   — scheduled care, wellness, elective procedures
SPECIALTY_TO_URGENCY = {
    # Emergency
    "Emergency Room - Reports": "Emergency",
    "Cardiovascular / Pulmonary": "Emergency",
    "Neurosurgery": "Emergency",
    "Surgery": "Emergency",
    "Neurology": "Emergency",
    # Urgent
    "Orthopedic": "Urgent",
    "Gastroenterology": "Urgent",
    "Urology": "Urgent",
    "Hematology - Oncology": "Urgent",
    "Nephrology": "Urgent",
    "Obstetrics / Gynecology": "Urgent",
    "Pain Management": "Urgent",
    "Pediatrics - Neonatal": "Urgent",
    "Pulmonary": "Urgent",
    "Infectious Disease": "Urgent",
    "Psychiatry / Psychology": "Urgent",
    # Routine
    "Consult - History and Phy.": "Routine",
    "General Medicine": "Routine",
    "Ophthalmology": "Routine",
    "Radiology": "Routine",
    "Dermatology": "Routine",
    "Dentistry": "Routine",
    "Physical Medicine - Rehab": "Routine",
    "Endocrinology": "Routine",
    "Allergy / Immunology": "Routine",
    "Chiropractic": "Routine",
    "Sleep Medicine": "Routine",
    "Rheumatology": "Routine",
    "Podiatry": "Routine",
    "ENT - Otolaryngology": "Routine",
    "Cosmetic / Plastic Surgery": "Routine",
    "Speech - Language": "Routine",
    "Autopsy": "Routine",
    "Bariatrics": "Routine",
    "Diets and Nutritions": "Routine",
    "Hospice - Palliative Care": "Routine",
    "IME-QME-Work Comp etc.": "Routine",
    "Lab Medicine - Pathology": "Routine",
    "Letters": "Routine",
    "Office Notes": "Routine",
    "SOAP / Chart / Progress Notes": "Routine",
    "Discharge Summary": "Routine",
}

# Target total per class after synthetic augmentation
TARGET_PER_CLASS = 300

# MLflow experiment name
EXPERIMENT_NAME = "medical-triage-classifier"


def _load_csv(csv_path: str) -> pd.DataFrame:
    """Load MTSamples CSV and return cleaned DataFrame with 'text' and 'specialty' columns."""
    df = pd.read_csv(csv_path)

    # MTSamples has columns like: description, medical_specialty, transcription, etc.
    # We use 'transcription' as the clinical text.
    text_col = None
    for candidate in ["transcription", "description", "keywords"]:
        if candidate in df.columns:
            text_col = candidate
            break

    if text_col is None:
        raise ValueError(
            f"CSV must have one of: transcription, description, keywords. "
            f"Found columns: {list(df.columns)}"
        )

    specialty_col = "medical_specialty"
    if specialty_col not in df.columns:
        raise ValueError(f"CSV must have '{specialty_col}' column.")

    df = df[[specialty_col, text_col]].copy()
    df.columns = ["specialty", "text"]
    df = df.dropna(subset=["text", "specialty"])
    df["text"] = df["text"].astype(str).str.strip()
    df["specialty"] = df["specialty"].astype(str).str.strip()
    df = df[df["text"].str.len() > 20]  # drop trivially short entries
    df["text"] = df["text"].str[:2000]  # truncate to model context limit

    return df.reset_index(drop=True)


def load_and_label(csv_path: str) -> pd.DataFrame:
    """Load MTSamples CSV, map specialties to urgency labels, drop unmapped rows.
    DEPRECATED: Use load_and_label_with_llm() for content-based labeling."""
    df = _load_csv(csv_path)

    df["urgency"] = df["specialty"].map(SPECIALTY_TO_URGENCY)
    unmapped = df["urgency"].isna().sum()
    if unmapped > 0:
        unmapped_specs = df[df["urgency"].isna()]["specialty"].unique()
        logger.warning(
            "Dropping %d rows with unmapped specialties: %s",
            unmapped, unmapped_specs[:10],
        )
    df = df.dropna(subset=["urgency"])
    df["source"] = "mtsamples"

    return df[["text", "urgency", "source"]].reset_index(drop=True)


def load_and_label_with_llm(csv_path: str, batch_size: int = 10) -> pd.DataFrame:
    """
    Load MTSamples CSV and label urgency using Claude Sonnet.

    Instead of mapping by specialty (a heuristic that mislabels ~30-40% of notes),
    Sonnet reads each clinical note and assigns urgency based on the actual content:
    vital signs, symptoms, acuity, and clinical presentation.

    Labels are assigned in batches to reduce API calls and cost.
    ~500 notes ÷ 10 per batch = ~50 API calls (~$0.30 total with Sonnet).

    Args:
        csv_path: Path to MTSamples CSV.
        batch_size: Number of notes to label per API call.

    Returns:
        DataFrame with columns: text, urgency, source
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field

    class UrgencyLabels(BaseModel):
        labels: list[str] = Field(
            description="List of urgency labels, one per note. "
            "Each must be exactly one of: Routine, Urgent, Emergency"
        )

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0,
    )

    chain = (
        ChatPromptTemplate.from_messages([
            ("system",
             "You are a clinical triage expert. For each clinical note, assign exactly "
             "one urgency label based on the CONTENT of the note — not the department "
             "or specialty it came from.\n\n"
             "Label definitions:\n"
             "- Routine: scheduled care, wellness visits, stable chronic conditions, "
             "follow-up appointments, medication refills, normal test results\n"
             "- Urgent: needs timely intervention within hours to days to prevent "
             "deterioration — infections requiring antibiotics, fractures, worsening "
             "symptoms, abnormal lab results needing follow-up\n"
             "- Emergency: acute, potentially life-threatening, needs immediate "
             "attention — chest pain, stroke symptoms, severe trauma, respiratory "
             "distress, active hemorrhage, loss of consciousness, overdose\n\n"
             "Read each note carefully. Base your decision on clinical severity, "
             "not on which department the note is from. A cardiology note about a "
             "stable follow-up is Routine. An orthopedic note about a compound "
             "fracture with hemorrhage is Emergency.\n\n"
             "Return exactly one label per note in the same order they were provided."),
            ("human",
             "Label the urgency for each of these {count} clinical notes:\n\n{notes}"),
        ])
        | llm.with_structured_output(UrgencyLabels)
    )

    df = _load_csv(csv_path)
    logger.info("Loaded %d notes from %s. Labeling with Sonnet...", len(df), csv_path)

    all_labels = []
    total = len(df)
    valid_labels = {"Routine", "Urgent", "Emergency"}

    for i in range(0, total, batch_size):
        batch_df = df.iloc[i:i + batch_size]
        batch_texts = batch_df["text"].tolist()

        # Format notes with indices for clarity
        notes_str = "\n\n".join(
            f"--- Note {j+1} ---\n{text[:1500]}"
            for j, text in enumerate(batch_texts)
        )

        try:
            result = chain.invoke({
                "count": len(batch_texts),
                "notes": notes_str,
            })

            batch_labels = result.labels

            # Validate label count matches batch size
            if len(batch_labels) != len(batch_texts):
                logger.warning(
                    "Batch %d-%d: expected %d labels, got %d. Padding with 'Routine'.",
                    i, i + len(batch_texts), len(batch_texts), len(batch_labels),
                )
                while len(batch_labels) < len(batch_texts):
                    batch_labels.append("Routine")
                batch_labels = batch_labels[:len(batch_texts)]

            # Validate each label
            for k, label in enumerate(batch_labels):
                if label not in valid_labels:
                    logger.warning("Invalid label '%s' at index %d, defaulting to Routine.", label, i + k)
                    batch_labels[k] = "Routine"

            all_labels.extend(batch_labels)

        except Exception as e:
            logger.error("Batch %d-%d failed: %s. Defaulting to Routine.", i, i + len(batch_texts), e)
            all_labels.extend(["Routine"] * len(batch_texts))

        # Progress logging
        labeled_so_far = len(all_labels)
        if labeled_so_far % 100 == 0 or labeled_so_far == total:
            logger.info("Labeled %d / %d (%.0f%%)", labeled_so_far, total, 100 * labeled_so_far / total)

    df["urgency"] = all_labels
    df["source"] = "mtsamples"

    # Log distribution
    counts = df["urgency"].value_counts()
    logger.info("LLM labeling complete. Distribution:\n%s", counts.to_string())

    return df[["text", "urgency", "source"]].reset_index(drop=True)


def generate_synthetic(
    class_counts: dict[str, int],
    target_per_class: int = TARGET_PER_CLASS,
) -> pd.DataFrame:
    """
    Generate synthetic clinical notes via Claude to balance classes.
    Only generates for classes below target_per_class.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field

    class SyntheticBatch(BaseModel):
        examples: list[str] = Field(
            description="List of synthetic clinical notes, each 50-200 words"
        )

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        temperature=0.7,
    )
    chain = (
        ChatPromptTemplate.from_messages([
            ("system",
             "You are a medical data augmentation assistant. Generate realistic but "
             "fully fictional clinical notes for training a triage classifier. "
             "Each note should read like a brief chief complaint or clinical summary "
             "that a triage nurse would write. Do NOT include real patient names or "
             "identifiers. Vary the writing style, length (50-200 words), and clinical "
             "presentation across examples."),
            ("human",
             "Generate {count} clinical notes that would be classified as {urgency} urgency.\n\n"
             "Urgency definitions:\n"
             "- Routine: scheduled care, wellness visits, stable chronic conditions\n"
             "- Urgent: needs timely intervention within hours/days to prevent deterioration\n"
             "- Emergency: acute, potentially life-threatening, needs immediate attention\n\n"
             "Return exactly {count} examples."),
        ])
        | llm.with_structured_output(SyntheticBatch)
    )

    all_synthetic = []
    for urgency, current_count in class_counts.items():
        needed = max(0, target_per_class - current_count)
        if needed == 0:
            logger.info("Class '%s' already has %d >= %d, skipping.", urgency, current_count, target_per_class)
            continue

        logger.info("Generating %d synthetic examples for '%s'...", needed, urgency)
        # Generate in batches of 25
        generated = 0
        while generated < needed:
            batch_size = min(25, needed - generated)
            try:
                result = chain.invoke({"count": batch_size, "urgency": urgency})
                for text in result.examples:
                    text = text.strip()
                    if len(text) > 20:
                        all_synthetic.append({
                            "text": text[:2000],
                            "urgency": urgency,
                            "source": "synthetic",
                        })
                        generated += 1
            except Exception as e:
                logger.error("Synthetic generation failed for %s: %s", urgency, e)
                break

    return pd.DataFrame(all_synthetic) if all_synthetic else pd.DataFrame(columns=["text", "urgency", "source"])


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split (70/15/15)."""
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["urgency"], random_state=42,
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["urgency"], random_state=42,
    )
    return train_df, val_df, test_df


def upload_to_s3(df: pd.DataFrame, s3_key: str) -> None:
    """Upload a DataFrame as CSV to S3."""
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    bucket = os.getenv("S3_BUCKET_NAME", "")
    if not bucket:
        logger.warning("S3_BUCKET_NAME not set — skipping S3 upload for %s", s3_key)
        return

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    client = boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    try:
        client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=csv_bytes,
            ContentType="text/csv",
        )
        logger.info("Uploaded %d rows to s3://%s/%s", len(df), bucket, s3_key)
    except (BotoCoreError, ClientError) as e:
        logger.error("S3 upload failed for %s: %s", s3_key, e)
        raise RuntimeError(f"S3 upload failed: {e}") from e


def log_to_mlflow(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
) -> None:
    """Log dataset statistics to MLflow."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="data_prep"):
        # Overall stats
        mlflow.log_param("total_samples", len(full_df))
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("val_samples", len(val_df))
        mlflow.log_param("test_samples", len(test_df))

        # Source breakdown
        source_counts = full_df["source"].value_counts().to_dict()
        for source, count in source_counts.items():
            mlflow.log_param(f"source_{source}", count)
        mlflow.log_param(
            "synthetic_ratio",
            round(source_counts.get("synthetic", 0) / len(full_df), 3),
        )

        # Class distribution
        class_counts = full_df["urgency"].value_counts().to_dict()
        for urgency, count in class_counts.items():
            mlflow.log_metric(f"class_{urgency.lower()}", count)

        # Log class balance info
        mlflow.log_param("class_distribution", json.dumps(class_counts))

        logger.info("Logged dataset stats to MLflow experiment '%s'.", EXPERIMENT_NAME)


def run(
    csv_path: str,
    skip_synthetic: bool = False,
    skip_s3: bool = False,
    use_specialty_labels: bool = False,
    llm_batch_size: int = 10,
    output_dir: str = "data",
) -> dict:
    """
    Full data preparation pipeline.

    Args:
        csv_path: Path to MTSamples CSV.
        skip_synthetic: If True, skip Claude synthetic generation.
        skip_s3: If True, skip S3 upload (save locally only).
        use_specialty_labels: If True, use old specialty-based mapping instead of LLM.
        llm_batch_size: Notes per API call for LLM labeling (default 10).
        output_dir: Local directory to save processed splits.

    Returns:
        dict with dataset statistics.
    """
    # Step 1: Load and label
    logger.info("Loading CSV from %s...", csv_path)
    if use_specialty_labels:
        logger.info("Using specialty-based labeling (heuristic).")
        df = load_and_label(csv_path)
    else:
        logger.info("Using Sonnet content-based labeling.")
        df = load_and_label_with_llm(csv_path, batch_size=llm_batch_size)
    class_counts = df["urgency"].value_counts().to_dict()
    logger.info("Loaded %d labeled samples. Distribution: %s", len(df), class_counts)

    # Step 2: Synthetic augmentation
    if not skip_synthetic:
        synthetic_df = generate_synthetic(class_counts)
        if len(synthetic_df) > 0:
            logger.info("Generated %d synthetic samples.", len(synthetic_df))
            df = pd.concat([df, synthetic_df], ignore_index=True)
    else:
        logger.info("Skipping synthetic generation.")

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    final_counts = df["urgency"].value_counts().to_dict()
    logger.info("Final dataset: %d samples. Distribution: %s", len(df), final_counts)

    # Step 3: Split
    train_df, val_df, test_df = split_dataset(df)
    logger.info("Split: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))

    # Step 4: Save locally
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    df.to_csv(os.path.join(output_dir, "full.csv"), index=False)
    logger.info("Saved splits to %s/", output_dir)

    # Step 5: Upload to S3
    if not skip_s3:
        s3_prefix = "medical-triage-classifier/datasets"
        upload_to_s3(train_df, f"{s3_prefix}/train.csv")
        upload_to_s3(val_df, f"{s3_prefix}/val.csv")
        upload_to_s3(test_df, f"{s3_prefix}/test.csv")
    else:
        logger.info("Skipping S3 upload.")

    # Step 6: Log to MLflow
    try:
        log_to_mlflow(train_df, val_df, test_df, df)
    except Exception as e:
        logger.warning("MLflow logging failed (non-fatal): %s", e)

    stats = {
        "total": len(df),
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "class_distribution": final_counts,
        "source_breakdown": df["source"].value_counts().to_dict(),
    }
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Prepare medical triage dataset")
    parser.add_argument("csv_path", nargs="?", default="data/mtsamples.csv",
                        help="Path to MTSamples CSV (default: data/mtsamples.csv)")
    parser.add_argument("--skip-synthetic", action="store_true",
                        help="Skip Claude synthetic data generation")
    parser.add_argument("--skip-s3", action="store_true",
                        help="Skip S3 upload (save locally only)")
    parser.add_argument("--use-specialty-labels", action="store_true",
                        help="Use old specialty-based mapping instead of Sonnet labeling")
    parser.add_argument("--llm-batch-size", type=int, default=10,
                        help="Notes per Sonnet API call for labeling (default: 10)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate CSV loading only, no generation or upload")
    parser.add_argument("--output-dir", default="data",
                        help="Local output directory (default: data)")
    args = parser.parse_args()

    if args.dry_run:
        print("=== Dry Run: validating CSV load and labeling ===\n")
        if not os.path.exists(args.csv_path):
            print(f"CSV not found at {args.csv_path}. Place your MTSamples CSV there.")
            sys.exit(1)
        # Dry run uses specialty labels (free, no API call)
        df = load_and_label(args.csv_path)
        counts = df["urgency"].value_counts()
        print(f"Loaded {len(df)} labeled samples from {args.csv_path}")
        print(f"\nClass distribution (specialty-based):\n{counts.to_string()}")
        print(f"\nSample texts per class:")
        for urgency in ["Routine", "Urgent", "Emergency"]:
            subset = df[df["urgency"] == urgency]
            if len(subset) > 0:
                sample = subset.iloc[0]["text"][:100]
                print(f"  {urgency}: \"{sample}...\"")
        print("\nDry run complete. No LLM labeling, synthetic generation, or upload.")
        print("Run without --dry-run for Sonnet content-based labeling.")
    else:
        if not os.path.exists(args.csv_path):
            print(f"ERROR: CSV not found at {args.csv_path}")
            sys.exit(1)
        stats = run(
            args.csv_path,
            skip_synthetic=args.skip_synthetic,
            skip_s3=args.skip_s3,
            use_specialty_labels=args.use_specialty_labels,
            llm_batch_size=args.llm_batch_size,
            output_dir=args.output_dir,
        )
        print(f"\n=== Data Preparation Complete ===")
        print(json.dumps(stats, indent=2))
