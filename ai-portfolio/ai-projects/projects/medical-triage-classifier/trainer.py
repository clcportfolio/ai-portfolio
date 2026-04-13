"""
trainer.py — Medical Triage Classifier
Fine-tunes distilbert-base-uncased with PEFT/LoRA for 3-class urgency
classification (Routine / Urgent / Emergency).

Designed to run on Google Colab (free GPU) or any CUDA-capable machine.
Logs metrics to MLflow, saves checkpoints to S3, registers final model
to MLflow Model Registry.

Run:  python trainer.py                         # full training (needs GPU)
      python trainer.py --dry-run                # validate config only
      python trainer.py --local-only             # skip S3, local save only
      python trainer.py --epochs 3 --lr 2e-4     # custom hyperparams
"""

import argparse
import json
import logging
import os
import sys

import mlflow
import numpy as np
import pandas as pd
import torch
from dotenv import find_dotenv, load_dotenv
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
LABEL_MAP = {"Routine": 0, "Urgent": 1, "Emergency": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)
EXPERIMENT_NAME = "medical-triage-classifier"
REGISTERED_MODEL_NAME = "triage-classifier-distilbert-lora"

# Default hyperparameters
DEFAULT_EPOCHS = 5
DEFAULT_LR = 3e-4
DEFAULT_BATCH_SIZE = 16
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.1
MAX_SEQ_LENGTH = 512


class TriageDataset(torch.utils.data.Dataset):
    """Simple PyTorch dataset for tokenized clinical texts."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def load_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test splits from local CSV files."""
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    return train_df, val_df, test_df


def load_data_from_s3(s3_prefix: str = "medical-triage-classifier/datasets") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test splits from S3."""
    import io

    import boto3

    bucket = os.getenv("S3_BUCKET_NAME", "")
    if not bucket:
        raise ValueError("S3_BUCKET_NAME not set.")

    client = boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    dfs = []
    for split in ["train", "val", "test"]:
        key = f"{s3_prefix}/{split}.csv"
        response = client.get_object(Bucket=bucket, Key=key)
        csv_bytes = response["Body"].read()
        dfs.append(pd.read_csv(io.StringIO(csv_bytes.decode("utf-8"))))
        logger.info("Loaded %s from s3://%s/%s (%d rows)", split, bucket, key, len(dfs[-1]))

    return dfs[0], dfs[1], dfs[2]


def tokenize_data(
    tokenizer,
    df: pd.DataFrame,
) -> TriageDataset:
    """Tokenize a DataFrame and return a TriageDataset."""
    texts = df["text"].tolist()
    labels = df["urgency"].map(LABEL_MAP).tolist()

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )
    return TriageDataset(encodings, torch.tensor(labels, dtype=torch.long))


def compute_metrics(eval_pred):
    """Compute accuracy and macro F1 for HuggingFace Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from training data.
    Underrepresented classes get higher weight so the loss penalizes
    mistakes on them more heavily, preventing the model from collapsing
    to the majority class.
    """
    from sklearn.utils.class_weight import compute_class_weight

    labels = df["urgency"].map(LABEL_MAP).values
    weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=labels)
    logger.info("Class weights: %s", {ID_TO_LABEL[i]: round(w, 3) for i, w in enumerate(weights)})
    return torch.tensor(weights, dtype=torch.float32)


class WeightedTrainer(Trainer):
    """Custom Trainer that applies class weights to the cross-entropy loss."""

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def build_lora_model(
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
):
    """Load base model and apply LoRA adapters."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id=LABEL_MAP,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_lin", "v_lin"],  # DistilBERT attention layers
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA applied. Trainable: %d / %d (%.2f%%)",
        trainable_params, total_params, 100 * trainable_params / total_params,
    )
    return model, lora_config


def upload_checkpoint_to_s3(local_path: str, s3_key: str) -> None:
    """Upload a training checkpoint directory to S3."""
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    bucket = os.getenv("S3_BUCKET_NAME", "")
    if not bucket:
        logger.warning("S3_BUCKET_NAME not set — skipping checkpoint upload.")
        return

    client = boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    for root, _, files in os.walk(local_path):
        for fname in files:
            filepath = os.path.join(root, fname)
            relative = os.path.relpath(filepath, local_path)
            key = f"{s3_key}/{relative}"
            try:
                client.upload_file(filepath, bucket, key)
            except (BotoCoreError, ClientError) as e:
                logger.warning("Failed to upload %s: %s", key, e)

    logger.info("Uploaded checkpoint to s3://%s/%s", bucket, s3_key)


def run(
    data_dir: str = "data",
    from_s3: bool = False,
    local_only: bool = False,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    output_dir: str = "model_artifacts",
) -> dict:
    """
    Full training pipeline.

    Returns:
        dict with training results and MLflow run ID.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training on device: %s", device)

    # Load data
    if from_s3:
        train_df, val_df, test_df = load_data_from_s3()
    else:
        train_df, val_df, test_df = load_data(data_dir)
    logger.info("Data loaded: train=%d, val=%d, test=%d", len(train_df), len(val_df), len(test_df))

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = tokenize_data(tokenizer, train_df)
    val_dataset = tokenize_data(tokenizer, val_df)
    logger.info("Tokenization complete.")

    # Build model
    model, lora_config = build_lora_model(lora_r=lora_r, lora_alpha=lora_alpha)
    model.to(device)

    # MLflow setup
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=10,
        report_to=[],  # disable default reporters; we use MLflow directly
        fp16=torch.cuda.is_available(),
    )

    # Compute class weights from training data to prevent majority-class collapse
    class_weights = compute_class_weights(train_df)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train with MLflow logging
    with mlflow.start_run(run_name="lora_finetune") as mlflow_run:
        # Log hyperparams
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": DEFAULT_LORA_DROPOUT,
            "lora_target_modules": "q_lin,v_lin",
            "max_seq_length": MAX_SEQ_LENGTH,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "device": device,
        })

        # Train
        logger.info("Starting training: %d epochs, lr=%s, batch=%d", epochs, lr, batch_size)
        train_result = trainer.train()

        # Log final training metrics
        mlflow.log_metrics({
            "train_loss": train_result.training_loss,
            "train_runtime_seconds": train_result.metrics.get("train_runtime", 0),
        })

        # Evaluate on validation set
        eval_metrics = trainer.evaluate()
        mlflow.log_metrics({
            "val_accuracy": eval_metrics.get("eval_accuracy", 0),
            "val_f1_macro": eval_metrics.get("eval_f1_macro", 0),
            "val_loss": eval_metrics.get("eval_loss", 0),
        })
        logger.info("Validation — accuracy: %.4f, F1: %.4f",
                     eval_metrics.get("eval_accuracy", 0), eval_metrics.get("eval_f1_macro", 0))

        # Save model locally
        final_model_dir = os.path.join(output_dir, "final")
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        logger.info("Model saved to %s", final_model_dir)

        # Upload checkpoint to S3
        if not local_only:
            s3_key = f"medical-triage-classifier/models/lora-{mlflow_run.info.run_id}"
            try:
                upload_checkpoint_to_s3(final_model_dir, s3_key)
            except Exception as e:
                logger.warning("S3 checkpoint upload failed (non-fatal): %s", e)

        # Register model to MLflow Model Registry
        try:
            model_uri = f"runs:/{mlflow_run.info.run_id}/model"
            # Log the model artifact first
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=None,
                artifacts={"model_dir": final_model_dir},
            )
            # Register
            mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
            logger.info(
                "Registered model '%s' version %s",
                REGISTERED_MODEL_NAME, mv.version,
            )
            mlflow.log_param("registered_model_version", mv.version)
        except Exception as e:
            logger.warning("MLflow model registration failed (non-fatal): %s", e)

        run_id = mlflow_run.info.run_id

    return {
        "run_id": run_id,
        "val_accuracy": eval_metrics.get("eval_accuracy", 0),
        "val_f1_macro": eval_metrics.get("eval_f1_macro", 0),
        "train_loss": train_result.training_loss,
        "model_dir": final_model_dir,
        "device": device,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train medical triage classifier with LoRA")
    parser.add_argument("--data-dir", default="data", help="Local data directory")
    parser.add_argument("--from-s3", action="store_true", help="Load data from S3")
    parser.add_argument("--local-only", action="store_true", help="Skip S3 upload")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--output-dir", default="model_artifacts")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config and model init only — no training")
    args = parser.parse_args()

    if args.dry_run:
        print("=== Dry Run: validating trainer configuration ===\n")
        print(f"Model: {MODEL_NAME}")
        print(f"Labels: {LABEL_MAP}")
        print(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}")
        print(f"Training: epochs={args.epochs}, lr={args.lr}, batch={args.batch_size}")
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        print(f"MLFLOW_TRACKING_URI: {os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')}")

        # Verify tokenizer loads
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

        # Verify model + LoRA init
        print("Loading model + LoRA adapters...")
        model, lora_cfg = build_lora_model(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
        print(f"  Model loaded. LoRA target modules: {lora_cfg.target_modules}")
        model.print_trainable_parameters()

        # Verify data exists
        train_path = os.path.join(args.data_dir, "train.csv")
        if os.path.exists(train_path):
            df = pd.read_csv(train_path)
            print(f"\nTrain data: {len(df)} rows, columns: {list(df.columns)}")
        else:
            print(f"\nTrain data not found at {train_path}. Run data_prep.py first.")

        print("\nDry run passed. Config is valid.")
    else:
        result = run(
            data_dir=args.data_dir,
            from_s3=args.from_s3,
            local_only=args.local_only,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            output_dir=args.output_dir,
        )
        print(f"\n=== Training Complete ===")
        print(json.dumps(result, indent=2, default=str))
