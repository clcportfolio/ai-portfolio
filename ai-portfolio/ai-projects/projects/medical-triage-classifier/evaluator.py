"""
evaluator.py — Medical Triage Classifier
Three-way comparison: baseline (pre-trained) vs fine-tuned (LoRA) vs Claude.
Computes accuracy, F1, per-class metrics, latency, and cost estimates.
Logs all results to MLflow.

Run:  python evaluator.py                          # full eval
      python evaluator.py --model-dir model_artifacts/final  # custom model path
      python evaluator.py --skip-claude             # skip Claude (saves API cost)
      python evaluator.py --dry-run                 # validate config only
"""

import argparse
import json
import logging
import os
import time

import mlflow
import numpy as np
import pandas as pd
import torch
from dotenv import find_dotenv, load_dotenv
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased"
LABEL_MAP = {"Routine": 0, "Urgent": 1, "Emergency": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)
MAX_SEQ_LENGTH = 512
EXPERIMENT_NAME = "medical-triage-classifier"

# Cost estimate: Claude Haiku per 1K input tokens (approximate)
CLAUDE_COST_PER_1K_INPUT = 0.001
CLAUDE_COST_PER_1K_OUTPUT = 0.005
AVG_TOKENS_PER_SAMPLE = 150  # rough estimate for clinical notes


def predict_baseline(texts: list[str], device: str = "cpu") -> tuple[list[str], float]:
    """
    Classify with vanilla pre-trained distilbert (no fine-tuning).
    Returns (predictions, avg_latency_ms).
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id=LABEL_MAP,
    )
    model.to(device)
    model.eval()

    predictions = []
    total_time = 0

    for text in texts:
        inputs = tokenizer(
            text, truncation=True, padding="max_length",
            max_length=MAX_SEQ_LENGTH, return_tensors="pt",
        ).to(device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        pred_id = torch.argmax(outputs.logits, dim=-1).item()
        predictions.append(ID_TO_LABEL[pred_id])

    avg_latency = (total_time / len(texts)) * 1000  # ms
    return predictions, avg_latency


def predict_finetuned(
    texts: list[str],
    model_dir: str,
    device: str = "cpu",
) -> tuple[list[str], float]:
    """
    Classify with LoRA fine-tuned model.
    Returns (predictions, avg_latency_ms).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load base model, then apply LoRA adapters
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id=LABEL_MAP,
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.to(device)
    model.eval()

    predictions = []
    total_time = 0

    for text in texts:
        inputs = tokenizer(
            text, truncation=True, padding="max_length",
            max_length=MAX_SEQ_LENGTH, return_tensors="pt",
        ).to(device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        pred_id = torch.argmax(outputs.logits, dim=-1).item()
        predictions.append(ID_TO_LABEL[pred_id])

    avg_latency = (total_time / len(texts)) * 1000  # ms
    return predictions, avg_latency


def predict_claude(texts: list[str]) -> tuple[list[str], float]:
    """
    Classify with Claude via LangChain structured output.
    Returns (predictions, avg_latency_ms).
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field

    class TriageResult(BaseModel):
        urgency: str = Field(description="One of: Routine, Urgent, Emergency")
        confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence")

    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        max_tokens=128,
        temperature=0,
    )

    chain = (
        ChatPromptTemplate.from_messages([
            ("system",
             "You are a medical triage classifier. Classify the clinical note into "
             "exactly one urgency level:\n"
             "- Routine: scheduled care, wellness visits, stable chronic conditions\n"
             "- Urgent: needs timely intervention within hours/days\n"
             "- Emergency: acute, potentially life-threatening, needs immediate attention\n\n"
             "Respond with the urgency level and your confidence (0.0 to 1.0)."),
            ("human", "{text}"),
        ])
        | llm.with_structured_output(TriageResult)
    )

    predictions = []
    total_time = 0

    for text in texts:
        start = time.perf_counter()
        try:
            result = chain.invoke({"text": text[:2000]})
            urgency = result.urgency
            # Normalize to expected labels
            if urgency not in LABEL_MAP:
                urgency = "Routine"  # fallback
        except Exception as e:
            logger.warning("Claude classification failed: %s", e)
            urgency = "Routine"
        elapsed = time.perf_counter() - start
        total_time += elapsed
        predictions.append(urgency)

    avg_latency = (total_time / len(texts)) * 1000  # ms
    return predictions, avg_latency


def compute_cost_estimate(n_samples: int) -> dict:
    """Estimate cost of Claude classification for n_samples."""
    total_input_tokens = n_samples * AVG_TOKENS_PER_SAMPLE
    total_output_tokens = n_samples * 20  # short structured output
    input_cost = (total_input_tokens / 1000) * CLAUDE_COST_PER_1K_INPUT
    output_cost = (total_output_tokens / 1000) * CLAUDE_COST_PER_1K_OUTPUT
    return {
        "total_cost": round(input_cost + output_cost, 4),
        "cost_per_1000": round((input_cost + output_cost) / n_samples * 1000, 4),
        "input_tokens_est": total_input_tokens,
    }


def run(
    test_csv: str = "data/test.csv",
    model_dir: str = "model_artifacts/final",
    skip_claude: bool = False,
) -> dict:
    """
    Full evaluation pipeline. Runs 3-way comparison and logs to MLflow.

    Returns:
        dict with metrics for each classifier.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Evaluating on device: %s", device)

    # Load test data
    test_df = pd.read_csv(test_csv)
    texts = test_df["text"].tolist()
    true_labels = test_df["urgency"].tolist()
    logger.info("Loaded %d test samples.", len(texts))

    results = {}

    # 1. Baseline
    logger.info("Running baseline (pre-trained distilbert)...")
    baseline_preds, baseline_latency = predict_baseline(texts, device)
    baseline_acc = accuracy_score(true_labels, baseline_preds)
    baseline_f1 = f1_score(true_labels, baseline_preds, average="macro")
    results["baseline"] = {
        "accuracy": round(baseline_acc, 4),
        "f1_macro": round(baseline_f1, 4),
        "avg_latency_ms": round(baseline_latency, 2),
        "cost_per_1000": 0.0,
        "report": classification_report(true_labels, baseline_preds, output_dict=True),
    }
    logger.info("Baseline — acc: %.4f, F1: %.4f, latency: %.1fms",
                baseline_acc, baseline_f1, baseline_latency)

    # 2. Fine-tuned
    if os.path.exists(model_dir):
        logger.info("Running fine-tuned model from %s...", model_dir)
        ft_preds, ft_latency = predict_finetuned(texts, model_dir, device)
        ft_acc = accuracy_score(true_labels, ft_preds)
        ft_f1 = f1_score(true_labels, ft_preds, average="macro")
        results["finetuned"] = {
            "accuracy": round(ft_acc, 4),
            "f1_macro": round(ft_f1, 4),
            "avg_latency_ms": round(ft_latency, 2),
            "cost_per_1000": 0.0,
            "report": classification_report(true_labels, ft_preds, output_dict=True),
        }
        logger.info("Fine-tuned — acc: %.4f, F1: %.4f, latency: %.1fms",
                    ft_acc, ft_f1, ft_latency)
    else:
        logger.warning("Fine-tuned model not found at %s. Skipping.", model_dir)
        results["finetuned"] = None

    # 3. Claude
    if not skip_claude:
        logger.info("Running Claude classification (Haiku)...")
        claude_preds, claude_latency = predict_claude(texts)
        claude_acc = accuracy_score(true_labels, claude_preds)
        claude_f1 = f1_score(true_labels, claude_preds, average="macro")
        cost_est = compute_cost_estimate(len(texts))
        results["claude"] = {
            "accuracy": round(claude_acc, 4),
            "f1_macro": round(claude_f1, 4),
            "avg_latency_ms": round(claude_latency, 2),
            "cost_per_1000": cost_est["cost_per_1000"],
            "report": classification_report(true_labels, claude_preds, output_dict=True),
        }
        logger.info("Claude — acc: %.4f, F1: %.4f, latency: %.1fms",
                    claude_acc, claude_f1, claude_latency)
    else:
        logger.info("Skipping Claude evaluation.")
        results["claude"] = None

    # Log to MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    try:
        with mlflow.start_run(run_name="model_comparison"):
            mlflow.log_param("test_samples", len(texts))
            mlflow.log_param("device", device)
            for model_name, metrics in results.items():
                if metrics is None:
                    continue
                mlflow.log_metric(f"{model_name}_accuracy", metrics["accuracy"])
                mlflow.log_metric(f"{model_name}_f1_macro", metrics["f1_macro"])
                mlflow.log_metric(f"{model_name}_latency_ms", metrics["avg_latency_ms"])
                mlflow.log_metric(f"{model_name}_cost_per_1000", metrics["cost_per_1000"])
            logger.info("Logged comparison metrics to MLflow.")
    except Exception as e:
        logger.warning("MLflow logging failed (non-fatal): %s", e)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate triage classifiers (3-way comparison)")
    parser.add_argument("--test-csv", default="data/test.csv", help="Path to test CSV")
    parser.add_argument("--model-dir", default="model_artifacts/final",
                        help="Path to fine-tuned model directory")
    parser.add_argument("--skip-claude", action="store_true",
                        help="Skip Claude evaluation (saves API cost)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config only — no evaluation")
    args = parser.parse_args()

    if args.dry_run:
        print("=== Dry Run: validating evaluator config ===\n")
        print(f"Model: {MODEL_NAME}")
        print(f"Labels: {LABEL_MAP}")
        print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
        if os.path.exists(args.test_csv):
            df = pd.read_csv(args.test_csv)
            print(f"Test data: {len(df)} rows")
            print(f"Class distribution:\n{df['urgency'].value_counts().to_string()}")
        else:
            print(f"Test CSV not found at {args.test_csv}. Run data_prep.py first.")
        if os.path.exists(args.model_dir):
            print(f"Fine-tuned model found at {args.model_dir}")
        else:
            print(f"Fine-tuned model not found at {args.model_dir}. Run trainer.py first.")
        print("\nDry run passed.")
    else:
        results = run(
            test_csv=args.test_csv,
            model_dir=args.model_dir,
            skip_claude=args.skip_claude,
        )
        # Pretty print summary table
        print("\n=== Model Comparison Results ===\n")
        print(f"{'Model':<15} {'Accuracy':>10} {'F1 (macro)':>12} {'Latency (ms)':>14} {'Cost/1K':>10}")
        print("-" * 65)
        for name, m in results.items():
            if m is None:
                print(f"{name:<15} {'(skipped)':>10}")
                continue
            print(f"{name:<15} {m['accuracy']:>10.4f} {m['f1_macro']:>12.4f} "
                  f"{m['avg_latency_ms']:>14.1f} ${m['cost_per_1000']:>9.4f}")
