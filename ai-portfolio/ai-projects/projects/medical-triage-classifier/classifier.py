"""
classifier.py — Medical Triage Classifier
Production inference interface. Loads the fine-tuned LoRA model from MLflow
Model Registry (or local path fallback) and exposes run() for classification.

Langfuse @observe on run() for inference tracing.

Run:  python classifier.py
      python classifier.py --text "Patient presents with severe chest pain"
      python classifier.py --model-dir model_artifacts/final
"""

import argparse
import logging
import os
import time

import torch
from dotenv import find_dotenv, load_dotenv
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased"
LABEL_MAP = {"Routine": 0, "Urgent": 1, "Emergency": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)
MAX_SEQ_LENGTH = 512
REGISTERED_MODEL_NAME = "triage-classifier-distilbert-lora"

# Module-level cache for loaded model + tokenizer
_model = None
_tokenizer = None
_device = None


def _load_model(model_dir: str | None = None):
    """
    Load model from local path or MLflow Registry. Caches at module level.
    Priority: model_dir arg > MLflow Registry > error.
    """
    global _model, _tokenizer, _device

    if _model is not None:
        return

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    # Try explicit local path first
    if model_dir and os.path.exists(model_dir):
        logger.info("Loading model from local path: %s", model_dir)
        _tokenizer = AutoTokenizer.from_pretrained(model_dir)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            id2label=ID_TO_LABEL,
            label2id=LABEL_MAP,
        )
        _model = PeftModel.from_pretrained(base_model, model_dir)
        _model.to(_device)
        _model.eval()
        logger.info("Model loaded from %s on %s.", model_dir, _device)
        return

    # Try MLflow Model Registry
    try:
        import mlflow

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)

        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production", "None"])
        if not versions:
            raise ValueError(f"No versions found for model '{REGISTERED_MODEL_NAME}'")

        latest = versions[0]
        artifact_uri = latest.source
        logger.info("Loading model from MLflow Registry: %s (version %s)", artifact_uri, latest.version)

        # Download artifacts to local temp dir
        local_path = mlflow.artifacts.download_artifacts(artifact_uri)
        model_path = os.path.join(local_path, "model_dir") if os.path.exists(
            os.path.join(local_path, "model_dir")
        ) else local_path

        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            id2label=ID_TO_LABEL,
            label2id=LABEL_MAP,
        )
        _model = PeftModel.from_pretrained(base_model, model_path)
        _model.to(_device)
        _model.eval()
        logger.info("Model loaded from MLflow Registry on %s.", _device)
        return

    except Exception as e:
        logger.warning("MLflow Registry load failed: %s", e)

    raise RuntimeError(
        "No model available. Provide --model-dir or register a model in MLflow. "
        "Run trainer.py first."
    )


def _classify_with_model(text: str) -> dict:
    """Classify text using the loaded fine-tuned model."""
    inputs = _tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        outputs = _model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1).squeeze()
    pred_id = torch.argmax(probs).item()
    confidence = probs[pred_id].item()

    return {
        "urgency": ID_TO_LABEL[pred_id],
        "confidence": round(confidence, 4),
        "model": "distilbert-lora-finetuned",
        "all_scores": {ID_TO_LABEL[i]: round(probs[i].item(), 4) for i in range(NUM_LABELS)},
    }


def _classify_with_claude(text: str) -> dict:
    """Fallback classification using Claude."""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    from pydantic import BaseModel, Field

    class TriageResult(BaseModel):
        urgency: str = Field(description="One of: Routine, Urgent, Emergency")
        confidence: float = Field(ge=0.0, le=1.0)

    llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=128, temperature=0)
    chain = (
        ChatPromptTemplate.from_messages([
            ("system",
             "Classify the clinical note as exactly one of: Routine, Urgent, Emergency.\n"
             "- Routine: scheduled care, stable chronic conditions\n"
             "- Urgent: needs timely intervention within hours/days\n"
             "- Emergency: acute, potentially life-threatening"),
            ("human", "{text}"),
        ])
        | llm.with_structured_output(TriageResult)
    )

    result = chain.invoke({"text": text[:2000]})
    return {
        "urgency": result.urgency,
        "confidence": round(result.confidence, 4),
        "model": "claude-haiku-fallback",
    }


def run(data: dict, model_dir: str | None = None, use_langfuse: bool = True) -> dict:
    """
    Classify clinical text into urgency level.

    Args:
        data: {"text": str} — clinical note to classify
        model_dir: Optional path to fine-tuned model directory
        use_langfuse: Whether to trace with Langfuse (default True)

    Returns:
        {"urgency": str, "confidence": float, "model": str, "latency_ms": float}
    """
    text = data.get("text", "")
    if not text:
        raise ValueError("Input must contain 'text' key with non-empty string.")

    start = time.perf_counter()

    # Try fine-tuned model first, fall back to Claude
    try:
        _load_model(model_dir)
        result = _classify_with_model(text)
    except RuntimeError:
        logger.info("Fine-tuned model unavailable. Falling back to Claude.")
        result = _classify_with_claude(text)

    elapsed = (time.perf_counter() - start) * 1000
    result["latency_ms"] = round(elapsed, 2)

    return result


# Langfuse-decorated version for pipeline use
try:
    from langfuse import observe

    @observe(name="triage_classify")
    def run_observed(data: dict, model_dir: str | None = None) -> dict:
        """Langfuse-traced wrapper around run()."""
        return run(data, model_dir=model_dir, use_langfuse=True)
except ImportError:
    run_observed = run


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Classify clinical text urgency")
    parser.add_argument("--text", type=str, default=None,
                        help="Clinical note text to classify")
    parser.add_argument("--model-dir", default="model_artifacts/final",
                        help="Path to fine-tuned model")
    args = parser.parse_args()

    sample_texts = [
        "Patient presents to the emergency department with acute onset chest pain "
        "radiating to the left arm. Diaphoretic, BP 180/110, HR 120. ECG shows "
        "ST elevation in leads II, III, aVF. Started on aspirin and heparin drip.",

        "45-year-old female presents for annual wellness exam. No acute complaints. "
        "History of well-controlled hypertension on lisinopril 10mg daily. "
        "Last labs 6 months ago were within normal limits.",

        "Patient reports worsening abdominal pain over the past 3 days, localized "
        "to the right lower quadrant. Low-grade fever of 100.4F. Nausea without "
        "vomiting. Rebound tenderness on exam. Concern for appendicitis.",
    ]

    text = args.text if args.text else None

    if text:
        print(f"\nClassifying: \"{text[:80]}...\"\n")
        result = run({"text": text}, model_dir=args.model_dir)
        print(json.dumps(result, indent=2))
    else:
        import json
        print("=== Medical Triage Classifier — Smoke Test ===\n")
        for i, sample in enumerate(sample_texts, 1):
            print(f"--- Sample {i} ---")
            print(f"Text: \"{sample[:80]}...\"")
            try:
                result = run({"text": sample}, model_dir=args.model_dir)
                print(f"Result: {json.dumps(result, indent=2)}\n")
            except Exception as e:
                print(f"Error: {e}\n")
