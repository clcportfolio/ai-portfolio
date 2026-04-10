"""
Streamlit UI — Medical Triage Classifier

Three-way comparison dashboard: baseline (pre-trained) vs fine-tuned (LoRA)
vs Claude (Haiku). Text input for clinical notes, side-by-side results,
metrics table, and MLflow/dataset details in expanders.

Run:  streamlit run app.py
"""

import json
import logging
import os
import time

import streamlit as st
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Triage Classifier",
    page_icon="🏥",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in [
    ("last_results", None),
    ("model_available", None),
    ("eval_metrics", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Medical Triage Classifier")
    st.markdown(
        "Fine-tunes DistilBERT with LoRA to classify clinical notes "
        "into **Routine**, **Urgent**, or **Emergency** urgency levels."
    )
    st.markdown("---")
    st.markdown("### Tech Stack")
    st.markdown("""
- **Model:** DistilBERT + PEFT/LoRA
- **Comparison:** Claude Haiku baseline
- **Tracking:** MLflow (local or EC2)
- **Storage:** AWS S3
- **Observability:** Langfuse
- **UI:** Streamlit
    """)
    st.markdown("---")

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    st.markdown(f"**MLflow UI:** [{mlflow_uri}]({mlflow_uri})")
    st.markdown("**GitHub:** [ai-portfolio](https://github.com/clcportfolio/ai-portfolio)")

    st.markdown("---")
    model_dir = st.text_input(
        "Model directory",
        value="model_artifacts/final",
        help="Path to fine-tuned LoRA model. Leave default if trained locally.",
    )


# ── Helper functions ──────────────────────────────────────────────────────────

def classify_text(text: str, model_dir: str) -> dict:
    """Run all three classifiers and return results."""
    from guardrails import validate_input, sanitize_output

    # Validate
    validate_input({"text": text})

    results = {}

    # 1. Fine-tuned model
    try:
        from classifier import run as classify_run
        result = classify_run({"text": text}, model_dir=model_dir)
        results["finetuned"] = sanitize_output(result)
    except Exception as e:
        results["finetuned"] = {"urgency": "N/A", "confidence": 0, "model": "unavailable", "error": str(e)}

    # 2. Baseline (pre-trained, no LoRA)
    try:
        import torch as _torch
        from transformers import AutoModelForSequenceClassification as _AutoModel
        from transformers import AutoTokenizer as _AutoTokenizer
        from evaluator import MODEL_NAME, LABEL_MAP, ID_TO_LABEL, NUM_LABELS, MAX_SEQ_LENGTH

        _tok = _AutoTokenizer.from_pretrained(MODEL_NAME)
        _base = _AutoModel.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS, id2label=ID_TO_LABEL, label2id=LABEL_MAP,
        )
        _base.eval()
        _inputs = _tok(text, truncation=True, padding="max_length",
                       max_length=MAX_SEQ_LENGTH, return_tensors="pt")
        _start = time.perf_counter()
        with _torch.no_grad():
            _out = _base(**_inputs)
        _elapsed = (time.perf_counter() - _start) * 1000
        _probs = _torch.softmax(_out.logits, dim=-1).squeeze()
        _pred_id = _torch.argmax(_probs).item()
        results["baseline"] = sanitize_output({
            "urgency": ID_TO_LABEL[_pred_id],
            "confidence": round(_probs[_pred_id].item(), 4),
            "model": "distilbert-base-uncased",
            "latency_ms": round(_elapsed, 2),
            "all_scores": {ID_TO_LABEL[i]: round(_probs[i].item(), 4) for i in range(NUM_LABELS)},
        })
    except Exception as e:
        results["baseline"] = {"urgency": "N/A", "confidence": 0, "model": "unavailable", "error": str(e)}

    # 3. Claude
    try:
        from langchain_anthropic import ChatAnthropic as _ChatAnthropic
        from langchain_core.prompts import ChatPromptTemplate as _ChatPromptTemplate
        from pydantic import BaseModel as _BaseModel, Field as _Field

        class _TriageResult(_BaseModel):
            urgency: str = _Field(description="One of: Routine, Urgent, Emergency")
            confidence: float = _Field(ge=0.0, le=1.0, description="Classification confidence")

        _llm = _ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=128, temperature=0)
        _chain = (
            _ChatPromptTemplate.from_messages([
                ("system",
                 "Classify the clinical note as exactly one of: Routine, Urgent, Emergency.\n"
                 "- Routine: scheduled care, stable chronic conditions\n"
                 "- Urgent: needs timely intervention within hours/days\n"
                 "- Emergency: acute, potentially life-threatening\n\n"
                 "Also provide your confidence from 0.0 to 1.0."),
                ("human", "{text}"),
            ])
            | _llm.with_structured_output(_TriageResult)
        )
        _start = time.perf_counter()
        _result = _chain.invoke({"text": text[:2000]})
        _elapsed = (time.perf_counter() - _start) * 1000
        results["claude"] = sanitize_output({
            "urgency": _result.urgency,
            "confidence": round(_result.confidence, 4),
            "model": "claude-haiku-4-5",
            "latency_ms": round(_elapsed, 2),
        })
    except Exception as e:
        results["claude"] = {"urgency": "N/A", "confidence": 0, "model": "unavailable", "error": str(e)}

    return results


def urgency_color(urgency: str) -> str:
    """Return color for urgency badge."""
    return {
        "Emergency": "#FF4B4B",
        "Urgent": "#FFA500",
        "Routine": "#4CAF50",
    }.get(urgency, "#888888")


def urgency_card(title: str, result: dict):
    """Render a classification result card."""
    urgency = result.get("urgency", "N/A")
    confidence = result.get("confidence", 0)
    latency = result.get("latency_ms", 0)
    color = urgency_color(urgency)
    error = result.get("error")

    st.markdown(f"**{title}**")
    if error:
        st.error(f"Model unavailable: {error}")
        return

    st.markdown(
        f'<div style="background-color:{color}22; border-left:4px solid {color}; '
        f'padding:12px; border-radius:4px; margin-bottom:8px;">'
        f'<span style="color:{color}; font-size:1.4em; font-weight:bold;">{urgency}</span><br>'
        f'<span style="color:#666;">Confidence: {confidence:.1%}</span><br>'
        f'<span style="color:#666;">Latency: {latency:.0f}ms</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Show all scores if available
    all_scores = result.get("all_scores")
    if all_scores:
        st.caption("Score breakdown:")
        for label, score in all_scores.items():
            st.progress(score, text=f"{label}: {score:.1%}")


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("Medical Triage Classifier")
st.markdown(
    "Enter a clinical note below to classify its urgency level. "
    "Compare results across three models: fine-tuned DistilBERT (LoRA), "
    "pre-trained baseline, and Claude Haiku."
)

# Sample texts
SAMPLES = {
    "Emergency — Chest pain": (
        "Patient presents to the emergency department with acute onset chest pain "
        "radiating to the left arm. Diaphoretic, BP 180/110, HR 120. ECG shows "
        "ST elevation in leads II, III, aVF."
    ),
    "Routine — Annual wellness": (
        "45-year-old female presents for annual wellness exam. No acute complaints. "
        "History of well-controlled hypertension on lisinopril 10mg daily. "
        "Last labs 6 months ago were within normal limits."
    ),
    "Urgent — Abdominal pain": (
        "Patient reports worsening abdominal pain over the past 3 days, localized "
        "to the right lower quadrant. Low-grade fever of 100.4F. Nausea without "
        "vomiting. Rebound tenderness on exam."
    ),
    "Custom": "",
}

sample_choice = st.selectbox("Choose a sample or enter custom text:", list(SAMPLES.keys()))
default_text = SAMPLES.get(sample_choice, "")

text_input = st.text_area(
    "Clinical Note",
    value=default_text,
    height=150,
    placeholder="Enter a clinical note, chief complaint, or patient summary...",
)

col_btn, col_info = st.columns([1, 3])
with col_btn:
    run_button = st.button("Classify", type="primary", use_container_width=True)
with col_info:
    st.caption("Runs all three classifiers and compares results.")

# ── Results ───────────────────────────────────────────────────────────────────

if run_button and text_input.strip():
    try:
        with st.spinner("Running classifiers..."):
            start = time.perf_counter()
            results = classify_text(text_input.strip(), model_dir)
            total_time = (time.perf_counter() - start) * 1000

        st.session_state["last_results"] = results

        st.success(f"Classification complete in {total_time:.0f}ms")

        # Three-column results
        col1, col2, col3 = st.columns(3)
        with col1:
            urgency_card("Fine-tuned (LoRA)", results.get("finetuned", {}))
        with col2:
            urgency_card("Baseline (pre-trained)", results.get("baseline", {}))
        with col3:
            urgency_card("Claude Haiku", results.get("claude", {}))

    except ValueError as e:
        st.error(f"Input validation failed: {e}")
    except Exception as e:
        st.error(f"Classification failed: {e}")
        logger.exception("Classification error")

elif run_button:
    st.warning("Please enter a clinical note before classifying.")

# ── Show previous results if available ────────────────────────────────────────
if not run_button and st.session_state["last_results"]:
    results = st.session_state["last_results"]
    st.markdown("### Previous Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        urgency_card("Fine-tuned (LoRA)", results.get("finetuned", {}))
    with col2:
        urgency_card("Baseline (pre-trained)", results.get("baseline", {}))
    with col3:
        urgency_card("Claude Haiku", results.get("claude", {}))

# ── Expanders ─────────────────────────────────────────────────────────────────
st.markdown("---")

with st.expander("Model Details & Hyperparameters"):
    st.markdown("""
| Parameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Fine-tuning | PEFT/LoRA |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA target modules | `q_lin`, `v_lin` |
| Max sequence length | 512 tokens |
| Training epochs | 5 |
| Learning rate | 3e-4 |
| Batch size | 16 |
| Optimizer | AdamW (weight decay 0.01) |
| Labels | Routine, Urgent, Emergency |
    """)

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    st.markdown(f"**MLflow Tracking UI:** [{mlflow_uri}]({mlflow_uri})")

with st.expander("Dataset Composition"):
    st.markdown("""
**Source:** MTSamples (mtsamples.com) — anonymized medical transcription samples.

**Labeling:** Medical specialties mapped to urgency levels:
- **Emergency:** Emergency Room, Cardiovascular/Pulmonary, Neurosurgery, Surgery, Neurology
- **Urgent:** Orthopedic, Gastroenterology, Urology, Hematology-Oncology, Nephrology, OB/GYN, etc.
- **Routine:** General Medicine, Ophthalmology, Radiology, Dermatology, Dentistry, etc.

**Augmentation:** Synthetic examples generated via Claude to balance classes (~300 per class target).

**Split:** 70% train / 15% validation / 15% test (stratified).
    """)

    # Try to load actual stats
    stats_path = "data/full.csv"
    if os.path.exists(stats_path):
        import pandas as pd
        df = pd.read_csv(stats_path)
        st.markdown("### Actual Dataset Stats")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Total Samples", len(df))
            source_counts = df["source"].value_counts()
            for source, count in source_counts.items():
                st.metric(f"Source: {source}", count)
        with col_b:
            class_counts = df["urgency"].value_counts()
            for urgency, count in class_counts.items():
                st.metric(f"Class: {urgency}", count)

with st.expander("Cost Comparison"):
    st.markdown("""
| Model | Inference Cost (per 1K) | Latency | Notes |
|---|---|---|---|
| Fine-tuned DistilBERT | **$0.00** | ~5-15ms (GPU) | One-time training cost only |
| Claude Haiku | ~$0.18 | ~300-500ms | Per-request API cost |
| Claude Sonnet | ~$1.80 | ~500-1000ms | Higher accuracy, 10x cost |

**Value proposition:** Fine-tuned model is ~100x cheaper and ~30x faster than Claude
for this specific classification task, with comparable accuracy on in-distribution data.
    """)

with st.expander("Integration with Clinical Intake Router"):
    st.markdown("""
This classifier is designed to replace the LLM-based `classification_agent` in
`projects/clinical-intake-router/`. The interface is compatible:

**Current (Claude-based):**
```python
# classification_agent.py uses ChatAnthropic for every classification
result = chain.invoke({"text": clinical_note})  # ~$0.003/call, ~400ms
```

**Proposed (fine-tuned):**
```python
from classifier import run
result = run({"text": clinical_note})  # $0/call, ~10ms
```

Same input shape (`{"text": str}`), same output shape
(`{"urgency": str, "confidence": float}`). Drop-in replacement.
    """)


if __name__ == "__main__":
    pass
