# Medical Triage Classifier

Fine-tunes DistilBERT with LoRA to classify clinical notes into Routine / Urgent / Emergency urgency levels. Compares against a pre-trained baseline and Claude Haiku to demonstrate the cost and latency advantages of task-specific fine-tuning.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env   # add your API keys
streamlit run app.py
```

## What you'll see

A three-column dashboard comparing classification results across the fine-tuned model, the baseline, and Claude. Enter a clinical note (or pick a sample), click "Classify," and see urgency labels with confidence scores and latency. Expanders show model details, dataset composition, and cost comparison.

## How it works

```
MTSamples CSV
    │
    ▼
data_prep.py ── label + synthetic augment ──► train/val/test CSVs (local + S3)
    │
    ▼
trainer.py ──── LoRA fine-tuning ──────────► MLflow (metrics + model registry)
    │
    ▼
evaluator.py ── 3-way comparison ──────────► MLflow (comparison metrics)
    │
    ▼
classifier.py ─ inference interface ───────► app.py (Streamlit dashboard)
```

**Build order:**
1. `python provision_infra.py` — set up S3 bucket
2. `python data_prep.py data/mtsamples.csv` — prepare dataset
3. `python trainer.py` — train on Colab (GPU required)
4. `python evaluator.py` — compare all three models
5. `streamlit run app.py` — launch dashboard

## Tech stack

| Component | Technology |
|---|---|
| Base model | `distilbert-base-uncased` |
| Fine-tuning | PEFT / LoRA (r=16, alpha=32) |
| Training | HuggingFace Transformers + Trainer |
| Experiment tracking | MLflow (local or EC2) |
| Model registry | MLflow Model Registry |
| Artifact storage | AWS S3 |
| LLM comparison | Claude Haiku (LangChain) |
| Observability | Langfuse (inference tracing) |
| Demo UI | Streamlit |
| Data augmentation | Claude Haiku (synthetic generation) |
